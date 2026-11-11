"""优云远程实例 SSH/SFTP 通道。

paramiko 惰性导入：后端仅在 provider=youyun 时才需要该依赖
（requirements-storage.txt 已声明）。凭据来自实例查询返回的
IP/端口/Base64 密码，绝不写日志。
"""
from __future__ import annotations

import logging
import posixpath
import stat as stat_module
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SSHCommandError(RuntimeError):
    def __init__(self, command: str, exit_code: int, output: str) -> None:
        super().__init__(f"远程命令失败（exit={exit_code}）：{command[:120]}")
        self.command = command
        self.exit_code = exit_code
        self.output = output


class YouyunSSHSession:
    """长驻远程进程会话：向 stdin 写请求行，从 stdout 读响应行。

    通道关闭即结束远端进程；stderr 必须由调用方在命令里重定向到文件，
    避免无人读取的 stderr 管道塞满后远端进程被阻塞。
    """

    def __init__(self, stdin: Any, stdout: Any) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self.channel = stdout.channel
        self._buffer = b""

    @property
    def alive(self) -> bool:
        return bool(self.channel is not None and not self.channel.closed)

    def write_line(self, text: str) -> None:
        if not self.alive:
            raise RuntimeError("SSH 会话已断开。")
        try:
            self._stdin.write((text + "\n").encode("utf-8"))
            self._stdin.flush()
        except Exception as exc:  # noqa: BLE001 - 通道写入失败统一转为可重建信号
            raise RuntimeError(f"SSH 会话写入失败：{exc}") from exc

    def readline(self, timeout: float) -> str:
        """读取一行响应；超时抛 TimeoutError，进程退出抛 RuntimeError。"""
        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            if b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                return line.decode("utf-8", errors="replace").rstrip("\r")
            if self.channel.recv_ready():
                self._buffer += self.channel.recv(4096)
                continue
            if self.channel.exit_status_ready():
                if self._buffer:
                    tail = self._buffer.decode("utf-8", errors="replace").rstrip("\r")
                    self._buffer = b""
                    return tail
                raise RuntimeError("远程推理进程已退出。")
            if time.monotonic() > deadline:
                raise TimeoutError("等待远程推理响应超时。")
            time.sleep(0.02)

    def close(self) -> None:
        for resource in (self._stdin, self.channel):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响主流程
                pass


class YouyunSSHClient:
    """以密码认证连接一次性训练实例，提供命令执行与文件传输。"""

    def __init__(self, transport: Any | None = None, connect_timeout: float = 20.0) -> None:
        # transport 仅供测试注入；生产路径按需创建 paramiko SSHClient
        self._transport = transport
        self._client: Any = None
        self._sftp: Any = None
        self._credentials: tuple[str, int, str, str] | None = None
        self.connect_timeout = connect_timeout

    def connect(self, host: str, port: int, user: str, password: str) -> None:
        self._credentials = (host, port, user, password)
        if self._client is not None and self._is_active():
            return
        self._close_connection()
        if self._transport is None:
            import paramiko  # 惰性导入

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=host,
                port=port,
                username=user,
                password=password,
                timeout=self.connect_timeout,
                banner_timeout=self.connect_timeout,
                auth_timeout=self.connect_timeout,
                look_for_keys=False,
                allow_agent=False,
            )
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(15)
            self._client = client
        else:
            self._client = self._transport.connect(host=host, port=port, user=user, password=password)

    def _is_active(self) -> bool:
        if self._client is None:
            return False
        get_transport = getattr(self._client, "get_transport", None)
        if not callable(get_transport):
            return True
        transport = get_transport()
        return bool(transport is not None and transport.is_active())

    def _ensure_connected(self) -> None:
        if self._is_active():
            return
        if self._credentials is None:
            raise RuntimeError("SSH 尚未连接。")
        self.connect(*self._credentials)

    @staticmethod
    def _is_transient_connection_error(exc: Exception) -> bool:
        detail = str(exc).lower()
        return any(
            marker in detail
            for marker in (
                "server connection dropped",
                "ssh session not active",
                "connection reset",
                "connection closed",
                "channel closed",
                "timeout opening channel",
                "timed out",
                "eof",
            )
        )

    def _reconnect(self) -> None:
        self._close_connection()
        self._ensure_connected()

    def run(self, command: str, timeout: float = 120.0, check: bool = True) -> tuple[int, str]:
        """执行命令，返回 (exit_code, stdout+stderr)。check=True 时非零退出抛异常。"""
        self._ensure_connected()
        try:
            stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - Paramiko may race with transport shutdown
            if not self._is_transient_connection_error(exc):
                raise
            self._reconnect()
            stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        stdin.close()
        output = (stdout.read() or b"") + (stderr.read() or b"")
        text = output.decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if check and code != 0:
            raise SSHCommandError(command, code, text[-2000:])
        return code, text

    def stream(self, command: str, timeout: float = 600.0):
        """Yield stdout bytes from a long-running remote command as they arrive."""
        self._ensure_connected()
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        stdin.close()
        channel = stdout.channel
        started = time.monotonic()
        while True:
            if channel.recv_ready():
                yield channel.recv(4096)
                continue
            if channel.exit_status_ready():
                while channel.recv_ready():
                    yield channel.recv(4096)
                break
            if time.monotonic() - started > timeout:
                channel.close()
                raise TimeoutError(f"远程命令超时：{command[:120]}")
            time.sleep(0.05)

    def log_connection(self) -> "YouyunSSHClient":
        """Independent transport: log reconnections must not close control I/O."""
        if self._credentials is None:
            raise RuntimeError("SSH 尚未连接。")
        client = YouyunSSHClient(transport=self._transport, connect_timeout=self.connect_timeout)
        try:
            client.connect(*self._credentials)
        except BaseException:
            client.close()
            raise
        return client

    def follow_log(self, command: str, stop: threading.Event):
        """Yield framed stdout with cancellation and no total command deadline.

        stdin stays open: EOF is the remote helper's parent-death signal.
        Only call on a dedicated log transport, not the chat/control client.
        """
        self._ensure_connected()
        stdin, stdout, stderr = self._client.exec_command(command, timeout=20)
        channel = stdout.channel
        buffer = b""
        try:
            while not stop.is_set():
                if channel.recv_stderr_ready():
                    channel.recv_stderr(4096)  # drain without leaking command/credentials
                if channel.recv_ready():
                    buffer += channel.recv(65536)
                    while b"\n" in buffer:
                        line, _, buffer = buffer.partition(b"\n")
                        yield line
                    if len(buffer) > 100_000:
                        raise ValueError("日志协议帧超过上限。")
                    continue
                if channel.exit_status_ready():
                    if channel.recv_exit_status() != 0:
                        raise RuntimeError("远端日志跟随进程退出。")
                    if buffer.strip():
                        raise ValueError("不完整日志协议帧。")
                    return
                stop.wait(.02)
        finally:
            for resource in (stdin, channel, stdout, stderr):
                try:
                    resource.close()
                except Exception:
                    pass

    def open_session(self, command: str, timeout: float = 600.0) -> YouyunSSHSession:
        """启动长驻远程进程，返回可交互读写 stdin/stdout 的会话。

        与 run()/stream() 不同，stdin 保持打开；调用方负责用 close() 结束，
        或在会话断开后重建。timeout 只作用于底层通道 socket 读超时。
        """
        self._ensure_connected()
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        stderr.close()
        return YouyunSSHSession(stdin, stdout)

    # ---- 文件传输 ----

    def _ensure_sftp(self) -> Any:
        self._ensure_connected()
        if self._sftp is None:
            try:
                self._sftp = self._client.open_sftp()
            except Exception as exc:  # noqa: BLE001 - cloud SSH may drop the first child channel
                if not self._is_transient_connection_error(exc) or self._credentials is None:
                    raise
                logger.warning("SFTP 子通道建立失败，重连 SSH 后重试：%s", exc)
                self._reconnect()
                self._sftp = self._client.open_sftp()
        return self._sftp

    def mkdirs(self, remote_dir: str) -> None:
        for attempt in range(2):
            try:
                parts = posixpath.normpath(remote_dir).split("/")
                current = ""
                for part in parts:
                    if not part:
                        current = "/"
                        continue
                    current = posixpath.join(current, part) if current != "/" else f"/{part}"
                    try:
                        self._ensure_sftp().stat(current)
                    except FileNotFoundError:
                        self._ensure_sftp().mkdir(current)
                return
            except Exception as exc:  # noqa: BLE001 - retry only transport failures
                if attempt > 0 or not self._is_transient_connection_error(exc):
                    raise
                logger.warning("SFTP 建目录时连接中断，重连 SSH 后重试：%s", exc)
                self._reconnect()

    def put_file(
        self,
        local_path: str | Path,
        remote_path: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._ensure_sftp().put(str(local_path), remote_path, callback=callback)

    def put_text(self, text: str, remote_path: str) -> None:
        import io

        sftp = self._ensure_sftp()
        with sftp.open(remote_path, "w") as remote_file:
            remote_file.write(text)

    def get_file(self, remote_path: str, local_path: str | Path) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_sftp().get(remote_path, str(target))

    def get_tree(self, remote_dir: str, local_dir: str | Path) -> list[str]:
        """递归下载远端目录到本地，返回下载的本地文件路径列表。"""
        downloaded: list[str] = []
        sftp = self._ensure_sftp()
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)

        def walk(remote: str, local: Path) -> None:
            for attr in sftp.listdir_attr(remote):
                remote_child = posixpath.join(remote, attr.filename)
                local_child = local / attr.filename
                if stat_module.S_ISDIR(attr.st_mode):
                    local_child.mkdir(parents=True, exist_ok=True)
                    walk(remote_child, local_child)
                else:
                    sftp.get(remote_child, str(local_child))
                    downloaded.append(str(local_child))

        walk(posixpath.normpath(remote_dir), root)
        return downloaded

    def _close_connection(self) -> None:
        for resource in (self._sftp, self._client):
            if resource is not None:
                try:
                    resource.close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响主流程
                    pass
        self._sftp = None
        self._client = None

    def close(self) -> None:
        self._close_connection()
        self._credentials = None

    def __enter__(self) -> "YouyunSSHClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
