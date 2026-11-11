"""优云保留实例上的训练前后效果对比推理。

训练成功后实例保留在 awaiting_comparison 状态。对比请求通过 SSH exec
通道在实例上运行 chat_inference.py（与本地子进程完全相同的
__XUNTUI_CHAT__ JSON 行协议），基模与 Adapter 直接复用训练时已就位的
公共模型库路径（或 ModelScope 兜底缓存）与 output 产物。实例释放后不回退本地：上层调用方负责
把「已释放」转换为明确的用户提示。

凭据每次通过 describe_instance 现查，backend 重启后首个对比请求会自动
重连并重新拉起远端进程；会话断开时丢弃 handle 并在下一次请求重建。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

from backend.store import resolve_youyun_instance_artifact

from .instance_manager import YouyunInstanceManager
from .ssh import YouyunSSHClient, YouyunSSHSession

logger = logging.getLogger(__name__)

CHAT_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "chat_inference.py"
PYTHON_PATH_FILE = "/root/cppt/python-path"
EVENT_PREFIX = "__XUNTUI_CHAT__"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class RemoteComparisonError(RuntimeError):
    """对比失败，message 直接展示给用户。"""


def remote_adapter_path(local_adapter_path: str, task_id: str, remote_root: str) -> str:
    """把本地镜像 Adapter 路径映射为实例上的远程路径。

    本地镜像 .../remote-jobs/<task-id>/output[/checkpoint-N] 与远程
    /root/cppt/<task-id>/output[/checkpoint-N] 目录树一致，仅前缀不同。
    先把反斜杠归一化，Windows 机器上产生的路径也能解析。
    """
    normalized = str(local_adapter_path).replace("\\", "/")
    remote_prefix = remote_root.rstrip("/")
    # When training has completed remotely but the local mirror is still
    # downloading, the artifact carries the already-authoritative remote path.
    # Trust only paths inside the configured remote root; callers still scope
    # the path to the current task before reaching this helper.
    if normalized == remote_prefix or normalized.startswith(remote_prefix + "/"):
        return normalized
    parts = [part for part in Path(normalized).parts if part not in ("/", "\\")]
    if task_id not in parts:
        raise RemoteComparisonError("Adapter 路径与训练任务不匹配。")
    index = len(parts) - parts[::-1].index(task_id)
    suffix = "/".join(parts[index:])
    remote = f"{remote_root}/{task_id}"
    return f"{remote}/{suffix}" if suffix else remote


class _RemoteChatHandle:
    def __init__(
        self,
        task_id: str,
        ssh: YouyunSSHClient,
        session: YouyunSSHSession,
        key: tuple[str, str, bool, str],
    ) -> None:
        self.task_id = task_id
        self.ssh = ssh
        self.session = session
        self.key = key

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响重建
            pass
        try:
            self.ssh.close()
        except Exception:  # noqa: BLE001
            pass


class RemoteComparisonManager:
    """按任务缓存远程对比会话；SSH/进程操作均为阻塞调用，统一走线程。"""

    def __init__(
        self,
        *,
        manager: YouyunInstanceManager,
        ssh_factory: Any = YouyunSSHClient,
        remote_root: str = "/root/cppt",
        ssh_ready_timeout: float = 300.0,
        ssh_connect_timeout: float = 20.0,
        ready_timeout: float = 600.0,
        request_timeout: float = 180.0,
        max_handles: int = 4,
        chat_script: Path = CHAT_SCRIPT_PATH,
    ) -> None:
        self.manager = manager
        self.ssh_factory = ssh_factory
        self.remote_root = remote_root.rstrip("/") or "/root/cppt"
        self.ssh_ready_timeout = ssh_ready_timeout
        self.ssh_connect_timeout = ssh_connect_timeout
        self.ready_timeout = ready_timeout
        self.request_timeout = request_timeout
        self.max_handles = max_handles
        self.chat_script = chat_script
        self._handles: OrderedDict[str, _RemoteChatHandle] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    # ---- 对外接口 ----

    def _task_lock(self, task_id: str) -> asyncio.Lock:
        lock = self._locks.get(task_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[task_id] = lock
        return lock

    async def compare(
        self,
        task: Mapping[str, Any],
        adapter_local_path: str,
        *,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
        adapter_enabled: bool,
        load_in_4bit: bool,
    ) -> dict[str, Any]:
        task_id = str(task.get("id") or "")
        marker = resolve_youyun_instance_artifact(list(task.get("artifacts") or []))
        if marker is None or not marker.get("instanceId"):
            raise RemoteComparisonError("任务没有可用的优云实例记录。")
        if str(marker.get("state")) != "awaiting_comparison":
            raise RemoteComparisonError("优云实例已释放，无法进行效果对比；请重新运行任务后再试。")
        model_id = str(task.get("model") or "").strip()
        if not model_id:
            raise RemoteComparisonError("任务缺少基模名称，无法进行效果对比。")
        adapter_remote = remote_adapter_path(adapter_local_path, task_id, self.remote_root)
        public_model_path = self._public_model_path(task)
        key = (model_id, adapter_remote, bool(load_in_4bit), public_model_path)
        payload = {
            "requestId": uuid.uuid4().hex,
            "messages": messages,
            "parameters": parameters,
            "adapterEnabled": bool(adapter_enabled),
        }
        lock = self._task_lock(task_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=5)
        except asyncio.TimeoutError:
            # 锁被占用说明预热正在建立会话：立刻返回友好提示，
            # 避免请求挂在代理超时上被掐成 500。
            raise RemoteComparisonError(
                "优云对比模型正在预热，请稍候几秒后再点一次。"
            ) from None
        try:
            for attempt in (0, 1):
                handle = await self._ensure_handle(task_id, task, str(marker["instanceId"]), key)
                try:
                    await asyncio.to_thread(handle.session.write_line, json.dumps(payload, ensure_ascii=False))
                except Exception:
                    # 会话可能在两次请求之间断开：重建一次后重试。
                    self._drop_handle(task_id)
                    if attempt == 0:
                        continue
                    raise RemoteComparisonError("优云推理会话已断开，请重新发起对比。")
                try:
                    return await asyncio.to_thread(
                        self._read_response, handle, str(payload["requestId"]), self.request_timeout
                    )
                except Exception:
                    # 读取阶段失败后会话状态不可信：丢弃 handle，下次请求重建。
                    self._drop_handle(task_id)
                    raise
        finally:
            lock.release()
        raise RemoteComparisonError("优云推理会话已断开，请重新发起对比。")

    async def warmup(
        self,
        task: Mapping[str, Any],
        adapter_local_path: str,
        *,
        load_in_4bit: bool,
    ) -> bool:
        """在实例保留期间预建远程会话并加载模型，让首次对比点击无需等待。

        会话已就绪时直接返回 True；建立失败抛出原始异常，由调用方记录。
        """
        task_id = str(task.get("id") or "")
        model_id = str(task.get("model") or "").strip()
        marker = resolve_youyun_instance_artifact(list(task.get("artifacts") or []))
        if not task_id or not model_id:
            return False
        if marker is None or str(marker.get("state")) != "awaiting_comparison":
            return False
        adapter_remote = remote_adapter_path(adapter_local_path, task_id, self.remote_root)
        public_model_path = self._public_model_path(task)
        key = (model_id, adapter_remote, bool(load_in_4bit), public_model_path)
        async with self._task_lock(task_id):
            handle = self._handles.get(task_id)
            if handle is not None and handle.session.alive and handle.key == key:
                return True
            handle = await asyncio.to_thread(
                self._build_handle_sync, task_id, str(marker["instanceId"]), key
            )
            self._handles[task_id] = handle
            self._handles.move_to_end(task_id)
            while len(self._handles) > self.max_handles:
                oldest_id, oldest = self._handles.popitem(last=False)
                logger.info("关闭超限的远程对比会话：task=%s", oldest.task_id)
                oldest.close()
            return True

    def close_handle(self, task_id: str) -> None:
        handle = self._handles.pop(task_id, None)
        if handle is not None:
            handle.close()

    def close_all(self) -> None:
        for task_id in list(self._handles):
            self.close_handle(task_id)

    # ---- 会话生命周期 ----

    def _drop_handle(self, task_id: str) -> None:
        handle = self._handles.pop(task_id, None)
        if handle is not None:
            handle.close()

    async def _ensure_handle(
        self,
        task_id: str,
        task: Mapping[str, Any],
        instance_id: str,
        key: tuple[str, str, bool, str],
    ) -> _RemoteChatHandle:
        handle = self._handles.get(task_id)
        if handle is not None and handle.session.alive and handle.key == key:
            self._handles.move_to_end(task_id)
            return handle
        if handle is not None:
            self._drop_handle(task_id)
        handle = await asyncio.to_thread(
            self._build_handle_sync, task_id, instance_id, key
        )
        self._handles[task_id] = handle
        self._handles.move_to_end(task_id)
        while len(self._handles) > self.max_handles:
            oldest_id, oldest = self._handles.popitem(last=False)
            logger.info("关闭超限的远程对比会话：task=%s", oldest.task_id)
            oldest.close()
        return handle

    def _build_handle_sync(
        self, task_id: str, instance_id: str, key: tuple[str, str, bool, str]
    ) -> _RemoteChatHandle:
        ssh: Any = None
        try:
            ssh = self._connect_with_retry(instance_id)
            self._upload_chat_script(ssh)
            model_id, adapter_remote, load_in_4bit, public_model_path = key
            self._preflight(ssh, task_id, adapter_remote)
            cache = self._resolve_model_cache(ssh, model_id, public_model_path)
            session = self._launch(ssh, task_id, model_id, adapter_remote, cache, load_in_4bit)
            handle = _RemoteChatHandle(task_id, ssh, session, key)
            self._wait_ready(handle)
            return handle
        except Exception:
            if ssh is not None:
                try:
                    ssh.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    def _connect_with_retry(self, instance_id: str) -> Any:
        deadline = time.monotonic() + self.ssh_ready_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            # 优云可能轮换 SSH 转发端口，每次尝试都现查实例与凭据。
            try:
                instance = self.manager.client.describe_instance(instance_id)
                credentials = self.manager.ssh_credentials(instance)
                host = str(credentials.get("host") or "")
                password = str(credentials.get("password") or "")
                if not host or not password:
                    raise RemoteComparisonError("优云实例缺少 SSH 连接信息（IP 或密码为空）。")
                ssh = self.ssh_factory(connect_timeout=self.ssh_connect_timeout)
                ssh.connect(
                    host,
                    int(credentials.get("port") or 22),
                    str(credentials.get("user") or "root"),
                    password,
                )
                return ssh
            except Exception as exc:  # noqa: BLE001 - 实例刚启动时端口可能未就绪
                last_error = exc
                time.sleep(2.0)
        detail = str(last_error) if last_error else "未知错误"
        raise RemoteComparisonError(f"无法连接优云对比实例：{detail}")

    def _upload_chat_script(self, ssh: Any) -> None:
        if not self.chat_script.exists():
            raise RemoteComparisonError(f"缺少对比推理脚本：{self.chat_script}")
        ssh.put_file(str(self.chat_script), f"{self.remote_root}/chat_inference.py")

    def _preflight(self, ssh: Any, task_id: str, adapter_remote: str) -> None:
        _, output = ssh.run(
            f"test -f {shlex.quote(PYTHON_PATH_FILE)} || echo missing-python; "
            f"test -f {shlex.quote(adapter_remote + '/adapter_config.json')} || echo missing-adapter; "
            "echo preflight-done",
            30,
        )
        lines = {line.strip() for line in output.splitlines()}
        if "missing-python" in lines:
            raise RemoteComparisonError("实例上缺少训练 Python 环境（python-path 丢失），请重新运行任务。")
        if "missing-adapter" in lines:
            raise RemoteComparisonError("实例上未找到所选 Adapter 产物，请重新运行任务后再对比。")
        if "preflight-done" not in lines:
            raise RemoteComparisonError("实例环境检查失败，请重试或重新运行任务。")

    @staticmethod
    def _public_model_path(task: Mapping[str, Any]) -> str:
        resource = task.get("youyunResource") or {}
        if isinstance(resource, Mapping):
            configured = str(resource.get("publicModelPath") or "").strip()
            if configured:
                return configured
        model = str(task.get("model") or "").strip()
        if str(task.get("modelSource") or "") == "public" and model.startswith(("/model/", "/models/")):
            return model
        return ""

    def _resolve_model_cache(self, ssh: Any, model_id: str, public_model_path: str = "") -> str:
        model_name = model_id.rsplit("/", 1)[-1]
        _, output = ssh.run(
            f"MODEL_NAME={shlex.quote(model_name)}; "
            f"PUBLIC_MODEL_PATH={shlex.quote(public_model_path)}; "
            'if [ -n "$PUBLIC_MODEL_PATH" ] && [ -f "$PUBLIC_MODEL_PATH/config.json" ]; then echo "PUBLIC_MODEL=$PUBLIC_MODEL_PATH"; exit 0; fi; '
            f"CACHE={shlex.quote(self.remote_root)}/modelscope-cache; "
            "for c in /root/.cache/modelscope /workspace/model-cache /root/modelscope-cache; do "
            'if [ -d "$c" ] && find "$c" -maxdepth 6 -iname "*$MODEL_NAME*" 2>/dev/null | grep -q .; '
            'then CACHE="$c"; break; fi; done; '
            'echo "CACHE=$CACHE"',
            120,
        )
        if public_model_path and not any(line.strip() == f"PUBLIC_MODEL={public_model_path}" for line in output.splitlines()):
            raise RemoteComparisonError("原训练任务的公共模型路径不可用，不能替换 Base Model 进行评测。")
        for line in output.splitlines():
            if line.startswith("PUBLIC_MODEL="):
                model_path = line.removeprefix("PUBLIC_MODEL=").strip()
                if model_path:
                    return f"__PUBLIC__:{model_path}"
            if line.startswith("CACHE="):
                cache = line.removeprefix("CACHE=").strip()
                if cache:
                    return cache
        return f"{self.remote_root}/modelscope-cache"

    def _launch(
        self,
        ssh: Any,
        task_id: str,
        model_id: str,
        adapter_remote: str,
        cache: str,
        load_in_4bit: bool,
    ) -> YouyunSSHSession:
        err_log = f"{self.remote_root}/{task_id}/chat-server.err.log"
        script_remote = f"{self.remote_root}/chat_inference.py"
        public_model = cache.removeprefix("__PUBLIC__:") if cache.startswith("__PUBLIC__:") else ""
        model_arg = public_model or model_id
        model_source = "local" if public_model else "modelscope"
        command = (
            f"cd {shlex.quote(self.remote_root)} && "
            f"USE_MODELSCOPE_HUB=1 MODELSCOPE_CACHE={shlex.quote(cache)} "
            f"$(cat {PYTHON_PATH_FILE}) {shlex.quote(script_remote)} "
            f"--model {shlex.quote(model_arg)} --source {model_source} "
            f"--adapter {shlex.quote(adapter_remote)}"
            + (" --load-in-4bit" if load_in_4bit else "")
            + f" 2>>{shlex.quote(err_log)}"
        )
        return ssh.open_session(command, timeout=self.request_timeout)

    def _err_log_tail(self, ssh: Any, task_id: str) -> str:
        try:
            _, output = ssh.run(
                f"tail -c 2000 {shlex.quote(self.remote_root + '/' + task_id + '/chat-server.err.log')} 2>/dev/null || true",
                15,
                False,
            )
            return output.strip()[-2000:]
        except Exception:  # noqa: BLE001 - 诊断日志获取失败不掩盖原始错误
            return ""

    # ---- 协议事件 ----

    def _parse_event(self, raw: str) -> dict[str, Any] | None:
        text = ANSI_ESCAPE.sub("", raw).strip()
        if not text.startswith(EVENT_PREFIX):
            return None
        try:
            event = json.loads(text.removeprefix(EVENT_PREFIX))
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None
        if event.get("type") in {"fatal", "error"}:
            raise RuntimeError(str(event.get("message") or "远程推理失败。"))
        return event

    def _read_response(
        self, handle: _RemoteChatHandle, request_id: str, timeout: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("远程推理响应超时。")
            try:
                raw = handle.session.readline(timeout=remaining)
            except RuntimeError as exc:
                detail = self._err_log_tail(handle.ssh, handle.task_id)
                if detail:
                    raise RemoteComparisonError(f"远程推理进程退出：{exc}；远端日志：{detail}") from exc
                raise
            event = self._parse_event(raw)
            if event is None:
                continue
            if event.get("type") != "response":
                continue
            if str(event.get("requestId")) != request_id:
                continue
            return event

    def _wait_ready(self, handle: _RemoteChatHandle) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteComparisonError("远程对比模型加载超时，请重试。")
            try:
                raw = handle.session.readline(timeout=remaining)
            except RuntimeError as exc:
                detail = self._err_log_tail(handle.ssh, handle.task_id)
                if detail:
                    raise RemoteComparisonError(f"远程模型加载失败：{detail}") from exc
                raise RemoteComparisonError(f"远程模型加载失败：{exc}") from exc
            except TimeoutError as exc:
                raise RemoteComparisonError("远程对比模型加载超时，请重试。") from exc
            event = self._parse_event(raw)
            if event is not None and event.get("type") == "ready":
                return
