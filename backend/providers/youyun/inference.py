"""优云远程 vLLM 推理 Provider，负责实例、SSH 进程与删除确认。"""
from __future__ import annotations

import asyncio

import logging
import os
import posixpath
import re
import shlex
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.providers.youyun.log_follow import LogCursor, RemoteLogCollector

from .instance_manager import YouyunInstanceManager
from .ssh import YouyunSSHClient

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OOM_WARMUP_RE = re.compile(r"warming up sampler with\s+(\d+)\s+dummy requests", re.IGNORECASE)
_OOM_MEMORY_RE = re.compile(
    r"Tried to allocate\s+([\d.]+\s+[KMGT]i?B).*?"
    r"total capacity of\s+([\d.]+\s+[KMGT]i?B).*?"
    r"of which\s+([\d.]+\s+[KMGT]i?B)\s+is free",
    re.IGNORECASE | re.DOTALL,
)


class YouyunInferenceStartError(RuntimeError):
    def __init__(
        self,
        message: str,
        handle: "YouyunInferenceHandle",
        cleanup_report: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.handle = handle
        self.cleanup_report = dict(cleanup_report)


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    return detail or type(exc).__name__


def _clean_remote_log(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text).replace("\r", "").strip()


def _remote_failure_summary(log_text: str) -> str:
    clean = _clean_remote_log(log_text)
    if not clean:
        return ""
    if "CUDA out of memory" in clean:
        warmup = _OOM_WARMUP_RE.search(clean)
        memory = _OOM_MEMORY_RE.search(clean)
        context: list[str] = []
        if warmup:
            context.append(f"预热采样器使用 {warmup.group(1)} 个并发样本")
        if memory:
            requested, total, free = memory.groups()
            context.append(f"GPU 总显存 {total}、剩余 {free}，仍需申请 {requested}")
        detail = "；".join(context)
        suffix = "请降低最大并发数或 GPU 显存利用率。"
        return f"远端 vLLM 显存不足：{detail}。{suffix}" if detail else f"远端 vLLM 显存不足。{suffix}"

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    preferred = (
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "torch.OutOfMemoryError:",
        "ValueError:",
        "RuntimeError:",
    )
    for marker in preferred:
        for line in reversed(lines):
            if marker in line and "Engine core initialization failed" not in line:
                return line[-1000:]
    return lines[-1][-1000:] if lines else ""


@dataclass
class YouyunInferenceHandle:
    instance_id: str
    endpoint: str
    pid_path: str
    workdir: str
    log_path: str = ""
    ssh: YouyunSSHClient | None = None
    last_probe_error: str = ""
    log_offset: int = 0
    log_file_id: str = ""
    log_python: str = "python3"

    def persisted(self) -> dict[str, Any]:
        return {
            "instanceId": self.instance_id,
            "endpoint": self.endpoint,
            "pidPath": self.pid_path,
            "workdir": self.workdir,
            "logPath": self.log_path or posixpath.join(self.workdir, "vllm.log"),
            "logOffset": self.log_offset,
            "logFileId": self.log_file_id,
            "logPython": self.log_python,
        }

    @classmethod
    def restore(cls, value: Mapping[str, Any]) -> "YouyunInferenceHandle":
        return cls(
            log_offset=int(value.get("logOffset", 0) or 0),
            log_file_id=str(value.get("logFileId", "") or ""),
            log_python=str(value.get("logPython", "python3") or "python3"),
            instance_id=str(value["instanceId"]),
            endpoint=str(value.get("endpoint") or ""),
            pid_path=str(value.get("pidPath") or ""),
            workdir=str(value.get("workdir") or ""),
            log_path=str(value.get("logPath") or posixpath.join(str(value.get("workdir") or ""), "vllm.log")),
        )


_REMOTE_LOG_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_REMOTE_LOG_PROCESS_PREFIX = re.compile(
    r"^\s*(?:\([^)]*?\s+pid=\d+\)|\[[^\]]*?pid=\d+[^\]]*\])\s*"
)


def _clean_remote_log_line(line: str) -> str | None:
    """简化远端 vLLM 日志，但保留每一条有内容的运行记录。"""
    cleaned = _REMOTE_LOG_ANSI_PATTERN.sub("", line).replace("\r", "").strip()
    if not cleaned:
        return None
    return _REMOTE_LOG_PROCESS_PREFIX.sub("", cleaned).strip() or None


class YouyunInferenceProvider:
    def __init__(
        self,
        manager: YouyunInstanceManager,
        *,
        ssh_factory: Callable[[], YouyunSSHClient] = YouyunSSHClient,
        remote_root: str = "/root/cppt/inference",
        remote_python: str = "python3",
        port: int = 8011,
        startup_timeout: float = 900,
        poll_interval: float = 2,
        ssh_ready_timeout: float | None = None,
        ssh_connect_timeout: float | None = None,
        install_vllm: bool | None = None,
        vllm_package: str | None = None,
        model_prepare_timeout: float | None = None,
    ) -> None:
        self._log_collectors: dict[str, RemoteLogCollector] = {}
        self._closed_log_instances: set[str] = set()
        self.manager = manager
        self.ssh_factory = ssh_factory
        self.remote_root = remote_root.rstrip("/")
        self.remote_python = remote_python
        self.port = port
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.ssh_ready_timeout = (
            ssh_ready_timeout
            if ssh_ready_timeout is not None
            else float(os.environ.get("YOUYUN_SSH_READY_TIMEOUT_SECONDS", "300"))
        )
        self.ssh_connect_timeout = (
            ssh_connect_timeout
            if ssh_connect_timeout is not None
            else float(os.environ.get("YOUYUN_SSH_CONNECT_TIMEOUT_SECONDS", "20"))
        )
        self.install_vllm = (
            install_vllm
            if install_vllm is not None
            else os.environ.get("XUNTUI_REMOTE_VLLM_INSTALL", "0").strip().lower() not in {"0", "false", "no"}
        )
        self.vllm_package = (vllm_package or os.environ.get("XUNTUI_REMOTE_VLLM_PACKAGE", "vllm")).strip() or "vllm"
        self.model_prepare_timeout = (
            model_prepare_timeout
            if model_prepare_timeout is not None
            else float(os.environ.get("XUNTUI_REMOTE_MODEL_PREPARE_TIMEOUT_SECONDS", "1800"))
        )

    @staticmethod
    def _stage(
        callback: Callable[..., None] | None,
        stage: str,
        message: str,
        **details: Any,
    ) -> None:
        if callback is None:
            return
        try:
            callback(stage, message, details)
        except TypeError:
            # Keep compatibility with simple one-argument callbacks used by integrations.
            try:
                callback(message)
            except Exception:  # noqa: BLE001 - observability must not abort a GPU launch
                logger.exception("优云推理启动阶段写入失败：%s", stage)
        except Exception:  # noqa: BLE001 - observability must not abort a GPU launch
            logger.exception("优云推理启动阶段写入失败：%s", stage)

    @staticmethod
    def _emit_log(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is None:
            return
        try:
            callback(message)
        except Exception:  # noqa: BLE001 - remote log persistence is best effort
            logger.exception("优云推理远端日志写入失败")

    def _connect(self, instance: Any) -> YouyunSSHClient:
        credentials = self.manager.ssh_credentials(instance)
        if not credentials["host"] or not credentials["password"]:
            raise RuntimeError("优云实例缺少 SSH 连接信息（IP 或密码为空）。")
        deadline = time.monotonic() + self.ssh_ready_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            # 优云 may publish the initial SSH forwarding port before it is
            # reachable, so refresh the instance on every attempt.
            get_instance = getattr(self.manager, "get", None)
            if callable(get_instance):
                try:
                    refreshed = get_instance(instance.uhost_id)
                except Exception:  # noqa: BLE001 - retain previous credentials
                    refreshed = None
                if refreshed is not None:
                    refreshed_credentials = self.manager.ssh_credentials(refreshed)
                    if refreshed_credentials.get("host") and refreshed_credentials.get("password"):
                        credentials = refreshed_credentials

            host = str(credentials["host"])
            port = int(credentials.get("port") or 22)
            ssh: YouyunSSHClient | None = None
            try:
                # Only probe TCP for the real client; injected test transports
                # must remain deterministic and network-free.
                if self.ssh_factory is YouyunSSHClient:
                    with socket.create_connection(
                        (host, port),
                        timeout=min(self.ssh_connect_timeout, max(1.0, deadline - time.monotonic())),
                    ):
                        pass
                ssh = self._new_ssh_client()
                ssh.connect(host, port, credentials["user"], credentials["password"])
                return ssh
            except Exception as exc:  # noqa: BLE001 - sshd/forwarding may lag instance state
                last_error = exc
                try:
                    if ssh is not None:
                        ssh.close()
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(self.poll_interval)
        detail = str(last_error).strip() if last_error else "未收到 SSH 响应"
        host = str(credentials.get("host") or "")
        port = int(credentials.get("port") or 22)
        raise RuntimeError(f"SSH 连接超时（{host}:{port}）：{detail}")

    def _new_ssh_client(self) -> YouyunSSHClient:
        try:
            return self.ssh_factory(connect_timeout=self.ssh_connect_timeout)
        except TypeError:
            return self.ssh_factory()

    def _upload_tree(
        self,
        ssh: YouyunSSHClient,
        local_root: Path,
        remote_root: str,
        *,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        files = sorted(path for path in local_root.rglob("*") if path.is_file())
        total_bytes = sum(path.stat().st_size for path in files)
        completed_bytes = 0
        for path in files:
            relative_path = path.relative_to(local_root).as_posix()
            remote_path = posixpath.join(remote_root, relative_path)
            file_size = path.stat().st_size
            ssh.mkdirs(posixpath.dirname(remote_path))

            def report_file_progress(
                transferred: int,
                _file_total: int,
                *,
                completed: int = completed_bytes,
                relative: str = relative_path,
            ) -> None:
                if on_progress is not None:
                    on_progress(min(total_bytes, completed + transferred), total_bytes, relative)

            ssh.put_file(path, remote_path, callback=report_file_progress)
            completed_bytes += file_size
            if on_progress is not None:
                on_progress(completed_bytes, total_bytes, relative_path)

    async def _prepare_remote_model(
        self,
        ssh: YouyunSSHClient,
        *,
        model: str,
        workdir: str,
        on_stage: Callable[..., None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        public_model_path: str | None = None,
    ) -> str:
        """Return a remote directory containing Base Model weights.

        A repository id is not a usable ``--model`` value on a fresh cloud
        instance. Prefer uploading a local model, then reuse known ModelScope
        or Hugging Face caches, and finally download into the instance.
        """
        if cancel_requested and cancel_requested():
            raise RuntimeError("服务启动已取消。")
        model_ref = str(model or "").strip()
        if not model_ref:
            raise RuntimeError("基模权重准备失败：Base Model 为空。")
        remote_model = posixpath.join(workdir, "base-model")
        local_model = Path(model_ref).expanduser()
        try:
            local_exists = local_model.is_dir()
        except OSError:
            local_exists = False
        if local_exists:
            self._stage(on_stage, "uploading_model", "开始上传 Base Model 到优云实例。")
            await asyncio.to_thread(self._upload_tree, ssh, local_model, remote_model)
            if cancel_requested and cancel_requested():
                raise RuntimeError("服务启动已取消。")
            self._stage(on_stage, "model_ready", "Base Model 上传完成。", modelPath=remote_model)
            return remote_model

        if any(char in model_ref for char in "\r\n"):
            raise RuntimeError("基模权重准备失败：Base Model 标识包含非法换行符。")
        configured_public = str(public_model_path or "").strip()
        model_key = "".join(
            char if char.isalnum() or char in "-_." else "-"
            for char in model_ref.replace("/", "--")
        )
        cache_patterns = " ".join(
            (
                f"/root/cppt/modelscope-cache/models/{model_key}/snapshots/*",
                f"/root/.cache/modelscope/models/{model_key}/snapshots/*",
                f"/workspace/model-cache/{model_key}/snapshots/*",
                f"/root/modelscope-cache/{model_key}/snapshots/*",
                f"/root/.cache/huggingface/hub/models--{model_key}/snapshots/*",
            )
        )
        self._stage(on_stage, "preparing_model", f"检查远端 Base Model 缓存：{model_ref}。")
        model_env = shlex.quote(model_ref)
        public_probe = (
            f"if [ -f {shlex.quote(configured_public + '/config.json')} ]; then printf 'MODEL_READY:%s\\n' {shlex.quote(configured_public)}; exit 0; fi; "
            if configured_public
            else ""
        )
        quick_probe = (
            "set -u; "
            f"MODEL_ID={model_env}; "
            + public_probe
            + "if [ \"${MODEL_ID#/}\" != \"$MODEL_ID\" ] && [ -f \"$MODEL_ID/config.json\" ]; then "
            + "printf 'MODEL_READY:%s\\n' \"$MODEL_ID\"; exit 0; fi; "
            + f"for candidate in {cache_patterns}; do "
            + "if [ -f \"$candidate/config.json\" ]; then printf 'MODEL_READY:%s\\n' \"$candidate\"; exit 0; fi; "
            + "done; "
            + "echo MODEL_MISS"
        )
        code, output = await asyncio.to_thread(ssh.run, quick_probe, 30, False)
        ready = next(
            (
                line.removeprefix("MODEL_READY:").strip()
                for line in output.splitlines()
                if line.startswith("MODEL_READY:") and line.removeprefix("MODEL_READY:").strip()
            ),
            "",
        )
        if code != 0:
            detail = output.strip()[-2000:] if output.strip() else "远端缓存检查失败。"
            raise RuntimeError(f"基模权重准备失败：{model_ref}；{detail}")
        if ready:
            self._stage(on_stage, "model_ready", "远端 Base Model 已就绪。", modelPath=ready)
            return ready
        if configured_public:
            raise RuntimeError(
                f"原任务记录的优云公共模型路径不可用：{configured_public}；"
                "不能按名称替换为其他 Base Model。"
            )

        if cancel_requested and cancel_requested():
            raise RuntimeError("服务启动已取消。")
        prep_dir = posixpath.join(workdir, ".model-prep")
        pid_path = posixpath.join(prep_dir, "download.pid")
        done_path = posixpath.join(prep_dir, "download.done")
        result_path = posixpath.join(prep_dir, "download.result")
        log_path = posixpath.join(prep_dir, "download.log")
        target_env = shlex.quote(remote_model)
        cache_env = shlex.quote(posixpath.join(self.remote_root, "modelscope-cache"))
        modelscope_result_path = posixpath.join(prep_dir, "modelscope.result")
        huggingface_result_path = posixpath.join(prep_dir, "huggingface.result")
        download_script = (
            "set -u; "
            f"MODEL_ID={model_env}; TARGET={target_env}; CACHE_DIR={cache_env}; "
            f"mkdir -p {shlex.quote(prep_dir)} \"$TARGET\"; "
            f"PY={shlex.quote(self.remote_python)}; "
            f"MS_RESULT={shlex.quote(modelscope_result_path)}; HF_RESULT={shlex.quote(huggingface_result_path)}; "
            # Prefer ModelScope for the common cloud path, but retain a
            # Hugging Face fallback for images/environments where ModelScope
            # is unavailable or the repository is not mirrored there.
            "if \"$PY\" -c 'import modelscope' >/dev/null 2>&1 || \"$PY\" -m pip install --no-cache-dir modelscope >> \"$MS_RESULT\" 2>&1; then "
            "MODEL_ID=\"$MODEL_ID\" CACHE_DIR=\"$CACHE_DIR\" \"$PY\" -c "
            "'import os; from modelscope import snapshot_download; p=snapshot_download(os.environ[\"MODEL_ID\"], cache_dir=os.environ[\"CACHE_DIR\"]); print(p)' > \"$MS_RESULT\" 2>&1; "
            "MS_PATH=$(tail -n 1 \"$MS_RESULT\" 2>/dev/null || true); "
            "if [ -n \"$MS_PATH\" ] && [ -f \"$MS_PATH/config.json\" ]; then "
            f"echo \"$MS_PATH\" > {shlex.quote(result_path)}; echo ok > {shlex.quote(done_path)}; exit 0; fi; "
            "else echo 'ModelScope 不可用，准备降级到 Hugging Face。' >> \"$MS_RESULT\"; fi; "
            "if \"$PY\" -c 'import huggingface_hub' >/dev/null 2>&1 || \"$PY\" -m pip install --no-cache-dir huggingface_hub >> \"$HF_RESULT\" 2>&1; then "
            "MODEL_ID=\"$MODEL_ID\" TARGET=\"$TARGET\" \"$PY\" -c "
            "'import os; from huggingface_hub import snapshot_download; p=snapshot_download(repo_id=os.environ[\"MODEL_ID\"], local_dir=os.environ[\"TARGET\"]); print(p)' > \"$HF_RESULT\" 2>&1; "
            "HF_PATH=$(tail -n 1 \"$HF_RESULT\" 2>/dev/null || true); "
            "if [ -n \"$HF_PATH\" ] && [ -f \"$HF_PATH/config.json\" ]; then "
            f"echo \"$HF_PATH\" > {shlex.quote(result_path)}; echo ok > {shlex.quote(done_path)}; exit 0; fi; "
            "else echo 'Hugging Face 不可用，无法完成 Base Model 下载。' >> \"$HF_RESULT\"; fi; "
            "echo 'ModelScope 与 Hugging Face 均未能下载 Base Model。' > "
            f"{shlex.quote(result_path)}; "
            f"echo 'ModelScope: ' >> {shlex.quote(result_path)}; cat \"$MS_RESULT\" >> {shlex.quote(result_path)} 2>/dev/null || true; "
            f"echo 'Hugging Face: ' >> {shlex.quote(result_path)}; cat \"$HF_RESULT\" >> {shlex.quote(result_path)} 2>/dev/null || true; "
            f"echo fail > {shlex.quote(done_path)}"
        )
        launch = (
            f"mkdir -p {shlex.quote(prep_dir)}; rm -f {shlex.quote(pid_path)} {shlex.quote(done_path)} "
            f"{shlex.quote(result_path)} {shlex.quote(log_path)} {shlex.quote(modelscope_result_path)} {shlex.quote(huggingface_result_path)}; "
            f"nohup bash -c {shlex.quote(download_script)} > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)}"
        )
        self._stage(
            on_stage,
            "downloading_model",
            f"远端未找到 Base Model，正在下载：{model_ref}。",
            modelId=model_ref,
            logPath=log_path,
        )
        await asyncio.to_thread(ssh.run, launch, 30, False)
        deadline = asyncio.get_running_loop().time() + self.model_prepare_timeout
        last_tail = ""
        while asyncio.get_running_loop().time() < deadline:
            if cancel_requested and cancel_requested():
                await self._stop_remote_model_prepare(ssh, pid_path)
                raise RuntimeError("服务启动已取消。")
            _, status_output = await asyncio.to_thread(
                ssh.run,
                f"echo status:$(cat {shlex.quote(done_path)} 2>/dev/null || echo pending); "
                f"echo result:$(cat {shlex.quote(result_path)} 2>/dev/null || true); "
                f"tail -n 4 {shlex.quote(log_path)} 2>/dev/null || true",
                30,
                False,
            )
            lines = status_output.splitlines()
            status = next((line.removeprefix("status:").strip() for line in lines if line.startswith("status:")), "pending")
            result_path_value = next((line.removeprefix("result:").strip() for line in lines if line.startswith("result:") and line.removeprefix("result:").strip()), "")
            tail = "\n".join(line for line in lines if not line.startswith(("status:", "result:"))).strip()
            if tail and tail != last_tail:
                last_tail = tail
                self._stage(on_stage, "downloading_model", f"Base Model 下载进度：{tail.splitlines()[-1][:200]}。")
            if status == "ok" and result_path_value:
                self._stage(on_stage, "model_ready", "远端 Base Model 已就绪。", modelPath=result_path_value)
                return result_path_value
            if status == "fail":
                raise RuntimeError(f"基模权重准备失败：{model_ref}；{result_path_value or last_tail or '下载失败'}")
            await asyncio.sleep(self.poll_interval)
        await self._stop_remote_model_prepare(ssh, pid_path)
        raise RuntimeError(f"基模权重准备超时：{model_ref}；{last_tail[-1000:]}")

    async def _stop_remote_model_prepare(self, ssh: YouyunSSHClient, pid_path: str) -> None:
        await asyncio.to_thread(
            ssh.run,
            f"if [ -f {shlex.quote(pid_path)} ]; then p=$(cat {shlex.quote(pid_path)}); "
            "kill -TERM -$p 2>/dev/null || kill -TERM $p 2>/dev/null || true; "
            "sleep 1; kill -KILL -$p 2>/dev/null || kill -KILL $p 2>/dev/null || true; fi; true",
            30,
            False,
        )

    async def _diagnose_remote_runtime(
        self,
        ssh: YouyunSSHClient,
        *,
        on_stage: Callable[..., None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> str:
        """Collect a compact runtime fingerprint before preparing vLLM.

        Images frequently contain several Python installations.  The probe is
        deliberately best-effort and never blocks startup on missing optional
        tools; its output is streamed into the service log for diagnosis.
        """
        self._stage(on_stage, "diagnosing_vllm_runtime", "采集远端 Python/Torch/CUDA/vLLM 运行时信息。")
        configured = shlex.quote(self.remote_python)
        script = (
            "set +e; "
            "echo runtime.diagnostic.begin; "
            "for candidate in "
            f"{configured} /usr/local/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python "
            "/opt/conda/bin/python /root/miniconda3/envs/*/bin/python /root/miniconda3/bin/python "
            "/usr/local/bin/python3 /usr/bin/python3; do "
            "if [ -x \"$candidate\" ]; then "
            "echo python.path:$candidate; \"$candidate\" --version 2>&1 | sed 's/^/python.version:/'; "
            "\"$candidate\" -c 'import sys; print(\"python.executable:\"+sys.executable)' 2>&1; "
            "if \"$candidate\" -c 'import torch; print(\"torch.version:\"+str(torch.__version__)); print(\"torch.cuda.version:\"+str(getattr(torch.version, \"cuda\", None))); print(\"torch.cuda.available:\"+str(torch.cuda.is_available())); print(\"torch.cuda.device_count:\"+str(torch.cuda.device_count()))' 2>&1; then :; else echo torch.error:unavailable; fi; "
            "if \"$candidate\" -c 'import vllm; print(\"vllm.version:\"+str(getattr(vllm, \"__version__\", \"unknown\")))' 2>&1; then :; else echo vllm.error:unavailable; fi; "
            "fi; done; "
            "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | sed 's/^/nvidia-smi:/'; "
            "else echo nvidia-smi:unavailable; fi; "
            "echo runtime.diagnostic.end"
        )
        try:
            code, output = await asyncio.to_thread(ssh.run, script, 45, False)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the real startup check
            message = f"远端运行时诊断不可用：{_exception_detail(exc)}"
            self._stage(on_stage, "diagnosing_vllm_runtime", message)
            self._emit_log(on_log, f"[runtime] {message}")
            return ""
        text = output.strip()
        if on_log is not None:
            for line in text.splitlines():
                if line.strip():
                    self._emit_log(on_log, f"[runtime] {line[-4000:]}")
        if code != 0:
            self._stage(on_stage, "diagnosing_vllm_runtime", "远端运行时诊断命令返回非零状态。", exitCode=code)
        return text

    async def _ensure_remote_ninja(
        self,
        ssh: YouyunSSHClient,
        *,
        on_stage: Callable[..., None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        """Best-effort install of ninja for official images that lack it.

        Loading a LoRA adapter can make torch compile its kernels at runtime,
        which requires ninja on PATH. Some official vLLM images (v0.25.1) do
        not ship it and the engine dies with FileNotFoundError('ninja') while
        loading the adapter. Best-effort by design: when the install fails,
        the original launch error is the more useful signal to surface.
        """
        python = shlex.quote(self.remote_python)
        check = (
            "set +e; "
            "if command -v ninja >/dev/null 2>&1; then echo ninja:present; "
            "elif "
            + python
            + " -c 'import shutil,sys; sys.exit(0 if shutil.which(\"ninja\") else 3)' 2>/dev/null; "
            "then echo ninja:present-in-env; else echo ninja:missing; fi; "
            "command -v nvcc >/dev/null 2>&1 && echo nvcc:present || echo nvcc:missing"
        )
        try:
            _, output = await asyncio.to_thread(ssh.run, check, 30, False)
        except Exception:  # noqa: BLE001 - diagnostics must not block startup
            return
        if "ninja:missing" not in output:
            return
        self._stage(
            on_stage,
            "installing_vllm",
            "远端缺少 ninja（LoRA 算子编译依赖），正在自动安装。",
        )
        install = (
            python
            + " -m pip install --no-cache-dir ninja >/dev/null 2>&1 || true; "
            + "ln -sf \"$(dirname "
            + python
            + ")/ninja\" /usr/local/bin/ninja 2>/dev/null || true; "
            "if command -v ninja >/dev/null 2>&1 || "
            + python
            + " -c 'import shutil,sys; sys.exit(0 if shutil.which(\"ninja\") else 3)' 2>/dev/null; "
            "then echo ninja:installed; else echo ninja:install-failed; fi"
        )
        try:
            _, output = await asyncio.to_thread(ssh.run, install, 180, False)
        except Exception:  # noqa: BLE001
            return
        if "ninja:installed" in output:
            self._emit_log(on_log, "[runtime] 远端 ninja 已安装（LoRA 算子编译依赖）。")
        else:
            self._emit_log(
                on_log,
                "[runtime] 远端 ninja 自动安装失败；若 vLLM 因缺少 ninja 退出，请更换镜像版本。",
            )

    async def _validate_remote_gpu(
        self,
        ssh: YouyunSSHClient,
        *,
        on_stage: Callable[..., None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> str:
        """Require both NVML and the selected PyTorch runtime to see a GPU."""
        self._stage(
            on_stage,
            "validating_gpu_runtime",
            "校验优云实例 GPU 驱动与 CUDA 可用性。",
        )
        python = shlex.quote(self.remote_python)
        script = (
            "set +e; "
            "echo gpu.preflight.begin; "
            "if ! command -v nvidia-smi >/dev/null 2>&1; then "
            "echo nvidia-smi:unavailable; exit 41; fi; "
            "SMI_OUTPUT=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1); "
            "SMI_CODE=$?; printf 'nvidia-smi:%s\\n' \"$SMI_OUTPUT\"; "
            "if [ \"$SMI_CODE\" -ne 0 ]; then exit 42; fi; "
            f"{python} -c 'import sys, torch; "
            "available=bool(torch.cuda.is_available()); count=int(torch.cuda.device_count()); "
            "print(\"torch.cuda.available:\"+str(available)); "
            "print(\"torch.cuda.device_count:\"+str(count)); "
            "sys.exit(0 if available and count > 0 else 43)' 2>&1; "
            "PY_CODE=$?; if [ \"$PY_CODE\" -ne 0 ]; then exit 43; fi; "
            "echo gpu.preflight.end"
        )
        code, output = await asyncio.to_thread(ssh.run, script, 45, False)
        text = output.strip()
        if on_log is not None:
            for line in text.splitlines():
                if line.strip():
                    self._emit_log(on_log, f"[gpu-preflight] {line[-4000:]}")
        if code == 0:
            self._stage(
                on_stage,
                "gpu_runtime_ready",
                "优云实例 GPU 驱动与 CUDA 校验通过。",
            )
            return text

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if code == 41 or any(line == "nvidia-smi:unavailable" for line in lines):
            reason = "实例中未找到 nvidia-smi。"
        elif code == 42:
            detail = next(
                (line.removeprefix("nvidia-smi:").strip() for line in lines if line.startswith("nvidia-smi:")),
                "nvidia-smi 返回非零状态",
            )
            reason = f"nvidia-smi 检查失败：{detail}。"
        elif any(line == "torch.cuda.available:False" for line in lines) or any(
            line == "torch.cuda.device_count:0" for line in lines
        ):
            reason = "PyTorch 未检测到可用 CUDA 设备。"
        else:
            detail = lines[-1][-1000:] if lines else "远端没有返回检查结果"
            reason = f"PyTorch CUDA 检查失败：{detail}。"
        message = f"优云实例 GPU 驱动不可用，请更换实例或镜像。{reason}"
        self._stage(
            on_stage,
            "gpu_runtime_failed",
            message,
            exitCode=code,
        )
        raise RuntimeError(message)

    def _prepare_vllm_runtime(
        self,
        ssh: YouyunSSHClient,
        on_stage: Callable[..., None] | None = None,
    ) -> str:
        configured = shlex.quote(self.remote_python)
        candidates = (
            f"{configured} "
            "/usr/local/miniconda3/envs/*/bin/python "
            "/opt/conda/envs/*/bin/python "
            "/opt/conda/bin/python "
            "/root/miniconda3/envs/*/bin/python "
            "/root/miniconda3/bin/python "
            "/usr/local/bin/python3 /usr/bin/python3"
        )
        probe = (
            "for candidate in " + candidates + "; do "
            "if [ -x \"$candidate\" ] && \"$candidate\" -c 'import vllm' >/dev/null 2>&1; then "
            "printf '%s\\n' \"$candidate\"; exit 0; fi; done; exit 42"
        )
        self._stage(on_stage, "checking_vllm_runtime", "检查远端 vLLM Python 环境。")
        code, output = ssh.run(probe, 60, False)
        if code == 0:
            selected = output.strip().splitlines()[-1]
            self.remote_python = selected
            self._stage(on_stage, "vllm_runtime_ready", f"远端 vLLM 环境已就绪：{selected}。")
            return selected
        if not self.install_vllm:
            raise RuntimeError(
                f"远端 Python 未安装 vLLM（当前配置：{self.remote_python}）。"
                "请设置 XUNTUI_REMOTE_VLLM_PYTHON，或启用 XUNTUI_REMOTE_VLLM_INSTALL。"
            )

        install = (
            "set -eu; PY=''; "
            "for candidate in "
            "/usr/local/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python "
            "/opt/conda/bin/python /root/miniconda3/envs/*/bin/python "
            "/root/miniconda3/bin/python /usr/local/bin/python3 /usr/bin/python3 " + configured + "; do "
            "if [ -x \"$candidate\" ] && \"$candidate\" -c 'import torch' >/dev/null 2>&1; then PY=\"$candidate\"; break; fi; done; "
            "if [ -z \"$PY\" ]; then echo '未找到包含 PyTorch 的远端 Python 环境' >&2; exit 43; fi; "
            f"echo \"使用 $PY 安装 vLLM\"; \"$PY\" -m pip install --no-cache-dir -U {shlex.quote(self.vllm_package)}; "
            "\"$PY\" -c 'import vllm; print(vllm.__version__)'; printf '%s\\n' \"$PY\""
        )
        self._stage(on_stage, "installing_vllm", "远端未安装 vLLM，正在安装运行环境。")
        code, output = ssh.run(install, 900, False)
        if code != 0:
            detail = output.strip()[-2000:] if output.strip() else "无安装输出"
            raise RuntimeError(f"远端 vLLM 安装失败：{detail}")
        selected = output.strip().splitlines()[-1]
        self.remote_python = selected
        self._stage(on_stage, "vllm_runtime_ready", f"远端 vLLM 安装完成：{selected}。")
        return selected

    async def _prepare_vllm_runtime_async(
        self,
        ssh: YouyunSSHClient,
        *,
        workdir: str,
        on_stage: Callable[..., None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> str:
        """Prepare vLLM without blocking cancellation on a long pip install."""
        def cancelled() -> bool:
            return bool(cancel_requested and cancel_requested())

        configured = shlex.quote(self.remote_python)
        candidates = (
            f"{configured} "
            "/usr/local/miniconda3/envs/*/bin/python "
            "/opt/conda/envs/*/bin/python "
            "/opt/conda/bin/python "
            "/root/miniconda3/envs/*/bin/python "
            "/root/miniconda3/bin/python "
            "/usr/local/bin/python3 /usr/bin/python3"
        )
        probe = (
            "for candidate in " + candidates + "; do "
            "if [ -x \"$candidate\" ] && \"$candidate\" -c 'import vllm' >/dev/null 2>&1; then "
            "printf '%s\\n' \"$candidate\"; exit 0; fi; done; exit 42"
        )
        if cancelled():
            raise RuntimeError("服务启动已取消。")
        self._stage(on_stage, "checking_vllm_runtime", "检查远端 vLLM Python 环境。")
        code, output = await asyncio.to_thread(ssh.run, probe, 60, False)
        if code == 0:
            selected = output.strip().splitlines()[-1]
            self.remote_python = selected
            self._stage(on_stage, "vllm_runtime_ready", f"远端 vLLM 环境已就绪：{selected}。")
            return selected
        if not self.install_vllm:
            raise RuntimeError(
                f"远端 Python 未安装 vLLM（当前配置：{self.remote_python}）。"
                "请设置 XUNTUI_REMOTE_VLLM_PYTHON，或启用 XUNTUI_REMOTE_VLLM_INSTALL。"
            )

        install_dir = posixpath.join(workdir, ".vllm-install")
        pid_path = posixpath.join(install_dir, "install.pid")
        done_path = posixpath.join(install_dir, "install.done")
        selected_path = posixpath.join(install_dir, "python")
        log_path = posixpath.join(install_dir, "install.log")
        install_script = (
            "set -u; PY=''; "
            "for candidate in "
            "/usr/local/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python "
            "/opt/conda/bin/python /root/miniconda3/envs/*/bin/python "
            "/root/miniconda3/bin/python /usr/local/bin/python3 /usr/bin/python3 " + configured + "; do "
            "if [ -x \"$candidate\" ] && \"$candidate\" -c 'import torch' >/dev/null 2>&1; then PY=\"$candidate\"; break; fi; done; "
            "if [ -z \"$PY\" ]; then echo '未找到包含 PyTorch 的远端 Python 环境' >&2; echo fail > "
            + shlex.quote(done_path)
            + "; exit 0; fi; "
            "if \"$PY\" -m pip install --no-cache-dir -U "
            + shlex.quote(self.vllm_package)
            + " && \"$PY\" -c 'import vllm; print(vllm.__version__)'; then "
            "printf '%s\\n' \"$PY\" > "
            + shlex.quote(selected_path)
            + "; echo ok > "
            + shlex.quote(done_path)
            + "; else echo fail > "
            + shlex.quote(done_path)
            + "; fi"
        )
        launch = (
            f"mkdir -p {shlex.quote(install_dir)}; "
            f"if [ ! -f {shlex.quote(done_path)} ]; then "
            f"rm -f {shlex.quote(pid_path)} {shlex.quote(selected_path)}; "
            f"nohup bash -c {shlex.quote(install_script)} > {shlex.quote(log_path)} 2>&1 & "
            f"echo $! > {shlex.quote(pid_path)}; fi"
        )
        self._stage(
            on_stage,
            "installing_vllm",
            "远端未安装 vLLM，正在安装运行环境。",
            installLogPath=log_path,
            installPidPath=pid_path,
            package=self.vllm_package,
        )
        await asyncio.to_thread(ssh.run, launch, 30, False)
        deadline = asyncio.get_running_loop().time() + 900
        last_tail = ""
        while asyncio.get_running_loop().time() < deadline:
            if cancelled():
                await self._stop_remote_install(ssh, pid_path)
                raise RuntimeError("服务启动已取消。")
            _, output = await asyncio.to_thread(
                ssh.run,
                f"echo status:$(cat {shlex.quote(done_path)} 2>/dev/null || echo pending); "
                f"echo python:$(cat {shlex.quote(selected_path)} 2>/dev/null || true); "
                f"tail -n 4 {shlex.quote(log_path)} 2>/dev/null || true",
                30,
                False,
            )
            lines = output.splitlines()
            status = next((line.removeprefix("status:").strip() for line in lines if line.startswith("status:")), "pending")
            selected = next((line.removeprefix("python:").strip() for line in lines if line.startswith("python:") and line.removeprefix("python:").strip()), "")
            tail = "\n".join(line for line in lines if not line.startswith(("status:", "python:"))).strip()
            if tail and tail != last_tail:
                last_tail = tail
                self._stage(on_stage, "installing_vllm", f"远端 vLLM 安装进度：{tail.splitlines()[-1][:200]}。")
            if status == "ok" and selected:
                self.remote_python = selected
                self._stage(on_stage, "vllm_runtime_ready", f"远端 vLLM 安装完成：{selected}。")
                return selected
            if status == "fail":
                raise RuntimeError(f"远端 vLLM 安装失败：{tail[-2000:] or '无安装输出'}")
            await asyncio.sleep(self.poll_interval)
        await self._stop_remote_install(ssh, pid_path)
        raise RuntimeError(f"远端 vLLM 安装超时：{last_tail[-1000:]}")

    async def _stop_remote_install(self, ssh: YouyunSSHClient, pid_path: str) -> None:
        await asyncio.to_thread(
            ssh.run,
            f"if [ -f {shlex.quote(pid_path)} ]; then p=$(cat {shlex.quote(pid_path)}); "
            "kill -TERM -$p 2>/dev/null || kill -TERM $p 2>/dev/null || true; "
            "sleep 1; kill -KILL -$p 2>/dev/null || kill -KILL $p 2>/dev/null || true; fi; true",
            30,
            False,
        )

    async def process_alive(self, handle: YouyunInferenceHandle) -> bool | None:
        try:
            ssh = await self._ensure_ssh(handle)
            code, _ = await asyncio.to_thread(
                ssh.run,
                f"if [ -s {shlex.quote(handle.pid_path)} ] && kill -0 $(cat {shlex.quote(handle.pid_path)}) 2>/dev/null; then exit 0; else exit 1; fi",
                10,
                False,
            )
            handle.last_probe_error = ""
            return code == 0
        except Exception as exc:  # noqa: BLE001 - transient SSH failures are not process exits
            self._drop_ssh(handle)
            handle.last_probe_error = _exception_detail(exc)
            return None

    @staticmethod
    def _background_launch_command(
        command: list[str],
        *,
        workdir: str,
        pid_path: str,
        log_path: str,
    ) -> str:
        quoted_pid = shlex.quote(pid_path)
        return (
            f"cd {shlex.quote(workdir)} || exit 1; "
            f"rm -f {quoted_pid}; "
            f"nohup {shlex.join(command)} > {shlex.quote(log_path)} 2>&1 </dev/null & "
            f"pid=$!; printf '%s\\n' \"$pid\" > {quoted_pid}"
        )

    async def start(
        self,
        service_id: str,
        *,
        model: str,
        adapter: str,
        resource: Mapping[str, Any],
        max_model_length: int,
        extra_args: list[str] | None = None,
        on_instance: Callable[[YouyunInferenceHandle], None] | None = None,
        on_stage: Callable[..., None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_log_cursor: Callable[[LogCursor], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> YouyunInferenceHandle:
        started_at = time.monotonic()
        last_stage_event_at = started_at

        def emit_stage(
            stage: str,
            message: str,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal last_stage_event_at
            now = time.monotonic()
            enriched = dict(details or {})
            enriched.setdefault("elapsedSeconds", int(now - started_at))
            enriched.setdefault("stageElapsedSeconds", int(now - last_stage_event_at))
            last_stage_event_at = now
            self._stage(on_stage, stage, message, **enriched)

        emit_stage("creating_instance", "开始检查库存并创建优云实例。")
        create_done = asyncio.Event()

        async def _report_instance_wait() -> None:
            # 实例创建 + 等待 Running 可能耗时数分钟：周期性写心跳日志，
            # 避免页面在静默期被误判为卡死。
            waited = 0
            while not create_done.is_set():
                try:
                    await asyncio.wait_for(create_done.wait(), timeout=20)
                    return
                except asyncio.TimeoutError:
                    waited += 20
                    emit_stage(
                        "creating_instance",
                        f"优云实例创建中：已等待 {waited} 秒（等待实例进入 Running 状态）。",
                    )

        instance_waiter = asyncio.create_task(_report_instance_wait())
        try:
            instance = await asyncio.to_thread(
                self.manager.create_one_shot,
                service_id,
                gpu=resource.get("gpuCount"),
                cpu=resource.get("cpuCores"),
                memory_gb=resource.get("memoryGb"),
                gpu_type=resource.get("gpuType"),
                image_id=resource.get("imageId"),
                allow_resource_fallback=resource.get("mode") == "auto",
                instance_mode=resource.get("instanceMode"),
                zone=resource.get("zone"),
            )
        finally:
            create_done.set()
            await asyncio.gather(instance_waiter, return_exceptions=True)
        workdir = posixpath.join(self.remote_root, service_id)
        handle = YouyunInferenceHandle(
            instance_id=instance.uhost_id,
            endpoint=f"http://{instance.ip}:{self.port}" if instance.ip else "",
            pid_path=posixpath.join(workdir, "vllm.pid"),
            workdir=workdir,
            log_path=posixpath.join(workdir, "vllm.log"),
        )
        if on_instance is not None:
            on_instance(handle)
        emit_stage(
            "instance_running",
            f"优云实例已就绪：{handle.instance_id}。",
            {"instanceId": handle.instance_id},
        )
        try:
            emit_stage(
                "ssh_connecting",
                "等待 SSH 连接可用。",
                {"instanceId": handle.instance_id},
            )
            ssh_done = asyncio.Event()

            async def _report_ssh_wait() -> None:
                waited = 0
                while not ssh_done.is_set():
                    try:
                        await asyncio.wait_for(ssh_done.wait(), timeout=20)
                        return
                    except asyncio.TimeoutError:
                        waited += 20
                        emit_stage(
                            "ssh_connecting",
                            f"等待 SSH 可用：已等待 {waited} 秒（实例开机后 SSH 端口才对外就绪）。",
                            {"instanceId": handle.instance_id},
                        )

            ssh_waiter = asyncio.create_task(_report_ssh_wait())
            try:
                ssh = await asyncio.to_thread(self._connect, instance)
            finally:
                ssh_done.set()
                await asyncio.gather(ssh_waiter, return_exceptions=True)
            handle.ssh = ssh
            emit_stage(
                "ssh_connected",
                "SSH 连接已建立。",
                {"instanceId": handle.instance_id},
            )
            await asyncio.to_thread(ssh.mkdirs, workdir)
            await self._diagnose_remote_runtime(ssh, on_stage=emit_stage, on_log=on_log)
            await self._prepare_vllm_runtime_async(
                ssh,
                workdir=workdir,
                on_stage=emit_stage,
                cancel_requested=cancel_requested,
            )
            await self._ensure_remote_ninja(ssh, on_stage=emit_stage, on_log=on_log)
            await self._validate_remote_gpu(
                ssh,
                on_stage=emit_stage,
                on_log=on_log,
            )
            if cancel_requested and cancel_requested():
                raise RuntimeError("服务启动已取消。")
            remote_model = await self._prepare_remote_model(
                ssh,
                model=model,
                workdir=workdir,
                on_stage=emit_stage,
                cancel_requested=cancel_requested,
                public_model_path=str(resource.get("publicModelPath") or "") or None,
            )
            remote_adapter = posixpath.join(workdir, "adapter")
            adapter_path = Path(adapter).expanduser()
            if not adapter_path.is_dir():
                raise RuntimeError(f"Adapter 路径不存在或不是目录：{adapter_path}")
            emit_stage("uploading_adapter", "开始上传 Adapter 到优云实例。")
            last_upload_percent = -1
            last_upload_report_at = time.monotonic()

            def report_adapter_progress(transferred: int, total: int, relative_path: str) -> None:
                nonlocal last_upload_percent, last_upload_report_at
                percent = 100 if total <= 0 else min(100, int(transferred * 100 / total))
                now = time.monotonic()
                if percent == last_upload_percent and now - last_upload_report_at < 10:
                    return
                if (
                    percent < 100
                    and percent < last_upload_percent + 5
                    and now - last_upload_report_at < 10
                ):
                    return
                last_upload_percent = percent
                last_upload_report_at = now
                transferred_mb = transferred / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                emit_stage(
                    "uploading_adapter",
                    f"Adapter 上传中：{percent}%（{transferred_mb:.1f} / {total_mb:.1f} MiB）· {relative_path}",
                    {
                        "progress": min(73, 68 + int(percent * 6 / 100)),
                        "bytesTransferred": transferred,
                        "bytesTotal": total,
                    },
                )

            await asyncio.to_thread(
                self._upload_tree,
                ssh,
                adapter_path,
                remote_adapter,
                on_progress=report_adapter_progress,
            )
            if cancel_requested and cancel_requested():
                raise RuntimeError("服务启动已取消。")
            emit_stage("adapter_uploaded", "Adapter 上传完成。")
            command = [
                self.remote_python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                remote_model,
                "--served-model-name",
                service_id,
                "--enable-lora",
                "--lora-modules",
                f"{service_id}={remote_adapter}",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.port),
                "--max-model-len",
                str(max_model_length),
                "--no-enable-log-requests",
                *(extra_args or []),
            ]
            log_path = posixpath.join(workdir, "vllm.log")
            launch = self._background_launch_command(
                command,
                workdir=workdir,
                pid_path=handle.pid_path,
                log_path=log_path,
            )
            emit_stage(
                "vllm_starting",
                "开始启动 vLLM 服务。",
                {"logPath": log_path},
            )
            await asyncio.to_thread(ssh.run, launch, 60)
            handle.log_python = self.remote_python
            self.ensure_log_collector(handle, on_log=on_log, on_cursor=on_log_cursor)
            emit_stage(
                "loading_model",
                "远端 vLLM 已启动，等待模型加载。",
                {"logPath": log_path},
            )
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            last_stage_at = 0.0
            while asyncio.get_running_loop().time() < deadline:
                process_alive = await self.process_alive(handle)
                if process_alive is False:
                    tail = await self.read_log_tail(handle)
                    detail = _remote_failure_summary(tail) or "远端未返回进程日志。"
                    raise RuntimeError(f"远端 vLLM 进程已退出：{detail}")
                if await self.health(handle):
                    if not handle.endpoint:
                        raise RuntimeError("优云实例已启动，但没有可用 IP。")
                    emit_stage("health_check_passed", "远端 /health 检查通过。")
                    return handle
                now = time.monotonic()
                if now - last_stage_at >= 10:
                    last_stage_at = now
                    message = "远端 vLLM 仍在加载模型。"
                    if process_alive is None:
                        message = (
                            "远端 SSH 探测暂时不可用，继续等待模型加载："
                            f"{handle.last_probe_error or '未知连接错误'}"
                        )
                    emit_stage(
                        "loading_model",
                        message,
                    )
                await asyncio.sleep(self.poll_interval)
            tail = await self.read_log_tail(handle)
            summary = _remote_failure_summary(tail)
            detail = f"；远端日志摘要：{summary}" if summary else "；远端未返回 vLLM 日志。"
            emit_stage(
                "health_check_failed",
                "远端 /health 检查超时。",
                {"logTail": tail[-4000:]},
            )
            raise RuntimeError(f"优云远程 vLLM 启动超时，未通过 /health 检查{detail}")
        except asyncio.CancelledError:
            await self.close_log_collector(handle)
            raise
        except Exception as exc:
            detail = _exception_detail(exc)
            tail = await self.read_log_tail(handle)
            summary = _remote_failure_summary(tail)
            if summary and summary not in detail:
                detail = f"{detail}；远端日志摘要：{summary}"
            try:
                report = await self.stop(handle)
            finally:
                await self.close_log_collector(handle, drain=True)
            raise YouyunInferenceStartError(detail, handle, report) from exc

    def ensure_log_collector(
        self, handle: YouyunInferenceHandle, *,
        on_log: Callable[[str], None] | None,
        on_cursor: Callable[[LogCursor], None] | None = None,
    ) -> RemoteLogCollector | None:
        if handle.instance_id in self._closed_log_instances:
            return None
        existing = self._log_collectors.get(handle.instance_id)
        if existing is not None:
            return existing

        def connect():
            # A fresh dedicated transport, not the health/chat/SFTP connection.
            instance = self.manager.client.describe_instance(handle.instance_id)
            credentials = self.manager.ssh_credentials(instance)
            ssh = self._new_ssh_client()
            try:
                ssh.connect(**credentials)
            except BaseException:
                ssh.close()
                raise
            return ssh

        def consume(line: str) -> None:
            cleaned = _clean_remote_log_line(line)
            if cleaned and on_log is not None:
                on_log(cleaned[-4000:])

        def checkpoint(cursor: LogCursor) -> None:
            if on_cursor is not None:
                on_cursor(cursor)
            handle.log_offset, handle.log_file_id = cursor.offset, cursor.file_id

        collector = RemoteLogCollector(
            connect, handle.log_path or posixpath.join(handle.workdir, "vllm.log"),
            python=handle.log_python, on_line=consume,
            cursor=LogCursor(handle.log_offset, handle.log_file_id),
            on_cursor=checkpoint, on_status=on_log,
        )
        self._log_collectors[handle.instance_id] = collector
        collector.start()
        return collector

    async def close_log_collector(self, handle: YouyunInferenceHandle, *, drain: bool = False) -> None:
        self._closed_log_instances.add(handle.instance_id)
        collector = self._log_collectors.get(handle.instance_id)
        if collector is not None:
            await collector.close(drain=drain)
            if self._log_collectors.get(handle.instance_id) is collector:
                self._log_collectors.pop(handle.instance_id, None)

    async def read_logs(
        self,
        handle: YouyunInferenceHandle,
        offset: int = 0,
        *,
        max_bytes: int = 256_000,
    ) -> tuple[str, int]:
        """Read newly appended bytes from remote vllm.log and return (text, offset)."""
        path = shlex.quote(handle.log_path or posixpath.join(handle.workdir, "vllm.log"))
        try:
            ssh = await self._ensure_ssh(handle)
            code, size_text = await asyncio.to_thread(
                ssh.run, f"if [ -f {path} ]; then wc -c < {path}; else echo 0; fi", 10, False
            )
            if code != 0:
                return "", offset
            try:
                size = int(size_text.strip().splitlines()[-1])
            except (ValueError, IndexError):
                return "", offset
            if size < offset:
                offset = 0
            if size <= offset:
                return "", offset
            start = offset
            code, text = await asyncio.to_thread(
                ssh.run,
                f"tail -c +{start + 1} {path} | head -c {int(max_bytes)}",
                20,
                False,
            )
            if code != 0:
                return "", offset
            consumed = len(text.encode("utf-8", errors="replace"))
            return text, start + consumed
        except Exception as exc:  # noqa: BLE001 - logging must not break lifecycle
            self._drop_ssh(handle)
            handle.last_probe_error = _exception_detail(exc)
            return "", offset

    async def read_log_tail(self, handle: YouyunInferenceHandle, lines: int = 80) -> str:
        path = shlex.quote(handle.log_path or posixpath.join(handle.workdir, "vllm.log"))
        try:
            ssh = await self._ensure_ssh(handle)
            code, output = await asyncio.to_thread(ssh.run, f"tail -n {int(lines)} {path}", 20, False)
            return output if code == 0 else ""
        except Exception as exc:  # noqa: BLE001
            self._drop_ssh(handle)
            handle.last_probe_error = _exception_detail(exc)
            return ""

    @staticmethod
    def _drop_ssh(handle: YouyunInferenceHandle) -> None:
        if handle.ssh is not None:
            try:
                handle.ssh.close()
            except Exception:  # noqa: BLE001 - connection cleanup is best effort
                pass
        handle.ssh = None

    async def _ensure_ssh(self, handle: YouyunInferenceHandle) -> YouyunSSHClient:
        if handle.ssh is not None:
            return handle.ssh
        instance = await asyncio.to_thread(self.manager.client.describe_instance, handle.instance_id)
        handle.ssh = await asyncio.to_thread(self._connect, instance)
        return handle.ssh

    async def health(self, handle: YouyunInferenceHandle) -> bool:
        ssh = handle.ssh
        if ssh is None:
            try:
                instance = await asyncio.to_thread(
                    self.manager.client.describe_instance, handle.instance_id
                )
                ssh = await asyncio.to_thread(self._connect, instance)
                handle.ssh = ssh
            except Exception:  # noqa: BLE001 - 健康探测返回布尔值
                return False
        try:
            code, _ = await asyncio.to_thread(
                ssh.run,
                f"curl -fsS http://127.0.0.1:{self.port}/health",
                10,
                False,
            )
            return code == 0
        except Exception as exc:  # noqa: BLE001
            self._drop_ssh(handle)
            handle.last_probe_error = _exception_detail(exc)
            return False

    async def stop(self, handle: YouyunInferenceHandle) -> dict[str, Any]:
        process_error: str | None = None
        ssh = handle.ssh
        try:
            if ssh is None:
                instance = await asyncio.to_thread(
                    self.manager.client.describe_instance, handle.instance_id
                )
                ssh = await asyncio.to_thread(self._connect, instance)
            if handle.pid_path:
                await asyncio.to_thread(
                    ssh.run,
                    f"if [ -f {shlex.quote(handle.pid_path)} ]; then "
                    f"kill -TERM $(cat {shlex.quote(handle.pid_path)}) 2>/dev/null || true; fi",
                    30,
                    False,
                )
        except Exception as exc:  # noqa: BLE001 - API 删除仍必须执行
            process_error = str(exc)
        finally:
            await self.close_log_collector(handle, drain=True)
            if ssh is not None:
                ssh.close()
            handle.ssh = None
        try:
            report = await asyncio.to_thread(
                self.manager.cleanup_after_task,
                handle.instance_id,
                stop_first=False,
            )
        except Exception as exc:  # noqa: BLE001
            report = {"terminated": False, "error": str(exc)}
        if process_error:
            report["processError"] = process_error
        return report


__all__ = [
    "YouyunInferenceHandle",
    "YouyunInferenceProvider",
    "YouyunInferenceStartError",
]
