"""优云远程 LLaMA-Factory 训练编排器。

生命周期（需求 10.2 / 29 节）：
创建一次性实例 → SSH 就绪 → 安装 LLaMA-Factory → 上传数据集与 YAML →
远程启动训练 → SSH 轮询日志回传（日志 + 指标，解析逻辑与本地执行器一致）→
下载产物到本地镜像 → 成功或停止且有 Adapter 时保留实例等待人工对比；其余路径清理并删除实例。

sink 协议与 backend.store 一致：append_log / append_metric / update /
is_cancel_requested / metrics。凭据与实例密码不写入日志。
"""
from __future__ import annotations

import asyncio

import json
import logging
import os
import posixpath
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.providers.youyun.log_follow import RemoteLogCollector

from backend.distributed import build_deepspeed_zero2_config
from backend.store import (
    NUM_EPOCHS_PATTERN,
    TOTAL_STEPS_PATTERN,
    infer_metric_step,
    parse_trainer_values,
)

from .instance_manager import YouyunInstanceManager
from .models import YouyunApiError, YouyunInstance
from .ssh import YouyunSSHClient

logger = logging.getLogger(__name__)

_INSTALL_COMMAND = (
    "mkdir -p /root/cppt; PY=''; "
    # 优云 CUDA 镜像的 torch 位于 miniconda env（如 /usr/local/miniconda3/envs/py312）
    "for c in /usr/local/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python "
    "/opt/conda/bin/python /root/miniconda3/envs/*/bin/python /root/miniconda3/bin/python "
    "/usr/local/bin/python3 /usr/bin/python3; do "
    "if [ -x \"$c\" ] && \"$c\" -c 'import torch' >/dev/null 2>&1; then PY=\"$c\"; break; fi; done; "
    "if [ -n \"$PY\" ]; then "
    "echo \"$PY\" > /root/cppt/python-path; "
    "\"$PY\" -m pip --version >/dev/null 2>&1 || \"$PY\" -m ensurepip --upgrade; "
    "NEED=''; "
    "\"$PY\" -c 'import llamafactory' >/dev/null 2>&1 || NEED=\"$NEED llamafactory\"; "
    "\"$PY\" -c 'import modelscope' >/dev/null 2>&1 || NEED=\"$NEED modelscope\"; "
    "\"$PY\" -c 'import tensorboard' >/dev/null 2>&1 || NEED=\"$NEED tensorboard\"; "
    "\"$PY\" -c 'import bitsandbytes' >/dev/null 2>&1 || NEED=\"$NEED bitsandbytes\"; "
    "if [ -n \"$NEED\" ]; then \"$PY\" -m pip install --no-cache-dir $NEED; fi; "
    # Single-card LLaMA-Factory training does not require DeepSpeed. Some
    # images ship the optional package without the CUDA toolkit's nvcc binary;
    # importing transformers then fails before training starts. Remove it only
    # in that unsupported-toolkit case so images with a real nvcc keep it.
    "if ! command -v nvcc >/dev/null 2>&1 && \"$PY\" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec(\"deepspeed\") else 1)' >/dev/null 2>&1; then "
    "echo 'deepspeed detected without nvcc; removing optional dependency'; \"$PY\" -m pip uninstall -y deepspeed >/dev/null 2>&1 || true; fi; "
    "echo \"python=$PY\"; \"$PY\" -c 'import sys; print(\"python_version=\" + sys.version.split()[0])'; "
    "\"$PY\" -c 'import torch; print(\"torch_version=\" + torch.__version__ + \" cuda=\" + str(torch.version.cuda))'; "
    "\"$PY\" -c 'import llamafactory; print(\"llamafactory=\" + str(getattr(llamafactory, \"__version__\", \"installed\")))' 2>/dev/null || true; "
    "for c in /root/cppt/modelscope-cache /root/.cache/modelscope /workspace/model-cache /root/modelscope-cache; do "
    "if [ -d \"$c\" ]; then echo \"modelscope_cache=$c size=$(du -sh \"$c\" 2>/dev/null | cut -f1)\"; fi; done; "
    "else echo 'no torch python found' >&2; false; fi"
)
_PYTHON_PATH_FILE = "/root/cppt/python-path"
_MODEL_CONTEXT_KEYS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_sequence_length",
    "seq_length",
    "max_context_length",
)


def _gpu_supports_bf16(gpu_type: str | None) -> bool:
    """Return True only for GPU families with reliable native BF16 support.

    Unknown models stay on FP16 so adding a new inventory entry cannot make a
    remote task fail solely because the platform guessed an unsupported dtype.
    """
    normalized = re.sub(r"[^a-z0-9]", "", (gpu_type or "").lower())
    if not normalized:
        return False
    return (
        normalized.startswith(("a10", "a16", "a30", "a40", "a100", "a800"))
        or normalized.startswith(("h20", "h100", "h800", "l4", "l40"))
        or normalized.startswith(("rtx30", "rtx40", "rtx50", "3090", "4090", "5090"))
        or normalized.startswith(("30", "40", "50"))
    )


def _selected_gpu_count(task: Mapping[str, Any]) -> int:
    """任务选择的目标 GPU 数；缺省或非法值回退 1，保持单卡行为。"""
    try:
        return max(1, int((task.get("youyunResource") or {}).get("gpuCount") or 1))
    except (TypeError, ValueError):
        return 1


def _probed_gpu_count(gpu_info: str) -> int:
    """从 nvidia-smi 输出统计实际可见 GPU 数。"""
    return len([line for line in gpu_info.strip().splitlines() if line.strip()])


def _ddp_launch_env(gpu_count: int) -> str:
    """多卡任务的受控 DDP 启动前缀（需求 30.3.2）；单卡返回空串保持原启动方式。

    全部变量由平台拼接，用户输入不进入 Shell（需求 30.3.3）。LLaMA-Factory CLI
    读取 FORCE_TORCHRUN=1 后改用 torchrun 启动，world size 即 NPROC_PER_NODE。
    """
    if gpu_count <= 1:
        return ""
    visible = ",".join(str(index) for index in range(gpu_count))
    return (
        f"FORCE_TORCHRUN=1 NPROC_PER_NODE={gpu_count} "
        f"MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES={visible} "
    )


class _RemoteStopRequested(RuntimeError):
    """内部控制流：用户在远程准备阶段请求停止。"""


class _RemoteReadinessUnavailable(RuntimeError):
    """测试替身或旧 Worker 未返回 readiness probe 的兼容信号。"""


class _RemoteAdapterMissing(RuntimeError):
    """The remote probe positively confirmed that no loadable Adapter exists."""


class _LogParser:
    """与本地执行器 events() 相同的单行解析（总步数/轮数/指标）。"""

    def __init__(self, logging_steps: int) -> None:
        self.total_steps: int | None = None
        self.total_epochs: float | None = None
        self.last_train_step = 0
        self.logging_steps = max(1, logging_steps)

    def parse(self, text: str) -> tuple[str, dict[str, Any]]:
        """返回 (event_type, payload)；payload 为指标时含 step/total_steps/progress。"""
        steps_match = TOTAL_STEPS_PATTERN.search(text)
        if steps_match:
            self.total_steps = int(steps_match.group(1).replace(",", ""))
        epochs_match = NUM_EPOCHS_PATTERN.search(text)
        if epochs_match:
            self.total_epochs = float(epochs_match.group(1))
        values = parse_trainer_values(text)
        if values:
            step = infer_metric_step(
                values,
                total_steps=self.total_steps,
                total_epochs=self.total_epochs,
                last_train_step=self.last_train_step,
                logging_steps=self.logging_steps,
            )
            if "loss" in values:
                self.last_train_step = step
            payload: dict[str, Any] = {**values, "step": step}
            if self.total_steps:
                payload["total_steps"] = self.total_steps
                payload["progress"] = min(95, int(step * 95 / self.total_steps))
            return "metric", payload
        return "log", {"raw": text}


def _posix_quote(text: str) -> str:
    """POSIX 单引号转义（用于服务端控制的脚本片段）。"""
    return "'" + text.replace("'", "'\\''") + "'"


_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _validate_task_id(task_id: str) -> str:
    if not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("taskId 包含不安全字符。")
    return task_id


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _mask_host(host: str | None) -> str:
    """日志中只保留 IP 的前三段，避免把远程地址写入任务详情。"""
    if not host:
        return "<unknown>"
    parts = host.split(".")
    return ".".join([*parts[:3], "x"]) if len(parts) == 4 else "<present>"


def _connection_error(exc: Exception, stage: str) -> tuple[str, bool]:
    """把底层异常归类为可展示阶段和是否值得重试。"""
    name = type(exc).__name__.lower()
    detail = str(exc).strip() or name
    lowered = detail.lower()
    if "auth" in name or "authentication" in lowered or "permission denied" in lowered:
        return f"认证失败：{detail}", False
    if "hostkey" in name or "bad host key" in lowered:
        return f"主机密钥校验失败：{detail}", False
    if stage == "tcp":
        return f"TCP 连接失败：{detail}", True
    if "banner" in lowered or "eof" in name or "eof" in lowered:
        return f"SSH 协议握手未完成：{detail}", True
    if "timeout" in name or "timed out" in lowered or "connection reset" in lowered or "connection refused" in lowered:
        return f"SSH 连接失败：{detail}", True
    # Paramiko 的 SSHException 还可能包装云端 sshd 尚未就绪的瞬态错误。
    return f"SSH 连接失败：{detail}", True


class YouyunTrainingRunner:
    def __init__(
        self,
        *,
        manager: YouyunInstanceManager,
        sink: Any,
        ssh_factory: Callable[[], YouyunSSHClient] = YouyunSSHClient,
        remote_root: str = "/root/cppt",
        mirror_root: Path | None = None,
        prepare_local: Callable[[Mapping[str, Any]], Any] | None = None,
        collect_local: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None = None,
        poll_interval: float = 5.0,
        ssh_ready_timeout: float | None = None,
        ssh_connect_timeout: float | None = None,
        install_timeout: float = 3600.0,
        training_timeout: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_retained: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.manager = manager
        self.sink = sink
        self.ssh_factory = ssh_factory
        self.remote_root = remote_root.rstrip("/")
        self.mirror_root = (mirror_root or Path(os.environ.get("XUNTUI_REMOTE_MIRROR_ROOT", "./runtime/remote-jobs"))).expanduser().resolve()
        self.prepare_local = prepare_local
        self.collect_local = collect_local
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
        self.install_timeout = install_timeout
        self.training_timeout = training_timeout if training_timeout is not None else float(os.environ.get("YOUYUN_TRAINING_TIMEOUT_SECONDS", 86400))
        self._sleep = sleep
        self._clock = clock
        self._stopped = False
        self.on_retained = on_retained
        self._retained_callbacks: set[asyncio.Task[None]] = set()

    # ---- 对外主入口 ----

    async def run(self, task_id: str, task: Mapping[str, Any]) -> str:
        _validate_task_id(task_id)
        self._stopped = False
        instance: YouyunInstance | None = None
        ssh: YouyunSSHClient | None = None
        keep_instance = False
        try:
            self.sink.update(task_id, current_step="正在检查库存并创建优云 GPU 实例")
            resource = dict(task.get("youyunResource") or {})
            gpu_count = _selected_gpu_count(task)
            strategy = str(resource.get("distributedStrategy") or "single")
            if gpu_count != 1 or strategy != "single":
                raise RuntimeError("当前版本暂不支持多卡训练，请选择 1 张 GPU。")
            # Historical one-card tasks may carry ddp/zero2 metadata. Keep
            # them runnable, but normalize the active execution to single-card.
            resource["distributedStrategy"] = "single"
            try:
                instance = await asyncio.to_thread(
                    self.manager.create_one_shot,
                    task_id,
                    gpu=resource.get("gpuCount"),
                    cpu=resource.get("cpuCores"),
                    memory_gb=resource.get("memoryGb"),
                    gpu_type=resource.get("gpuType"),
                    image_id=resource.get("imageId"),
                    allow_resource_fallback=resource.get("mode") == "auto",
                    instance_mode=resource.get("instanceMode"),
                    zone=resource.get("zone"),
                )
            except YouyunApiError as exc:
                stage = "优云库存检查" if exc.action == "CheckCompShareResourceCapacity" else "优云实例创建"
                message = f"{stage}失败：{exc.message}（动作={exc.action}，RetCode={exc.ret_code}）"
                self.sink.append_log(task_id, message)
                self.sink.update(task_id, state="failed", current_step=stage + "失败", error=message)
                return "failed"
            self.sink.append_log(
                task_id,
                f"优云实例已就绪：{instance.uhost_id}（{instance.gpu_type} ×{instance.gpu}，{instance.zone}）",
            )
            self._record_instance(task_id, instance, state="running")

            if self.sink.is_cancel_requested(task_id):
                self._stopped = True
                raise _RemoteStopRequested()
            ssh = await asyncio.to_thread(self._connect_ssh, instance, task_id)
            _, gpu_info = await asyncio.to_thread(ssh.run, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true", 60)
            probed_count = _probed_gpu_count(gpu_info)
            if gpu_info.strip():
                self.sink.append_log(task_id, f"GPU（探测到 {probed_count} 张）：{gpu_info.strip().splitlines()[0]}")
            if gpu_count > 1:
                # 需求 30.3.1：探测数不足时任务失败，走 finally 的实例清理。
                if probed_count < gpu_count:
                    raise RuntimeError(
                        f"实例仅探测到 {probed_count} 张 GPU，少于任务选择的 {gpu_count} 张，无法执行单实例多卡 DDP 训练。"
                    )
                await self._log_distributed_diagnostics(task_id, ssh, gpu_count)
                self._set_instance_state(
                    task_id,
                    instance.uhost_id,
                    "running",
                    extra={"probedGpuCount": probed_count, "distributedMode": "ddp", "worldSize": gpu_count},
                )
            if self.sink.is_cancel_requested(task_id):
                self._stopped = True
                raise _RemoteStopRequested()

            normalized_task = dict(task)
            normalized_task["youyunResource"] = resource
            await self._install_environment(task_id, normalized_task, ssh)
            public_model_path = await self._inspect_model_cache(task_id, normalized_task, ssh)
            handle = await self._upload_task(task_id, normalized_task, ssh, model_path=public_model_path)
            returncode = await self._train_and_follow(task_id, normalized_task, ssh, handle)

            if self._stopped:
                retained = await self._try_collect_artifacts(task_id, task, ssh, handle)
                if retained:
                    self._retain_instance(task_id, instance.uhost_id)
                    self._notify_retained(task_id, task)
                    keep_instance = True
                elif retained is None:
                    # An unavailable SSH/download is not evidence that weights
                    # are absent. Keep the only possible copy until retry or expiry.
                    self._set_instance_state(task_id, instance.uhost_id, "awaiting_comparison", extra={
                        "retainedAt": datetime.now(timezone.utc).isoformat(),
                        "comparisonReady": False,
                    })
                    self.sink.update(task_id, current_step="训练已停止，权重状态待确认（实例保留，请重试回传或手动释放）")
                    self.sink.append_log(task_id, "无法确认远端权重状态，实例暂不删除；请重试产物回传或手动释放，保留超时后会自动释放。")
                    keep_instance = True
                return "stopped"
            if returncode != 0:
                message = f"远程训练进程异常退出，返回码 {returncode}。"
                self.sink.update(
                    task_id,
                    state="failed",
                    current_step="远程训练失败",
                    error=message,
                )
                return "failed"
            # The remote files are authoritative as soon as the training
            # process exits and the adapter readiness check succeeds. Do not
            # block the user on the potentially large SSH download.
            try:
                remote_artifacts = await self._inspect_remote_artifacts(task_id, task, ssh, handle)
            except _RemoteReadinessUnavailable:
                # Older agents and lightweight test doubles do not understand
                # the probe command. Keep the old safe behavior for them: do
                # the full collection before marking the task complete.
                await self._collect_artifacts(task_id, task, ssh, handle)
                remote_artifacts = [
                    {**item, "syncState": "available", "syncError": None}
                    for item in self._artifacts(task_id)
                    if item.get("type") == "adapter"
                ]
            existing = [item for item in self._artifacts(task_id) if item.get("type") != "adapter"]
            self.sink.update(task_id, artifacts=[*existing, *remote_artifacts])
            self._set_instance_state(
                task_id,
                instance.uhost_id,
                "awaiting_comparison",
                extra={
                    "retainedAt": datetime.now(timezone.utc).isoformat(),
                    "comparisonReady": True,
                },
            )
            self.sink.update(
                task_id,
                progress=100,
                current_step="训练完成，等待效果对比（已可用，权重正在后台回传）",
            )
            self._notify_retained(task_id, task)
            keep_instance = True
            return "completed"
        except _RemoteStopRequested:
            self._stopped = True
            self.sink.update(
                task_id,
                state="stopped",
                current_step="训练已停止",
                cancel_requested=False,
                stop_reason="manual",
            )
            return "stopped"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("优云远程训练失败：%s", task_id)
            self.sink.append_log(task_id, f"优云远程训练错误：{exc}")
            self.sink.update(task_id, state="failed", current_step="远程训练失败", error=str(exc))
            return "failed"
        finally:
            if ssh is not None:
                ssh.close()
            if instance is not None and not keep_instance:
                await self._cleanup_instance(task_id, instance.uhost_id)
            elif instance is not None:
                self.sink.append_log(
                    task_id,
                    f"优云实例 {instance.uhost_id} 暂不释放，等待效果对比完成后人工确认。",
                )

    # ---- 各阶段 ----

    def _notify_retained(self, task_id: str, task: Mapping[str, Any]) -> None:
        """训练成功保留实例后异步触发预热回调；失败只记日志，不影响任务终态。"""
        if self.on_retained is None:
            return

        async def _runner() -> None:
            try:
                # The callback must observe the just-persisted remote adapter
                # artifacts rather than the stale pre-training task snapshot.
                callback_task = self.sink.get(task_id) if callable(getattr(self.sink, "get", None)) else None
                await self.on_retained(task_id, callback_task or task)
            except Exception as exc:  # noqa: BLE001 - 预热失败不影响训练结果
                logger.warning("优云对比预热回调失败：%s %s", task_id, exc)
                try:
                    self.sink.append_log(task_id, f"优云对比模型预热失败：{exc}")
                except Exception:  # noqa: BLE001
                    pass


        try:
            callback_task = asyncio.get_running_loop().create_task(_runner())
        except RuntimeError:
            return
        self._retained_callbacks.add(callback_task)
        callback_task.add_done_callback(self._retained_callbacks.discard)

    async def _log_distributed_diagnostics(self, task_id: str, ssh: YouyunSSHClient, gpu_count: int) -> None:
        """多卡任务启动前记录执行拓扑与环境诊断（需求 30.3.4，输出已脱敏）。"""
        self.sink.append_log(task_id, f"分布式模式：单实例多卡 DDP/ZeRO，world size（NPROC_PER_NODE）={gpu_count}。")
        _, versions = await asyncio.to_thread(
            ssh.run,
            f"PY=$(cat {_PYTHON_PATH_FILE} 2>/dev/null); if [ -n \"$PY\" ]; then \"$PY\" -c "
            "'import torch; print(\"torch=\" + torch.__version__, \"cuda=\" + str(torch.version.cuda), "
            "\"nccl=\" + \".\".join(map(str, torch.cuda.nccl.version())))' 2>/dev/null || true; fi",
            30,
            False,
        )
        for line in versions.strip().splitlines():
            if line.strip():
                self.sink.append_log(task_id, f"分布式环境：{line.strip()}")
        _, topology = await asyncio.to_thread(ssh.run, "nvidia-smi topo -m 2>/dev/null | head -30 || true", 30, False)
        if topology.strip():
            self.sink.append_log(task_id, "GPU 拓扑（nvidia-smi topo -m）：\n" + topology.strip())

    def _connect_ssh(self, instance: YouyunInstance, task_id: str | None = None) -> YouyunSSHClient:
        creds = self.manager.ssh_credentials(instance)
        if not creds["host"] or not creds["password"]:
            raise RuntimeError("实例缺少 SSH 连接信息（IP 或密码为空）。")
        deadline = self._clock() + self.ssh_ready_timeout
        last_error: Exception | None = None
        last_stage = "准备连接"
        attempts = 0
        started_at = self._clock()
        while self._clock() < deadline:
            if task_id and self.sink.is_cancel_requested(task_id):
                self._stopped = True
                raise _RemoteStopRequested()

            # Running 后再次查询，避免优云稍后才补齐/变更公网 SSH 转发信息。
            get_instance = getattr(self.manager, "get", None)
            if callable(get_instance):
                try:
                    refreshed = get_instance(instance.uhost_id)
                except Exception:  # noqa: BLE001 - 保留旧凭据继续重试
                    refreshed = None
                if refreshed is not None:
                    refreshed_creds = self.manager.ssh_credentials(refreshed)
                    if refreshed_creds.get("host") and refreshed_creds.get("password"):
                        creds = refreshed_creds

            attempts += 1
            endpoint = f"{_mask_host(str(creds['host']))}:{int(creds.get('port') or 22)}"
            elapsed = max(0, int(self._clock() - started_at))
            if task_id:
                self.sink.update(task_id, current_step=f"等待 SSH 连接（第 {attempts} 次，{endpoint}，已等待 {elapsed}s）")

            # 真实 Paramiko 路径先探测 TCP；注入的测试/集成工厂不做外部网络探测。
            if self.ssh_factory is YouyunSSHClient:
                try:
                    with socket.create_connection(
                        (str(creds["host"]), int(creds.get("port") or 22)),
                        timeout=min(self.ssh_connect_timeout, max(1.0, deadline - self._clock())),
                    ):
                        pass
                except Exception as exc:  # noqa: BLE001 - 端口尚未可达时继续等待
                    last_stage = "TCP 连接"
                    last_error = exc
                    message, retryable = _connection_error(exc, "tcp")
                    if task_id:
                        self.sink.append_log(task_id, f"SSH 第 {attempts} 次（{endpoint}）：{message}")
                    if not retryable:
                        raise RuntimeError(f"SSH 连接失败（{endpoint}）：{message}") from exc
                    self._sleep(self.poll_interval)
                    continue

            ssh = self._new_ssh_client()
            try:
                ssh.connect(creds["host"], creds["port"], creds["user"], creds["password"])
                if task_id and attempts > 1:
                    self.sink.append_log(task_id, f"SSH 连接已建立（{endpoint}，第 {attempts} 次）。")
                return ssh
            except Exception as exc:  # noqa: BLE001 - sshd 可能尚未就绪，重试
                last_stage = "SSH 协议握手"
                last_error = exc
                ssh.close()
                message, retryable = _connection_error(exc, "ssh")
                if task_id:
                    self.sink.append_log(task_id, f"SSH 第 {attempts} 次（{endpoint}）：{message}")
                if not retryable:
                    raise RuntimeError(f"SSH 连接失败（{endpoint}）：{message}") from exc
                self._sleep(self.poll_interval)
        endpoint = f"{_mask_host(str(creds['host']))}:{int(creds.get('port') or 22)}"
        waited = max(0, int(self._clock() - started_at))
        detail = str(last_error).strip() if last_error else "未收到 SSH 响应"
        message = f"SSH 连接超时（{endpoint}）：阶段={last_stage}，已等待 {waited}s，尝试 {attempts} 次，最后错误：{detail}"
        if task_id:
            self.sink.append_log(task_id, message)
        raise RuntimeError(message)

    def _new_ssh_client(self) -> YouyunSSHClient:
        try:
            return self.ssh_factory(connect_timeout=self.ssh_connect_timeout)
        except TypeError:
            return self.ssh_factory()

    async def _stop_remote_install(self, ssh: YouyunSSHClient) -> None:
        await asyncio.to_thread(
            ssh.run,
            "if [ -f /tmp/env-install.pid ]; then "
            "p=$(cat /tmp/env-install.pid); "
            "kill -TERM -$p 2>/dev/null || kill -TERM $p 2>/dev/null || true; "
            "sleep 1; kill -KILL -$p 2>/dev/null || kill -KILL $p 2>/dev/null || true; "
            "fi; true",
            60,
            False,
        )

    async def _install_environment(self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient) -> None:
        self.sink.update(task_id, current_step="正在安装 LLaMA-Factory 训练环境（首次约 3-10 分钟）")
        if self.sink.is_cancel_requested(task_id):
            self._stopped = True
            raise _RemoteStopRequested()
        # 后台化整段安装（含完成标记），SSH 立即返回。started 标记让
        # SSH 抖动后的补发保持幂等，避免重复启动 pip 安装进程。
        launch = (
            "if [ ! -f /tmp/env-install.started ]; then "
            "rm -f /tmp/env-install.done /tmp/env-install.pid; date -Iseconds > /tmp/env-install.started; "
            f"nohup bash -c {_posix_quote(_INSTALL_COMMAND + ' && echo ok > /tmp/env-install.done || echo fail > /tmp/env-install.done')} "
            "> /tmp/env-install.log 2>&1 & echo $! > /tmp/env-install.pid; fi"
        )
        await asyncio.to_thread(ssh.run, launch, 30)
        deadline = self._clock() + self.install_timeout
        last_tail = ""
        restart_attempts = 0
        while self._clock() < deadline:
            if self.sink.is_cancel_requested(task_id):
                self._stopped = True
                self.sink.update(task_id, state="stopping", current_step="正在停止远程环境安装")
                await self._stop_remote_install(ssh)
                raise _RemoteStopRequested()
            _, output = await asyncio.to_thread(
                ssh.run,
                "cat /tmp/env-install.done 2>/dev/null; echo ---; "
                "tail -n 3 /tmp/env-install.log 2>/dev/null; "
                "if [ -f /tmp/env-install.pid ]; then "
                "p=$(cat /tmp/env-install.pid); "
                "if kill -0 \"$p\" 2>/dev/null && "
                "! ps -o stat= -p \"$p\" 2>/dev/null | grep -q '^[[:space:]]*Z'; then "
                "status=alive; else status=dead; fi; "
                "else status=missing; fi; "
                "echo \"__XUNTUI_INSTALL_STATUS__=$status\"",
                60,
                False,
            )
            status_match = re.search(
                r"__XUNTUI_INSTALL_STATUS__=(alive|dead|missing)", output
            )
            install_status = status_match.group(1) if status_match else "unknown"
            if status_match:
                output = output[: status_match.start()]
            head, _, tail = output.partition("---")
            tail = tail.strip()
            if tail and tail != last_tail:
                last_tail = tail
                self.sink.append_log(task_id, f"安装进度：{tail.splitlines()[-1][:200]}")
            if "ok" in head:
                _, diagnostics = await asyncio.to_thread(
                    ssh.run,
                    "tail -n 20 /tmp/env-install.log 2>/dev/null || true",
                    60,
                    False,
                )
                if diagnostics.strip():
                    self.sink.append_log(task_id, f"远端环境指纹：{diagnostics.strip()[-3000:]}")
                self.sink.append_log(task_id, "LLaMA-Factory 环境安装完成。")
                return
            if "fail" in head:
                raise RuntimeError(f"远程依赖安装失败：{tail[-1500:]}")
            if install_status in {"dead", "missing"}:
                restart_attempts += 1
                if restart_attempts > 2:
                    detail = tail[-1500:] or "安装日志为空"
                    raise RuntimeError(f"远程依赖安装进程反复异常退出：{detail}")
                self.sink.append_log(
                    task_id,
                    f"检测到远程环境安装进程已退出，正在自动重试（{restart_attempts}/2）。",
                )
                await asyncio.to_thread(
                    ssh.run,
                    "rm -f /tmp/env-install.started /tmp/env-install.pid /tmp/env-install.done; "
                    + launch,
                    30,
                )
                last_tail = ""
            self._sleep(self.poll_interval)
        raise RuntimeError("远程依赖安装超时。")

    async def _inspect_model_cache(self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient) -> str | None:
        """Resolve the exact recorded model on the remote instance.

        Public catalog models must exist at their recorded path. Repository IDs
        use the shared remote downloader, which tries ModelScope and then the
        same ID on Hugging Face without substituting a similarly named model.
        """
        model_id = str(task.get("model") or task.get("model_name_or_path") or "").strip()
        if not model_id:
            return None
        resource = dict(task.get("youyunResource") or {})
        configured = str(resource.get("publicModelPath") or "").strip()
        _, python_output = await asyncio.to_thread(
            ssh.run, f"cat {_PYTHON_PATH_FILE} 2>/dev/null || command -v python3", 30, False
        )
        remote_python = next((line.strip() for line in python_output.splitlines() if line.strip()), "python3")
        from backend.providers.youyun.inference import YouyunInferenceProvider

        resolver = YouyunInferenceProvider(
            self.manager,
            remote_root=self.remote_root,
            remote_python=remote_python,
            poll_interval=self.poll_interval,
            model_prepare_timeout=self.install_timeout,
        )

        def report(stage: str, message: str, **_details: Any) -> None:
            if stage in {"downloading_model", "model_ready"}:
                self.sink.append_log(task_id, message)

        return await resolver._prepare_remote_model(
            ssh,
            model=model_id,
            workdir=f"{self.remote_root}/{task_id}",
            public_model_path=configured or None,
            on_stage=report,
            cancel_requested=lambda: self.sink.is_cancel_requested(task_id),
        )

    async def _upload_task(self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient, *, model_path: str | None = None) -> Any:
        if self.prepare_local is None:
            raise RuntimeError("缺少本地镜像 prepare 回调。")
        if self.sink.is_cancel_requested(task_id):
            self._stopped = True
            raise _RemoteStopRequested()
        self.sink.update(task_id, current_step="正在准备配置并上传数据集")
        handle = await self.prepare_local(task)
        if self.sink.is_cancel_requested(task_id):
            self._stopped = True
            raise _RemoteStopRequested()
        workdir = Path(handle.workdir)
        remote_workdir = f"{self.remote_root}/{task_id}"
        await asyncio.to_thread(ssh.mkdirs, f"{remote_workdir}/dataset")
        await asyncio.to_thread(ssh.mkdirs, f"{remote_workdir}/output")

        # YAML 在本地镜像上完成路径校验后，把 dataset_dir/output_dir 改写为
        # 结构一致的远端镜像路径，再上传。仅在已知支持的 GPU 上启用 BF16，
        # 其余设备使用 FP16，兼容 2080/P40/V100S 等较旧架构。
        from backend.llamafactory import dump_yaml

        config = _load_yaml_dict(workdir / "train.yaml")
        if model_path:
            config["model_name_or_path"] = model_path
        config["dataset_dir"] = f"{remote_workdir}/dataset"
        config["output_dir"] = f"{remote_workdir}/output"
        resource = dict(task.get("youyunResource") or {})
        distributed_strategy = str(resource.get("distributedStrategy") or "single")
        if distributed_strategy == "deepspeed_zero2":
            if _selected_gpu_count(task) < 2:
                raise RuntimeError("DeepSpeed ZeRO-2 至少需要 2 张 GPU。")
            config["deepspeed"] = f"{remote_workdir}/ds_zero2.json"
        gpu_type = str(resource.get("gpuType") or "")
        use_bf16 = _gpu_supports_bf16(gpu_type)
        config["fp16"] = not use_bf16
        config["bf16"] = use_bf16
        # LLaMA-Factory rejects val_size when an explicit validation dataset
        # is present. The local mirror may generate both for split datasets;
        # the uploaded validation file is authoritative on remote workers.
        if config.get("eval_dataset"):
            config.pop("val_size", None)
        remote_yaml = dump_yaml(config)
        deepspeed_config = build_deepspeed_zero2_config() if distributed_strategy == "deepspeed_zero2" else None

        def upload() -> None:
            for item in sorted((workdir / "dataset").iterdir()):
                if item.is_file():
                    ssh.put_file(item, f"{remote_workdir}/dataset/{item.name}")
            ssh.put_text(remote_yaml, f"{remote_workdir}/train.yaml")
            if deepspeed_config is not None:
                ssh.put_text(deepspeed_config, f"{remote_workdir}/ds_zero2.json")

        await asyncio.to_thread(upload)
        if self.sink.is_cancel_requested(task_id):
            self._stopped = True
            raise _RemoteStopRequested()
        self.sink.append_log(task_id, f"数据集与训练配置已上传：{remote_workdir}")
        handle.remote_workdir = remote_workdir  # type: ignore[attr-defined]
        return handle

    async def _train_and_follow(self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient, handle: Any) -> int:
        remote_workdir = getattr(handle, "remote_workdir", f"{self.remote_root}/{task_id}")
        log_path = f"{remote_workdir}/train.log"
        pid_path = f"{remote_workdir}/train.pid"
        code_path = f"{remote_workdir}/exit.code"
        model_id = str(task.get("model") or task.get("model_name_or_path") or "")
        model_name = model_id.rsplit("/", 1)[-1]
        ddp_env = _ddp_launch_env(_selected_gpu_count(task))
        runner = (
            f"cd {remote_workdir} && PY=$(cat {_PYTHON_PATH_FILE}) && "
            f"MODEL_NAME={_posix_quote(model_name)}; CACHE={self.remote_root}/modelscope-cache; "
            "for c in /root/.cache/modelscope /workspace/model-cache /root/modelscope-cache; do "
            "if [ -d \"$c\" ] && find \"$c\" -maxdepth 6 -iname \"*$MODEL_NAME*\" 2>/dev/null | grep -q .; then CACHE=\"$c\"; break; fi; done; "
            f"{ddp_env}"
            # nohup redirects stdout to a file; flush each loss immediately,
            # including Python workers spawned by the training launcher.
            "PYTHONUNBUFFERED=1 USE_MODELSCOPE_HUB=1 MODELSCOPE_CACHE=\"$CACHE\" "
            f"$PY -m llamafactory.cli train train.yaml; echo $? > {code_path}"
        )
        launch = f"rm -f {code_path}; nohup bash -c {_posix_quote(runner)} >> {log_path} 2>&1 & echo $! > {pid_path}"
        self.sink.update(task_id, current_step="正在启动远程训练进程")
        await asyncio.to_thread(ssh.run, launch, 30)

        parameters = dict(task.get("parameters") or {})
        parser = _LogParser(int(parameters.get("loggingSteps", 10)))
        try:
            _, python_path = await asyncio.to_thread(
                ssh.run, f"cat {_PYTHON_PATH_FILE} 2>/dev/null || command -v python3", 30, False,
            )
        except Exception:
            # Log setup must not fail an otherwise healthy training process.
            python_path = "python3"
            self.sink.append_log(task_id, "日志 Python 路径探测失败，采集器使用 python3 并自动重试。")
        collector = RemoteLogCollector(
            ssh.log_connection, log_path,
            python=python_path.strip().splitlines()[-1] if python_path.strip() else "python3",
            on_line=lambda line: self._ingest_chunk(task_id, parser, line),
            on_status=lambda message: self.sink.append_log(task_id, message),
        )
        collector.start()
        deadline = self._clock() + self.training_timeout
        try:
            while self._clock() < deadline:
                if self.sink.is_cancel_requested(task_id):
                    self.sink.update(task_id, state="stopping", current_step="正在停止远程训练进程")
                    await asyncio.to_thread(
                        ssh.run,
                        f"p=$(cat {pid_path} 2>/dev/null || true); "
                        f"if [ -n \"$p\" ]; then kill -TERM -$p 2>/dev/null || kill -TERM $p 2>/dev/null || true; "
                        f"sleep 5; kill -KILL -$p 2>/dev/null || kill -KILL $p 2>/dev/null || true; fi; "
                        f"echo 130 > {code_path}",
                        60, False,
                    )
                    self._stopped = True
                _, output = await asyncio.to_thread(
                    ssh.run,
                    f"echo code:$(cat {code_path} 2>/dev/null || echo none)",
                    60, False,
                )
                exit_code = None
                for line in output.splitlines():
                    if line.startswith("code:") and line[5:].strip() != "none":
                        exit_code = int(line[5:].strip())
                if exit_code is not None:
                    # Stop the live reader and finish a finite tail before terminal state.
                    await collector.close(drain=True)
                    if self._stopped:
                        self.sink.update(task_id, state="stopped", current_step="训练已停止", cancel_requested=False, stop_reason="manual")
                        return 130
                    self.sink.append_log(task_id, f"远程训练进程退出，返回码 {exit_code}。")
                    return exit_code
                if self._sleep is time.sleep:
                    await asyncio.sleep(self.poll_interval)
                else:
                    # Preserve injected test clocks without blocking the event loop.
                    await asyncio.to_thread(self._sleep, self.poll_interval)
            raise RuntimeError(f"远程训练超过时限（{int(self.training_timeout)} 秒）。")
        finally:
            await collector.close()

    def _ingest_chunk(self, task_id: str, parser: _LogParser, chunk: str) -> None:
        for raw_line in chunk.splitlines():
            text = raw_line.strip()
            if not text:
                continue
            self.sink.append_log(task_id, text)
            if "***** Running training *****" in text:
                self.sink.update(task_id, current_step="训练已开始，等待首个训练指标")
            event_type, payload = parser.parse(text)
            if event_type != "metric":
                continue
            values = payload
            if not ("loss" in values or "eval_loss" in values):
                continue
            step = int(values.get("step") or 0)
            if step <= 0:
                previous = self.sink.metrics(task_id)
                step = (previous[-1]["step"] if previous else 0) + 1
            self.sink.append_metric(
                task_id,
                step=step,
                epoch=float(values["epoch"]) if values.get("epoch") is not None else None,
                train_loss=float(values["loss"]) if values.get("loss") is not None else None,
                eval_loss=float(values["eval_loss"]) if values.get("eval_loss") is not None else None,
                learning_rate=float(values["learning_rate"]) if values.get("learning_rate") is not None else None,
            )
            total_steps = int(payload.get("total_steps") or 0)
            self.sink.update(
                task_id,
                progress=min(95, int(payload.get("progress", 0))),
                current_step=f"训练中 · Step {step}/{total_steps}" if total_steps > 0 else f"训练中 · Step {step}",
                loss=(float(values["loss"]) if values.get("loss") is not None else None),
            )

    async def _inspect_remote_artifacts(
        self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient, handle: Any
    ) -> list[dict[str, Any]]:
        """Create lightweight artifact records without downloading weights.

        The shell only returns metadata and paths. The actual bytes remain on
        the retained GPU instance so comparison can start immediately.
        """
        remote_workdir = getattr(handle, "remote_workdir", f"{self.remote_root}/{task_id}")
        output = f"{remote_workdir}/output"
        command = (
            f"OUT={_posix_quote(output)}; "
            "usable() { [ -f \"$1/adapter_config.json\" ] && "
            "{ [ -s \"$1/adapter_model.safetensors\" ] || [ -s \"$1/adapter_model.bin\" ]; }; }; "
            "final=$OUT; if ! usable \"$final\"; then final=''; "
            "final=$(find \"$OUT\" -maxdepth 1 -type d -name 'checkpoint-*' | sort -Vr | "
            "while IFS= read -r candidate; do "
            "if usable \"$candidate\"; then printf '%s' \"$candidate\"; break; fi; done); fi; "
            "[ -n \"$final\" ] || { echo 'adapter_missing'; exit 2; }; "
            "state=$OUT/trainer_state.json; [ -f \"$state\" ] || state=$final/trainer_state.json; "
            "best=''; if [ -f \"$state\" ]; then "
            "best=$(grep -o '\"best_model_checkpoint\"[[:space:]]*:[[:space:]]*\"[^\"]*\"' \"$state\" | head -1 | cut -d '\"' -f4); fi; "
            "case \"$best\" in checkpoint-*) best=$OUT/$best;; esac; "
            "usable \"$best\" || best=''; "
            "printf 'ready=1\\n'; printf 'best=%s\\n' \"$best\"; printf 'final=%s\\n' \"$final\""
        )
        _, result = await asyncio.to_thread(ssh.run, command, 60, False)
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if "adapter_missing" in lines:
            raise _RemoteAdapterMissing("远程输出目录及检查点中均未找到完整 Adapter。")
        if "weights_missing" in lines:
            raise RuntimeError("远程训练已退出，但没有找到 safetensors/bin 权重文件。")
        if not lines:
            raise _RemoteReadinessUnavailable("远程 Worker 未返回 readiness probe 结果。")
        if "ready=1" not in lines:
            raise RuntimeError("远程 Adapter 完整性检查未通过。")
        local_output = Path(handle.workdir) / "output"
        best_remote = next((line.removeprefix("best=").strip() for line in lines if line.startswith("best=")), "")
        if best_remote and not best_remote.startswith("/"):
            best_remote = f"{output}/{best_remote.lstrip('./')}"
        if best_remote:
            best_remote = posixpath.normpath(best_remote)
        if best_remote and not best_remote.startswith(output + "/"):
            best_remote = ""
        final_remote = next((line.removeprefix("final=").strip() for line in lines if line.startswith("final=")), output)
        final_remote = posixpath.normpath(final_remote)
        if final_remote != output and not final_remote.startswith(output + "/"):
            raise RuntimeError("远程 Adapter 路径不在任务输出目录内。")
        final_suffix = posixpath.relpath(final_remote, output)
        artifacts: list[dict[str, Any]] = []
        if best_remote:
            suffix = best_remote.removeprefix(output).lstrip("/")
            artifacts.append({
                "type": "adapter", "name": "best", "variant": "best", "selected": True,
                "path": str(local_output / suffix), "remotePath": best_remote,
                "syncState": "remote_ready",
            })
        artifacts.append({
            "type": "adapter", "name": "final", "variant": "final", "selected": not bool(best_remote),
            "path": str(local_output if final_suffix == "." else local_output / final_suffix), "remotePath": final_remote,
            "syncState": "remote_ready",
        })
        self.sink.append_log(task_id, "远程 Adapter 完整性检查通过：效果对比已提前开放，权重回传转入后台。")
        return artifacts

    async def _collect_artifacts(self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient, handle: Any) -> None:
        if self.collect_local is None:
            raise RuntimeError("缺少本地产物收集回调。")
        self.sink.update(task_id, current_step="正在回传训练产物")
        remote_workdir = getattr(handle, "remote_workdir", f"{self.remote_root}/{task_id}")
        local_output = Path(handle.workdir) / "output"
        await asyncio.to_thread(ssh.get_tree, f"{remote_workdir}/output", local_output)
        # Pass the trusted remote prefix without rewriting the historical YAML
        # or requiring the original dataset to still exist during a retry.
        artifacts = await self.collect_local({**dict(task), "_remoteOutputDir": f"{remote_workdir}/output"})
        for artifact in artifacts:
            if artifact.get("type") != "adapter":
                continue
            relative = Path(artifact["path"]).resolve().relative_to(local_output.resolve()).as_posix()
            artifact["remotePath"] = f"{remote_workdir}/output" + (f"/{relative}" if relative != "." else "")
            artifact["syncState"] = "available"
            artifact["syncError"] = None
        base_model_metadata = await self._read_base_model_metadata(ssh, local_output)
        if base_model_metadata:
            artifacts = [
                {**artifact, "baseModelMetadata": base_model_metadata}
                if artifact.get("type") == "adapter"
                else artifact
                for artifact in artifacts
            ]
        self.sink.append_log(task_id, f"产物已回传本地镜像：{local_output}（{len(artifacts)} 项）")
        existing = self._artifacts(task_id)
        instance_markers = [item for item in existing if item.get("type") == "youyun-instance"]
        self.sink.update(task_id, artifacts=[*artifacts, *instance_markers])

    async def _read_base_model_metadata(
        self, ssh: YouyunSSHClient, local_output: Path
    ) -> dict[str, Any] | None:
        adapter_config = local_output / "adapter_config.json"
        try:
            adapter = json.loads(adapter_config.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        base_path = adapter.get("base_model_name_or_path") if isinstance(adapter, dict) else None
        if not isinstance(base_path, str) or not posixpath.isabs(base_path):
            return None
        temporary_config = local_output.parent / ".xuntui-base-model-config.json"
        try:
            await asyncio.to_thread(
                ssh.get_file,
                posixpath.join(base_path, "config.json"),
                temporary_config,
            )
            config = json.loads(temporary_config.read_text(encoding="utf-8-sig"))
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        finally:
            temporary_config.unlink(missing_ok=True)
        if not isinstance(config, dict):
            return None
        sections = [config]
        sections.extend(
            value for value in (config.get("text_config"), config.get("llm_config"))
            if isinstance(value, dict)
        )
        for section in sections:
            for key in _MODEL_CONTEXT_KEYS:
                try:
                    value = int(section.get(key))
                except (TypeError, ValueError, OverflowError):
                    continue
                if 128 <= value <= 1_048_576:
                    return {"maxContextLength": value, "source": f"remote-config.json:{key}"}
        return None

    def _artifacts(self, task_id: str) -> list[dict[str, Any]]:
        getter = getattr(self.sink, "get", None)
        if not callable(getter):
            return []
        try:
            return list((getter(task_id) or {}).get("artifacts") or [])
        except Exception:  # noqa: BLE001 - 生命周期日志不能被兼容 store 读取失败阻断
            return []

    def _record_instance(self, task_id: str, instance: YouyunInstance, *, state: str) -> None:
        artifacts = [item for item in self._artifacts(task_id) if item.get("type") != "youyun-instance"]
        artifacts.append(
            {
                "type": "youyun-instance",
                "provider": "youyun",
                "instanceId": instance.uhost_id,
                "gpuType": instance.gpu_type,
                "gpuCount": instance.gpu,
                "distributedMode": "single",
                "distributedStrategy": "single",
                "worldSize": max(1, instance.gpu or 1),
                "zone": instance.zone,
                "state": state,
            }
        )
        self.sink.update(task_id, artifacts=artifacts)

    def _set_instance_state(
        self,
        task_id: str,
        uhost_id: str,
        state: str,
        *,
        error: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        artifacts = self._artifacts(task_id)
        updated = False
        for index, item in enumerate(artifacts):
            if item.get("type") == "youyun-instance" and item.get("instanceId") == uhost_id:
                marker = {**item, "state": state}
                if error:
                    marker["cleanupError"] = error
                else:
                    marker.pop("cleanupError", None)
                if extra:
                    marker.update(extra)
                artifacts[index] = marker
                updated = True
                break
        if updated:
            self.sink.update(task_id, artifacts=artifacts)

    def _retain_instance(self, task_id: str, uhost_id: str) -> None:
        """Keep a stopped run comparable when a usable Adapter was saved."""
        self._set_instance_state(
            task_id,
            uhost_id,
            "awaiting_comparison",
            extra={
                "retainedAt": datetime.now(timezone.utc).isoformat(),
                "comparisonReady": True,
            },
        )
        self.sink.update(
            task_id,
            current_step="训练已停止，可进行效果对比（实例待手动释放）",
        )
        self.sink.append_log(
            task_id,
            "训练已停止且 Adapter 已保存：优云实例暂不释放，可先进行效果对比；完成后请手动释放。",
        )

    async def sync_retained_artifacts(self, task_id: str, task: Mapping[str, Any]) -> None:
        """Download artifacts from a retained instance after comparison starts."""
        marker = next(
            (item for item in self._artifacts(task_id) if item.get("type") == "youyun-instance"),
            None,
        )
        if not marker or not marker.get("instanceId"):
            raise RuntimeError("任务没有可同步的优云实例。")
        if marker.get("state") not in {"awaiting_comparison", "debugging"}:
            raise RuntimeError("优云实例已释放，无法继续回传权重。")
        ssh: YouyunSSHClient | None = None
        try:
            instance = await asyncio.to_thread(
                self.manager.client.describe_instance, str(marker["instanceId"])
            )
            ssh = await asyncio.to_thread(self._connect_ssh, instance, task_id)
            _validate_task_id(task_id)
            workdir = self.mirror_root / task_id
            workdir.mkdir(parents=True, exist_ok=True)
            handle = type("RetainedHandle", (), {"workdir": str(workdir), "remote_workdir": f"{self.remote_root}/{task_id}"})()
            await self._collect_artifacts(task_id, task, ssh, handle)
        finally:
            if ssh is not None:
                ssh.close()

    async def _try_collect_artifacts(self, task_id: str, task: Mapping[str, Any], ssh: YouyunSSHClient, handle: Any) -> bool | None:
        try:
            await self._collect_artifacts(task_id, task, ssh, handle)
        except Exception as exc:  # noqa: BLE001 - 停止场景尽力回传，失败不阻断
            self.sink.append_log(task_id, f"停止后回传产物未完成：{exc}")
            try:
                remote_artifacts = await self._inspect_remote_artifacts(task_id, task, ssh, handle)
            except _RemoteAdapterMissing:
                self.sink.append_log(task_id, "已确认远端没有完整 Adapter，停止后释放实例。")
                return False
            except Exception as probe_error:
                self.sink.append_log(task_id, f"停止后远端权重检查未完成：{probe_error}")
                return None
            existing = [item for item in self._artifacts(task_id) if item.get("type") != "adapter"]
            self.sink.update(task_id, artifacts=[*existing, *[
                {**item, "syncError": str(exc)} for item in remote_artifacts
            ]])
            return bool(remote_artifacts)
        return any(
            item.get("type") == "adapter"
            and (item.get("path") or item.get("remotePath"))
            for item in self._artifacts(task_id)
        )

    async def _cleanup_instance(self, task_id: str, uhost_id: str) -> None:
        try:
            report = await asyncio.to_thread(self.manager.cleanup_after_task, uhost_id, stop_first=False)
        except Exception as exc:  # noqa: BLE001 - 清理失败也要记录并继续
            report = {"terminated": False, "error": str(exc)}
        if report.get("terminated"):
            self._set_instance_state(task_id, uhost_id, "released")
            self.sink.append_log(task_id, f"优云实例 {uhost_id} 已释放删除。")
        else:
            error = str(report.get("error") or "实例删除未确认")
            self._set_instance_state(task_id, uhost_id, "cleanup_failed", error=error)
            self.sink.append_log(
                task_id,
                f"优云实例 {uhost_id} 清理未完成：{error}（请到控制台手动删除，避免持续计费）",
            )


__all__ = ["YouyunTrainingRunner"]
