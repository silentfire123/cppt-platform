"""一次性优云 GPU 实例生命周期管理。

策略（需求 10.2/10.3、29 节确认）：
- 每个远程任务临时创建实例；任务结束、失败、停止、超时后必须删除实例。
- 创建前强制库存检查（官方要求，勿用 Describe 类接口代替）。
- 创建后轮询至 Running 并拿到 SSH 信息（IP/端口/密码）。
- terminate 永不抛异常：清理失败也要返回报告，由调用方落库。
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable

from .client import YouyunClient
from .models import GpuSpecCombination, YouyunApiError, YouyunInstance, resolve_region_zone

logger = logging.getLogger(__name__)

_PENDING_STATES = {"", "Installing", "Install", "Starting", "Pending", "Creating"}


def _decode_password(encoded: str | None) -> str | None:
    if not encoded:
        return None
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _validate_mode_zone(client: YouyunClient, *, mode: str, zone: str, zones: list[dict[str, Any]] | None = None) -> None:
    """Validate the provider's mode-specific zone capability before creation."""
    describe_support_zones = getattr(client, "describe_support_zones", None)
    if not callable(describe_support_zones):
        return
    zones = zones if zones is not None else describe_support_zones(client.config.region)
    selected = next((item for item in zones if item.get("zone") == zone), None)
    if selected is None:
        raise YouyunApiError("Config", None, f"优云可用区 {zone} 不在当前区域支持列表中。")
    # DescribeCompShareSupportZone 会忽略 Region 过滤返回全部区域的可用区，
    # 因此必须显式比对 zone 所属区域，避免 Region 与 Zone 错配被优云以
    # RetCode=230 (Params [Zone] not available) 拒绝。
    zone_region = str(selected.get("region") or "")
    configured_region = str(client.config.region or "")
    if configured_region and zone_region and zone_region != configured_region:
        raise YouyunApiError(
            "Config",
            None,
            f"优云可用区 {zone} 属于区域 {zone_region}，与当前配置区域 {configured_region} 不一致，请同步选择该区域的可用区。",
        )
    is_pod = bool(selected.get("is_pod"))
    if mode == "pod" and not is_pod:
        raise YouyunApiError("Config", None, f"Pod 模式不能使用整机可用区 {zone}。")
    if mode == "uhost" and is_pod:
        raise YouyunApiError("Config", None, f"整机 UHost 模式不能使用 Pod 可用区 {zone}。")


class YouyunInstanceManager:
    def __init__(
        self,
        client: YouyunClient,
        poll_interval: float = 5.0,
        boot_timeout: float = 900.0,
        terminate_timeout: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.poll_interval = poll_interval
        self.boot_timeout = boot_timeout
        self.terminate_timeout = terminate_timeout
        self._sleep = sleep
        self._clock = clock

    # ---- 创建 ----

    def create_one_shot(
        self,
        task_id: str,
        gpu: int | None = None,
        cpu: int | None = None,
        memory_gb: int | None = None,
        gpu_type: str | None = None,
        image_id: str | None = None,
        allow_resource_fallback: bool = False,
        name: str | None = None,
        instance_mode: str | None = None,
        zone: str | None = None,
    ) -> YouyunInstance:
        """库存检查 → 创建 → 轮询至 Running。失败抛 YouyunApiError。

        ``allow_resource_fallback`` is used by automatic scheduling. The
        selected shape is only a preference in that mode, so a retry may use
        another currently available shape. Manual selections remain strict.
        """
        selected_mode = (instance_mode if instance_mode in {"uhost", "pod"} else None) or self.client.config.instance_mode or "uhost"
        selected_mode = selected_mode.lower()
        if selected_mode not in {"uhost", "pod"}:
            selected_mode = "uhost"
        selected_zone = zone
        if not selected_zone:
            zones = self.client.describe_support_zones(self.client.config.region)
            preferred_zone = self.client.config.pod_zone if selected_mode == "pod" else self.client.config.zone
            selected_zone = resolve_region_zone(
                zones,
                region=self.client.config.region,
                mode=selected_mode,
                preferred_zone=preferred_zone,
            )
        else:
            zones = None
        if not selected_zone:
            raise YouyunApiError(
                "Config",
                None,
                f"区域 {self.client.config.region} 没有可用的 {selected_mode.upper()} 可用区。",
            )
        _validate_mode_zone(self.client, mode=selected_mode, zone=selected_zone, zones=zones)
        capacity_kwargs: dict[str, Any] = {"gpu_type": gpu_type, "image_id": image_id}
        capacity_kwargs["zone"] = selected_zone
        logger.info(
            f"create_one_shot: mode={selected_mode}, input_zone={zone}, selected_zone={selected_zone}, "
            f"capacity_kwargs={capacity_kwargs}"
        )
        capacity = self.client.check_resource_capacity(**capacity_kwargs)
        spec = self._pick_spec(
            capacity.specs,
            gpu,
            cpu,
            memory_gb,
            allow_fallback=allow_resource_fallback,
        )
        if spec is None:
            detail = (
                f"GPU {capacity.gpu_type} 在 {selected_zone} 当前无可用规格库存。"
                if not any(item.resource_enough for item in capacity.specs)
                else "所选优云 GPU 规格当前无库存。"
            )
            raise YouyunApiError(
                "CheckCompShareResourceCapacity",
                None,
                detail,
            )
        instance_name = (name or f"cppt-{task_id}")[:63]
        create_kwargs: dict[str, Any] = {
            "name": instance_name,
            "gpu": spec.gpu,
            "cpu": spec.cpu,
            "memory_gb": spec.memory_gb,
            "gpu_type": gpu_type,
            "image_id": image_id,
        }
        create_kwargs["zone"] = selected_zone
        ids = self.client.create_instance(**create_kwargs)
        if not ids:
            raise YouyunApiError("CreateCompShareInstance", None, "创建实例未返回 ID。")
        uhost_id = ids[0]
        logger.info("优云实例已创建：%s（%s，规格 %s卡/%s核/%sGB）", uhost_id, capacity.gpu_type, spec.gpu, spec.cpu, spec.memory_gb)
        try:
            return self.wait_running(uhost_id)
        except Exception as exc:
            # A boot timeout still leaves a billable instance. Best-effort cleanup here,
            # before the provider receives a handle, prevents orphaned GPU resources.
            report = self.cleanup_after_task(uhost_id, stop_first=False)
            detail = f"；实例清理结果：{'已释放' if report.get('terminated') else '失败：' + str(report.get('error'))}"
            if isinstance(exc, YouyunApiError):
                raise YouyunApiError(exc.action, exc.ret_code, f"{exc.message}{detail}") from exc
            raise RuntimeError(f"优云实例 {uhost_id} 启动失败：{exc}{detail}") from exc

    def _pick_spec(
        self,
        specs: list[GpuSpecCombination],
        gpu: int | None,
        cpu: int | None,
        memory_gb: int | None,
        *,
        allow_fallback: bool = False,
    ) -> GpuSpecCombination | None:
        available = [s for s in specs if s.resource_enough]
        all_available = available
        if gpu is not None:
            available = [s for s in available if s.gpu == gpu]
        if cpu is not None:
            available = [s for s in available if s.cpu == cpu]
        if memory_gb is not None:
            available = [s for s in available if s.memory_gb == memory_gb]
        if not available and allow_fallback:
            # Automatic retries may relax CPU/memory, but keep an explicitly
            # requested card count (the training UI currently allows one card).
            available = [s for s in all_available if gpu is None or s.gpu == gpu]
        return min(available, key=lambda s: (s.gpu, s.cpu, s.memory_gb)) if available else None

    def wait_running(self, uhost_id: str) -> YouyunInstance:
        deadline = self._clock() + self.boot_timeout
        last_state = ""
        while self._clock() < deadline:
            instance = self.client.describe_instance(uhost_id)
            last_state = instance.state
            if instance.running and instance.ip:
                return instance
            if instance.state in {"Fail", "Error", "Terminated", "Deleted"}:
                raise YouyunApiError("DescribeCompShareInstance", None, f"实例 {uhost_id} 进入异常状态 {instance.state}。")
            self._sleep(self.poll_interval)
        raise YouyunApiError("DescribeCompShareInstance", None, f"实例 {uhost_id} 等待 Running 超时（最后状态 {last_state}）。")

    # ---- 查询 ----

    def get(self, uhost_id: str) -> YouyunInstance | None:
        try:
            return self.client.describe_instance(uhost_id)
        except YouyunApiError:
            return None

    def ssh_credentials(self, instance: YouyunInstance) -> dict[str, Any]:
        return {
            "host": instance.ip,
            "port": instance.ssh_port or 22,
            "user": instance.ssh_user or "root",
            "password": _decode_password(instance.password_b64),
        }

    # ---- 释放（永不抛异常） ----

    def terminate(self, uhost_id: str) -> dict[str, Any]:
        """删除一次性实例并等待消失。返回清理报告（含失败原因），不抛异常。"""
        report: dict[str, Any] = {"uhost_id": uhost_id, "terminated": False, "error": None}
        try:
            self.client.delete_instance(uhost_id)
        except YouyunApiError as exc:
            report["error"] = str(exc)
            logger.error("优云实例 %s 删除指令失败：%s", uhost_id, exc)
        deadline = self._clock() + self.terminate_timeout
        while self._clock() < deadline:
            if self.get(uhost_id) is None:
                report["terminated"] = True
                return report
            self._sleep(self.poll_interval)
        if report["error"] is None:
            report["error"] = f"删除后 {self.terminate_timeout}s 内实例仍存在。"
        return report

    def cleanup_after_task(self, uhost_id: str, *, stop_first: bool = True) -> dict[str, Any]:
        """任务收尾清理：可选先停机再删除，任何失败都记录进报告。"""
        report: dict[str, Any] = {"uhost_id": uhost_id, "terminated": False, "error": None, "steps": []}
        if stop_first:
            try:
                self.client.stop_instance(uhost_id)
                report["steps"].append({"step": "stop", "ok": True})
            except YouyunApiError as exc:
                # 停机失败不阻断删除，仅记录
                report["steps"].append({"step": "stop", "ok": False, "error": str(exc)})
        term = self.terminate(uhost_id)
        report["terminated"] = term["terminated"]
        report["error"] = term["error"]
        return report
