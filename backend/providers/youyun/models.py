"""优云 Provider 的配置与数据模型。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


_YOUYUN_REGION_LABELS = {
    "cn-bj2": "华北一区",
    "cn-wlcb": "华北二区",
    "cn-sh2": "上海二区",
}


def build_region_capabilities(zones: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate zone-level Pod/UHost data into region-level capabilities.

    ``DescribeCompShareSupportZone`` reports the instance form for a *zone*,
    not for the whole region. A region is dual-mode when it has at least one
    Pod zone and one UHost zone. Keep the raw zone lists so callers can choose
    a deterministic zone while still exposing the complete provider response.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for item in zones:
        region = str(item.get("region") or "").strip()
        zone = str(item.get("zone") or "").strip()
        if not region or not zone:
            continue
        mode = "pod" if bool(item.get("is_pod")) else "uhost"
        capability = grouped.setdefault(
            region,
            {
                "region": region,
                "displayName": _YOUYUN_REGION_LABELS.get(region, region),
                "modes": {"pod": None, "uhost": None},
                "zones": {"pod": [], "uhost": []},
            },
        )
        if capability["modes"][mode] is None:
            capability["modes"][mode] = zone
        capability["zones"][mode].append(zone)

    return list(grouped.values())


def resolve_region_zone(
    zones: Iterable[dict[str, Any]],
    *,
    region: str,
    mode: str,
    preferred_zone: str | None = None,
) -> str | None:
    """Return the zone in ``region`` that matches the requested instance form."""
    expected_pod = mode == "pod"
    candidates = [
        str(item.get("zone") or "").strip()
        for item in zones
        if str(item.get("region") or "").strip() == region
        and bool(item.get("is_pod")) == expected_pod
        and str(item.get("zone") or "").strip()
    ]
    if preferred_zone and preferred_zone in candidates:
        return preferred_zone
    return candidates[0] if candidates else None


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


@dataclass(frozen=True)
class YouyunConfig:
    api_base_url: str = "https://api.compshare.cn"
    public_key: str = ""
    private_key: str = ""
    region: str = ""
    zone: str = ""
    image_id: str = ""
    training_image_id: str = "compshareImage-1bvy5l0kjip2"
    gpu_type: str = "4090"
    charge_type: str = "Postpay"
    boot_disk_size_gb: int = 100
    timeout_seconds: float = 30.0
    max_retries: int = 3
    instance_mode: str = "uhost"
    pod_zone: str = ""
    public_model_roots: str = "/models,/model"

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "YouyunConfig":
        _load_dotenv(dotenv_path or Path(__file__).resolve().parents[3] / ".env")
        return cls(
            api_base_url=os.environ.get("YOUYUN_API_BASE_URL", "https://api.compshare.cn").strip().rstrip("/"),
            public_key=os.environ.get("YOUYUN_PUBLIC_KEY", "").strip(),
            private_key=os.environ.get("YOUYUN_PRIVATE_KEY", "").strip(),
            region=os.environ.get("YOUYUN_REGION", "").strip(),
            zone=os.environ.get("YOUYUN_ZONE", "").strip(),
            image_id=os.environ.get("YOUYUN_IMAGE_ID", "").strip(),
            training_image_id=os.environ.get(
                "YOUYUN_TRAINING_IMAGE_ID", "compshareImage-1bvy5l0kjip2"
            ).strip(),
            gpu_type=os.environ.get("YOUYUN_GPU_TYPE", "4090").strip(),
            charge_type=os.environ.get("YOUYUN_CHARGE_TYPE", "Postpay").strip(),
            boot_disk_size_gb=int(os.environ.get("YOUYUN_BOOT_DISK_SIZE_GB", "100")),
            timeout_seconds=float(os.environ.get("YOUYUN_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.environ.get("YOUYUN_MAX_RETRIES", "2")),
            instance_mode=os.environ.get("YOUYUN_INSTANCE_MODE", "uhost").strip().lower() or "uhost",
            pod_zone=os.environ.get("YOUYUN_POD_ZONE", "").strip(),
            public_model_roots=os.environ.get("YOUYUN_PUBLIC_MODEL_ROOTS", "/models,/model").strip() or "/models,/model",
        )

    def missing_keys(self) -> list[str]:
        required = {
            "YOUYUN_PUBLIC_KEY": self.public_key,
            "YOUYUN_PRIVATE_KEY": self.private_key,
            "YOUYUN_REGION": self.region,
        }
        missing = [key for key, value in required.items() if not value]
        if not self.zone and not (self.instance_mode == "pod" and self.pod_zone):
            missing.append("YOUYUN_ZONE")
        return missing

    def with_overrides(self, region: str | None = None, zone: str | None = None) -> "YouyunConfig":
        """用资源选择覆盖区域/可用区；空值回落到基础配置。

        优云要求请求中 Region 与 Zone 属于同一区域，创建/库存检查/实例操作
        都必须使用服务创建时选定的那一组，否则会被 RetCode=230 拒绝。
        """
        overrides: dict[str, str] = {}
        if region and region.strip():
            overrides["region"] = region.strip()
        if zone and zone.strip():
            overrides["zone"] = zone.strip()
        return replace(self, **overrides) if overrides else self


class YouyunApiError(RuntimeError):
    """优云 API 返回 RetCode != 0 或网络层失败。"""

    def __init__(self, action: str, ret_code: int | None, message: str) -> None:
        super().__init__(f"优云 API {action} 失败（RetCode={ret_code}）：{message}")
        self.action = action
        self.ret_code = ret_code
        self.message = message


@dataclass
class GpuSpecCombination:
    gpu: int
    cpu: int
    memory_gb: int
    resource_enough: bool


@dataclass
class GpuMachineType:
    name: str
    zone: str
    status: str
    description: str
    graphics_memory_gb: float | None
    machine_sizes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sold_out(self) -> bool:
        return self.status != "Normal"


@dataclass
class ResourceCapacity:
    gpu_type: str
    specs: list[GpuSpecCombination]

    def first_available(self) -> GpuSpecCombination | None:
        return next((s for s in self.specs if s.resource_enough), None)


@dataclass
class YouyunPrice:
    charge_type: str
    instance: float
    disks: float
    system_disks: float
    image: float
    currency: str = "CNY"

    @property
    def total(self) -> float:
        return self.instance + self.disks + self.system_disks + self.image


@dataclass
class YouyunInstance:
    uhost_id: str
    name: str
    state: str
    region: str
    zone: str
    gpu: int
    gpu_type: str
    cpu: int
    memory_mb: int
    ip: str | None
    ssh_port: int | None
    ssh_user: str | None
    password_b64: str | None
    image_id: str
    os_type: str
    charge_type: str
    instance_type: str
    raw: dict[str, Any]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "YouyunInstance":
        command = str(data.get("SshLoginCommand") or "")
        ip: str | None = None
        port: int | None = None
        user: str | None = None
        # 兼容 "ssh root@x.x.x.x -p 60022" 与 "ssh -p 23 root@x.x.x.x" 两种顺序
        if command:
            parts = command.split()
            for index, token in enumerate(parts):
                if "@" in token and not token.startswith("-"):
                    maybe_user, _, host = token.partition("@")
                    if host:
                        user, ip = maybe_user, host
                elif index > 0 and parts[index - 1] == "-p" and token.isdigit():
                    port = int(token)
        return cls(
            uhost_id=str(data.get("UHostId") or ""),
            name=str(data.get("Name") or ""),
            state=str(data.get("State") or ""),
            region=str(data.get("Region") or ""),
            zone=str(data.get("Zone") or ""),
            gpu=int(data.get("GPU") or 0),
            gpu_type=str(data.get("GpuType") or ""),
            cpu=int(data.get("CPU") or 0),
            memory_mb=int(data.get("Memory") or 0),
            ip=ip,
            ssh_port=port,
            ssh_user=user or "root",
            password_b64=(data.get("Password") or None),
            image_id=str(data.get("CompShareImageId") or ""),
            os_type=str(data.get("OsType") or ""),
            charge_type=str(data.get("ChargeType") or ""),
            instance_type=str(data.get("InstanceType") or ""),
            raw=data,
        )

    @property
    def running(self) -> bool:
        return self.state == "Running"
