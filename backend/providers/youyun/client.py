"""优云 API 客户端：签名、请求、重试与错误码处理。

统一 POST 到 api_base_url，表单编码，RetCode != 0 抛 YouyunApiError。
数组参数按 UCloud 协议拍平为 Key.0.Field 形式。
"""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import GpuMachineType, GpuSpecCombination, ResourceCapacity, YouyunApiError, YouyunConfig, YouyunInstance, YouyunPrice
from .signer import sign

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {500, 502, 503, 504}
_RETRYABLE_RET_CODES = {226645}


def _resource_params(
    config: YouyunConfig,
    *,
    gpu_type: str | None = None,
    image_id: str | None = None,
    charge_type: str | None = None,
    zone: str | None = None,
) -> dict[str, Any]:
    """Build the common resource identity shared by capacity and create calls."""
    final_zone = zone or config.zone
    logger.debug(f"_resource_params: input zone={zone}, config.zone={config.zone}, final Zone={final_zone}")
    return {
        "Region": config.region,
        "Zone": final_zone,
        "GpuType": gpu_type or config.gpu_type,
        "CompShareImageId": image_id or config.image_id,
        "ChargeType": charge_type or config.charge_type,
    }


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(_flatten(item, f"{name}.{index}"))
                else:
                    flat[f"{name}.{index}"] = item
        else:
            flat[name] = value
    return flat


class YouyunClient:
    def __init__(self, config: YouyunConfig | None = None, require_full: bool = True) -> None:
        self.config = config or YouyunConfig.from_env()
        missing = self.config.missing_keys()
        if require_full and missing:
            raise YouyunApiError("Config", None, f"优云配置缺少：{', '.join(missing)}")

    # ---- 发现类接口：只需要公钥/私钥 ----

    def describe_support_zones(self, region: str | None = None) -> list[dict[str, Any]]:
        """列出支持的可用区。不传 region 返回全部。"""
        params: dict[str, Any] = {}
        # ``None`` intentionally means all regions; callers that need the
        # configured region pass it explicitly.
        if region is not None and region.strip():
            params["Region"] = region.strip()
        payload = self._post("DescribeCompShareSupportZone", params)
        return [
            {
                "region": str(item.get("Region") or ""),
                "zone": str(item.get("Zone") or ""),
                "describe": str(item.get("Describe") or ""),
                "is_pod": bool(item.get("IsPod")),
                "unsupported_image_types": list(item.get("UnsupportedImageTypes") or []),
                "disable_image_sync": bool(item.get("DisableImageSync")),
            }
            for item in payload.get("ZoneInfo") or []
        ]

    def describe_images(
        self,
        region: str | None = None,
        zone: str | None = None,
        image_type: str | None = None,
        limit: int = 100,
        use_config_zone: bool = True,
    ) -> list[dict[str, Any]]:
        """列出优云镜像并保留实例类型元数据。"""
        params: dict[str, Any] = {"Region": region or self.config.region, "Limit": limit}
        if zone or (use_config_zone and self.config.zone):
            params["Zone"] = zone or self.config.zone
        if image_type:
            params["ImageType"] = image_type
        payload = self._post("DescribeCompShareImages", params)
        return [
            {
                "image_id": str(item.get("CompShareImageId") or ""),
                "name": str(item.get("Name") or ""),
                "image_type": str(item.get("ImageType") or ""),
                "container": item.get("Container"),
                "author": str(item.get("Author") or ""),
                "image_owner_alias": str(item.get("ImageOwnerAlias") or ""),
                "softwares": dict(item.get("Softwares") or {}),
                "zone": str(item.get("Zone") or ""),
                "image_support_zone": item.get("ImageSupportZone"),
                "status": str(item.get("Status") or ""),
                "tags": [str(tag) for tag in item.get("Tags") or []],
                "description": str(item.get("Description") or ""),
            }
            for item in payload.get("ImageSet") or []
        ]

    def describe_model_repository_models(
        self,
        name: str | None = None,
        tags: str | None = None,
        region: str | None = None,
    ) -> list[dict[str, Any]]:
        """List models preloaded in the provider's public model repository."""
        params: dict[str, Any] = {"Region": region or self.config.region}
        if name and name.strip():
            params["name"] = name.strip()
        if tags and tags.strip():
            params["tags"] = tags.strip()
        payload = self._post("DescribeModelRepositoryModels", params)
        models: list[dict[str, Any]] = []
        for item in payload.get("Models") or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("Path") or "").strip()
            model_name = str(item.get("Name") or "").strip()
            if not path or not model_name:
                continue
            models.append(
                {
                    "name": model_name,
                    "path": path,
                    "tag": str(item.get("Tag") or "").strip(),
                    "size": str(item.get("Size") or "").strip(),
                    "createTime": item.get("CreateTime"),
                    "updateTime": item.get("UpdateTime"),
                    "id": item.get("ModelId") or item.get("ModelID") or item.get("Id") or item.get("ID"),
                    "sizeBytes": item.get("SizeBytes"),
                    "status": item.get("Status"),
                    "tags": item.get("Tags") or [],
                }
            )
        return models

    def _post(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        import json

        form = {
            "Action": action,
            "PublicKey": self.config.public_key,
            **_flatten(params),
        }
        signed = sign(form, self.config.private_key)
        body = urllib.parse.urlencode(signed).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                request = urllib.request.Request(
                    self.config.api_base_url,
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("RetCode") != 0:
                    ret_code = payload.get("RetCode")
                    error = YouyunApiError(action, ret_code, str(payload.get("Message") or "未知错误"))
                    if ret_code in _RETRYABLE_RET_CODES and attempt < self.config.max_retries:
                        logger.warning(
                            "优云 API 暂时不可用，将重试 action=%s ret_code=%s attempt=%s",
                            action,
                            ret_code,
                            attempt + 1,
                        )
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise error
                return payload
            except YouyunApiError:
                raise
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise YouyunApiError(action, None, f"网络请求失败：{last_error}")

    # ---- 规格与库存 ----

    def describe_instance_types(
        self, gpu_type: str | None = None, instance_type: str = "uhost"
    ) -> list[GpuMachineType]:
        params: dict[str, Any] = {
            "Region": self.config.region,
            "InstanceType": instance_type,
        }
        if gpu_type:
            params["MachineTypes"] = [gpu_type]
        payload = self._post("DescribeAvailableCompShareInstanceTypes", params)
        return [
            GpuMachineType(
                name=str(item.get("Name") or ""),
                zone=str(item.get("Zone") or ""),
                status=str(item.get("Status") or ""),
                description=str(item.get("Description") or ""),
                graphics_memory_gb=(item.get("GraphicsMemory") or {}).get("Value"),
                machine_sizes=list(item.get("MachineSizes") or []),
            )
            for item in payload.get("AvailableInstanceTypes") or []
        ]

    def check_resource_capacity(
        self,
        gpu_type: str | None = None,
        image_id: str | None = None,
        charge_type: str | None = None,
        cpu_platform: str = "Intel/Auto",
        zone: str | None = None,
    ) -> ResourceCapacity:
        gpu_type = gpu_type or self.config.gpu_type
        payload = self._post(
            "CheckCompShareResourceCapacity",
            {
                **_resource_params(
                    self.config,
                    gpu_type=gpu_type,
                    image_id=image_id,
                    charge_type=charge_type,
                    zone=zone,
                ),
                "MachineType": "G",
                "MinimalCpuPlatform": cpu_platform,
                "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": self.config.boot_disk_size_gb}],
            },
        )
        specs = [
            GpuSpecCombination(
                gpu=int(spec.get("Gpu") or 0),
                cpu=int(spec.get("Cpu") or 0),
                memory_gb=int(spec.get("Mem") or 0),
                resource_enough=bool(spec.get("ResourceEnough")),
            )
            for spec in payload.get("Specs") or []
        ]
        return ResourceCapacity(gpu_type=gpu_type, specs=specs)

    def get_instance_price(
        self,
        gpu: int,
        cpu: int,
        memory_gb: int,
        gpu_type: str | None = None,
        charge_type: str | None = None,
        quantity: int = 1,
        image_id: str | None = None,
        zone: str | None = None,
    ) -> YouyunPrice:
        """查询实例当前计费价格，保留优云返回的分项价格。"""
        payload = self._post(
            "GetCompShareInstancePrice",
            {
                **_resource_params(
                    self.config,
                    gpu_type=gpu_type,
                    image_id=image_id,
                    charge_type=charge_type,
                    zone=zone,
                ),
                "GPU": gpu,
                "CPU": cpu,
                "Memory": memory_gb * 1024,
                "Quantity": quantity,
                "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": self.config.boot_disk_size_gb}],
            },
        )
        details = payload.get("PriceDetails") or payload.get("PriceDetail") or payload.get("ListPriceDetails") or []
        if isinstance(details, dict):
            details = [details]
        detail = next(
            (item for item in details if str(item.get("ChargeType") or "") == str(charge_type or self.config.charge_type)),
            next(iter(details), {}),
        ) if isinstance(details, list) else {}

        def amount(name: str) -> float:
            value = detail.get(name, 0)
            if isinstance(value, dict):
                value = value.get("Price") or value.get("Amount") or value.get("Value") or 0
            try:
                return float(value or 0)
            except (TypeError, ValueError):
                return 0.0

        return YouyunPrice(
            charge_type=str(detail.get("ChargeType") or charge_type or self.config.charge_type),
            instance=amount("Instance"),
            disks=amount("Disks"),
            system_disks=amount("SystemDisks"),
            image=amount("CompShareImage"),
            currency=str(detail.get("Currency") or "CNY"),
        )

    # ---- 实例生命周期 ----

    def create_instance(
        self,
        name: str,
        gpu: int,
        cpu: int,
        memory_gb: int,
        image_id: str | None = None,
        gpu_type: str | None = None,
        charge_type: str | None = None,
        zone: str | None = None,
    ) -> list[str]:
        payload = self._post(
            "CreateCompShareInstance",
            {
                **_resource_params(
                    self.config,
                    gpu_type=gpu_type,
                    image_id=image_id,
                    charge_type=charge_type,
                    zone=zone,
                ),
                "GPU": gpu,
                "CPU": cpu,
                "Memory": memory_gb * 1024,
                "Name": name,
                "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": self.config.boot_disk_size_gb}],
            },
        )
        return [str(item) for item in payload.get("UHostIds") or []]

    def describe_instances(self, uhost_ids: list[str] | None = None) -> list[YouyunInstance]:
        params: dict[str, Any] = {"Region": self.config.region, "Limit": 100}
        if uhost_ids:
            params["UHostIds"] = uhost_ids
        payload = self._post("DescribeCompShareInstance", params)
        return [YouyunInstance.from_api(item) for item in payload.get("UHostSet") or []]

    def describe_instance(self, uhost_id: str) -> YouyunInstance:
        instances = self.describe_instances([uhost_id])
        if not instances:
            raise YouyunApiError("DescribeCompShareInstance", None, f"实例 {uhost_id} 不存在")
        return instances[0]

    def _instance_action(self, action: str, uhost_id: str) -> None:
        # 实例操作类接口（Start/Stop/Terminate）需要 Region+Zone+单数 UHostId
        self._post(action, {"Region": self.config.region, "Zone": self.config.zone, "UHostId": uhost_id})

    def start_instance(self, uhost_id: str) -> None:
        self._instance_action("StartCompShareInstance", uhost_id)

    def stop_instance(self, uhost_id: str) -> None:
        self._instance_action("StopCompShareInstance", uhost_id)

    def delete_instance(self, uhost_id: str) -> None:
        self._instance_action("TerminateCompShareInstance", uhost_id)

    def get_instance_monitor(self, uhost_id: str) -> dict[str, Any]:
        payload = self._post(
            "GetCompShareInstanceMonitor",
            {"Region": self.config.region, "ResourceIds": [uhost_id]},
        )
        return payload
