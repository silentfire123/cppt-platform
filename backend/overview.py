from __future__ import annotations

from typing import Any


ACTIVE_TRAINING_STATES = {
    "provisioning",
    "preparing",
    "running",
    "stopping",
    "collecting",
    "uploading_artifact",
}
ACTIVE_INFERENCE_STATES = {"starting", "running", "stopping", "recovering"}


def _count(items: list[dict[str, Any]], **conditions: Any) -> int:
    return sum(
        1
        for item in items
        if all(item.get(key) == value for key, value in conditions.items())
    )


def _time_key(value: Any) -> str:
    return str(value or "").replace("T", " ").replace("Z", "+00:00")


def _summary_task(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if task is None:
        return None
    resource = task.get("youyunResource") or {}
    return {
        "id": task.get("id"),
        "name": task.get("name"),
        "state": task.get("state"),
        "progress": task.get("progress", 0),
        "currentStep": task.get("currentStep", ""),
        "model": task.get("model", ""),
        "provider": task.get("provider", "local"),
        "executionMode": task.get("executionMode", "local"),
        "gpuType": resource.get("gpuType"),
        "createdAt": task.get("createdAt"),
        "error": task.get("error"),
    }


def _summary_service(service: dict[str, Any] | None) -> dict[str, Any] | None:
    if service is None:
        return None
    config = service.get("config") or {}
    return {
        "id": service.get("id"),
        "name": service.get("name"),
        "state": service.get("state"),
        "exposure": service.get("exposure"),
        "model": service.get("model") or config.get("model", ""),
        "provider": config.get("provider", "local"),
        "adapterVariant": config.get("adapterVariant", "best"),
        "gpu": service.get("gpu", ""),
        "memory": service.get("memory", ""),
        "updatedAt": service.get("updatedAt"),
        "error": service.get("error"),
    }


def _gpu_payload(gpu: dict[str, Any]) -> dict[str, Any]:
    result = dict(gpu)
    total = gpu.get("memoryTotalMb")
    used = gpu.get("memoryUsedMb")
    if isinstance(total, (int, float)) and total > 0 and isinstance(used, (int, float)):
        result["memoryUsagePercent"] = round(used / total * 100, 1)
    else:
        result["memoryUsagePercent"] = None
    return result


def _youyun_summary(
    tasks: list[dict[str, Any]], services: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate active Youyun instances and their postpaid hourly rates."""
    active_tasks = [
        task
        for task in tasks
        if task.get("provider") == "youyun"
        and task.get("state") in ACTIVE_TRAINING_STATES
    ]
    active_services = [
        service
        for service in services
        if (service.get("config") or {}).get("provider") == "youyun"
        and service.get("state") in ACTIVE_INFERENCE_STATES
    ]

    gpu_types: set[str] = set()
    postpaid_hourly = 0.0
    prepaid_count = 0
    currency = "CNY"
    for resource in (
        *(task.get("youyunResource") or {} for task in active_tasks),
        *(
            (service.get("config") or {}).get("youyunResource") or {}
            for service in active_services
        ),
    ):
        gpu_type = resource.get("gpuType")
        if gpu_type:
            gpu_types.add(str(gpu_type))
        if resource.get("chargeType") == "Postpay":
            price = resource.get("price") or {}
            total = price.get("total")
            if isinstance(total, (int, float)):
                postpaid_hourly += float(total)
            if price.get("currency"):
                currency = str(price["currency"])
        else:
            prepaid_count += 1
    return {
        "activeCount": len(active_tasks) + len(active_services),
        "trainingCount": len(active_tasks),
        "inferenceCount": len(active_services),
        "gpuTypes": sorted(gpu_types),
        "postpaidHourlyTotal": round(postpaid_hourly, 2),
        "prepaidCount": prepaid_count,
        "currency": currency,
    }


def _activity(
    *,
    item_type: str,
    item: dict[str, Any],
    timestamp_key: str,
    module: str,
) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or ""),
        "type": item_type,
        "module": module,
        "name": str(item.get("name") or item.get("id") or "未命名记录"),
        "state": str(item.get("state") or item.get("status") or ""),
        "timestamp": item.get(timestamp_key),
    }


def build_overview_snapshot(
    *,
    datasets: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    model_assets: list[dict[str, Any]],
    services: list[dict[str, Any]],
    health: dict[str, Any],
    gpu: dict[str, Any],
    shared_storage: dict[str, Any],
    inference_metrics: dict[str, Any] | None,
    generated_at: str,
) -> dict[str, Any]:
    active_task_ids = set(health.get("activeTaskIds") or [])
    if health.get("activeTaskId"):
        active_task_ids.add(health["activeTaskId"])
    active_task = next(
        (
            task
            for task in tasks
            if task.get("id") == health.get("activeTaskId")
        ),
        None,
    )
    if active_task is None:
        active_task = next(
            (task for task in tasks if task.get("state") in ACTIVE_TRAINING_STATES),
            next((task for task in tasks if task.get("state") == "queued"), None),
        )
    active_service = next(
        (service for service in services if service.get("state") == "running"),
        next(
            (
                service
                for service in services
                if service.get("state") in ACTIVE_INFERENCE_STATES
            ),
            None,
        ),
    )

    active_local_task = next(
        (
            task
            for task in tasks
            if task.get("state") in ACTIVE_TRAINING_STATES
            and task.get("provider", "local") != "youyun"
            and task.get("executionMode", "local") != "remote"
            and task.get("id") in active_task_ids
        ),
        None,
    )
    active_local_service = next(
        (
            service
            for service in services
            if service.get("state") in ACTIVE_INFERENCE_STATES
            and (service.get("config") or {}).get("provider", "local") != "youyun"
        ),
        None,
    )

    stale_active_task = (
        active_task is not None
        and active_task.get("state") in ACTIVE_TRAINING_STATES
        and active_task.get("id") not in active_task_ids
    )

    resource_owner = "idle"
    if active_local_task is not None:
        resource_owner = "training"
    elif active_local_service is not None:
        resource_owner = "inference"

    unavailable_models = [
        asset for asset in model_assets if asset.get("status") == "unavailable"
    ]
    alerts: list[dict[str, Any]] = []
    if health.get("status") != "ok":
        alerts.append(
            {
                "id": "backend-health",
                "severity": "critical",
                "module": "overview",
                "title": "后端服务状态异常",
                "detail": str(health.get("status") or "状态未知"),
            }
        )
    if shared_storage.get("status") == "unavailable":
        alerts.append(
            {
                "id": "shared-storage-unavailable",
                "severity": "warning",
                "module": "models",
                "title": "模型共享存储不可用",
                "detail": str(shared_storage.get("error") or "共享存储尚未初始化"),
            }
        )
    if stale_active_task:
        alerts.append(
            {
                "id": f"stale-task-{active_task.get('id')}",
                "severity": "critical",
                "module": "training",
                "entityId": active_task.get("id"),
                "title": f"训练任务状态与 Worker 不一致：{active_task.get('name')}",
                "detail": "任务记录仍为运行中，但当前 Worker 没有执行该任务，请检查任务恢复或手动重试。",
            }
        )
    if unavailable_models:
        alerts.append(
            {
                "id": "models-unavailable",
                "severity": "warning",
                "module": "models",
                "title": f"{len(unavailable_models)} 个模型资产不可用",
                "detail": "请检查本地路径或训练产物是否仍然存在。",
            }
        )
    for dataset in (item for item in datasets if item.get("state") == "failed"):
        alerts.append(
            {
                "id": f"dataset-{dataset.get('id')}",
                "severity": "warning",
                "module": "datasets",
                "entityId": dataset.get("id"),
                "title": f"数据集校验失败：{dataset.get('name')}",
                "detail": str(dataset.get("validationMessage") or "请检查数据格式。"),
            }
        )
    for task in (item for item in tasks if item.get("state") == "failed"):
        alerts.append(
            {
                "id": f"task-{task.get('id')}",
                "severity": "critical",
                "module": "training",
                "entityId": task.get("id"),
                "title": f"训练任务失败：{task.get('name')}",
                "detail": str(task.get("error") or task.get("currentStep") or "训练失败"),
            }
        )
    for service in (item for item in services if item.get("state") == "failed"):
        alerts.append(
            {
                "id": f"service-{service.get('id')}",
                "severity": "critical",
                "module": "inference",
                "entityId": service.get("id"),
                "title": f"推理服务启动失败：{service.get('name')}",
                "detail": str(service.get("error") or "推理服务启动失败"),
            }
        )
    alerts.sort(key=lambda item: item["severity"] != "critical")

    activities = [
        *[
            _activity(item_type="training", item=item, timestamp_key="createdAt", module="training")
            for item in tasks[:6]
        ],
        *[
            _activity(item_type="inference", item=item, timestamp_key="updatedAt", module="inference")
            for item in services[:6]
        ],
        *[
            _activity(item_type="model", item=item, timestamp_key="updatedAt", module="models")
            for item in model_assets[:6]
        ],
        *[
            _activity(item_type="dataset", item=item, timestamp_key="updatedAt", module="datasets")
            for item in datasets[:6]
        ],
    ]
    activities.sort(key=lambda item: _time_key(item.get("timestamp")), reverse=True)

    return {
        "status": "attention" if alerts else "healthy",
        "generatedAt": generated_at,
        "counts": {
            "datasets": {
                "total": len(datasets),
                "published": _count(datasets, state="published"),
                "ready": _count(datasets, state="ready"),
                "failed": _count(datasets, state="failed"),
            },
            "training": {
                "total": len(tasks),
                "active": sum(
                    1 for task in tasks if task.get("state") in ACTIVE_TRAINING_STATES
                ),
                "queued": _count(tasks, state="queued"),
                "completed": _count(tasks, state="completed"),
                "failed": _count(tasks, state="failed"),
                "stopped": _count(tasks, state="stopped"),
            },
            "models": {
                "total": len(model_assets),
                "base": _count(model_assets, kind="base"),
                "trained": _count(model_assets, kind="trained"),
                "unavailable": len(unavailable_models),
            },
            "inference": {
                "total": len(services),
                "running": _count(services, state="running"),
                "online": _count(services, exposure="online"),
                "failed": _count(services, state="failed"),
            },
        },
        "runtime": {
            "worker": health.get("worker", "unknown"),
            "activeTaskId": health.get("activeTaskId"),
            "activeTaskIds": list(health.get("activeTaskIds") or []),
            "queue": health.get("queue", "unknown"),
            "storageBackend": health.get("storageBackend", "unknown"),
            "inference": health.get("inference", "unknown"),
            "resourceOwner": resource_owner,
            "gpu": _gpu_payload(gpu),
            "youyun": _youyun_summary(tasks, services),
            "sharedStorage": shared_storage,
        },
        "activeTraining": _summary_task(active_task),
        "activeInference": _summary_service(active_service),
        "inferenceMetrics": inference_metrics if active_service else None,
        "alerts": alerts[:10],
        "recentActivity": activities[:8],
    }
