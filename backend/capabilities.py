"""Server-owned execution policy; independent of optional ML runtimes."""

import os

from fastapi import HTTPException

LOCAL_DISABLED = "生产环境已禁用本地运行，请使用优云远程实例。"
EVALUATION_DISABLED = "生产环境暂未开放模型评测。"


def local_runtime_enabled() -> bool:
    return os.getenv("XUNTUI_LOCAL_RUNTIME_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def evaluation_enabled() -> bool:
    configured = os.getenv("XUNTUI_EVALUATION_ENABLED")
    if configured is None:
        return local_runtime_enabled()
    return configured.strip().lower() in {"1", "true", "yes", "on"}


def require_evaluation_enabled() -> None:
    if not evaluation_enabled():
        raise HTTPException(status_code=409, detail=EVALUATION_DISABLED)


def require_local_runtime() -> None:
    if not local_runtime_enabled():
        raise HTTPException(status_code=409, detail=LOCAL_DISABLED)


def require_provider(provider: str | None) -> None:
    if provider != "youyun":
        require_local_runtime()


def runtime_capabilities() -> dict:
    local = local_runtime_enabled()
    configured_sources = os.getenv(
        "XUNTUI_MODEL_CATALOG", "youyun-public,remote-modelscope"
    ).split(",")
    sources = list(
        dict.fromkeys(
            value.strip()
            for value in configured_sources
            if value.strip() in {"youyun-public", "remote-modelscope"}
        )
    )
    # Remote repository IDs remain an allowed fallback even when the catalog is public-only.
    if "remote-modelscope" not in sources:
        sources.append("remote-modelscope")
    return {
        "localRuntimeEnabled": local,
        "trainingProviders": ["local", "youyun"] if local else ["youyun"],
        "inferenceProviders": ["local", "youyun"] if local else ["youyun"],
        "cliTrainingEnabled": local,
        "modelSources": sources,
        "evaluationEnabled": evaluation_enabled(),
        "evaluationGenerationProvider": "youyun",
        "evaluationReusesTrainingInstance": True,
    }
