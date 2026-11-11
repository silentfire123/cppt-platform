"""Dormant distributed-training helpers reserved for a later release.

Production currently enforces single-GPU training. These helpers remain as
historical design scaffolding and must not be called by the active path until
real multi-GPU acceptance is complete.
"""
from __future__ import annotations

import json
from typing import Any, Literal

DistributedStrategy = Literal["ddp", "deepspeed_zero2"]


def validate_single_gpu_training(gpu_count: int, strategy: str) -> None:
    """Enforce the temporary production boundary: one GPU, one process."""
    if gpu_count != 1 or strategy != "single":
        raise ValueError("当前版本暂不支持多卡训练，请选择 1 张 GPU。")


def validate_distributed_strategy(strategy: str, gpu_count: int) -> DistributedStrategy:
    """Validate the supported strategy/GPU combinations before provisioning."""
    if strategy not in {"ddp", "deepspeed_zero2"}:
        raise ValueError(f"不支持的分布式策略：{strategy}")
    if gpu_count < 1:
        raise ValueError("GPU 数量必须至少为 1。")
    if strategy == "deepspeed_zero2" and gpu_count < 2:
        raise ValueError("DeepSpeed ZeRO-2 至少需要 2 张 GPU。")
    return strategy  # type: ignore[return-value]


def build_deepspeed_zero2_config() -> str:
    """Return the conservative ZeRO-2 config used by remote LLaMA-Factory."""
    config: dict[str, Any] = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
        "bf16": {"enabled": "auto"},
        "wall_clock_breakdown": False,
    }
    return json.dumps(config, ensure_ascii=False, indent=2) + "\n"


__all__ = ["DistributedStrategy", "build_deepspeed_zero2_config", "validate_distributed_strategy", "validate_single_gpu_training"]
