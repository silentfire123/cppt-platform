from __future__ import annotations

"""LLaMA-Factory configuration models and safe platform-to-YAML conversion.

This module deliberately contains no process or storage code.  It is the
single place where the business-facing training payload is translated into a
LLaMA-Factory configuration, which keeps local and future remote executors in
sync.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TaskType = Literal["SFT", "DPO"]
TuningMode = Literal["lora", "qlora"]

# BitsAndBytes optimizers require the bitsandbytes package. The production
# executor currently runs on a CPU-only server, where the portable Torch
# optimizer is the supported default and must not depend on bitsandbytes.
_BITSANDBYTES_OPTIMIZERS = frozenset({"adamw_8bit", "paged_adamw_8bit"})

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}$")
# Backslashes are valid Windows path separators; paths are still constrained
# to approved roots before they are passed to a subprocess.
_SHELL_META = re.compile(r"[;&|<>`$\n\r\x00]")


class LlamaFactoryParameters(BaseModel):
    """Supported first-version parameters, with extension points for DPO/QLoRA."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    epochs: float = Field(default=1, gt=0, le=100)
    learning_rate: float = Field(default=2e-4, gt=0, le=10, alias="learningRate")
    batch_size: int = Field(default=1, ge=1, le=64, alias="batchSize")
    gradient_accumulation: int = Field(default=1, ge=1, le=4096, alias="gradientAccumulation")
    max_sequence_length: int = Field(default=512, ge=8, le=32768, alias="maxSequenceLength")
    lora_rank: int = Field(default=8, ge=1, le=256, alias="loraRank")
    lora_alpha: int = Field(default=16, ge=1, le=512, alias="loraAlpha")
    lora_dropout: float = Field(default=0.0, ge=0, lt=1, alias="loraDropout")
    logging_steps: int = Field(default=10, ge=1, le=100000, alias="loggingSteps")
    validation_split: float = Field(default=0.0, ge=0, lt=1, alias="validationSplit")
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    lr_scheduler: str = Field(default="cosine", min_length=1, alias="lrScheduler")
    lora_plus_enabled: bool = Field(default=False, alias="loraPlusEnabled")
    lora_plus_ratio: float = Field(default=16, gt=0, alias="loraPlusRatio")
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1, alias="warmupRatio")
    weight_decay: float = Field(default=0.0, ge=0, le=10, alias="weightDecay")
    max_grad_norm: float = Field(default=1.0, gt=0, alias="maxGradNorm")
    packing: bool = False
    gradient_checkpointing: bool = Field(default=False, alias="gradientCheckpointing")
    optimizer: str = "adamw_torch"
    target_modules: str = Field(default="attention", alias="targetModules")
    quantization_bit: int | None = Field(default=None, ge=4, le=8, alias="quantizationBit")
    template: str | None = None
    save_steps: int | None = Field(default=None, ge=1, alias="saveSteps")
    save_strategy: str = Field(default="steps", alias="saveStrategy")
    checkpoint_limit: int = Field(default=2, ge=1, le=20, alias="checkpointLimit")
    checkpoint_strategy: Literal["epoch", "steps"] = Field(default="epoch", alias="checkpointStrategy")
    checkpoint_interval: int = Field(default=100, ge=1, alias="checkpointInterval")
    eval_steps: int | None = Field(default=None, ge=1, alias="evalSteps")
    max_steps: int = Field(default=-1, ge=-1, alias="maxSteps")
    report_to: str = Field(default="none", alias="reportTo")
    use_cpu: bool = Field(default=True, alias="useCpu")

    @model_validator(mode="after")
    def normalize_cpu_optimizer(self) -> "LlamaFactoryParameters":
        # The deployed LLaMA-Factory environment is CPU-only and deliberately
        # does not install bitsandbytes. Keep accepting legacy/UI payloads that
        # selected an 8-bit optimizer, but emit a runnable Torch optimizer.
        if self.use_cpu and self.optimizer in _BITSANDBYTES_OPTIMIZERS:
            self.optimizer = "adamw_torch"
        return self

    @field_validator("learning_rate", mode="before")
    @classmethod
    def parse_learning_rate(cls, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("learningRate 必须是数字。") from exc

    @field_validator("validation_split", mode="before")
    @classmethod
    def normalize_validation_split(cls, value: Any) -> float:
        number = float(value or 0)
        # The UI historically sends a percentage integer (for example 10).
        return number / 100 if number > 1 else number


class LlamaFactoryTask(BaseModel):
    """Canonical executor input shared by local and future remote execution."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_id: str = Field(alias="taskId", min_length=1, max_length=128)
    framework: Literal["llamafactory"] = "llamafactory"
    task_type: TaskType = Field(default="SFT", alias="taskType")
    tuning_mode: TuningMode = Field(default="lora", alias="tuningMode")
    model: str = Field(min_length=1, max_length=512)
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=128)
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    parameters: LlamaFactoryParameters = Field(default_factory=LlamaFactoryParameters)
    output_name: str = Field(default="adapter", alias="outputName", min_length=1, max_length=128)
    execution_mode: Literal["local", "remote"] = Field(default="local", alias="executionMode")
    provider: Literal["local", "youyun"] = "local"
    dataset: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "dataset_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("任务 ID 和数据集 ID 只能包含字母、数字、点、下划线和短横线。")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value or _SHELL_META.search(value):
            raise ValueError("模型标识包含不允许的 Shell 字符。")
        if not _SAFE_MODEL.fullmatch(value) and not Path(value).is_absolute():
            raise ValueError("模型标识格式不安全。")
        return value


def _as_mapping(value: LlamaFactoryTask | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, LlamaFactoryTask):
        return value.model_dump(by_alias=True)
    return dict(value)


def _safe_name(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} 不安全：只能包含字母、数字、点、下划线和短横线。")
    return value


def _safe_path(path_value: str | os.PathLike[str], *, label: str, roots: Iterable[Path] = ()) -> Path:
    raw = str(path_value).strip()
    if not raw or _SHELL_META.search(raw):
        raise ValueError(f"{label} 路径为空或包含不允许的字符。")
    path = Path(raw).expanduser()
    resolved = path.resolve()
    allowed = [Path(root).expanduser().resolve() for root in roots]
    if allowed and not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError(f"{label} 必须位于受控工作目录内。")
    return resolved


def validate_training_paths(
    *,
    dataset_paths: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    allowed_roots: Iterable[Path] = (),
) -> dict[str, Path]:
    """Validate all file/directory paths used by the generated config."""
    result: dict[str, Path] = {}
    for key in ("trainPath", "validationPath", "testPath", "challengePath"):
        value = dataset_paths.get(key)
        if not value:
            continue
        path = _safe_path(value, label=key, roots=allowed_roots)
        if not path.is_file():
            raise ValueError(f"{key} 不存在或不是文件：{path}")
        result[key] = path
    if "trainPath" not in result:
        raise ValueError("训练数据必须提供 trainPath。")
    result["outputDir"] = _safe_path(output_dir, label="outputDir", roots=allowed_roots)
    return result


def _target_modules(value: str) -> list[str]:
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    all_linear = attention + ["gate_proj", "up_proj", "down_proj"]
    if value in {"all", "all-linear", "all_linear"}:
        return all_linear
    if value in {"attention", "attn", "auto"}:
        return attention
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in items):
        raise ValueError("targetModules 包含不安全的模块名。")
    return items


def platform_to_llamafactory_config(
    task: LlamaFactoryTask | Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    allowed_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Convert a platform task to a deterministic LLaMA-Factory config dict."""
    raw = _as_mapping(task)
    canonical = LlamaFactoryTask.model_validate(raw)
    params = canonical.parameters
    dataset = dict(canonical.dataset)
    dataset_paths = dict(dataset.get("paths") or dataset)
    chosen_output = output_dir or dataset.get("outputDir") or raw.get("outputDir")
    if not chosen_output:
        raise ValueError("必须提供 outputDir，避免训练产物写入未受控目录。")
    paths = validate_training_paths(
        dataset_paths=dataset_paths,
        output_dir=chosen_output,
        allowed_roots=allowed_roots,
    )
    dataset_name = _safe_name(canonical.dataset_id, "datasetId")
    model_name = canonical.model
    # LLaMA-Factory 只认 lora/oft/freeze/full；QLoRA = lora + quantization_bit 4。
    # CPU 环境未装 bitsandbytes，量化位仅在 GPU（use_cpu=False）时下发。
    finetuning_type = "lora" if canonical.tuning_mode in {"lora", "qlora"} else canonical.tuning_mode
    config: dict[str, Any] = {
        "model_name_or_path": model_name,
        "stage": canonical.task_type.lower(),
        "do_train": True,
        "finetuning_type": finetuning_type,
        "dataset_dir": str(paths["trainPath"].parent),
        "dataset": dataset_name,
        "template": params.template or dataset.get("template") or "default",
        "cutoff_len": params.max_sequence_length,
        "output_dir": str(paths["outputDir"]),
        "per_device_train_batch_size": params.batch_size,
        "gradient_accumulation_steps": params.gradient_accumulation,
        "learning_rate": params.learning_rate,
        "num_train_epochs": params.epochs,
        "lr_scheduler_type": params.lr_scheduler,
        "warmup_ratio": params.warmup_ratio,
        "weight_decay": params.weight_decay,
        "max_grad_norm": params.max_grad_norm,
        "logging_steps": params.logging_steps,
        "logging_first_step": True,
        "save_strategy": params.checkpoint_strategy,
        "save_total_limit": params.checkpoint_limit,
        "seed": params.seed,
        "packing": params.packing,
        "gradient_checkpointing": params.gradient_checkpointing,
        "optim": params.optimizer,
        "report_to": params.report_to,
        "use_cpu": params.use_cpu,
        "fp16": False,
        "bf16": False,
        "ddp_timeout": 180000000,
    }
    if canonical.task_type == "SFT":
        config["val_size"] = params.validation_split
        config["do_eval"] = params.validation_split > 0 or bool(paths.get("validationPath"))
    else:
        config["do_eval"] = bool(paths.get("validationPath"))
    if paths.get("validationPath"):
        config.pop("val_size", None)
        config["eval_dataset"] = f"{dataset_name}_validation"
    if params.checkpoint_strategy == "steps":
        config["save_steps"] = params.checkpoint_interval
    elif params.save_steps:
        config["save_steps"] = params.save_steps
    if params.lora_plus_enabled:
        config["loraplus_lr_ratio"] = params.lora_plus_ratio
    if params.eval_steps:
        config["eval_steps"] = params.eval_steps
    if params.max_steps >= 0:
        config["max_steps"] = params.max_steps
    if canonical.tuning_mode == "lora" or params.use_cpu:
        # CPU 部署未安装 bitsandbytes，QLoRA 在 CPU 上退化为纯 LoRA
        config.update(
            lora_rank=params.lora_rank,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            lora_target=_target_modules(params.target_modules),
        )
    else:
        config.update(
            quantization_bit=params.quantization_bit or 4,
            lora_rank=params.lora_rank,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            lora_target=_target_modules(params.target_modules),
        )
    # Preserve explicitly supported extra LLaMA-Factory flags while filtering
    # the legacy platform-only fields and unsafe path/command overrides.
    reserved = set(config)
    for key, value in (params.model_extra or {}).items():
        if key in reserved or key in {"output_dir", "dataset_dir", "model_name_or_path"}:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"额外训练参数名不安全：{key}")
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
    if config["do_eval"]:
        # Every validation point must have recoverable weights. Align saves
        # with validation; the UI explains this override of the raw save schedule.
        # Do NOT reload best at the end: output/ must remain the final model.
        interval = params.eval_steps or 50
        config.update(
            eval_strategy="steps", eval_steps=interval,
            save_strategy="steps", save_steps=interval,
            metric_for_best_model="eval_loss", greater_is_better=False,
            load_best_model_at_end=False,
            save_total_limit=max(2, params.checkpoint_limit),
        )
    return config


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:@+\\-]+", text) and text.lower() not in {"true", "false", "null", "yes", "no"}:
        return text
    return json.dumps(text, ensure_ascii=False)


def dump_yaml(config: Mapping[str, Any]) -> str:
    """Small dependency-free YAML emitter for the scalar/list config we create."""
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {_yaml_scalar(item)}" for item in value)
        elif isinstance(value, Mapping):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {_yaml_scalar(child_value)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def platform_to_llamafactory_yaml(
    task: LlamaFactoryTask | Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    allowed_roots: Iterable[Path] = (),
) -> str:
    return dump_yaml(
        platform_to_llamafactory_config(
            task, output_dir=output_dir, allowed_roots=allowed_roots
        )
    )


def dataset_info_for_task(
    task: LlamaFactoryTask | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the dataset_info.json entries consumed by LLaMA-Factory."""
    raw = _as_mapping(task)
    canonical = LlamaFactoryTask.model_validate(raw)
    dataset = dict(canonical.dataset)
    paths = dict(dataset.get("paths") or dataset)
    formatting = str(dataset.get("formatting") or "sharegpt")
    columns = dict(dataset.get("columns") or {"messages": "messages"})
    tags = dict(dataset.get("tags") or {
        "role_tag": "role", "content_tag": "content",
        "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system",
    })
    result: dict[str, dict[str, Any]] = {}
    result[canonical.dataset_id] = {
        "file_name": Path(str(paths["trainPath"])).name,
        "formatting": formatting,
        "columns": columns,
        "tags": tags,
    }
    if paths.get("validationPath"):
        result[f"{canonical.dataset_id}_validation"] = {
            "file_name": Path(str(paths["validationPath"])).name,
            "formatting": formatting,
            "columns": columns,
            "tags": tags,
        }
    return result


__all__ = [
    "LlamaFactoryParameters",
    "LlamaFactoryTask",
    "dataset_info_for_task",
    "dump_yaml",
    "platform_to_llamafactory_config",
    "platform_to_llamafactory_yaml",
    "validate_training_paths",
]
