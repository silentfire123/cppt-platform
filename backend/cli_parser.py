from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shlex
from typing import Any


class CliValidationError(ValueError):
    """Raised when a submitted training command is invalid or unsafe."""


DEFAULT_PARAMETERS: dict[str, Any] = {
    "epochs": 3,
    "learningRate": "2e-4",
    "batchSize": 1,
    "gradientAccumulation": 8,
    "validationSteps": 50,
    "maxSequenceLength": 1024,
    "loraRank": 16,
    "loraAlpha": 16,
    "loraDropout": 0,
    "lrScheduler": "linear",
    "loraPlusEnabled": False,
    "loraPlusRatio": 16,
    "targetModules": "auto",
    "optimizer": "adamw_torch",
    "warmupRatio": 0.1,
    "weightDecay": 0.01,
    "maxGradNorm": 1,
    "seed": 42,
    "loggingSteps": 10,
    "packing": False,
    "gradientCheckpointing": True,
}


EXTRA_PARAM_TYPES: dict[str, type] = {
    "save_safetensors": bool,
    "tf32": bool,
    "torch_compile": bool,
    "eval_accumulation_steps": int,
    "dataloader_num_workers": int,
    "group_by_length": bool,
    "include_tokens_per_second": bool,
    "include_num_input_tokens_seen": bool,
    "logging_nan_inf_filter": bool,
    "remove_unused_columns": bool,
}

SHELL_META = re.compile(r"(?:&&|[;&|<>`]|\$\()")
KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", default="modelscope")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task-type", choices=("SFT", "DPO"), default="SFT")
    parser.add_argument("--owner", default="平台管理员")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--tuning-mode", choices=("lora", "qlora"), default="qlora")
    parser.add_argument("--merge-model", action="store_true")
    parser.add_argument("--validation-split", type=int, default=10)
    parser.add_argument("--checkpoint-limit", type=int, default=2)
    parser.add_argument("--checkpoint-strategy", choices=("epoch", "steps"), default="epoch")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lr-scheduler", choices=("linear", "cosine", "constant"))
    parser.add_argument("--enable-lora-plus", action="store_true")
    parser.add_argument("--lora-plus-ratio", type=float)
    parser.add_argument("--target-modules", choices=("auto", "attention", "all-linear"))
    parser.add_argument("--optimizer", choices=("adamw_8bit", "adamw_torch", "paged_adamw_8bit"))
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chat-max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-temperature", type=float, default=0.2)
    parser.add_argument("--chat-top-p", type=float, default=0.9)
    parser.add_argument("--chat-seed", type=int, default=42)
    parser.add_argument("--trainer-param", action="append", default=[])
    return parser


def _strip_prefix(tokens: list[str]) -> list[str]:
    if not tokens:
        raise CliValidationError("命令不能为空。")
    normalized = [token.replace("\\", "/") for token in tokens]
    if normalized[0] in {"python", "python3", "py"}:
        normalized = normalized[1:]
    if normalized and normalized[0].endswith("train_cli.py"):
        normalized = normalized[1:]
    if normalized and normalized[0] == "submit":
        normalized = normalized[1:]
    if not normalized or normalized[0].startswith("-") is False and normalized[0] != "--help":
        return normalized
    return normalized


def _parse_extra(items: list[str], line_numbers: list[int] | None = None) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for index, item in enumerate(items):
        prefix = f"第 {line_numbers[index]} 行" if line_numbers and index < len(line_numbers) else ""
        if "=" not in item:
            raise CliValidationError(f"{prefix + '，' if prefix else ''}高级参数必须使用 KEY=VALUE 格式：{item}")
        key, raw_value = item.split("=", 1)
        if key not in EXTRA_PARAM_TYPES or not KEY_PATTERN.fullmatch(key):
            allowed = ", ".join(sorted(EXTRA_PARAM_TYPES))
            raise CliValidationError(f"{prefix + '，' if prefix else ''}不支持的高级训练参数 {key}。允许值：{allowed}")
        expected_type = EXTRA_PARAM_TYPES[key]
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if expected_type is bool and not isinstance(value, bool):
            raise CliValidationError(f"{prefix + '，' if prefix else ''}高级参数 {key} 必须是 true 或 false。")
        if expected_type is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise CliValidationError(f"{prefix + '，' if prefix else ''}高级参数 {key} 必须是整数。")
        extras[key] = value
    return extras


def validate_extra_parameters(values: dict[str, Any]) -> dict[str, Any]:
    encoded = [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]
    return _parse_extra(encoded)


def _tokenize_command(command: str) -> tuple[list[str], list[int]]:
    lines = command.splitlines() or [command]
    tokens: list[str] = []
    token_lines: list[int] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if SHELL_META.search(line):
            raise CliValidationError(
                f"第 {line_number} 行包含不允许的 Shell 操作符，只能提交训练参数。"
            )
        try:
            line_tokens = shlex.split(line, posix=True)
        except ValueError as exc:
            raise CliValidationError(
                f"第 {line_number} 行引号格式错误：{exc}"
            ) from exc
        tokens.extend(line_tokens)
        token_lines.extend([line_number] * len(line_tokens))
    return tokens, token_lines


def parse_cli_command(command: str) -> dict[str, Any]:
    if not command.strip():
        raise CliValidationError("命令不能为空。")
    tokens, token_lines = _tokenize_command(command)
    tokens = _strip_prefix(tokens)
    if token_lines:
        prefix_count = len(token_lines) - len(tokens)
        token_lines = token_lines[prefix_count:]
    if not tokens:
        raise CliValidationError("命令缺少训练参数。")
    parser = _parser()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            namespace, unknown = parser.parse_known_args(tokens)
    except SystemExit as exc:
        line_number = next(
            (token_lines[index] for index, token in enumerate(tokens) if token.startswith("--")),
            1,
        )
        raise CliValidationError(
            f"第 {line_number} 行参数格式不正确，请检查参数名称和值。"
        ) from exc
    if unknown:
        unknown_line = next(
            (
                token_lines[index]
                for index, token in enumerate(tokens)
                if token in unknown
            ),
            1,
        )
        raise CliValidationError(
            f"第 {unknown_line} 行存在未识别的命令参数：{' '.join(unknown)}"
        )
    values = vars(namespace)
    parameters = dict(DEFAULT_PARAMETERS)
    mapping = {
        "epochs": "epochs",
        "learning_rate": "learningRate",
        "batch_size": "batchSize",
        "gradient_accumulation": "gradientAccumulation",
        "validation_steps": "validationSteps",
        "max_sequence_length": "maxSequenceLength",
        "lora_rank": "loraRank",
        "lora_alpha": "loraAlpha",
        "lora_dropout": "loraDropout",
        "lr_scheduler": "lrScheduler",
        "lora_plus_ratio": "loraPlusRatio",
        "target_modules": "targetModules",
        "optimizer": "optimizer",
        "warmup_ratio": "warmupRatio",
        "weight_decay": "weightDecay",
        "max_grad_norm": "maxGradNorm",
        "seed": "seed",
        "logging_steps": "loggingSteps",
        "packing": "packing",
        "gradient_checkpointing": "gradientCheckpointing",
    }
    for source, target in mapping.items():
        if values[source] is not None:
            parameters[target] = values[source]
    parameters["loraPlusEnabled"] = bool(values["enable_lora_plus"] or values["lora_plus_ratio"] is not None)
    payload = {
        "name": values["name"].strip(),
        "model": values["model"].strip(),
        "modelSource": values["model_source"],
        "tuningMode": values["tuning_mode"],
        "datasetId": values["dataset_id"].strip(),
        "datasetName": values["dataset_name"].strip(),
        "taskType": values["task_type"],
        "owner": values["owner"].strip(),
        "mergeModel": values["merge_model"],
        "validationSplit": values["validation_split"],
        "outputName": values["output_name"].strip(),
        "checkpointLimit": values["checkpoint_limit"],
        "checkpointStrategy": values["checkpoint_strategy"],
        "checkpointInterval": values["checkpoint_interval"],
        "maxSteps": values["max_steps"],
        "earlyStopping": values["early_stopping"],
        "patience": values["patience"],
        "minDelta": values["min_delta"],
        "parameters": parameters,
        "extraParameters": _parse_extra(
            values["trainer_param"],
            [
                token_lines[index]
                for index, token in enumerate(tokens[:-1])
                if token == "--trainer-param"
            ],
        ),
        "chatParameters": {
            "maxNewTokens": values["chat_max_new_tokens"],
            "temperature": values["chat_temperature"],
            "topP": values["chat_top_p"],
            "seed": values["chat_seed"],
        },
    }
    if not payload["name"] or not payload["model"] or not payload["datasetId"] or not payload["datasetName"] or not payload["outputName"]:
        raise CliValidationError("name、model、dataset-id、dataset-name、output-name 不能为空。")
    if not 0 <= payload["validationSplit"] <= 30:
        raise CliValidationError("validation-split 必须在 0 到 30 之间。")
    if payload["patience"] < 1 or payload["minDelta"] < 0:
        raise CliValidationError("patience 必须大于等于 1，min-delta 必须大于等于 0。")
    if payload["maxSteps"] < -1 or payload["checkpointLimit"] < 1 or payload["checkpointInterval"] < 1:
        raise CliValidationError("max-steps、checkpoint-limit、checkpoint-interval 的值不合法。")
    return payload
