"""Read LLaMA-Factory artifact metadata without loading model weights.

The final evaluation is saved AFTER trainer_state.json by LLaMA-Factory.
A final-only evaluation must never be promoted to a historical best checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _step(value: Any) -> int | None:
    number = finite_number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match[1]) if match else None


def _best_path(value: Any, output: Path, remote_output: str | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidates = [Path(value)] if Path(value).is_absolute() else [output / value, output.parent / value]
    if remote_output:
        remote_path, remote_root = PurePosixPath(value), PurePosixPath(remote_output)
        if remote_path.is_absolute() and remote_path.is_relative_to(remote_root):
            relative = remote_path.relative_to(remote_root)
            if ".." not in relative.parts:
                candidates.append(output.joinpath(*relative.parts))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate != output and candidate.is_relative_to(output) and _checkpoint_step(candidate) is not None and (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def read_training_artifacts(output: Path, *, remote_output: str | None = None) -> list[dict[str, Any]]:
    output = output.resolve()
    state = read_json_object(output / "trainer_state.json")
    checkpoints = sorted(
        (p for p in output.glob("checkpoint-*") if _checkpoint_step(p) is not None and p.resolve().is_relative_to(output) and (p / "adapter_config.json").is_file()),
        key=lambda p: _checkpoint_step(p) or 0,
    )
    final_dir = output if (output / "adapter_config.json").is_file() else (checkpoints[-1] if checkpoints else None)
    if final_dir is None:
        raise FileNotFoundError(f"未找到 LLaMA-Factory Adapter 产物：{output}")
    # For a stopped run without a root model, describe the saved checkpoint,
    # not later log rows or an unrelated root evaluation.
    final_state = state if final_dir == output else read_json_object(final_dir / "trainer_state.json")
    history = final_state.get("log_history") or []
    history = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
    final_step = _step(final_state.get("global_step"))
    if final_step is None:
        final_step = _checkpoint_step(final_dir) if final_dir != output else max((_step(x.get("step")) or 0 for x in history), default=0) or None
    final_eval = finite_number(read_json_object(final_dir / "eval_results.json").get("eval_loss"))
    if final_eval is None:
        final_eval = finite_number(read_json_object(final_dir / "all_results.json").get("eval_loss"))
    if final_eval is None:
        final_eval = next((finite_number(x.get("eval_loss")) for x in reversed(history) if _step(x.get("step")) == final_step and finite_number(x.get("eval_loss")) is not None), None)
    final_train = next((finite_number(x.get("loss")) for x in reversed(history) if finite_number(x.get("loss")) is not None), None)
    eval_history = state.get("log_history") or []
    eval_history = [x for x in eval_history if isinstance(x, dict) and finite_number(x.get("eval_loss")) is not None] if isinstance(eval_history, list) else []
    best_dir = _best_path(state.get("best_model_checkpoint"), output, remote_output) if eval_history else None
    artifacts: list[dict[str, Any]] = []
    if best_dir:
        best_step = _checkpoint_step(best_dir)
        best_eval = next((finite_number(x["eval_loss"]) for x in reversed(eval_history) if _step(x.get("step")) == best_step), None)
        if best_eval is not None:
            # Include the end-of-training evaluation in the comparison only
            # when there is an actual historical best to compare against.
            if final_eval is not None and final_eval < best_eval:
                best_dir, best_step, best_eval = final_dir, final_step, final_eval
            artifacts.append({"type": "adapter", "name": "best", "variant": "best", "selected": True, "path": str(best_dir), "checkpointStep": best_step, "evalLoss": best_eval, "loss": best_eval, "metric": "eval_loss"})
    has_best = bool(artifacts)
    final: dict[str, Any] = {"type": "adapter", "name": "final", "variant": "final", "selected": not has_best, "path": str(final_dir), "checkpointStep": final_step, "evalLoss": final_eval, "trainLoss": final_train}
    if final_eval is not None:
        final.update(loss=final_eval, metric="eval_loss")
    if not has_best:
        if final_eval is not None and not eval_history:
            final["selectionReason"] = "仅训练结束后验证，未生成历史最佳检查点，使用最终 Adapter"
        elif eval_history:
            final["selectionReason"] = "存在验证记录，但最佳检查点不可用，使用最终 Adapter"
    artifacts.append(final)
    for name in ("trainer_state.json", "eval_results.json", "train_results.json", "all_results.json"):
        path = output / name
        if path.is_file():
            artifacts.append({"type": "metrics", "name": name, "path": str(path)})
    return artifacts


def verified_local_training_artifacts(workdir: Path, *, expected_manifest_sha256: str | None = None) -> bool:
    """A manifest-verified mirror can be reindexed without a live GPU instance."""
    workdir = workdir.resolve()
    manifest_path = workdir / "manifest.json"
    if expected_manifest_sha256 is not None:
        try:
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha256:
                return False
        except OSError:
            return False
    manifest = read_json_object(manifest_path)
    entries = manifest.get("files")
    if manifest.get("taskId") != workdir.name or not isinstance(entries, list) or not entries:
        return False
    verified: set[str] = set()
    try:
        for entry in entries:
            relative = PurePosixPath(entry["relativePath"])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "output":
                return False
            path = workdir.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(workdir / "output") or not path.is_file() or path.stat().st_size != entry["sizeBytes"]:
                return False
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != entry["sha256"]:
                    return False
            verified.add(relative.as_posix())
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return "output/trainer_state.json" in verified and "output/adapter_config.json" in verified and any(
        PurePosixPath(name).parent.as_posix() == "output" and PurePosixPath(name).suffix in {".safetensors", ".bin"} and PurePosixPath(name).name.startswith("adapter_model")
        for name in verified
    )
