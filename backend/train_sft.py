from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


EVENT_PREFIX = "__XUNTUI_EVENT__"


def base_model_metadata(model_path: str | Path) -> dict[str, Any] | None:
    """Capture the Base Model capability while its files are available on the trainer."""
    root = Path(str(model_path)).expanduser()
    keys = ("max_position_embeddings", "max_seq_len", "max_sequence_length", "seq_length", "max_context_length")
    for filename in ("config.json", "tokenizer_config.json"):
        path = root / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        sections = [data]
        sections.extend(section for name in ("text_config", "llm_config", "language_config") if isinstance((section := data.get(name)), dict))
        for section in sections:
            for key in keys:
                try:
                    value = int(section.get(key))
                except (TypeError, ValueError, OverflowError):
                    continue
                if 128 <= value <= 1_048_576:
                    return {
                        "maxContextLength": value,
                        "source": f"{filename}:{key}",
                        "configPath": str(path),
                    }
    return None


def dpo_length_kwargs(max_sequence_length: int) -> dict[str, int]:
    """Keep DPO prompt/completions bounded by the selected final sequence length."""
    length = max(128, int(max_sequence_length))
    return {
        "max_prompt_length": length,
        "max_completion_length": length,
        "max_length": length,
    }


def emit(event_type: str, **payload: Any) -> None:
    print(
        EVENT_PREFIX
        + json.dumps({"type": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def resolve_model(model_id: str, source: str) -> str:
    if source != "modelscope":
        emit("status", message=f"从 Hugging Face 加载模型 {model_id}")
        return model_id

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "当前 unsloth_env 未安装 modelscope。请安装 modelscope，"
            "或把任务模型来源改为 huggingface。"
        ) from exc

    emit("status", message=f"从 ModelScope 检查并下载模型 {model_id}")
    cache_dir = os.environ.get("XUNTUI_MODELSCOPE_CACHE", "/root/.cache/modelscope")
    local_path = snapshot_download(model_id, cache_dir=cache_dir)
    emit("status", message=f"基础模型已就绪：{local_path}")
    return local_path


def target_modules(value: str) -> list[str]:
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if value == "attention":
        return attention
    return attention + ["gate_proj", "up_proj", "down_proj"]


def normalize_dpo_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    messages_values = batch.get("messages", [])
    prompt_values = batch.get("prompt", [])
    chosen_values = batch.get("chosen", [])
    rejected_values = batch.get("rejected", [])
    size = max(
        len(messages_values),
        len(prompt_values),
        len(chosen_values),
        len(rejected_values),
    )
    prompts: list[Any] = []
    chosens: list[Any] = []
    rejecteds: list[Any] = []

    def as_messages(value: Any, role: str) -> list[dict[str, str]]:
        if isinstance(value, str):
            return [{"role": role, "content": value}]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return value
        return []

    for index in range(size):
        context = messages_values[index] if index < len(messages_values) else None
        if not context and index < len(prompt_values):
            context = prompt_values[index]
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]
        elif isinstance(context, dict):
            context = [context]
        elif not isinstance(context, list):
            context = []
        chosen = chosen_values[index] if index < len(chosen_values) else None
        rejected = rejected_values[index] if index < len(rejected_values) else None
        prompts.append(context)
        chosens.append(as_messages(chosen, "assistant"))
        rejecteds.append(as_messages(rejected, "assistant"))
    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


def build_optimizer(model: Any, parameters: dict[str, Any]) -> Any:
    import torch
    from peft.optimizers import create_loraplus_optimizer

    optimizer_name = parameters["optimizer"]
    if optimizer_name == "adamw_torch":
        optimizer_cls = torch.optim.AdamW
    else:
        import bitsandbytes as bnb

        optimizer_cls = (
            bnb.optim.PagedAdamW8bit
            if optimizer_name == "paged_adamw_8bit"
            else bnb.optim.AdamW8bit
        )
    return create_loraplus_optimizer(
        model,
        optimizer_cls,
        lr=float(parameters["learningRate"]),
        loraplus_lr_ratio=float(parameters["loraPlusRatio"]),
        weight_decay=float(parameters["weightDecay"]),
    )


def run(config: dict[str, Any]) -> None:
    emit("status", message="加载 Unsloth 与 CUDA 运行环境")
    from unsloth import FastLanguageModel, is_bfloat16_supported

    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    task_type = config.get("taskType", "SFT")
    if task_type == "DPO":
        try:
            # TRL 0.24 receives `(available, version)` tuples from
            # Transformers 5.x for some optional integrations and treats the
            # non-empty tuple as truthy. Disable integrations unused by DPO.
            import trl.import_utils as trl_import_utils

            trl_import_utils._llm_blender_available = False
            trl_import_utils._weave_available = False
            from trl import DPOConfig, DPOTrainer
        except Exception as exc:
            raise RuntimeError(
                "当前训练环境无法加载 TRL DPOTrainer，请先安装其依赖（包括 mergekit）。"
            ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("WSL 中未检测到可用 CUDA GPU。")

    parameters = config["parameters"]
    tuning_mode = config.get("tuningMode", "qlora")
    load_in_4bit = tuning_mode == "qlora"
    model_dtype = (
        None
        if load_in_4bit
        else (torch.bfloat16 if is_bfloat16_supported() else torch.float16)
    )
    model_path = resolve_model(config["model"], config.get("modelSource", "modelscope"))
    emit(
        "status",
        message=(
            f"加载 {'4-bit' if load_in_4bit else '16-bit'} 基础模型到 "
            f"{torch.cuda.get_device_name(0)}"
        ),
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=int(parameters["maxSequenceLength"]),
        dtype=model_dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=False,
        random_state=int(parameters["seed"]),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(parameters["loraRank"]),
        target_modules=target_modules(parameters["targetModules"]),
        lora_alpha=int(parameters["loraAlpha"]),
        lora_dropout=float(parameters["loraDropout"]),
        bias="none",
        use_gradient_checkpointing=(
            "unsloth" if parameters["gradientCheckpointing"] else False
        ),
        random_state=int(parameters["seed"]),
        max_seq_length=int(parameters["maxSequenceLength"]),
    )

    data_files = {"train": config["trainPath"]}
    if config.get("validationPath") and int(config.get("validationSplit", 0)) > 0:
        data_files["validation"] = config["validationPath"]
    emit("status", message=f"读取数据集：{config['datasetName']}")
    dataset = load_dataset("json", data_files=data_files)
    validation_split = int(config.get("validationSplit", 0))
    if validation_split > 0 and "validation" not in dataset:
        split_dataset = dataset["train"].train_test_split(
            test_size=validation_split / 100,
            seed=int(parameters["seed"]),
        )
        dataset = DatasetDict(
            train=split_dataset["train"],
            validation=split_dataset["test"],
        )

    def format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
        return {"text": texts}

    if task_type == "DPO":
        for split_name in dataset:
            source_columns = dataset[split_name].column_names
            dataset[split_name] = dataset[split_name].map(
                normalize_dpo_batch,
                batched=True,
                remove_columns=source_columns,
                desc=f"规范化 {split_name} DPO 偏好数据",
            )
    else:
        dataset = dataset.map(format_batch, batched=True, desc="应用 Qwen 对话模板")
    emit(
        "status",
        message=(
            f"训练样本 {len(dataset['train'])} 条"
            + (
                f"，使用验证集 {len(dataset['validation'])} 条"
                if "validation" in dataset
                else "，未启用训练期验证"
            )
        ),
    )

    output_dir = Path(config["artifactRoot"]) / config["id"]
    checkpoint_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    final_adapter_dir = output_dir / "final_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_strategy = config.get("checkpointStrategy", "epoch")
    max_steps = int(config.get("maxSteps", -1))
    evaluation_enabled = "validation" in dataset
    bf16 = bool(is_bfloat16_supported())
    steps_per_epoch = max(
        1,
        math.ceil(
            len(dataset["train"])
            / (
                int(parameters["batchSize"])
                * int(parameters["gradientAccumulation"])
            )
        ),
    )
    total_steps = (
        max_steps
        if max_steps > 0
        else math.ceil(steps_per_epoch * float(parameters["epochs"]))
    )
    warmup_steps = int(total_steps * float(parameters["warmupRatio"]))
    save_steps = (
        steps_per_epoch * max(1, int(config.get("checkpointInterval", 1)))
        if save_strategy == "epoch"
        else max(1, int(config.get("checkpointInterval", 1)))
    )

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=int(parameters["batchSize"]),
        gradient_accumulation_steps=int(parameters["gradientAccumulation"]),
        num_train_epochs=float(parameters["epochs"]),
        max_steps=max_steps,
        learning_rate=float(parameters["learningRate"]),
        lr_scheduler_type=parameters["lrScheduler"],
        warmup_steps=warmup_steps,
        weight_decay=float(parameters["weightDecay"]),
        max_grad_norm=float(parameters["maxGradNorm"]),
        bf16=bf16,
        fp16=not bf16,
        logging_steps=max(1, int(parameters["loggingSteps"])),
        logging_first_step=True,
        report_to="none",
        eval_strategy="steps" if evaluation_enabled else "no",
        eval_steps=(
            max(1, int(parameters["validationSteps"]))
            if evaluation_enabled
            else None
        ),
        per_device_eval_batch_size=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=max(1, int(config.get("checkpointLimit", 1))),
        seed=int(parameters["seed"]),
        data_seed=int(parameters["seed"]),
        max_length=int(parameters["maxSequenceLength"]),
        gradient_checkpointing=bool(parameters["gradientCheckpointing"]),
        dataset_num_proc=1,
        pad_token=tokenizer.pad_token or tokenizer.eos_token,
    )
    if task_type == "SFT":
        training_kwargs.update(
            packing=bool(parameters["packing"]),
            dataset_text_field="text",
            eos_token=tokenizer.eos_token,
        )
    else:
        training_kwargs.update(
            beta=float(config.get("dpoBeta", 0.1)),
            loss_type="sigmoid",
        )
        training_kwargs.update(
            dpo_length_kwargs(int(parameters["maxSequenceLength"]))
        )
    training_kwargs.update(config.get("extraParameters", {}))
    training_args = (SFTConfig if task_type == "SFT" else DPOConfig)(**training_kwargs)

    stop_flag_path = Path(str(config.get("stopFlagPath", ""))) if config.get("stopFlagPath") else None
    early_stopping = bool(config.get("earlyStopping", False))
    patience = max(1, int(config.get("patience", 3)))
    min_delta = max(0.0, float(config.get("minDelta", 0)))
    metric_name = "eval_loss" if evaluation_enabled else "train_loss"

    class ProgressCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best_state: dict[str, Any] | None = None
            self.best_loss: float | None = None
            self.best_step: int | None = None
            self.best_train_loss: float | None = None
            self.best_eval_loss: float | None = None
            self.final_train_loss: float | None = None
            self.final_eval_loss: float | None = None
            self.last_eval_step: int | None = None
            self.bad_evaluations = 0
            self.stop_requested = False
            self.stop_reason: str | None = None
            self.model: Any | None = None

        @staticmethod
        def _trainable_state(model: Any) -> dict[str, Any]:
            # LoRA/QLoRA keeps the trainable state small enough to retain one
            # CPU snapshot for reliable best-checkpoint promotion.
            return {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        def _stop_file_exists(self) -> bool:
            return bool(stop_flag_path and stop_flag_path.exists())

        def _request_stop(self, control: Any, reason: str) -> None:
            self.stop_requested = True
            self.stop_reason = reason
            control.should_training_stop = True

        def _record_best(self, loss: float, step: int, model: Any) -> None:
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss
                self.best_step = step
                self.best_state = self._trainable_state(model)
                if metric_name == "eval_loss":
                    self.best_eval_loss = loss
                else:
                    self.best_train_loss = loss

        def on_train_begin(self, args: Any, state: Any, control: Any, model=None, **kwargs: Any):
            self.model = model
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics=None,
            model=None,
            **kwargs: Any,
        ):
            values = metrics or {}
            raw_eval_loss = values.get("eval_loss")
            model = model or self.model
            if raw_eval_loss is None or model is None:
                return control
            eval_loss = float(raw_eval_loss)
            self.final_eval_loss = eval_loss
            self.last_eval_step = int(state.global_step)
            previous_best = self.best_loss
            self._record_best(eval_loss, int(state.global_step), model)
            if previous_best is None or eval_loss < previous_best - min_delta:
                self.bad_evaluations = 0
            else:
                self.bad_evaluations += 1
            if self._stop_file_exists():
                self._request_stop(control, "manual")
            elif early_stopping and self.bad_evaluations >= patience:
                self._request_stop(control, "automatic")
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
            if self._stop_file_exists():
                self._request_stop(control, "manual")
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs=None, **kwargs: Any):
            values = logs or {}
            total = max(1, int(state.max_steps or 1))
            step = int(state.global_step)
            progress = min(95, int(step * 95 / total))
            train_loss = values.get("loss")
            eval_loss = values.get("eval_loss")
            model = kwargs.get("model") or self.model
            if train_loss is not None:
                self.final_train_loss = float(train_loss)
                if not evaluation_enabled and model is not None:
                    previous_best = self.best_loss
                    self._record_best(float(train_loss), step, model)
                    if previous_best is None or float(train_loss) < previous_best - min_delta:
                        self.bad_evaluations = 0
                    else:
                        self.bad_evaluations += 1
                    if self._stop_file_exists():
                        self._request_stop(control, "manual")
                    elif early_stopping and self.bad_evaluations >= patience:
                        self._request_stop(control, "automatic")
            emit(
                "progress",
                progress=progress,
                step=step,
                totalSteps=total,
                epoch=values.get("epoch"),
                trainLoss=train_loss,
                evalLoss=eval_loss,
                learningRate=values.get("learning_rate"),
                message=f"训练中 · Step {step}/{total}",
            )
            return control

    progress_callback = ProgressCallback()
    optimizer = build_optimizer(model, parameters)
    trainer_class = SFTTrainer if task_type == "SFT" else DPOTrainer
    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=training_args,
        callbacks=[progress_callback],
        optimizers=(optimizer, None),
    )

    emit(
        "status",
        message=f"开始 {task_type} {'QLoRA' if load_in_4bit else 'LoRA'} 训练",
    )
    result = trainer.train()
    final_step = int(trainer.state.global_step)
    if evaluation_enabled and progress_callback.last_eval_step != final_step:
        emit("status", message=f"评估最终 Adapter · Step {final_step}")
        trainer.evaluate()
    best_state = progress_callback.best_state
    has_best_checkpoint = best_state is not None and progress_callback.best_step is not None

    emit("status", message="保存最终 LoRA Adapter 与 tokenizer")
    model.save_pretrained(str(final_adapter_dir if has_best_checkpoint else adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir if has_best_checkpoint else adapter_dir))

    if has_best_checkpoint:
        emit(
            "status",
            message=(
                "恢复验证集 Loss 最低的 Adapter"
                if metric_name == "eval_loss"
                else "恢复训练 Loss 最低的 Adapter"
            ),
        )
        assert best_state is not None
        for name, parameter in model.named_parameters():
            snapshot = best_state.get(name)
            if snapshot is not None:
                parameter.data.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    selected_adapter_dir = adapter_dir
    selected_variant = "best" if has_best_checkpoint else "final"
    selected_step = progress_callback.best_step if has_best_checkpoint else final_step
    selected_loss = progress_callback.best_loss if has_best_checkpoint else (
        progress_callback.final_eval_loss
        if evaluation_enabled
        else progress_callback.final_train_loss
    )
    final_loss = (
        progress_callback.final_eval_loss
        if evaluation_enabled
        else progress_callback.final_train_loss
    )
    model_metadata = base_model_metadata(model_path)
    artifacts = [
        {
            "type": "adapter",
            "name": config["outputName"],
            "path": str(selected_adapter_dir),
            "variant": selected_variant,
            "selected": True,
            "checkpointStep": selected_step,
            "evalLoss": progress_callback.best_eval_loss,
            "trainLoss": progress_callback.best_train_loss,
            "loss": selected_loss,
            "metric": metric_name,
            "baseModelPath": model_path,
            "baseModelMetadata": model_metadata,
        }
    ]
    if has_best_checkpoint:
        artifacts.append(
            {
                "type": "adapter",
                "name": f"{config['outputName']}-final",
                "path": str(final_adapter_dir),
                "variant": "final",
                "selected": False,
                "checkpointStep": final_step,
                "evalLoss": progress_callback.final_eval_loss,
                "trainLoss": progress_callback.final_train_loss,
                "loss": final_loss,
                "metric": metric_name,
                "baseModelPath": model_path,
                "baseModelMetadata": model_metadata,
            }
        )

    if config.get("mergeModel"):
        merged_dir = output_dir / "merged"
        emit("status", message="合并 LoRA Adapter 与基础模型")
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )
        artifacts.append(
            {
                "type": "merged",
                "name": f"{config['outputName']}-merged",
                "path": str(merged_dir),
                "sourceAdapterVariant": selected_variant,
            }
        )

    metrics_path = output_dir / "train_metrics.json"
    result_metrics = {
        **result.metrics,
        "best_step": progress_callback.best_step,
        "best_eval_loss": progress_callback.best_eval_loss,
        "best_train_loss": progress_callback.best_train_loss,
        "best_loss": selected_loss,
        "final_eval_loss": progress_callback.final_eval_loss,
        "final_train_loss": progress_callback.final_train_loss,
        "final_loss": final_loss,
        "final_step": final_step,
        "selected_adapter": selected_variant,
        "selection_metric": metric_name,
        "early_stopping": early_stopping,
        "patience": patience,
        "min_delta": min_delta,
        "stop_reason": progress_callback.stop_reason,
    }
    metrics_path.write_text(
        json.dumps(result_metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    artifacts.append(
        {"type": "metrics", "name": "训练指标", "path": str(metrics_path)}
    )
    event_type = "stopped" if progress_callback.stop_requested else "completed"
    emit(
        event_type,
        message=(
            "训练已停止，Adapter 已保存"
            if progress_callback.stop_requested
            else "训练完成，Adapter 已保存"
        ),
        progress=100,
        loss=selected_loss,
        artifacts=artifacts,
        stopReason=progress_callback.stop_reason,
        selection={
            "bestStep": progress_callback.best_step,
            "bestEvalLoss": progress_callback.best_eval_loss,
            "bestTrainLoss": progress_callback.best_train_loss,
            "bestLoss": selected_loss,
            "finalEvalLoss": progress_callback.final_eval_loss,
            "finalTrainLoss": progress_callback.final_train_loss,
            "finalLoss": final_loss,
            "finalStep": final_step,
            "selectedAdapter": selected_variant,
            "selectionMetric": metric_name,
            "selectionReason": (
                (
                    "验证集 Loss 最低"
                    if metric_name == "eval_loss"
                    else "训练 Loss 最低"
                )
                if has_best_checkpoint
                else "未产生有效验证 Loss，使用最终 Adapter"
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        run(config)
        return 0
    except Exception as exc:
        diagnostics = ""
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                diagnostics = (
                    f" GPU 显存：free={free / 1024**3:.2f}GB，"
                    f"total={total / 1024**3:.2f}GB，"
                    f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB，"
                    f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB。"
                )
        except Exception:
            pass
        emit(
            "failed",
            message=f"{exc}{diagnostics}",
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
