from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from backend.llamafactory import (
    LlamaFactoryTask,
    dataset_info_for_task,
    platform_to_llamafactory_config,
    platform_to_llamafactory_yaml,
)
from backend.local_llamafactory_executor import LocalLlamaFactoryExecutor


class LlamaFactoryConfigTest(unittest.TestCase):
    def test_loraplus_and_checkpoint_controls_reach_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            task = LlamaFactoryTask(
                taskId="ft-controls",
                model="tiny-model",
                datasetId="smoke_dataset",
                dataset={"paths": {"trainPath": str(train)}},
                parameters={
                    "loraPlusEnabled": True,
                    "loraPlusRatio": 8,
                    "checkpointLimit": 3,
                    "checkpointStrategy": "steps",
                    "checkpointInterval": 25,
                },
            )
            config = platform_to_llamafactory_config(task, output_dir=root / "output", allowed_roots=[root])
            self.assertEqual(config["loraplus_lr_ratio"], 8)
            self.assertEqual(config["save_strategy"], "steps")
            self.assertEqual(config["save_steps"], 25)
            self.assertEqual(config["save_total_limit"], 3)

    def test_loraplus_is_omitted_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            task = LlamaFactoryTask(
                taskId="ft-controls-off",
                model="tiny-model",
                datasetId="smoke_dataset",
                dataset={"paths": {"trainPath": str(train)}},
                parameters={"loraPlusRatio": 16},
            )
            config = platform_to_llamafactory_config(task, output_dir=root / "output", allowed_roots=[root])
            self.assertNotIn("loraplus_lr_ratio", config)

    def test_platform_parameters_are_translated_to_safe_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text('{"messages": []}\n', encoding="utf-8")
            task = LlamaFactoryTask(
                taskId="ft-smoke-1",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                datasetId="smoke_dataset",
                taskType="SFT",
                tuningMode="lora",
                outputName="smoke",
                dataset={"paths": {"trainPath": str(train)}},
                parameters={
                    "batchSize": 1,
                    "gradientAccumulation": 2,
                    "learningRate": "2e-4",
                    "epochs": 1,
                    "maxSequenceLength": 128,
                    "loraRank": 4,
                    "loraAlpha": 8,
                    "targetModules": "attention",
                    "useCpu": True,
                },
            )
            config = platform_to_llamafactory_config(task, output_dir=root / "output", allowed_roots=[root])
            self.assertEqual(config["stage"], "sft")
            self.assertEqual(config["finetuning_type"], "lora")
            self.assertEqual(config["per_device_train_batch_size"], 1)
            self.assertEqual(config["gradient_accumulation_steps"], 2)
            self.assertEqual(config["lora_rank"], 4)
            self.assertEqual(config["lora_target"], ["q_proj", "k_proj", "v_proj", "o_proj"])
            self.assertTrue(config["logging_first_step"])
            self.assertIn("output_dir:", platform_to_llamafactory_yaml(task, output_dir=root / "output", allowed_roots=[root]))
            self.assertIn("smoke_dataset", dataset_info_for_task(task))

    def test_auto_target_modules_resolves_to_attention_modules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text('{}\n', encoding="utf-8")
            task = LlamaFactoryTask(
                taskId="ft-auto-target",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                datasetId="smoke_dataset",
                dataset={"paths": {"trainPath": str(train)}},
                parameters={"targetModules": "auto"},
            )
            config = platform_to_llamafactory_config(task, output_dir=root / "output", allowed_roots=[root])
            self.assertEqual(config["lora_target"], ["q_proj", "k_proj", "v_proj", "o_proj"])

    def test_cpu_falls_back_from_bitsandbytes_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text('{}\n', encoding="utf-8")
            task = LlamaFactoryTask(
                taskId="ft-cpu-optimizer",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                datasetId="smoke_dataset",
                dataset={"paths": {"trainPath": str(train)}},
                parameters={
                    "optimizer": "adamw_8bit",
                    "useCpu": True,
                },
            )
            config = platform_to_llamafactory_config(task, output_dir=root / "output", allowed_roots=[root])
            self.assertEqual(task.parameters.optimizer, "adamw_torch")
            self.assertEqual(config["optim"], "adamw_torch")
            self.assertNotIn("adamw_8bit", platform_to_llamafactory_yaml(task, output_dir=root / "output", allowed_roots=[root]))

    def test_cpu_falls_back_from_paged_bitsandbytes_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text('{}\n', encoding="utf-8")
            task = LlamaFactoryTask(
                taskId="ft-cpu-paged-optimizer",
                model="tiny-model",
                datasetId="smoke_dataset",
                dataset={"paths": {"trainPath": str(train)}},
                parameters={
                    "optimizer": "paged_adamw_8bit",
                    "useCpu": True,
                },
            )
            config = platform_to_llamafactory_config(task, output_dir=root / "output", allowed_roots=[root])
            self.assertEqual(config["optim"], "adamw_torch")

    def test_paths_cannot_escape_allowed_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            task = {
                "taskId": "ft-safe",
                "model": "tiny-model",
                "datasetId": "safe",
                "dataset": {"paths": {"trainPath": str(train)}},
            }
            with self.assertRaises(ValueError):
                platform_to_llamafactory_config(task, output_dir=Path(directory).parent / "escape", allowed_roots=[root])


class LocalExecutorSmokeTest(unittest.IsolatedAsyncioTestCase):
    def test_console_script_is_not_prefixed_with_python(self) -> None:
        executor = LocalLlamaFactoryExecutor(
            command="llamafactory-cli",
            python_executable="/opt/venvs/llamafactory/bin/python",
        )
        self.assertEqual(
            executor._launch_argv("/tmp/train.yaml"),
            ["llamafactory-cli", "train", "/tmp/train.yaml"],
        )

    def test_explicit_python_module_is_prefixed(self) -> None:
        executor = LocalLlamaFactoryExecutor(
            command="-m llamafactory.cli",
            python_executable="/opt/venvs/llamafactory/bin/python",
        )
        self.assertEqual(
            executor._launch_argv("/tmp/train.yaml"),
            ["/opt/venvs/llamafactory/bin/python", "-m", "llamafactory.cli", "train", "/tmp/train.yaml"],
        )

    async def test_prepare_uses_isolated_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text('{"messages": [{"role":"user","content":"hi"}]}\n', encoding="utf-8")
            executor = LocalLlamaFactoryExecutor(work_root=root / "work", command=[sys.executable, "-c", "pass"])
            task = {
                "id": "ft-isolated-paths",
                "model": "tiny-model",
                "datasetId": "tiny",
                "dataset": {"paths": {"trainPath": str(source)}},
                "trainPath": str(source),
            }
            handle = await executor.prepare(task)
            config = (Path(handle.workdir) / "train.yaml").read_text(encoding="utf-8")
            self.assertIn(f"dataset_dir: {Path(handle.workdir) / 'dataset'}", config)
            self.assertNotIn(f"dataset_dir: {source.parent}\n", config)
            self.assertIn("dataset: tiny", config)

    async def test_events_infers_step_and_progress_from_llamafactory_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "tiny.jsonl"
            train.write_text('{"messages": [{"role":"user","content":"hi"}]}\n', encoding="utf-8")
            script = (
                "print('Num Epochs = 1', flush=True); "
                "print('Total optimization steps = 10', flush=True); "
                "print(\"{'loss': '1.5', 'learning_rate': '1e-4', 'epoch': '0.5'}\", flush=True)"
            )
            executor = LocalLlamaFactoryExecutor(
                work_root=root / "work",
                command=[sys.executable, "-u", "-c", script],
            )
            task = {
                "id": "ft-progress-events",
                "model": "tiny-model",
                "datasetId": "tiny",
                "parameters": {"loggingSteps": 5},
                "dataset": {"paths": {"trainPath": str(train)}},
            }
            await executor.start(task)
            events = [event async for event in executor.events(task)]
            metric = next(event for event in events if event.type == "metric")
            self.assertEqual(metric.payload["step"], 5)
            self.assertEqual(metric.payload["total_steps"], 10)
            self.assertEqual(metric.payload["progress"], 47)

    async def test_events_waits_for_quick_process_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "tiny.jsonl"
            train.write_text('{"messages": [{"role":"user","content":"hi"}]}\n', encoding="utf-8")
            executor = LocalLlamaFactoryExecutor(
                work_root=root / "work",
                command=[sys.executable, "-c", "pass"],
            )
            task = {
                "id": "ft-quick-exit",
                "model": "tiny-model",
                "datasetId": "tiny",
                "dataset": {"paths": {"trainPath": str(train)}},
            }
            await executor.start(task)
            self.assertEqual([event async for event in executor.events(task)], [])
            self.assertEqual((await executor.status(task))["state"], "completed")

    async def test_fake_cli_runs_same_prepare_start_collect_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "tiny.jsonl"
            train.write_text('{"messages": [{"role":"user","content":"hi"}]}\n', encoding="utf-8")
            smoke_cli = Path(__file__).resolve().parents[1] / "scripts" / "llamafactory_smoke_cli.py"
            executor = LocalLlamaFactoryExecutor(
                work_root=root / "work",
                command=[sys.executable, str(smoke_cli)],
            )
            task = {
                "id": "ft-smoke-executor",
                "model": "tiny-model",
                "datasetId": "tiny",
                "taskType": "SFT",
                "tuningMode": "lora",
                "outputName": "smoke",
                "parameters": {"useCpu": True, "maxSequenceLength": 128},
                "dataset": {"paths": {"trainPath": str(train)}},
            }
            handle = await executor.prepare(task)
            self.assertTrue(Path(handle.config_path).is_file())
            await executor.start(task)
            events = [event async for event in executor.events(task)]
            self.assertEqual([event.type for event in events], ["status", "metric", "completed"])
            self.assertEqual((await executor.status(task))["state"], "completed")
            artifacts = await executor.collect_artifacts(task)
            variants = {item.get("variant") for item in artifacts if item.get("type") == "adapter"}
            self.assertEqual(variants, {"best", "final"})
            manifest = next(item for item in artifacts if item["type"] == "manifest")
            self.assertTrue(Path(manifest["path"]).is_file())

    async def test_no_eval_history_selects_final_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "tiny.jsonl"
            train.write_text('{"messages": [{"role":"user","content":"hi"}]}\n', encoding="utf-8")
            executor = LocalLlamaFactoryExecutor(
                work_root=root / "work",
                command=[sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "llamafactory_smoke_cli.py")],
            )
            task = {
                "id": "ft-final-only",
                "model": "tiny-model",
                "datasetId": "tiny",
                "dataset": {"paths": {"trainPath": str(train)}},
            }
            handle = await executor.prepare(task)
            output = Path(handle.workdir) / "output"
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output / "trainer_state.json").write_text(
                json.dumps({"best_model_checkpoint": str(checkpoint), "log_history": [{"loss": 1.2, "step": 1}]}),
                encoding="utf-8",
            )
            artifacts = await executor.collect_artifacts(task)
            adapters = [item for item in artifacts if item.get("type") == "adapter"]
            self.assertEqual([item["variant"] for item in adapters], ["final"])
            self.assertTrue(adapters[0]["selected"])


if __name__ == "__main__":
    unittest.main()
