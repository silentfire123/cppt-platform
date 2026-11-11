from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from backend.local_llamafactory_executor import LocalLlamaFactoryExecutor
from backend.llamafactory import platform_to_llamafactory_config


class TrainingOutputRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        train = self.root / "train.jsonl"
        train.write_text('{}\n', encoding="utf-8")
        self.task = {"id": "ft-regression", "model": "tiny", "datasetId": "tiny", "dataset": {"paths": {"trainPath": str(train)}}, "validationSplit": 0.1, "parameters": {"validationSteps": 25}}
        self.executor = LocalLlamaFactoryExecutor(work_root=self.root / "work")
        self.handle = await self.executor.prepare(self.task)
        self.output = Path(self.handle.workdir) / "output"
        self.output.mkdir(exist_ok=True)
        (self.output / "adapter_config.json").write_text("{}")
        (self.output / "adapter_model.safetensors").write_bytes(b"test-final")

    def write_state(self, **values):
        (self.output / "trainer_state.json").write_text(json.dumps(values), encoding="utf-8")

    async def test_final_evaluation_and_actual_step_survive_missing_eval_history(self):
        self.write_state(global_step=897, best_model_checkpoint=None, log_history=[{"step":895,"loss":0.8433},{"step":897,"train_loss":0.995}])
        (self.output / "eval_results.json").write_text('{"eval_loss":1.0750216245651245}')
        artifacts = await self.executor.collect_artifacts(self.task)
        final = next(a for a in artifacts if a.get("variant") == "final")
        self.assertEqual(final["checkpointStep"], 897)
        self.assertAlmostEqual(final["evalLoss"], 1.0750216245651245)
        self.assertFalse(any(a.get("variant") == "best" for a in artifacts))
        self.assertTrue(any(a["type"] == "metrics" and a["name"] == "eval_results.json" for a in artifacts))

    async def test_remote_best_path_is_mapped_to_local_mirror(self):
        checkpoint = self.output / "checkpoint-25"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}")
        self.handle.remote_workdir = "/root/cppt/ft-regression"
        self.write_state(global_step=50, best_model_checkpoint="/root/cppt/ft-regression/output/checkpoint-25", log_history=[{"step":25,"eval_loss":0.8},{"step":50,"eval_loss":1.2}])
        artifacts = await self.executor.collect_artifacts(self.task)
        best = next((a for a in artifacts if a.get("variant") == "best"), None)
        self.assertIsNotNone(best)
        self.assertEqual(best["path"], str(checkpoint))
        self.assertEqual(best["evalLoss"], 0.8)

    async def test_validation_interval_reaches_eval_and_checkpoint_schedule(self):
        canonical = self.executor._canonical_task(self.task, self.output)
        config = platform_to_llamafactory_config(canonical, output_dir=self.output)
        self.assertEqual(config.get("eval_strategy"), "steps")
        self.assertEqual(config.get("eval_steps"), 25)
        self.assertEqual(config.get("save_strategy"), "steps")
        self.assertEqual(config.get("save_steps"), 25)
        self.assertEqual(config.get("metric_for_best_model"), "eval_loss")
        self.assertIs(config.get("greater_is_better"), False)
        self.assertIs(config.get("load_best_model_at_end"), False)


    async def test_collector_can_reindex_after_backend_restart_without_reprepare(self):
        self.write_state(global_step=897, log_history=[{"step":895,"loss":0.8}])
        original_yaml = Path(self.handle.config_path).read_bytes()
        self.executor._handles.clear()
        artifacts = await self.executor.collect_artifacts(self.task)
        self.assertEqual(next(a for a in artifacts if a.get("variant") == "final")["checkpointStep"], 897)
        self.assertEqual(Path(self.handle.config_path).read_bytes(), original_yaml)

    async def test_final_evaluation_can_improve_on_a_real_historical_best(self):
        checkpoint = self.output / "checkpoint-25"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}")
        self.write_state(global_step=50, best_model_checkpoint=str(checkpoint), log_history=[{"step":25,"eval_loss":0.8}])
        (self.output / "eval_results.json").write_text('{"eval_loss":0.7}')
        artifacts = await self.executor.collect_artifacts(self.task)
        best = next(a for a in artifacts if a.get("variant") == "best")
        self.assertEqual(best["path"], str(self.output))
        self.assertEqual(best["checkpointStep"], 50)
        self.assertEqual(best["evalLoss"], 0.7)

    async def test_final_loss_does_not_borrow_an_earlier_checkpoint_eval(self):
        self.write_state(global_step=897, log_history=[{"step":875,"eval_loss":0.5},{"step":895,"loss":0.8}])
        artifacts = await self.executor.collect_artifacts(self.task)
        self.assertIsNone(next(a for a in artifacts if a.get("variant") == "final")["evalLoss"])

    async def test_missing_or_external_best_checkpoint_is_not_invented(self):
        checkpoint = self.output / "checkpoint-25"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}")
        for best in ["/other/task/output/checkpoint-25", "../checkpoint-25", "checkpoint-99"]:
            with self.subTest(best=best):
                self.write_state(global_step=50, best_model_checkpoint=best, log_history=[{"step":25,"eval_loss":0.8}])
                artifacts = await self.executor.collect_artifacts(self.task)
                self.assertFalse(any(a.get("variant") == "best" for a in artifacts))

    async def test_relative_best_checkpoint_is_supported(self):
        checkpoint = self.output / "checkpoint-25"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}")
        for best in ["checkpoint-25", "output/checkpoint-25"]:
            self.write_state(global_step=50, best_model_checkpoint=best, log_history=[{"step":25,"eval_loss":0.8}])
            artifacts = await self.executor.collect_artifacts(self.task)
            self.assertEqual(next(a for a in artifacts if a.get("variant") == "best")["checkpointStep"], 25)

    async def test_nan_final_eval_is_ignored_and_summary_fallback_is_used(self):
        self.write_state(global_step=897, log_history=[None, {"step":895,"loss":0.8}])
        (self.output / "eval_results.json").write_text('{"eval_loss":NaN}')
        (self.output / "all_results.json").write_text('{"eval_loss":1.075}')
        final = next(a for a in await self.executor.collect_artifacts(self.task) if a.get("variant") == "final")
        self.assertEqual(final["evalLoss"], 1.075)

    async def test_stopped_run_describes_saved_checkpoint_not_unsaved_step(self):
        (self.output / "adapter_config.json").unlink()
        checkpoint = self.output / "checkpoint-25"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "trainer_state.json").write_text('{"global_step":25,"log_history":[{"step":25,"eval_loss":0.8}]}')
        self.write_state(global_step=30, log_history=[{"step":30,"loss":0.6}])
        (self.output / "eval_results.json").write_text('{"eval_loss":0.1}')
        final = next(a for a in await self.executor.collect_artifacts(self.task) if a.get("variant") == "final")
        self.assertEqual(final["checkpointStep"], 25)
        self.assertEqual(final["evalLoss"], 0.8)

    async def test_no_validation_keeps_original_checkpoint_schedule(self):
        task = {**self.task, "validationSplit":0, "checkpointStrategy":"steps", "checkpointInterval":100}
        config = platform_to_llamafactory_config(self.executor._canonical_task(task, self.output), output_dir=self.output)
        self.assertIs(config["do_eval"], False)
        self.assertNotIn("eval_strategy", config)
        self.assertEqual(config["save_steps"], 100)

    async def test_validation_keeps_two_checkpoints_and_overrides_conflicting_flags(self):
        task = {**self.task, "checkpointLimit":1, "extraParameters":{"load_best_model_at_end":True,"metric_for_best_model":"accuracy"}}
        config = platform_to_llamafactory_config(self.executor._canonical_task(task, self.output), output_dir=self.output)
        self.assertEqual(config["save_total_limit"], 2)
        self.assertIs(config["load_best_model_at_end"], False)
        self.assertEqual(config["metric_for_best_model"], "eval_loss")

    async def test_explicit_validation_dataset_does_not_emit_conflicting_val_size(self):
        validation = self.root / "validation.jsonl"
        validation.write_text("{}")
        self.task["dataset"]["paths"]["validationPath"] = str(validation)
        config = platform_to_llamafactory_config(self.executor._canonical_task(self.task, self.output), output_dir=self.output)
        self.assertNotIn("val_size", config)
        self.assertEqual(config["eval_dataset"], "tiny_validation")

    async def test_manifest_detects_corrupt_or_missing_mirror(self):
        from backend.training_artifacts import verified_local_training_artifacts
        self.write_state(global_step=897, log_history=[])
        await self.executor.collect_artifacts(self.task)
        workdir = self.output.parent
        self.assertTrue(verified_local_training_artifacts(workdir))
        (self.output / "adapter_model.safetensors").write_bytes(b"wrong-data")
        self.assertFalse(verified_local_training_artifacts(workdir))

    async def test_mirror_from_previous_attempt_is_not_used_without_current_manifest(self):
        from backend.test_app_cli import app_module
        self.write_state(global_step=25, log_history=[])
        artifacts = await self.executor.collect_artifacts(self.task)
        task = {**self.task, "state":"completed", "provider":"youyun", "artifacts":[a for a in artifacts if a["type"] != "manifest"]}
        with patch.object(app_module,"youyun_mirror_root",self.executor.work_root):
            self.assertFalse(app_module._verified_task_mirror(self.task["id"], task))
            manifest = next(a for a in artifacts if a["type"] == "manifest")
            task["artifacts"].append({**manifest, "sha256":"0" * 64})
            self.assertFalse(app_module._verified_task_mirror(self.task["id"], task))
            task["artifacts"][-1] = manifest
            self.assertTrue(app_module._verified_task_mirror(self.task["id"], task))

    async def test_released_instance_can_reindex_verified_mirror_without_gpu(self):
        from backend.test_app_cli import app_module
        self.write_state(global_step=897, log_history=[{"step":895,"loss":0.8}])
        (self.output / "eval_results.json").write_text('{"eval_loss":1.075}')
        artifacts = await self.executor.collect_artifacts(self.task)
        artifacts[0].update(checkpointStep=895, syncState="failed", syncError="missing method", baseModelMetadata={"maxContextLength":32768})
        task = {**self.task, "state":"completed", "provider":"youyun", "artifacts":[*artifacts,{"type":"youyun-instance","state":"released","instanceId":"gone"}]}
        logs = []
        store = SimpleNamespace(get=lambda _:task, update=lambda _, **kw:task.update(kw), append_log=lambda _, msg:logs.append(msg))
        runner = SimpleNamespace(remote_root="/root/cppt")  # No SSH/sync method: local recovery must not call it.
        with patch.object(app_module,"store",store), patch.object(app_module,"youyun_mirror_root",self.executor.work_root), patch.object(app_module,"youyun_mirror_executor",self.executor), patch.object(app_module,"build_youyun_runner",return_value=runner), patch.object(app_module,"upload_completed_task_artifacts",new=AsyncMock(return_value=[])), patch.object(app_module,"sync_model_assets"):
            self.executor._handles.clear()
            await app_module.sync_remote_task_artifacts(self.task["id"])
            final = next(a for a in task["artifacts"] if a.get("variant") == "final")
            self.assertEqual(final["checkpointStep"], 897)
            self.assertEqual(final["evalLoss"], 1.075)
            self.assertEqual(final["syncState"], "available")
            self.assertEqual(final["baseModelMetadata"]["maxContextLength"], 32768)
            self.assertEqual(final["remotePath"], "/root/cppt/ft-regression/output")
            with patch.object(app_module,"schedule_remote_artifact_sync") as schedule:
                await app_module.retry_task_artifact_sync(self.task["id"])
                schedule.assert_called_once_with(self.task["id"])


class ArtifactPreviewRegressionTest(unittest.TestCase):
    def test_single_json_is_opened_as_content_not_parent_directory(self):
        from backend.test_app_cli import app_module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "trainer_state.json"
            target.write_text('{"global_step":897}')
            (root / "adapter_config.json").write_text("{}")
            with patch.object(app_module, "_artifact_preview_roots", return_value=(root,)):
                result = asyncio.run(app_module.preview_artifact(app_module.ArtifactPreviewRequest(path=str(target))))
            self.assertEqual(result.get("kind"), "text")
            self.assertEqual(json.loads(result["content"])["global_step"], 897)
            self.assertNotIn("files", result)


class TrainingSelectionMetadataRegressionTest(unittest.TestCase):
    def test_final_only_eval_is_not_reported_as_no_validation(self):
        from backend.store import TaskStore
        from backend.test_store import payload
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "task.db")
            task_id = store.create(payload())["id"]
            reason = "仅训练结束后验证，未生成历史最佳检查点，使用最终 Adapter"
            store.update(task_id, state="completed", artifacts=[{
                "type":"adapter", "name":"final", "variant":"final", "selected":True,
                "path":"/output", "checkpointStep":897, "evalLoss":1.075,
                "loss":1.075, "metric":"eval_loss", "selectionReason":reason,
            }])
            result = store.get(task_id)
            self.assertEqual(result["selectionReason"], reason)
            self.assertEqual(result["selectionMetric"], "eval_loss")
            self.assertEqual(result["finalLoss"], 1.075)
            self.assertEqual(result["finalStep"], 897)
            self.assertIsNone(result["bestAdapterPath"])
            self.assertIsNone(result["bestStep"])
