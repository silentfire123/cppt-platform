from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
TEST_RUNTIME = tempfile.TemporaryDirectory()
for key, value in {"XUNTUI_STORAGE_BACKEND": "sqlite", "XUNTUI_DATABASE_PATH": str(Path(TEST_RUNTIME.name) / "test.db"), "XUNTUI_RUNTIME_ROOT": str(Path(TEST_RUNTIME.name) / "runtime"), "XUNTUI_DATASET_ROOT": str(Path(TEST_RUNTIME.name) / "datasets"), "XUNTUI_ARTIFACT_ROOT": str(Path(TEST_RUNTIME.name) / "artifacts")}.items():
    os.environ[key] = value

from backend import app as api
from backend.capabilities import runtime_capabilities
from backend.inference_service import InferenceCreate, InferenceRuntime


class RuntimePolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.policy = patch.dict(os.environ, {"XUNTUI_LOCAL_RUNTIME_ENABLED": "false"})
        self.policy.start()
        self.addCleanup(self.policy.stop)

    async def assert_disabled(self, operation):
        with self.assertRaises(HTTPException) as caught:
            await operation
        self.assertEqual(caught.exception.status_code, 409)

    async def test_capabilities_and_development_default(self):
        result = await api.capabilities()
        self.assertFalse(result["localRuntimeEnabled"])
        self.assertFalse(result["evaluationEnabled"])
        self.assertEqual(result["trainingProviders"], ["youyun"])
        with patch.dict(os.environ, {"XUNTUI_LOCAL_RUNTIME_ENABLED": "true"}):
            self.assertTrue(runtime_capabilities()["cliTrainingEnabled"])
            self.assertTrue(runtime_capabilities()["evaluationEnabled"])

    async def test_production_blocks_evaluation_mutations_and_dataset_creation(self):
        payload = SimpleNamespace(datasetId="dataset", targetTaskId="task", targetVariant="selected")
        review = SimpleNamespace(score=5, dimensions={}, attribution="other", note="")
        with patch.object(api, "store") as store, patch.object(api, "evaluation_store") as evaluations:
            await self.assert_disabled(api.create_evaluation_task(payload))
            await self.assert_disabled(api.start_evaluation_task("evaluation"))
            await self.assert_disabled(api.retry_evaluation_task("evaluation"))
            await self.assert_disabled(api.review_evaluation_item("evaluation", "item", review))
            await self.assert_disabled(api.upload_dataset(SimpleNamespace(), "eval", "SFT", "eval.jsonl", "evaluation"))
            store.get_dataset.assert_not_called()
            evaluations.get_task.assert_not_called()

    async def test_local_create_cli_and_search_never_reach_runtime(self):
        with patch.object(api, "store") as store:
            await self.assert_disabled(api.create_task(SimpleNamespace(provider="local")))
            await self.assert_disabled(api.create_cli_task(SimpleNamespace(command="anything")))
            await self.assert_disabled(api.search_models("Qwen", source="modelscope"))
            store.create.assert_not_called()

    async def test_local_chat_does_not_launch_process(self):
        with patch.object(api, "store") as store, patch.object(api, "ensure_chat_process", new_callable=AsyncMock) as launch:
            store.get.return_value = {"id": "local", "provider": "local"}
            payload = api.ChatRequest.model_validate({"model": "Qwen/test", "taskId": "local", "mode": "base", "messages": [{"role": "user", "content": "hello"}]})
            await self.assert_disabled(api.chat(payload))
            launch.assert_not_called()

    async def test_local_history_is_readable_but_cannot_retry_or_clone(self):
        with patch.object(api, "store") as store:
            store.get.return_value = {"id": "old", "provider": "local", "state": "completed"}
            self.assertEqual((await api.get_task("old"))["id"], "old")
            await self.assert_disabled(api.retry_task("old"))
            await self.assert_disabled(api.clone_task("old"))
            store.retry.assert_not_called()
            store.clone.assert_not_called()

    async def test_inference_local_create_and_lifecycle_are_blocked_before_mutation(self):
        runtime = object.__new__(InferenceRuntime)
        runtime.store = MagicMock()
        runtime.store.get.return_value = {"id": "old", "state": "stopped", "config": {"provider": "local"}}
        runtime.cancel_events = {}
        await self.assert_disabled(runtime.create(SimpleNamespace(provider="local")))
        for name in ("start", "start_background", "restart", "recover", "restore"):
            await self.assert_disabled(getattr(runtime, name)("old"))
        runtime.store.update.assert_not_called()

    async def test_worker_only_claims_youyun_queue(self):
        with patch.object(api, "store") as store, patch.object(api.asyncio, "sleep", side_effect=asyncio.CancelledError):
            store.claim_next.return_value = None
            with self.assertRaises(asyncio.CancelledError):
                await api.worker_loop()
            self.assertEqual([call.kwargs for call in store.claim_next.call_args_list], [{"provider": "youyun"}])

    async def test_evaluation_rechecks_target_before_start_and_retry(self):
        with patch.object(api, "store") as store, patch.object(api, "evaluation_store") as evaluations:
            evaluations.get_task.return_value = {"targetTaskId": "old", "state": "draft"}
            store.get.return_value = {"id": "old", "provider": "local", "state": "completed"}
            await self.assert_disabled(api.start_evaluation_task("evaluation"))
            await self.assert_disabled(api.retry_evaluation_task("evaluation"))
            evaluations.update_task.assert_not_called()

    async def test_evaluation_accepts_retained_adapter_and_rejects_released_instance(self):
        target = {"id": "remote", "state": "completed", "provider": "youyun", "artifacts": [{"type": "adapter", "path": "/mirror/adapter", "remotePath": "/workspace/adapter", "syncState": "remote_ready"}, {"type": "youyun-instance", "instanceId": "instance", "state": "awaiting_comparison"}]}
        api.require_evaluation_target(target)
        target["artifacts"][1]["state"] = "released"
        with self.assertRaises(HTTPException):
            api.require_evaluation_target(target)

    async def test_evaluation_targets_only_returns_reusable_youyun_tasks(self):
        reusable = {"id": "remote", "state": "completed", "provider": "youyun", "artifacts": [{"type": "adapter", "path": "/mirror/adapter", "remotePath": "/workspace/adapter", "syncState": "remote_ready"}, {"type": "youyun-instance", "instanceId": "instance", "state": "awaiting_comparison"}]}
        released = {**reusable, "id": "released", "artifacts": [*reusable["artifacts"][:-1], {"type": "youyun-instance", "instanceId": "old", "state": "released"}]}
        local = {**reusable, "id": "local", "provider": "local"}
        with patch.object(api, "store") as store:
            store.list.return_value = [reusable, released, local]
            self.assertEqual(await api.evaluation_targets(), [reusable])

    async def test_dataset_preview_reads_jsonl_without_datasets_package(self):
        path = Path(TEST_RUNTIME.name) / "preview.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": f"q{index}"}, {"role": "assistant", "content": f"a{index}"}]}
            for index in range(10)
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        dataset = {"id": "dataset", "paths": {"trainPath": str(path), "validationPath": None}}
        with patch.object(api, "store") as store:
            store.get_dataset.return_value = dataset
            training = await api.dataset_prompts("dataset", split="train", limit=100, validation_split=20, seed=42)
            validation = await api.dataset_prompts("dataset", split="validation", limit=100, validation_split=20, seed=42)
        self.assertEqual(len(training), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse({row["prompt"] for row in training} & {row["prompt"] for row in validation})

    async def test_evaluation_generation_passes_original_model_to_remote_chat(self):
        target = {"id": "remote", "state": "completed", "provider": "youyun", "model": "/models/original", "modelSource": "public", "artifacts": [{"type": "adapter", "path": "/mirror/adapter", "remotePath": "/workspace/adapter", "syncState": "remote_ready"}, {"type": "youyun-instance", "instanceId": "instance", "state": "awaiting_comparison"}]}
        evaluation = {"targetTaskId": "remote", "state": "queued", "judgeConfig": {"useTeacher": False, "useSimilarity": False}}
        with patch.object(api, "store") as store, patch.object(api, "evaluation_store") as evaluations, patch.object(api, "run_youyun_chat", new_callable=AsyncMock) as remote:
            store.get.return_value = target
            evaluations.get_task.return_value = evaluation
            evaluations.list_items.return_value = [{"id": "item", "prompt": "hello"}]
            remote.return_value = {"text": "answer"}
            await api._run_evaluation_task("evaluation")
            self.assertEqual(remote.await_args.args[0], target)
            self.assertEqual(remote.await_args.args[1].model, "/models/original")
            self.assertEqual(remote.await_count, 1)


if __name__ == "__main__":
    unittest.main()
