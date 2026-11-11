from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


TEST_RUNTIME = tempfile.TemporaryDirectory()
TEST_ROOT = Path(TEST_RUNTIME.name)
os.environ["XUNTUI_STORAGE_BACKEND"] = "sqlite"
os.environ["XUNTUI_DATABASE_PATH"] = str(TEST_ROOT / "training.db")
os.environ["XUNTUI_RUNTIME_ROOT"] = str(TEST_ROOT / "runtime")
os.environ["XUNTUI_DATASET_ROOT"] = str(TEST_ROOT / "datasets")
os.environ["XUNTUI_ARTIFACT_ROOT"] = str(TEST_ROOT / "artifacts")

from backend import app as app_module


class FakeStdin:
    def write(self, data: bytes) -> None:
        self.data = data

    async def drain(self) -> None:
        return None


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = FakeStdin()


def create_completed_task() -> dict:
    task = app_module.store.create(
        {
            "name": "chat adapter contract",
            "model": "Qwen/Qwen3-1.7B",
            "modelSource": "modelscope",
            "tuningMode": "qlora",
            "datasetId": "qwenruozhi_v1",
            "datasetName": "dataset",
            "parameters": {"epochs": 1},
        }
    )
    best_path = app_module.ARTIFACT_ROOT / task["id"] / "adapter"
    final_path = app_module.ARTIFACT_ROOT / task["id"] / "final_adapter"
    best_path.mkdir(parents=True)
    final_path.mkdir(parents=True)
    app_module.store.update(
        task["id"],
        state="completed",
        artifacts=[
            {
                "type": "adapter",
                "name": "best",
                "path": str(best_path),
                "variant": "best",
                "selected": True,
                "checkpointStep": 20,
            },
            {
                "type": "adapter",
                "name": "final",
                "path": str(final_path),
                "variant": "final",
                "selected": False,
                "checkpointStep": 30,
            },
        ],
    )
    return app_module.store.get(task["id"])


class ChatApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_selected_adapter_loads_best_artifact_and_reports_identity(self) -> None:
        task = create_completed_task()
        process = FakeProcess()
        ensure = AsyncMock(return_value=process)
        response_event = AsyncMock(
            return_value={
                "text": "answer",
                "elapsedSeconds": 0.5,
                "outputTokens": 3,
            }
        )
        request = app_module.ChatRequest(
            mode="adapter",
            adapterVariant="selected",
            taskId=task["id"],
            model=task["model"],
            messages=[{"role": "user", "content": "hello"}],
        )

        with patch.object(app_module, "ensure_chat_process", ensure), patch.object(
            app_module, "read_chat_event", response_event
        ):
            response = await app_module.chat(request)

        call = ensure.await_args.kwargs
        self.assertEqual(call["adapter_path"], task["bestAdapterPath"])
        self.assertEqual(response["adapterVariant"], "best")
        self.assertEqual(response["adapterStep"], 20)
        self.assertEqual(response["adapterPath"], task["bestAdapterPath"])

    async def test_adapter_under_task_training_workdir_is_allowed(self) -> None:
        task = create_completed_task()
        adapter_path = app_module.local_executor.work_root / task["id"] / "output"
        adapter_path.mkdir(parents=True)
        app_module.store.update(
            task["id"],
            artifacts=[
                {
                    "type": "adapter",
                    "name": "final",
                    "path": str(adapter_path),
                    "variant": "final",
                    "selected": True,
                    "checkpointStep": 1,
                }
            ],
        )
        process = FakeProcess()
        ensure = AsyncMock(return_value=process)
        response_event = AsyncMock(
            return_value={
                "text": "answer",
                "elapsedSeconds": 0.5,
                "outputTokens": 3,
            }
        )
        request = app_module.ChatRequest(
            mode="adapter",
            adapterVariant="final",
            taskId=task["id"],
            model=task["model"],
            messages=[{"role": "user", "content": "hello"}],
        )

        with patch.object(app_module, "ensure_chat_process", ensure), patch.object(
            app_module, "read_chat_event", response_event
        ):
            response = await app_module.chat(request)

        self.assertEqual(ensure.await_args.kwargs["adapter_path"], str(adapter_path.resolve()))
        self.assertEqual(response["adapterVariant"], "final")
        self.assertEqual(response["adapterStep"], 1)

    async def test_adapter_under_task_remote_mirror_workdir_is_allowed(self) -> None:
        task = create_completed_task()
        adapter_path = app_module.youyun_mirror_executor.work_root / task["id"] / "output"
        adapter_path.mkdir(parents=True)
        app_module.store.update(
            task["id"],
            artifacts=[
                {
                    "type": "adapter",
                    "name": "final",
                    "path": str(adapter_path),
                    "variant": "final",
                    "selected": True,
                    "checkpointStep": 1,
                }
            ],
        )
        process = FakeProcess()
        ensure = AsyncMock(return_value=process)
        response_event = AsyncMock(
            return_value={"text": "answer", "elapsedSeconds": 0.5, "outputTokens": 3}
        )
        request = app_module.ChatRequest(
            mode="adapter",
            adapterVariant="final",
            taskId=task["id"],
            model=task["model"],
            messages=[{"role": "user", "content": "hello"}],
        )

        with patch.object(app_module, "ensure_chat_process", ensure), patch.object(
            app_module, "read_chat_event", response_event
        ):
            response = await app_module.chat(request)

        self.assertEqual(ensure.await_args.kwargs["adapter_path"], str(adapter_path.resolve()))
        self.assertEqual(response["adapterVariant"], "final")
        self.assertEqual(response["adapterStep"], 1)

    async def test_adapter_from_another_task_workdir_is_rejected(self) -> None:
        task = create_completed_task()
        adapter_path = app_module.local_executor.work_root / "another-task" / "output"
        adapter_path.mkdir(parents=True)
        app_module.store.update(
            task["id"],
            artifacts=[
                {
                    "type": "adapter",
                    "name": "final",
                    "path": str(adapter_path),
                    "variant": "final",
                    "selected": True,
                    "checkpointStep": 1,
                }
            ],
        )
        request = app_module.ChatRequest(
            mode="adapter",
            adapterVariant="final",
            taskId=task["id"],
            model=task["model"],
            messages=[{"role": "user", "content": "hello"}],
        )

        with self.assertRaises(app_module.HTTPException) as raised:
            await app_module.chat(request)

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, "LoRA Adapter 路径无效。")

    async def test_base_comparison_reuses_adapter_process_but_disables_adapter(self) -> None:
        task = create_completed_task()
        process = FakeProcess()
        ensure = AsyncMock(return_value=process)
        response_event = AsyncMock(
            return_value={
                "text": "base answer",
                "elapsedSeconds": 0.5,
                "outputTokens": 3,
            }
        )
        request = app_module.ChatRequest(
            mode="base",
            taskId=task["id"],
            model=task["model"],
            messages=[{"role": "user", "content": "hello"}],
        )

        with patch.object(app_module, "ensure_chat_process", ensure), patch.object(
            app_module, "read_chat_event", response_event
        ):
            response = await app_module.chat(request)

        self.assertEqual(ensure.await_args.kwargs["adapter_path"], task["bestAdapterPath"])
        written_request = app_module.json.loads(process.stdin.data.decode())
        self.assertFalse(written_request["adapterEnabled"])
        self.assertNotIn("adapterVariant", response)


if __name__ == "__main__":
    unittest.main()
