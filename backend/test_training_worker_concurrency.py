from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import app as app_module


class _QueueStore:
    def __init__(self, *, local: list[dict], remote: list[dict]) -> None:
        self.local = list(local)
        self.remote = list(remote)

    def claim_next(self, provider: str | None = None) -> dict | None:
        queue = self.remote if provider == "youyun" else self.local
        return queue.pop(0) if queue else None


class _FlakyQueueStore(_QueueStore):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failures = 1

    def claim_next(self, provider: str | None = None) -> dict | None:
        if provider == "local" and self.failures:
            self.failures -= 1
            raise RuntimeError("temporary database disconnect")
        return super().claim_next(provider)


class TrainingWorkerConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        app_module.remote_training_tasks.clear()

    async def asyncTearDown(self) -> None:
        for task in app_module.remote_training_tasks.values():
            task.cancel()
        if app_module.remote_training_tasks:
            await asyncio.gather(
                *app_module.remote_training_tasks.values(),
                return_exceptions=True,
            )
        app_module.remote_training_tasks.clear()

    async def test_remote_tasks_dispatch_concurrently_while_local_inference_blocks_local_queue(self) -> None:
        store = _QueueStore(
            local=[{"id": "local-1", "provider": "local"}],
            remote=[
                {"id": "remote-1", "provider": "youyun", "executionMode": "remote"},
                {"id": "remote-2", "provider": "youyun", "executionMode": "remote"},
            ],
        )
        started: list[str] = []
        releases = {
            "remote-1": asyncio.Event(),
            "remote-2": asyncio.Event(),
        }

        async def execute(task: dict) -> None:
            started.append(task["id"])
            await releases[task["id"]].wait()

        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "inference_runtime",
            SimpleNamespace(local_service_active=True),
        ), patch.object(app_module, "execute_claimed_training_task", execute):
            worker = asyncio.create_task(app_module.worker_loop())
            try:
                async with asyncio.timeout(1):
                    while len(started) < 2:
                        await asyncio.sleep(0.01)
                self.assertCountEqual(started, ["remote-1", "remote-2"])
                self.assertEqual(store.local[0]["id"], "local-1")
                self.assertEqual(set(app_module.remote_training_tasks), {"remote-1", "remote-2"})

                releases["remote-1"].set()
                async with asyncio.timeout(1):
                    while "remote-1" in app_module.remote_training_tasks:
                        await asyncio.sleep(0.01)
                self.assertIn("remote-2", app_module.remote_training_tasks)
                self.assertFalse(app_module.remote_training_tasks["remote-2"].done())
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def test_local_training_keeps_single_execution_slot(self) -> None:
        store = _QueueStore(
            local=[
                {"id": "local-1", "provider": "local"},
                {"id": "local-2", "provider": "local"},
            ],
            remote=[],
        )
        started: list[str] = []
        release = asyncio.Event()

        async def execute(task: dict) -> None:
            started.append(task["id"])
            await release.wait()

        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "inference_runtime",
            SimpleNamespace(local_service_active=False),
        ), patch.object(app_module, "execute_claimed_training_task", execute):
            worker = asyncio.create_task(app_module.worker_loop())
            try:
                async with asyncio.timeout(1):
                    while not started:
                        await asyncio.sleep(0.01)
                await asyncio.sleep(0.2)
                self.assertEqual(started, ["local-1"])
                self.assertEqual(store.local[0]["id"], "local-2")
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def test_worker_survives_transient_queue_store_failure(self) -> None:
        store = _FlakyQueueStore(
            local=[{"id": "local-1", "provider": "local"}],
            remote=[],
        )
        started: list[str] = []
        release = asyncio.Event()

        async def execute(task: dict) -> None:
            started.append(task["id"])
            await release.wait()

        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "inference_runtime",
            SimpleNamespace(local_service_active=False),
        ), patch.object(app_module, "execute_claimed_training_task", execute):
            worker = asyncio.create_task(app_module.worker_loop())
            try:
                async with asyncio.timeout(2):
                    while not started:
                        await asyncio.sleep(0.01)
                self.assertEqual(started, ["local-1"])
            finally:
                release.set()
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
