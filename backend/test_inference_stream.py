from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from starlette.websockets import WebSocketDisconnect

from backend import app as app_module
from backend.inference_service import InferenceCreate, InferenceRuntime, InferenceStore, create_inference_router


class _FakeWebSocket:
    def __init__(self, *, after_id: int = 0) -> None:
        self.query_params = {"afterId": str(after_id)}
        self.messages: list[dict] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        json.dumps(payload)
        self.messages.append(payload)

    async def close(self, code: int | None = None) -> None:
        self.closed = True


class InferenceStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_stream_sends_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "inference.db")
            service = store.create(
                service_id="svc-heartbeat",
                payload=InferenceCreate(name="heartbeat", taskId="task-1"),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter="/tmp/adapter",
                endpoint="/v1/chat/completions",
            )
            store.update(service["id"], state="running")
            runtime = InferenceRuntime(
                store,
                object(),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            async def stable_metrics(service_id: str, *, run_id: str | None = None):
                return {"source": "test", "serviceId": service_id}

            runtime.metrics = stable_metrics  # type: ignore[method-assign]
            websocket = _FakeWebSocket()
            router = create_inference_router(runtime, websocket_auth=lambda _: True)
            endpoint = next(route.endpoint for route in router.routes if route.path.endswith("/stream"))
            clock = iter([0.0, 16.0, 16.0])
            sleeps = 0

            async def fake_sleep(_: float) -> None:
                nonlocal sleeps
                sleeps += 1
                if sleeps >= 2:
                    raise WebSocketDisconnect()

            with patch("backend.inference_service.monotonic", side_effect=lambda: next(clock, 16.0)), patch(
                "backend.inference_service.asyncio.sleep", side_effect=fake_sleep
            ):
                await endpoint(websocket, service["id"])

            self.assertTrue(any(message.get("type") == "heartbeat" for message in websocket.messages))

    async def test_metrics_failure_does_not_end_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "inference.db")
            service = store.create(
                service_id="svc-metrics-error",
                payload=InferenceCreate(name="metrics", taskId="task-1"),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter="/tmp/adapter",
                endpoint="/v1/chat/completions",
            )
            store.update(service["id"], state="running")
            runtime = InferenceRuntime(
                store,
                object(),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            async def failing_then_stable(service_id: str, *, run_id: str | None = None):
                if not getattr(failing_then_stable, "failed", False):
                    failing_then_stable.failed = True
                    raise RuntimeError("metrics unavailable")
                return {"source": "test", "serviceId": service_id}

            runtime.metrics = failing_then_stable  # type: ignore[method-assign]
            websocket = _FakeWebSocket()
            router = create_inference_router(runtime, websocket_auth=lambda _: True)
            endpoint = next(route.endpoint for route in router.routes if route.path.endswith("/stream"))
            sleeps = 0

            async def fake_sleep(_: float) -> None:
                nonlocal sleeps
                sleeps += 1
                if sleeps >= 2:
                    raise WebSocketDisconnect()

            with patch("backend.inference_service.asyncio.sleep", side_effect=fake_sleep):
                await endpoint(websocket, service["id"])

            self.assertGreaterEqual(sleeps, 2)
            self.assertTrue(any(message.get("type") == "state" for message in websocket.messages))

    async def test_startup_progress_change_triggers_state_push(self) -> None:
        """启动期间 state 恒为 starting,但 config.startup 进度持续变化:
        进度推送必须跟上,否则前端进度条冻结在首帧。"""

        class Store:
            def __init__(self) -> None:
                self.startup: dict = {
                    "stage": "creating_instance",
                    "progress": 15,
                    "message": "优云实例创建中：已等待 0 秒。",
                    "elapsedSeconds": 0,
                }

            def get(self, service_id: str):
                return {
                    "id": service_id,
                    "state": "starting",
                    "config": {"startup": dict(self.startup)},
                }

            def log_items(self, service_id: str, *, after_id: int = 0, limit: int = 500):
                return {"items": [], "hasMore": False}

            def append_log(self, service_id: str, message: str):
                return None

        class Runtime:
            def __init__(self, store: Store) -> None:
                self.store = store

            async def metrics(self, service_id: str):
                return {"source": "test"}

            async def refresh_remote_logs(self, service_id: str):
                # 模拟远端启动推进:首轮之后进度从创建实例切到 Adapter 上传。
                if self.store.startup.get("stage") == "creating_instance":
                    self.store.startup = {
                        "stage": "uploading_adapter",
                        "progress": 68,
                        "message": "Adapter 上传中：58%。",
                        "elapsedSeconds": 120,
                    }

        store = Store()
        runtime = Runtime(store)
        websocket = _FakeWebSocket()
        router = create_inference_router(runtime, websocket_auth=lambda _: True)  # type: ignore[arg-type]
        endpoint = next(route.endpoint for route in router.routes if route.path.endswith("/stream"))
        sleeps = 0

        async def fake_sleep(_: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps >= 4:
                raise WebSocketDisconnect()

        with patch("backend.inference_service.asyncio.sleep", side_effect=fake_sleep):
            await endpoint(websocket, "svc-progress")

        state_messages = [message for message in websocket.messages if message.get("type") == "state"]
        self.assertGreaterEqual(len(state_messages), 2)
        progresses = [message["service"]["config"]["startup"]["progress"] for message in state_messages]
        self.assertEqual(progresses[0], 15)
        self.assertIn(68, progresses)

    async def test_postgres_datetime_log_is_serialized_before_send(self) -> None:
        class Store:
            def get(self, service_id: str):
                return {"id": service_id, "state": "stopped", "config": {}}

            def log_items(self, service_id: str, *, after_id: int = 0, limit: int = 500):
                return {
                    "items": [{
                        "id": 1,
                        "created_at": datetime.now(timezone.utc),
                        "message": "服务已停止",
                    }],
                    "hasMore": False,
                    "runId": None,
                }

            def metrics(self, service_id: str):
                return {"source": "test"}

        class Runtime:
            store = Store()

            async def metrics(self, service_id: str):
                return {"source": "test"}

            async def refresh_remote_logs(self, service_id: str):
                return None

        runtime = Runtime()
        websocket = _FakeWebSocket()
        router = create_inference_router(runtime, websocket_auth=lambda _: True)
        endpoint = next(route.endpoint for route in router.routes if route.path.endswith("/stream"))

        await endpoint(websocket, "svc-datetime")

        self.assertEqual(websocket.messages[0]["createdAt"].__class__, str)

    async def test_training_idle_stream_sends_heartbeat(self) -> None:
        class TaskStore:
            def get(self, task_id: str):
                return {"id": task_id, "state": "running", "progress": 0, "currentStep": 0}

            def log_items(self, task_id: str, *, after_id: int = 0, limit: int = 500):
                return {"items": [], "hasMore": False}

            def metrics(self, task_id: str):
                return []

        websocket = _FakeWebSocket()
        endpoint = app_module.task_stream
        original_store = app_module.store
        original_auth = app_module.websocket_authenticated
        app_module.store = TaskStore()  # type: ignore[assignment]
        app_module.websocket_authenticated = lambda _: True
        sleeps = 0

        async def fake_sleep(_: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps >= 2:
                raise WebSocketDisconnect()

        try:
            with patch("backend.app.WEBSOCKET_HEARTBEAT_INTERVAL_SECONDS", 0.0), patch(
                "backend.app.asyncio.sleep", side_effect=fake_sleep
            ):
                await endpoint(websocket, "task-heartbeat")
        finally:
            app_module.store = original_store
            app_module.websocket_authenticated = original_auth

        self.assertTrue(any(message.get("type") == "heartbeat" for message in websocket.messages))

    async def test_training_metrics_read_failure_does_not_end_stream(self) -> None:
        class TaskStore:
            def get(self, task_id: str):
                return {"id": task_id, "state": "running", "progress": 40, "currentStep": "训练中"}

            def log_items(self, task_id: str, *, after_id: int = 0, limit: int = 500):
                return {"items": [], "hasMore": False}

            def metrics(self, task_id: str):
                if not getattr(self, "failed", False):
                    self.failed = True
                    raise RuntimeError("temporary metrics read failure")
                return [{"step": 2, "trainLoss": 1.2}]

        websocket = _FakeWebSocket()
        endpoint = app_module.task_stream
        original_store = app_module.store
        original_auth = app_module.websocket_authenticated
        app_module.store = TaskStore()  # type: ignore[assignment]
        app_module.websocket_authenticated = lambda _: True
        sleeps = 0

        async def fake_sleep(_: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps >= 2:
                raise WebSocketDisconnect()

        try:
            with patch("backend.app.asyncio.sleep", side_effect=fake_sleep):
                await endpoint(websocket, "task-metrics-error")
        finally:
            app_module.store = original_store
            app_module.websocket_authenticated = original_auth

        self.assertTrue(any(message.get("type") == "metrics" for message in websocket.messages))
