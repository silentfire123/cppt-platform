from __future__ import annotations

import os
import tempfile
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
from typing import Any

from backend.inference_service import (
    InferenceChatRequest,
    InferenceCreate,
    InferenceRuntime,
    InferenceStore,
)


def complete_adapter(path: Path) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


class FakeTaskStore:
    def __init__(self, task: dict[str, Any] | None) -> None:
        self.task = task

    def get(self, task_id: str) -> dict[str, Any] | None:
        if self.task and self.task.get("id") == task_id:
            return self.task
        return None


class InferenceStoreTests(unittest.TestCase):
    def test_list_page_supports_filters_and_empty_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "services.db")
            for service_id, name in (("svc-a", "alpha"), ("svc-b", "beta")):
                store.create(
                    service_id=service_id,
                    payload=InferenceCreate(name=name, taskId="ft-1"),
                    model="Qwen/Qwen2.5-3B-Instruct",
                    adapter="/tmp/best-adapter",
                    endpoint="/v1/chat/completions",
                )
            store.update("svc-b", state="failed")
            result = store.list_page(page=1, page_size=1, state="failed", query="beta")
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["items"][0]["id"], "svc-b")
            self.assertEqual(store.list_page(page=2, page_size=1, query="missing")["items"], [])

    def test_service_name_is_required_and_trimmed(self) -> None:
        with self.assertRaises(ValueError):
            InferenceCreate(name="   ", taskId="ft-1")
        self.assertEqual(InferenceCreate(name="  客服服务  ", taskId="ft-1").name, "客服服务")

    def test_create_update_logs_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "test.db")
            payload = InferenceCreate(name="客服服务", taskId="ft-1")
            service = store.create(
                service_id="svc-1",
                payload=payload,
                model="Qwen/Qwen2.5-3B-Instruct",
                adapter="/tmp/best-adapter",
                endpoint="/v1/chat/completions",
            )
            self.assertEqual(service["state"], "created")
            self.assertEqual(service["model"], "Qwen/Qwen2.5-3B-Instruct")
            self.assertEqual(service["adapter"], "/tmp/best-adapter")
            self.assertEqual(len(store.list()), 1)

            updated = store.update(
                "svc-1",
                state="running",
                exposure="online",
                recovery_attempts=1,
            )
            self.assertIsNotNone(updated)
            self.assertEqual(updated["state"], "running")
            self.assertEqual(updated["exposure"], "online")
            self.assertEqual(updated["recoveryAttempts"], 1)

            store.append_log("svc-1", "vLLM ready")
            self.assertTrue(any("vLLM ready" in line for line in store.logs("svc-1")))

            store.set_metrics("svc-1", {"kvCacheUsagePercent": 0.4})
            self.assertEqual(store.metrics("svc-1")["kvCacheUsagePercent"], 0.4)

    def test_log_items_starts_with_latest_window_for_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "logs.db")
            store.create(
                service_id="svc-logs",
                payload=InferenceCreate(name="日志服务", taskId="ft-1"),
                model="Qwen/Qwen2.5-3B-Instruct",
                adapter="/tmp/best-adapter",
                endpoint="/v1/chat/completions",
            )
            store.create_run("svc-logs")
            for index in range(205):
                store.append_log("svc-logs", f"line-{index}")

            result = store.log_items("svc-logs", limit=200)

            self.assertEqual(len(result["items"]), 200)
            self.assertEqual(result["items"][0]["message"], "line-5")
            self.assertEqual(result["items"][-1]["message"], "line-204")
            self.assertTrue(result["hasOlder"])

            older = store.log_items(
                "svc-logs",
                before_id=result["firstItemId"],
                limit=200,
            )
            self.assertEqual(
                [item["message"] for item in older["items"]],
                [f"line-{index}" for index in range(5)],
            )
            self.assertFalse(older["hasOlder"])

            store.append_log("svc-logs", "line-205")
            delta = store.log_items("svc-logs", after_id=result["nextAfterId"], limit=200)
            self.assertEqual([item["message"] for item in delta["items"]], ["line-205"])

    def test_export_logs_returns_complete_history_beyond_runtime_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "export.db")
            store.create(
                service_id="svc-export",
                payload=InferenceCreate(name="导出服务", taskId="ft-1"),
                model="Qwen/Qwen2.5-3B-Instruct",
                adapter="/tmp/best-adapter",
                endpoint="/v1/chat/completions",
            )
            store.create_run("svc-export")
            for index in range(205):
                store.append_log("svc-export", f"line-{index}")

            exported = store.export_logs("svc-export")

            self.assertEqual(len(exported), 206)  # includes the creation log
            self.assertIn("line-0", exported[1])
            self.assertIn("line-204", exported[-1])

    def test_archive_restore_and_delete_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "test.db")
            payload = InferenceCreate(name="归档服务", taskId="ft-1")
            store.create(
                service_id="svc-archive",
                payload=payload,
                model="Qwen/Qwen2.5-3B-Instruct",
                adapter="/tmp/best-adapter",
                endpoint="/v1/chat/completions",
            )
            store.append_log("svc-archive", "before archive")
            store.set_metrics("svc-archive", {"kvCacheUsagePercent": 0.4})

            store.update("svc-archive", state="running")
            with self.assertRaisesRegex(RuntimeError, "不能归档"):
                store.archive("svc-archive")

            store.update("svc-archive", state="stopped")
            archived = store.archive("svc-archive")
            self.assertEqual(archived["state"], "archived")
            self.assertEqual(archived["exposure"], "offline")
            self.assertEqual(store.list(), [])
            self.assertEqual(len(store.list(include_archived=True)), 1)
            self.assertTrue(any("before archive" in line for line in store.logs("svc-archive")))
            self.assertEqual(store.metrics("svc-archive")["kvCacheUsagePercent"], 0.4)

            restored = store.restore("svc-archive")
            self.assertEqual(restored["state"], "stopped")
            with self.assertRaisesRegex(RuntimeError, "只有已归档"):
                store.delete("svc-archive")

            store.archive("svc-archive")
            store.delete("svc-archive")
            self.assertIsNone(store.get("svc-archive"))
            self.assertEqual(store.logs("svc-archive"), [])
            self.assertEqual(store.metrics("svc-archive"), {"source": "not_started", "updatedAt": None})


class InferenceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_restores_missing_or_incomplete_adapter_and_prefers_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restored = root / "restored"
            complete_adapter(restored)
            local = root / "local"
            task = {"id": "ft-restore", "state": "completed", "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "artifacts": [{"type": "adapter", "variant": "final", "path": str(local)}]}
            restorer = Mock(return_value=str(restored))
            runtime = InferenceRuntime(InferenceStore(root / "services.db"), FakeTaskStore(task), port=8011,
                                       training_busy=lambda: False, artifact_restorer=restorer)
            for incomplete in (False, True):
                if incomplete:
                    local.mkdir()
                    (local / "adapter_config.json").write_text("{}", encoding="utf-8")
                service = await runtime.create(InferenceCreate(name="恢复权重", taskId="ft-restore", adapterVariant="final"))
                self.assertEqual(service["adapter"], str(restored))
                restorer.assert_called_with("ft-restore", "final")
            (local / "adapter_model.safetensors").write_bytes(b"local")
            restorer.reset_mock()
            service = await runtime.create(InferenceCreate(name="本地优先", taskId="ft-restore", adapterVariant="final"))
            self.assertEqual(service["adapter"], str(local))
            restorer.assert_not_called()

    async def test_start_existing_service_recovers_adapter_before_runtime_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restored = root / "restored"
            complete_adapter(restored)
            restorer = Mock(return_value=str(restored))
            runtime = InferenceRuntime(InferenceStore(root / "services.db"), FakeTaskStore(None), port=8011,
                                       training_busy=lambda: False, artifact_restorer=restorer)
            service = runtime.store.create(service_id="svc-restored", payload=InferenceCreate(name="恢复已有服务", taskId="ft-old", adapterVariant="final"),
                                           model="Qwen/Qwen2.5-0.5B-Instruct", adapter=str(root / "missing"), endpoint="/v1/chat/completions")
            with patch.object(runtime, "_validate_vllm_runtime", side_effect=RuntimeError("runtime boundary")):
                with self.assertRaisesRegex(RuntimeError, "runtime boundary"):
                    await runtime.start(service["id"])
            restorer.assert_called_once_with("ft-old", "final")
            saved = runtime.store.get(service["id"])
            self.assertEqual(saved["adapter"], str(restored))
            self.assertEqual(saved["config"]["adapter"], str(restored))

    async def test_restore_failure_or_nonlocal_result_blocks_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = {"id": "ft-missing", "state": "completed", "model": "Qwen/Qwen2.5-0.5B-Instruct", "artifacts": []}
            restorer = Mock(return_value="s3://rag-assets/model/missing")
            runtime = InferenceRuntime(InferenceStore(Path(directory) / "services.db"), FakeTaskStore(task), port=8011,
                                       training_busy=lambda: False, artifact_restorer=restorer)
            with self.assertRaisesRegex(ValueError, "完整的本地"):
                await runtime.create(InferenceCreate(name="失败恢复", taskId="ft-missing"))
            restorer.side_effect = ValueError("checksum mismatch")
            with self.assertRaisesRegex(ValueError, "MinIO 恢复失败.*checksum mismatch"):
                await runtime.create(InferenceCreate(name="失败恢复", taskId="ft-missing"))
            self.assertEqual(runtime.store.list(), [])

    async def test_startup_validation_failure_converges_to_failed_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-fail",
                "state": "completed",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "artifacts": [{"type": "adapter", "variant": "best", "path": str(adapter)}],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = await runtime.create(InferenceCreate(name="失败服务", taskId="ft-fail"))
            with patch.object(runtime, "_validate_vllm_runtime", side_effect=RuntimeError("import failed")):
                with self.assertRaisesRegex(RuntimeError, "import failed"):
                    await runtime.start(service["id"])
            failed = runtime.store.get(service["id"])
            assert failed is not None
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["exposure"], "offline")
            self.assertEqual(failed["gpu"], "未占用")

    def test_proxy_stream_forwards_small_sse_lines_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "test.db")
            service = store.create(
                service_id="svc-stream",
                payload=InferenceCreate(name="流式服务", taskId="ft-1"),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter="/tmp/adapter",
                endpoint="/v1/chat/completions",
            )
            store.update("svc-stream", state="running")
            runtime = InferenceRuntime(
                store,
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            runtime.service_id = service["id"]

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def readline(self):
                    return next(self.lines, b"")

                lines = iter(
                    [
                        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
                        b"\n",
                        b"data: [DONE]\n",
                        b"\n",
                    ]
                )

            with patch.object(urllib.request, "urlopen", return_value=Response()):
                chunks = list(
                    runtime.proxy_stream(
                        "svc-stream",
                        InferenceChatRequest(
                            messages=[{"role": "user", "content": "hello"}]
                        ),
                    )
                )

            self.assertEqual(chunks[0], b'data: {"choices":[{"delta":{"content":"hi"}}]}\n')
            self.assertEqual(chunks[1], b"\n")

    def test_remote_proxy_uses_private_vllm_endpoint_over_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InferenceStore(Path(directory) / "test.db")
            service = store.create(
                service_id="svc-remote-stream",
                payload=InferenceCreate(
                    name="远程流式服务",
                    taskId="ft-1",
                    provider="youyun",
                    youyunResource={"gpuType": "4090", "gpuCount": 1, "cpuCores": 8, "memoryGb": 32},
                ),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter="/tmp/adapter",
                endpoint="/v1/chat/completions",
            )
            store.update("svc-remote-stream", state="running")
            runtime = InferenceRuntime(
                store,
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )

            class SSH:
                def __init__(self) -> None:
                    self.command = ""

                def stream(self, command: str, timeout: float = 600.0):
                    self.command = command
                    yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

            ssh = SSH()
            runtime.remote_handles[service["id"]] = SimpleNamespace(ssh=ssh)
            runtime.remote_endpoints[service["id"]] = "http://remote-vllm"
            with patch.object(urllib.request, "urlopen") as urlopen:
                chunks = list(
                    runtime.proxy_stream(
                        service["id"],
                        InferenceChatRequest(
                            messages=[{"role": "user", "content": "hello"}]
                        ),
                    )
                )

            self.assertEqual(chunks, [b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'])
            self.assertIn("http://127.0.0.1:8011/v1/chat/completions", ssh.command)
            urlopen.assert_not_called()

    async def test_create_can_select_best_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "best-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-1",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "best",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            store = InferenceStore(Path(directory) / "test.db")
            runtime = InferenceRuntime(
                store,
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = await runtime.create(
                InferenceCreate(name="客服服务", taskId="ft-1")
            )
            self.assertEqual(service["state"], "created")
            self.assertEqual(service["config"]["model"], task["model"])
            self.assertEqual(service["config"]["adapter"], str(adapter))

    async def test_manual_retry_preserves_legacy_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            complete_adapter(adapter)

            async def resolve(resource: dict[str, object]) -> dict[str, object]:
                return {**resource, "region": "cn-wlcb", "zone": "cn-wlcb-01"}

            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore({
                    "id": "ft-retry",
                    "state": "completed",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "baseModelMetadata": {"maxContextLength": 4096},
                    "artifacts": [{"type": "adapter", "variant": "best", "path": str(adapter)}],
                }),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
                youyun_resource_resolver=resolve,
            )
            service = await runtime.create(
                InferenceCreate(
                    name="手动重试服务",
                    taskId="ft-retry",
                    provider="youyun",
                    youyunResource={"mode": "manual", "gpuType": "4090", "gpuCount": 1, "cpuCores": 8, "memoryGb": 32},
                )
            )
            runtime.store.update(service["id"], state="failed", recovery_attempts=1)
            expected = runtime.store.get(service["id"])
            assert expected is not None
            with patch.object(runtime, "start", new=AsyncMock(return_value=expected)) as start:
                await runtime.recover(
                    service["id"],
                    youyun_resource={"mode": "manual", "gpuType": "4090", "gpuCount": 1, "cpuCores": 16, "memoryGb": 64},
                )
            start.assert_awaited_once_with(service["id"], recovering=True)
            current = runtime.store.get(service["id"])
            assert current is not None
            self.assertEqual(current["recoveryAttempts"], 1)
            self.assertEqual(current["config"]["youyunResource"]["cpuCores"], 16)

    async def test_retry_inventory_failure_converges_to_failed_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            complete_adapter(adapter)

            calls = 0

            async def resolve(resource: dict[str, object]) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise ValueError("优云 4090 当前无可用库存")
                return {**resource, "region": "cn-wlcb", "zone": "cn-wlcb-01", "price": {"total": 1.0}}

            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore({
                    "id": "ft-inventory-fail",
                    "state": "completed",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "baseModelMetadata": {"maxContextLength": 4096},
                    "artifacts": [{"type": "adapter", "variant": "best", "path": str(adapter)}],
                }),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
                youyun_resource_resolver=resolve,
            )
            service = await runtime.create(InferenceCreate(
                name="库存失败服务",
                taskId="ft-inventory-fail",
                provider="youyun",
                youyunResource={"mode": "manual", "gpuType": "4090", "gpuCount": 1, "cpuCores": 8, "memoryGb": 32},
            ))
            runtime.store.update(service["id"], state="failed", exposure="test")
            with self.assertRaisesRegex(ValueError, "无可用库存"):
                await runtime.recover(service["id"])
            failed = runtime.store.get(service["id"])
            assert failed is not None
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["exposure"], "offline")
            self.assertEqual(failed["gpu"], "未占用")

    async def test_remote_create_persists_server_resolved_resource(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "remote-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-remote",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "baseModelMetadata": {"maxContextLength": 4096},
                "artifacts": [{"type": "adapter", "variant": "best", "path": str(adapter)}],
            }

            async def resolve(resource: dict[str, object]) -> dict[str, object]:
                self.assertEqual(resource["gpuType"], "4090")
                return {
                    "mode": "manual",
                    "region": "cn-wlcb",
                    "zone": "cn-wlcb-01",
                    "gpuType": "4090",
                    "gpuCount": 1,
                    "cpuCores": 16,
                    "memoryGb": 64,
                    "chargeType": "Postpay",
                    "price": {"currency": "CNY", "total": 2.05},
                }

            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
                youyun_resource_resolver=resolve,
            )
            service = await runtime.create(
                InferenceCreate(
                    name="优云推理服务",
                    taskId="ft-remote",
                    provider="youyun",
                    youyunResource={
                        "mode": "manual",
                        "gpuType": "4090",
                        "gpuCount": 1,
                        "cpuCores": 16,
                        "memoryGb": 64,
                    },
                )
            )
            self.assertEqual(service["config"]["youyunResource"]["zone"], "cn-wlcb-01")
            self.assertEqual(service["config"]["youyunResource"]["price"]["total"], 2.05)

    async def test_remote_create_preserves_training_public_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "remote-adapter"
            complete_adapter(adapter)
            public_path = "/models/Qwen/Qwen2.5-3B-Instruct"
            task = {
                "id": "ft-public",
                "state": "completed",
                "model": public_path,
                "modelSource": "public",
                "baseModelMetadata": {"maxContextLength": 4096},
                "youyunResource": {"publicModelPath": public_path},
                "artifacts": [
                    {"type": "adapter", "variant": "best", "path": str(adapter)}
                ],
            }

            async def resolve(resource: dict[str, object]) -> dict[str, object]:
                self.assertEqual(resource["publicModelPath"], public_path)
                return {**resource, "region": "cn-wlcb", "zone": "cn-wlcb-01"}

            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
                youyun_resource_resolver=resolve,
            )
            service = await runtime.create(
                InferenceCreate(
                    name="公共模型推理服务",
                    taskId="ft-public",
                    provider="youyun",
                    youyunResource={
                        "gpuType": "4090",
                        "gpuCount": 1,
                        "cpuCores": 16,
                        "memoryGb": 64,
                    },
                )
            )

            self.assertEqual(
                service["config"]["youyunResource"]["publicModelPath"],
                public_path,
            )

    async def test_final_only_task_can_be_deployed_when_final_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "final-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-legacy",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "final",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = await runtime.create(
                InferenceCreate(
                    name="历史服务",
                    taskId="ft-legacy",
                    adapterVariant="final",
                )
            )
            self.assertEqual(service["config"]["adapterVariant"], "final")
            self.assertEqual(service["config"]["adapter"], str(adapter))

    async def test_requested_adapter_variant_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "final-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-final-only",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "final",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            with self.assertRaisesRegex(ValueError, "Best Adapter"):
                await runtime.create(
                    InferenceCreate(
                        name="错误版本",
                        taskId="ft-final-only",
                        adapterVariant="best",
                    )
                )

    async def test_task_with_best_and_final_can_explicitly_deploy_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best-adapter"
            final = Path(directory) / "final-adapter"
            complete_adapter(best)
            complete_adapter(final)
            task = {
                "id": "ft-both",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {"type": "adapter", "variant": "best", "selected": True, "path": str(best)},
                    {"type": "adapter", "variant": "final", "selected": False, "path": str(final)},
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = await runtime.create(
                InferenceCreate(
                    name="最终版本服务",
                    taskId="ft-both",
                    adapterVariant="final",
                )
            )
            self.assertEqual(service["config"]["adapterVariant"], "final")
            self.assertEqual(service["adapter"], str(final))

    async def test_stopped_task_with_best_adapter_is_deployable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "best-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-stopped",
                "state": "stopped",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "best",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = await runtime.create(
                InferenceCreate(name="早停服务", taskId="ft-stopped")
            )
            self.assertEqual(service["config"]["adapter"], str(adapter))

    async def test_command_contains_scheduler_and_cache_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "best-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-1",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "best",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            store = InferenceStore(Path(directory) / "test.db")
            runtime = InferenceRuntime(
                store,
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = await runtime.create(
                InferenceCreate(
                    name="吞吐服务",
                    taskId="ft-1",
                    kvCacheMode="manual",
                    kvCacheMemoryGiB=4,
                    maxNumSeqs=32,
                    maxNumBatchedTokens=16384,
                    enablePrefixCaching=True,
                    enableChunkedPrefill=True,
                )
            )
            command = runtime._command(service)
            self.assertIn("--enable-lora", command)
            self.assertIn("--no-enable-log-requests", command)
            self.assertNotIn("--disable-log-requests", command)
            self.assertIn("--enable-prefix-caching", command)
            self.assertIn("--enable-chunked-prefill", command)
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "32")
            self.assertEqual(
                command[command.index("--max-num-batched-tokens") + 1], "16384"
            )
            self.assertEqual(
                command[command.index("--kv-cache-memory-bytes") + 1],
                str(4 * 1024**3),
            )

    async def test_custom_vllm_command_is_validated_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "best-adapter"
            complete_adapter(adapter)
            task = {
                "id": "ft-custom",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "best",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            raw_command = (
                "vllm serve Qwen/Qwen2.5-3B-Instruct "
                f"--lora-modules best={adapter} "
                "--max-num-seqs 32 --enable-prefix-caching"
            )
            preview = runtime.preview_custom_command("ft-custom", raw_command)
            self.assertTrue(preview["valid"])
            self.assertEqual(preview["port"], 8011)
            self.assertIn("--max-num-seqs", preview["flags"])
            self.assertIn("--enable-prefix-caching", preview["flags"])
            self.assertIn("<service-id>", preview["command"])

            service = await runtime.create(
                InferenceCreate(
                    name="自定义命令服务",
                    taskId="ft-custom",
                    deploymentMode="custom",
                    customCommand=raw_command,
                )
            )
            self.assertEqual(service["config"]["deploymentMode"], "custom")
            self.assertEqual(service["config"]["customCommand"], raw_command)
            command = runtime._command(service)
            self.assertEqual(command[0:3], [command[0], "-m", "vllm.entrypoints.openai.api_server"])
            self.assertIn("--enable-lora", command)
            self.assertIn("--served-model-name", command)

    async def test_custom_vllm_command_uses_requested_final_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best-adapter"
            final = Path(directory) / "final-adapter"
            complete_adapter(best)
            complete_adapter(final)
            task = {
                "id": "ft-custom-final",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {"type": "adapter", "variant": "best", "selected": True, "path": str(best)},
                    {"type": "adapter", "variant": "final", "selected": False, "path": str(final)},
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            raw_command = (
                "vllm serve Qwen/Qwen2.5-3B-Instruct "
                f"--lora-modules final={final}"
            )
            preview = runtime.preview_custom_command(
                "ft-custom-final", raw_command, "final"
            )
            self.assertEqual(preview["adapter"], str(final))

            service = await runtime.create(
                InferenceCreate(
                    name="Final 自定义命令服务",
                    taskId="ft-custom-final",
                    adapterVariant="final",
                    deploymentMode="custom",
                    customCommand=raw_command,
                )
            )
            self.assertEqual(service["config"]["adapterVariant"], "final")
            self.assertEqual(service["adapter"], str(final))

    async def test_custom_vllm_command_rejects_shell_and_wrong_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "best-adapter"
            wrong_adapter = Path(directory) / "wrong-adapter"
            complete_adapter(adapter)
            wrong_adapter.mkdir()
            task = {
                "id": "ft-custom-invalid",
                "state": "completed",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "best",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            with self.assertRaisesRegex(ValueError, "不支持|Shell"):
                runtime.preview_custom_command(
                    "ft-custom-invalid",
                    f"vllm serve Qwen/Qwen2.5-3B-Instruct --lora-modules best={adapter} && echo hacked",
                )
            with self.assertRaisesRegex(ValueError, "所选训练任务的 Adapter"):
                runtime.preview_custom_command(
                    "ft-custom-invalid",
                    f"vllm serve Qwen/Qwen2.5-3B-Instruct --lora-modules best={wrong_adapter}",
                )
            with self.assertRaisesRegex(ValueError, "Base Model"):
                runtime.preview_custom_command(
                    "ft-custom-invalid",
                    f"vllm serve Qwen/Qwen2.5-7B-Instruct --lora-modules best={adapter}",
                )


    def test_vllm_runtime_diagnostics_use_configured_python_and_preload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python"
            executable.write_text("", encoding="utf-8")
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            with patch.dict(
                os.environ,
                {
                    "XUNTUI_VLLM_PYTHON": str(executable),
                    "XUNTUI_VLLM_LD_PRELOAD": "/usr/lib/libiomp5.so",
                    "XUNTUI_INFERENCE_DEVICE": "cpu",
                },
                clear=False,
            ), patch.object(
                InferenceRuntime, "_cpu_flags", return_value={"avx", "avx2"}
            ), patch(
                "backend.inference_service.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="0.27.1\n", stderr=""
                ),
            ) as run:
                self.assertEqual(runtime._validate_vllm_runtime(), "0.27.1")

            run.assert_called_once()
            args, kwargs = run.call_args
            self.assertEqual(
                args[0],
                [str(executable), "-c", "import vllm; print(vllm.__version__)"],
            )
            self.assertEqual(kwargs["timeout"], 30)
            self.assertEqual(kwargs["env"]["LD_PRELOAD"], "/usr/lib/libiomp5.so")

    def test_cpu_vllm_runtime_diagnostics_reject_missing_avx2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python"
            executable.write_text("", encoding="utf-8")
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            with patch.dict(
                os.environ,
                {
                    "XUNTUI_VLLM_PYTHON": str(executable),
                    "XUNTUI_INFERENCE_DEVICE": "cpu",
                },
                clear=False,
            ), patch(
                "backend.inference_service.platform.machine", return_value="x86_64"
            ), patch.object(
                InferenceRuntime, "_cpu_flags", return_value={"avx", "sse4_2"}
            ), patch(
                "backend.inference_service.subprocess.run"
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "缺少 AVX2"):
                    runtime._validate_vllm_runtime()

            run.assert_not_called()

    def test_vllm_runtime_diagnostics_expose_import_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python"
            executable.write_text("", encoding="utf-8")
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            with patch.dict(
                os.environ,
                {
                    "XUNTUI_VLLM_PYTHON": str(executable),
                    "XUNTUI_INFERENCE_DEVICE": "auto",
                },
                clear=False,
            ), patch(
                "backend.inference_service.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="ModuleNotFoundError: No module named 'vllm'",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "ModuleNotFoundError"):
                    runtime._validate_vllm_runtime()

    def test_cpu_managed_command_filters_gpu_only_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            service = {
                "id": "svc-cpu",
                "config": {
                    "deploymentMode": "managed",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "adapter": "/workspace/training-work/ft-cpu/output",
                    "maxModelLength": 512,
                    "maxNumSeqs": 1,
                    "maxNumBatchedTokens": 512,
                    "gpuMemoryUtilization": 0.75,
                    "dtype": "bfloat16",
                    "tensorParallelSize": 1,
                    "quantization": "none",
                    "enablePrefixCaching": True,
                    "enableChunkedPrefill": True,
                    "kvCacheDtype": "auto",
                    "kvCacheMode": "auto",
                    "kvCacheMemoryGiB": None,
                },
            }
            with patch.dict(
                os.environ,
                {
                    "XUNTUI_INFERENCE_DEVICE": "cpu",
                    "XUNTUI_VLLM_PYTHON": "/opt/venvs/vllm/bin/python",
                },
                clear=False,
            ):
                command = runtime._command(service)

            self.assertEqual(command[0], "/opt/venvs/vllm/bin/python")
            self.assertNotIn("--gpu-memory-utilization", command)
            self.assertNotIn("--quantization", command)
            self.assertNotIn("--kv-cache-dtype", command)
            self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")

    def test_cpu_commands_reject_gpu_quantization_and_cache_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "test.db"),
                FakeTaskStore(None),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
            )
            base_config = {
                "deploymentMode": "managed",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "adapter": "/workspace/training-work/ft-cpu/output",
                "maxModelLength": 512,
                "maxNumSeqs": 1,
                "maxNumBatchedTokens": 512,
                "gpuMemoryUtilization": 0.75,
                "dtype": "bfloat16",
                "tensorParallelSize": 1,
                "quantization": "none",
                "enablePrefixCaching": False,
                "enableChunkedPrefill": False,
                "kvCacheDtype": "auto",
                "kvCacheMode": "auto",
                "kvCacheMemoryGiB": None,
            }
            with patch.dict(
                os.environ, {"XUNTUI_INFERENCE_DEVICE": "cpu"}, clear=False
            ):
                bitsandbytes = {**base_config, "quantization": "bitsandbytes"}
                with self.assertRaisesRegex(ValueError, "bitsandbytes"):
                    runtime._command({"id": "svc-bnb", "config": bitsandbytes})

                fp8 = {**base_config, "kvCacheDtype": "fp8"}
                with self.assertRaisesRegex(ValueError, "FP8"):
                    runtime._command({"id": "svc-fp8", "config": fp8})

                for option in (
                    "--gpu-memory-utilization 0.8",
                    "--quantization bitsandbytes",
                    "--kv-cache-dtype fp8",
                ):
                    with self.subTest(option=option), self.assertRaisesRegex(
                        ValueError, "CPU 冒烟推理不支持"
                    ):
                        runtime._normalized_custom_args(
                            "vllm serve Qwen/Qwen2.5-0.5B-Instruct "
                            "--lora-modules cpu=/workspace/training-work/ft-cpu/output "
                            + option,
                            model="Qwen/Qwen2.5-0.5B-Instruct",
                            adapter="/workspace/training-work/ft-cpu/output",
                            service_id="svc-cpu",
                            port=8011,
                        )


if __name__ == "__main__":
    unittest.main()
