from __future__ import annotations

import os
import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TEST_RUNTIME = tempfile.TemporaryDirectory()
TEST_ROOT = Path(TEST_RUNTIME.name)
os.environ["XUNTUI_STORAGE_BACKEND"] = "sqlite"
os.environ["XUNTUI_DATABASE_PATH"] = str(TEST_ROOT / "training.db")
os.environ["XUNTUI_RUNTIME_ROOT"] = str(TEST_ROOT / "runtime")
os.environ["XUNTUI_DATASET_ROOT"] = str(TEST_ROOT / "datasets")
os.environ["XUNTUI_ARTIFACT_ROOT"] = str(TEST_ROOT / "artifacts")

from fastapi import HTTPException

from backend import app as app_module


class YouyunTrainingImageFilterTest(unittest.TestCase):
    def test_public_model_search_returns_container_path(self) -> None:
        class _Client:
            def describe_model_repository_models(self, *, name=None, tags=None):
                self.args = (name, tags)
                return [{
                    "name": "Llama 3 8B",
                    "path": "/models/llama-3-8b",
                    "tag": "LLM",
                    "size": "16GB",
                    "createTime": "2026-01-01 00:00:00",
                }]

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(app_module.search_models("llama", source="public"))
        self.assertEqual(client.args, ("llama", None))
        self.assertEqual(result[0]["id"], "/models/llama-3-8b")
        self.assertEqual(result[0]["publicModelPath"], "/models/llama-3-8b")
        self.assertEqual(result[0]["source"], "youyun-public")

    def test_public_model_catalog_is_paged_five_per_page(self) -> None:
        class _Client:
            def describe_model_repository_models(self, *, name=None, tags=None):
                return [
                    {"name": f"Model {index}", "path": f"/models/model-{index}", "tag": "LLM", "size": "1GB"}
                    for index in range(7)
                ]

        runner = SimpleNamespace(manager=SimpleNamespace(client=_Client()))
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(app_module.list_public_models(page=2, pageSize=5))
        self.assertEqual(result["total"], 7)
        self.assertEqual(result["totalPages"], 2)
        self.assertEqual([item["path"] for item in result["items"]], ["/models/model-5", "/models/model-6"])

    def test_youyun_image_version_uses_explicit_metadata_only(self) -> None:
        self.assertEqual(
            app_module._youyun_image_version({"name": "vLLM v0.12.0", "tags": []}),
            "0.12.0",
        )
        self.assertEqual(
            app_module._youyun_image_version({"name": "vLLM CUDA 12.8", "tags": []}),
            None,
        )
        self.assertEqual(
            app_module._youyun_image_version({"name": "vLLM image", "tags": ["v0.25.1"]}),
            "v0.25.1",
        )

    def test_keeps_cuda_pytorch_and_vllm_but_excludes_unrelated_images(self) -> None:
        self.assertTrue(app_module._is_training_image({"name": "CUDA 12.8 PyTorch", "image_type": "App"}))
        self.assertTrue(app_module._is_training_image({"name": "vLLM v0.25.1", "image_type": "App"}))
        self.assertTrue(app_module._is_training_image({"name": "LLaMA-Factory 0.9.3", "image_type": "Community"}))
        self.assertFalse(app_module._is_training_image({"name": "Windows-nvidia 2022", "image_type": "System"}))
        self.assertFalse(app_module._is_training_image({"name": "ComfyUI 基础镜像", "image_type": "App"}))

    def test_inference_image_filter_keeps_only_official_vllm_images(self) -> None:
        self.assertTrue(app_module._is_official_vllm_image({"name": "vLLM official CUDA 12.8", "image_type": "App"}))
        self.assertTrue(app_module._is_official_vllm_image({"name": "vLLM system image", "image_type": "System"}))
        self.assertFalse(app_module._is_official_vllm_image({"name": "vLLM community image", "image_type": "Community"}))
        self.assertFalse(app_module._is_official_vllm_image({"name": "vLLM community image", "image_type": "Public"}))
        self.assertFalse(app_module._is_official_vllm_image({"name": "CUDA PyTorch official", "image_type": "App"}))

    def test_inference_images_endpoint_excludes_community_and_non_vllm(self) -> None:
        class _Client:
            config = SimpleNamespace(training_image_id="train-img", image_id="fallback-img")

            def describe_images(self):
                return [
                    {"image_id": "official-vllm", "name": "vLLM official", "image_type": "App", "status": "Available", "tags": ["v0.12"]},
                    {"image_id": "community-vllm", "name": "vLLM community", "image_type": "Community", "status": "Available", "tags": []},
                    {"image_id": "official-pytorch", "name": "PyTorch official", "image_type": "App", "status": "Available", "tags": []},
                ]

        app_module._YOUYUN_INFERENCE_IMAGES_CACHE = None
        with patch.object(app_module, "build_youyun_runner", return_value=SimpleNamespace(manager=SimpleNamespace(client=_Client()))):
            result = asyncio.run(app_module.youyun_inference_images())
        self.assertEqual([item["id"] for item in result["images"]], ["official-vllm"])
        self.assertEqual(result["defaultImageId"], "official-vllm")
        # Capability booleans were never returned by the provider; the endpoint
        # must not invent them.
        self.assertNotIn("capabilities", result["images"][0]["compatibility"])
        self.assertNotIn("acceptanceStatus", result["images"][0]["compatibility"])

    def test_inference_images_default_to_newest_general_provider_version(self) -> None:
        class _Client:
            config = SimpleNamespace(training_image_id="train-img", image_id="fallback-img")

            def describe_images(self):
                return [
                    {"image_id": "old", "name": "vLLM v0.24.0", "image_type": "App", "status": "Available", "tags": []},
                    {"image_id": "gpu5090", "name": "vLLM v0.27.0-5090", "image_type": "App", "status": "Available", "tags": []},
                    {"image_id": "newest", "name": "vLLM v0.26.1", "image_type": "App", "status": "Available", "tags": []},
                ]

        app_module._YOUYUN_INFERENCE_IMAGES_CACHE = None
        with patch.dict(os.environ, {"YOUYUN_INFERENCE_IMAGE_ID": ""}, clear=False):
            with patch.object(app_module, "build_youyun_runner", return_value=SimpleNamespace(manager=SimpleNamespace(client=_Client()))):
                result = asyncio.run(app_module.youyun_inference_images())
        # Newest version first, and the 5090-only image never becomes the
        # default even though it carries the highest version.
        self.assertEqual([item["id"] for item in result["images"]], ["gpu5090", "newest", "old"])
        self.assertEqual(result["defaultImageId"], "newest")
        versions = {item["id"]: item["compatibility"]["version"] for item in result["images"]}
        self.assertEqual(versions["newest"], "0.26.1")
        self.assertEqual(
            next(item["compatibility"]["recommendation"] for item in result["images"] if item["id"] == "gpu5090"),
            "rtx5090",
        )

    def test_inference_images_env_override_wins_over_newest_version(self) -> None:
        class _Client:
            config = SimpleNamespace(training_image_id="train-img", image_id="fallback-img")

            def describe_images(self):
                return [
                    {"image_id": "pinned", "name": "vLLM v0.24.0", "image_type": "App", "status": "Available", "tags": []},
                    {"image_id": "newest", "name": "vLLM v0.26.1", "image_type": "App", "status": "Available", "tags": []},
                ]

        app_module._YOUYUN_INFERENCE_IMAGES_CACHE = None
        with patch.dict(os.environ, {"YOUYUN_INFERENCE_IMAGE_ID": "pinned"}, clear=False):
            with patch.object(app_module, "build_youyun_runner", return_value=SimpleNamespace(manager=SimpleNamespace(client=_Client()))):
                result = asyncio.run(app_module.youyun_inference_images())
        self.assertEqual(result["defaultImageId"], "pinned")

    def test_training_images_keep_configured_llama_factory_when_provider_omits_it(self) -> None:
        class _Client:
            config = SimpleNamespace(training_image_id="llama-default", image_id="fallback-img")

            def describe_images(self):
                return [
                    {"image_id": "cuda-other", "name": "cuda126_python312", "image_type": "App", "status": "Available", "tags": []},
                ]

        app_module._YOUYUN_IMAGES_CACHE = None
        with patch.object(app_module, "build_youyun_runner", return_value=SimpleNamespace(manager=SimpleNamespace(client=_Client()))):
            result = asyncio.run(app_module.youyun_images())

        self.assertEqual(result["defaultImageId"], "llama-default")
        self.assertEqual(result["images"][0]["id"], "llama-default")
        self.assertEqual(result["images"][0]["name"], "LLaMA-Factory 社区训练镜像")
        self.assertIn("cuda-other", [item["id"] for item in result["images"]])

    def test_inference_resource_requires_explicit_image(self) -> None:
        with patch.object(
            app_module,
            "build_youyun_runner",
            return_value=SimpleNamespace(manager=SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace()))),
        ):
            with self.assertRaisesRegex(HTTPException, "必须选择官方 vLLM 镜像") as context:
                asyncio.run(
                    app_module.resolve_youyun_resource(
                        app_module.YouyunResourceSelection(purpose="inference")
                    )
                )
        self.assertEqual(context.exception.status_code, 422)


class _YouyunLifecycleStore:
    def __init__(self, task: dict) -> None:
        self.task = task
        self.messages: list[str] = []

    def get(self, task_id: str) -> dict | None:
        return self.task if task_id == self.task["id"] else None

    def list(self, *, include_details: bool = True) -> list[dict]:
        return [self.task]

    def update(self, task_id: str, **changes: object) -> dict:
        assert task_id == self.task["id"]
        mapping = {
            "current_step": "currentStep",
            "cancel_requested": "cancelRequested",
        }
        self.task.update({mapping.get(key, key): value for key, value in changes.items()})
        return self.task

    def append_log(self, task_id: str, message: str) -> None:
        assert task_id == self.task["id"]
        self.messages.append(message)


class _YouyunLifecycleManager:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.cleaned: list[str] = []

    def cleanup_after_task(self, instance_id: str, stop_first: bool = True) -> dict:
        self.cleaned.append(instance_id)
        return self.result


class YouyunLifecycleApiTest(unittest.IsolatedAsyncioTestCase):
    def task(self, *, state: str = "completed", marker_state: str = "awaiting_comparison") -> dict:
        return {
            "id": "ft-youyun",
            "provider": "youyun",
            "state": state,
            "currentStep": "done",
            "artifacts": [
                {
                    "type": "youyun-instance",
                    "instanceId": "uhost-1",
                    "state": marker_state,
                }
            ],
        }

    async def test_manual_release_persists_released_marker(self) -> None:
        store = _YouyunLifecycleStore(self.task())
        manager = _YouyunLifecycleManager({"terminated": True, "error": None})
        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "build_youyun_runner",
            return_value=SimpleNamespace(manager=manager),
        ):
            updated = await app_module.cleanup_youyun_task_instance(
                "ft-youyun", reason="结束任务并释放"
            )
        self.assertEqual(updated["artifacts"][0]["state"], "released")
        self.assertEqual(manager.cleaned, ["uhost-1"])
        self.assertTrue(any("已结束任务并释放" in message for message in store.messages))

    async def test_manual_release_allows_stopped_task_with_retained_instance(self) -> None:
        store = _YouyunLifecycleStore(self.task(state="stopped"))
        manager = _YouyunLifecycleManager({"terminated": True, "error": None})
        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "build_youyun_runner",
            return_value=SimpleNamespace(manager=manager),
        ):
            updated = await app_module.release_youyun_instance("ft-youyun")
        self.assertEqual(updated["state"], "stopped")
        self.assertEqual(updated["artifacts"][0]["state"], "released")
        self.assertEqual(manager.cleaned, ["uhost-1"])

    async def test_release_failure_remains_retryable(self) -> None:
        store = _YouyunLifecycleStore(self.task())
        manager = _YouyunLifecycleManager(
            {"terminated": False, "error": "billing lock"}
        )
        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "build_youyun_runner",
            return_value=SimpleNamespace(manager=manager),
        ):
            with self.assertRaisesRegex(RuntimeError, "billing lock"):
                await app_module.cleanup_youyun_task_instance(
                    "ft-youyun", reason="销毁"
                )
        self.assertEqual(store.task["artifacts"][0]["state"], "cleanup_failed")
        self.assertEqual(
            store.task["artifacts"][0]["cleanupError"], "billing lock"
        )

    async def test_restart_reconciliation_cleans_interrupted_instance(self) -> None:
        store = _YouyunLifecycleStore(self.task(state="failed", marker_state="running"))
        manager = _YouyunLifecycleManager({"terminated": True, "error": None})
        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "build_youyun_runner",
            return_value=SimpleNamespace(manager=manager),
        ):
            await app_module.reconcile_interrupted_youyun_instances()
        self.assertEqual(store.task["artifacts"][0]["state"], "released")
        self.assertEqual(manager.cleaned, ["uhost-1"])


class TrainingMetricsRecoveryTest(unittest.TestCase):
    def test_trainer_state_log_history_recovers_train_and_eval_loss(self) -> None:
        class MetricStore:
            def __init__(self) -> None:
                self.metrics: list[dict] = []

            def append_metric(self, task_id: str, **values: object) -> None:
                self.metrics.append({"taskId": task_id, **values})

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "trainer_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "log_history": [
                            {"loss": 2.3, "learning_rate": 0.0002, "epoch": 0.5, "step": 10},
                            {"eval_loss": 1.7, "epoch": 0.5, "step": 10},
                            {"loss": 1.9, "learning_rate": 0.0001, "epoch": 1.0, "step": 20},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            store = MetricStore()
            with patch.object(app_module, "store", store):
                recovered = app_module._recover_trainer_metrics("ft-metrics", state_path)

        self.assertEqual(recovered, 2)
        self.assertEqual(store.metrics[0]["step"], 10)
        self.assertEqual(store.metrics[0]["train_loss"], 2.3)
        self.assertEqual(store.metrics[0]["eval_loss"], 1.7)
        self.assertEqual(store.metrics[1]["step"], 20)


class YouyunAutoReleaseTest(unittest.IsolatedAsyncioTestCase):
    def task(self, *, retained_at: str | None, marker_state: str = "awaiting_comparison") -> dict:
        marker = {
            "type": "youyun-instance",
            "instanceId": "uhost-1",
            "state": marker_state,
        }
        if retained_at is not None:
            marker["retainedAt"] = retained_at
        return {
            "id": "ft-youyun",
            "provider": "youyun",
            "state": "completed",
            "artifacts": [marker],
        }

    async def run_scan(self, task: dict, retention_hours: int = 6):
        store = _YouyunLifecycleStore(task)
        manager = _YouyunLifecycleManager({"terminated": True, "error": None})
        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "build_youyun_runner",
            return_value=SimpleNamespace(manager=manager),
        ), patch.object(app_module, "YOUYUN_RETENTION_HOURS", retention_hours):
            released = await app_module.auto_release_expired_youyun_instances()
        return released, store, manager

    async def test_expired_instance_is_released(self) -> None:
        released, store, manager = await self.run_scan(
            self.task(retained_at="2026-01-01T00:00:00+00:00")
        )
        self.assertEqual(released, 1)
        self.assertEqual(manager.cleaned, ["uhost-1"])
        self.assertEqual(store.task["artifacts"][0]["state"], "released")
        self.assertTrue(any("自动释放" in message for message in store.messages))

    async def test_fresh_instance_is_kept(self) -> None:
        from datetime import datetime, timedelta, timezone

        fresh = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        released, store, manager = await self.run_scan(self.task(retained_at=fresh))
        self.assertEqual(released, 0)
        self.assertEqual(manager.cleaned, [])
        self.assertEqual(store.task["artifacts"][0]["state"], "awaiting_comparison")

    async def test_legacy_marker_gets_retained_at_stamp(self) -> None:
        released, store, manager = await self.run_scan(self.task(retained_at=None))
        self.assertEqual(released, 0)
        self.assertEqual(manager.cleaned, [])
        self.assertEqual(store.task["artifacts"][0]["state"], "awaiting_comparison")
        self.assertIn("retainedAt", store.task["artifacts"][0])

    async def test_non_awaiting_states_are_ignored(self) -> None:
        released, _, manager = await self.run_scan(
            self.task(retained_at="2026-01-01T00:00:00+00:00", marker_state="released")
        )
        self.assertEqual(released, 0)
        self.assertEqual(manager.cleaned, [])


class GatewayIdentityTest(unittest.TestCase):
    def test_gateway_headers_are_url_decoded(self) -> None:
        from types import SimpleNamespace

        request = SimpleNamespace(
            headers={
                "x-control-user-id": "u-123",
                "x-control-username": "%E5%88%98%E9%87%91%E6%AD%A6",
                "x-control-roles": "admin",
            }
        )
        user = app_module.gateway_user_from_request(request)
        self.assertEqual(user["name"], "刘金武")
        self.assertEqual(user["id"], "u-123")

    def test_plain_username_passes_through(self) -> None:
        from types import SimpleNamespace

        request = SimpleNamespace(
            headers={
                "x-control-user-id": "",
                "x-control-username": "刘金武",
                "x-control-roles": "",
            }
        )
        user = app_module.gateway_user_from_request(request)
        self.assertEqual(user["name"], "刘金武")


class DatasetInteractionLogicTest(unittest.TestCase):
    def test_storage_backend_prefers_postgres_and_keeps_local_fallback(self) -> None:
        self.assertEqual(app_module.resolve_storage_backend("", "postgresql://local"), "postgres")
        self.assertEqual(app_module.resolve_storage_backend("", ""), "sqlite")
        self.assertEqual(app_module.resolve_storage_backend("sqlite", "postgresql://local"), "sqlite")

    def test_clone_names_increment_from_one_and_ignore_existing_source_suffix(self) -> None:
        source = {"name": "数据集"}
        existing = [
            {"name": "数据集"},
            {"name": "数据集 副本1"},
            {"name": "数据集 副本3"},
        ]
        self.assertEqual(app_module.next_clone_name(source, existing), "数据集 副本2")
        self.assertEqual(
            app_module.next_clone_name({"name": "数据集 副本1"}, existing),
            "数据集 副本2",
        )

    def test_import_source_must_be_jsonl_under_exports(self) -> None:
        self.assertEqual(
            app_module.normalize_import_source_key("/exports/team/train.jsonl"),
            "exports/team/train.jsonl",
        )
        with self.assertRaisesRegex(ValueError, "exports"):
            app_module.normalize_import_source_key("datasets/train.jsonl")
        with self.assertRaisesRegex(ValueError, "JSONL"):
            app_module.normalize_import_source_key("exports/team/manifest.json")

    def test_import_sources_put_newest_supported_jsonl_first(self) -> None:
        client = SimpleNamespace(
            list_objects=lambda *_args, **_kwargs: [
                SimpleNamespace(
                    object_name="exports/sft/older.jsonl",
                    size=10,
                    last_modified=app_module.datetime.fromisoformat("2026-08-25T10:00:00+00:00"),
                ),
                SimpleNamespace(
                    object_name="exports/manifest.json",
                    size=20,
                    last_modified=app_module.datetime.fromisoformat("2026-08-26T12:00:00+00:00"),
                ),
                SimpleNamespace(
                    object_name="exports/dpo/newest.jsonl",
                    size=30,
                    last_modified=app_module.datetime.fromisoformat("2026-08-26T11:00:00+00:00"),
                ),
                SimpleNamespace(
                    object_name="exports/sft/unknown-time.jsonl",
                    size=40,
                    last_modified=None,
                ),
            ]
        )
        with patch.object(app_module, "require_dataset_minio_client", return_value=client):
            sources = app_module.list_dataset_import_sources()

        self.assertEqual(
            [source["key"] for source in sources],
            [
                "exports/dpo/newest.jsonl",
                "exports/sft/older.jsonl",
                "exports/sft/unknown-time.jsonl",
                "exports/manifest.json",
            ],
        )


class YouyunImageLocationTest(unittest.TestCase):
    def test_provider_container_metadata_controls_instance_mode(self) -> None:
        self.assertEqual(app_module._youyun_image_instance_mode({"container": "True", "image_type": "Other"}), "pod")
        self.assertEqual(app_module._youyun_image_instance_mode({"container": False, "image_type": "Other"}), "uhost")
        self.assertEqual(app_module._youyun_image_instance_mode({"Container": 1, "ImageType": "Other"}), "pod")
        self.assertEqual(app_module._youyun_image_instance_mode({"Container": 0, "ImageType": "Other"}), "uhost")
        self.assertEqual(app_module._youyun_image_instance_mode({"image_type": "System"}), "uhost")

    def test_container_image_keeps_all_five_display_locations(self) -> None:
        zones = [
            {"region": "cn-sh2", "zone": "cn-sh2-01", "describe": "上海二A", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "describe": "华北一C", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "describe": "华北二C", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "describe": "华北二A", "is_pod": False},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "describe": "上海二B", "is_pod": False},
        ]
        locations = app_module._youyun_location_options(
            zones, image_mode="pod", image={"container": True, "image_type": "App"}
        )
        self.assertEqual([item["displayName"] for item in locations], ["上海二A", "华北一C", "华北二C", "华北二A", "上海二B"])
        self.assertEqual([item["supported"] for item in locations], [True, True, True, False, False])
        self.assertIn("容器", locations[3]["disabledReason"])
        self.assertEqual(locations[3]["actualZone"], "cn-wlcb-01")
        self.assertEqual(locations[4]["actualZone"], "cn-sh2-02")

    def test_container_image_preserves_each_exact_zone_without_image_mode_remapping(self) -> None:
        zones = [
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "describe": "North China 2C", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "describe": "North China 2A", "is_pod": False, "unsupported_image_types": ["Community"]},
            {"region": "cn-sh2", "zone": "cn-sh2-01", "describe": "Shanghai 2A", "is_pod": True},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "describe": "Shanghai 2B", "is_pod": False, "unsupported_image_types": ["Community"]},
        ]
        locations = app_module._youyun_location_options(
            zones, image_mode="pod", image={"container": True, "image_type": "Community"}
        )
        by_zone = {item["zone"]: item for item in locations}
        self.assertEqual(by_zone["cn-wlcb-01"]["actualZone"], "cn-wlcb-01")
        self.assertEqual(by_zone["cn-sh2-02"]["actualZone"], "cn-sh2-02")
        self.assertFalse(by_zone["cn-wlcb-01"]["supported"])
        self.assertFalse(by_zone["cn-sh2-02"]["supported"])

    def test_virtual_machine_image_only_supports_uhost_zone(self) -> None:
        zones = [
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "describe": "华北二C", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "describe": "华北二A", "is_pod": False},
        ]
        locations = app_module._youyun_location_options(
            zones, image_mode="uhost", image={"container": False, "image_type": "Other"}
        )
        requested = next(item for item in locations if item["zone"] == "cn-wlcb-03")
        fallback = next(item for item in locations if item["zone"] == "cn-wlcb-01")
        self.assertFalse(requested["supported"])
        self.assertTrue(fallback["supported"])
        self.assertEqual(requested["actualZone"], "cn-wlcb-03")
        self.assertEqual(fallback["actualZone"], "cn-wlcb-01")

    def test_system_image_only_enables_dual_mode_uhost_locations(self) -> None:
        zones = [
            {"region": "cn-sh2", "zone": "cn-sh2-01", "describe": "上海二A", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "describe": "华北一C", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "describe": "华北二C", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "describe": "华北二A", "is_pod": False},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "describe": "上海二B", "is_pod": False},
        ]
        locations = app_module._youyun_location_options(
            zones, image_mode="uhost", image={"container": False, "image_type": "System"}
        )
        self.assertEqual([item["supported"] for item in locations], [False, False, False, True, True])


class YouyunResourceModelTest(unittest.TestCase):
    @staticmethod
    def _runner(*, zones: list[dict]) -> SimpleNamespace:
        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img",
                image_id="",
                gpu_type="4090",
                charge_type="Postpay",
                region="cn-wlcb",
                zone="cn-wlcb-01",
                instance_mode="uhost",
                pod_zone="",
            )

            def describe_support_zones(self, region=None):
                return zones

            def check_resource_capacity(self, **kwargs):
                return SimpleNamespace(
                    specs=[SimpleNamespace(gpu=1, cpu=16, memory_gb=64, resource_enough=True)]
                )

            def get_instance_price(self, **kwargs):
                return SimpleNamespace(
                    currency="CNY", instance=1.0, disks=0.0, system_disks=0.0, image=0.0,
                    total=1.0, charge_type="Postpay",
                )

        return SimpleNamespace(manager=SimpleNamespace(client=_Client()))

    def test_public_model_path_defaults_to_pod_mode(self) -> None:
        # 可用区列表模拟真实 API：忽略 Region 过滤返回全部区域，
        # 自动挑选必须落在当前区域的 Pod 可用区（cn-wlcb-03）。
        runner = self._runner(zones=[
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True},
        ])
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(
                app_module.resolve_youyun_resource(
                    app_module.YouyunResourceSelection(
                        publicModelPath="/models/qwen", cpuCores=16, memoryGb=64,
                    )
                )
            )
        self.assertEqual(result["instanceMode"], "pod")
        self.assertEqual(result["zone"], "cn-wlcb-03")

    def test_public_model_path_rejects_container_image_on_uhost_zone(self) -> None:
        runner = self._runner(zones=[{"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False}])
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            with self.assertRaises(app_module.HTTPException) as caught:
                asyncio.run(
                    app_module.resolve_youyun_resource(
                        app_module.YouyunResourceSelection(
                            publicModelPath="/models/qwen", instanceMode="uhost", zone="cn-wlcb-01", cpuCores=16, memoryGb=64,
                        )
                    )
                )
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("Pod", caught.exception.detail)

    def test_public_model_path_accepts_explicit_pod_mode(self) -> None:
        runner = self._runner(zones=[{"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True}])
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(
                app_module.resolve_youyun_resource(
                    app_module.YouyunResourceSelection(
                        publicModelPath="/models/qwen", instanceMode="pod", region="cn-bj2", zone="cn-bj2-03",
                        cpuCores=16, memoryGb=64,
                    )
                )
            )
        self.assertEqual(result["instanceMode"], "pod")

    def test_image_type_must_match_zone_mode(self) -> None:
        zones = [
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "is_pod": False},
            {"region": "cn-sh2", "zone": "cn-sh2-01", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True},
        ]
        for image, region, display_zone in (
            ("app-image", "cn-wlcb", "cn-wlcb-01"),
            ("ubuntu-image", "cn-sh2", "cn-sh2-01"),
        ):
            with self.subTest(image=image):
                runner = self._runner(zones=zones)
                client = runner.manager.client
                client.describe_images = lambda **kwargs: [
                    {"image_id": "app-image", "image_type": "App", "container": True, "name": "vLLM"},
                    {"image_id": "ubuntu-image", "image_type": "System", "container": "False", "name": "Ubuntu"},
                ]
                with patch.object(app_module, "build_youyun_runner", return_value=runner):
                    with self.assertRaises(app_module.HTTPException) as caught:
                        asyncio.run(
                            app_module.resolve_youyun_resource(
                                app_module.YouyunResourceSelection(
                                    imageId=image, region=region, zone=display_zone,
                                    cpuCores=16, memoryGb=64,
                                )
                            )
                        )
                self.assertEqual(caught.exception.status_code, 422)
                self.assertIn("镜像", caught.exception.detail)

    def test_resource_selection_resolves_dual_mode_region_to_matching_zone(self) -> None:
        zones = [
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "is_pod": False},
            {"region": "cn-sh2", "zone": "cn-sh2-01", "is_pod": True},
        ]
        for region, mode, expected_zone in (
            ("cn-wlcb", "pod", "cn-wlcb-03"),
            ("cn-wlcb", "uhost", "cn-wlcb-01"),
            ("cn-sh2", "pod", "cn-sh2-01"),
            ("cn-sh2", "uhost", "cn-sh2-02"),
        ):
            with self.subTest(region=region, mode=mode):
                runner = self._runner(zones=zones)
                image_id = None
                if mode == "uhost":
                    image_id = "ubuntu-image"
                    runner.manager.client.describe_images = lambda **kwargs: [
                        {"image_id": image_id, "image_type": "System", "container": False, "name": "Ubuntu"}
                    ]
                with patch.object(app_module, "build_youyun_runner", return_value=runner):
                    result = asyncio.run(
                        app_module.resolve_youyun_resource(
                            app_module.YouyunResourceSelection(
                                region=region, instanceMode=mode, imageId=image_id,
                                cpuCores=16, memoryGb=64,
                            )
                        )
                    )
                self.assertEqual(result["zone"], expected_zone)
                self.assertEqual(result["instanceMode"], mode)

    def test_gpu_specs_exposes_region_capabilities_for_both_instance_modes(self) -> None:
        zones = [
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False, "describe": "华北二A"},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "华北二C"},
        ]

        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-03", instance_mode="pod", pod_zone="cn-wlcb-03",
            )

            def describe_support_zones(self, region=None):
                return zones

            def describe_instance_types(self, *, instance_type="uhost"):
                return []

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(app_module.youyun_gpu_specs(region="cn-wlcb", instanceMode="pod"))

        capability = next(item for item in result["regionCapabilities"] if item["region"] == "cn-wlcb")
        self.assertEqual(capability["modes"], {"pod": "cn-wlcb-03", "uhost": "cn-wlcb-01"})

    def test_capacity_error_distinguishes_missing_gpu_from_stock_shortage(self) -> None:
        missing_gpu = SimpleNamespace(ret_code=230, message="Params [GpuType] not available")
        error = app_module._youyun_capacity_http_error(
            missing_gpu, gpu_type="4090", zone="cn-sh2-01"
        )
        self.assertEqual(error.status_code, 422)
        self.assertIn("不提供 GPU 4090", error.detail)

        zone_machine_missing = SimpleNamespace(ret_code=230, message="Params [GpuType] not available machine type not configured for zone")
        error = app_module._youyun_capacity_http_error(
            zone_machine_missing, gpu_type="3080Ti", zone="cn-wlcb-03"
        )
        self.assertEqual(error.status_code, 422)

        provider_failure = SimpleNamespace(ret_code=500, message="temporary failure")
        error = app_module._youyun_capacity_http_error(
            provider_failure, gpu_type="4090", zone="cn-sh2-01"
        )
        self.assertEqual(error.status_code, 502)

    def test_resource_selection_preserves_explicit_region_and_zone(self) -> None:
        runner = self._runner(zones=[{"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True}])
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(
                app_module.resolve_youyun_resource(
                    app_module.YouyunResourceSelection(
                        region="cn-bj2",
                        zone="cn-bj2-03",
                        instanceMode="pod",
                        cpuCores=16,
                        memoryGb=64,
                    )
                )
            )
        self.assertEqual(result["region"], "cn-bj2")
        self.assertEqual(result["zone"], "cn-bj2-03")
        self.assertEqual(result["instanceMode"], "pod")

    def test_resource_selection_self_heals_zone_shaped_region(self) -> None:
        # 前端状态串位时可能把可用区代码提交为 region（region=cn-bj2-03），
        # 与另一个区域的可用区（zone=cn-wlcb-03）组合。后端必须按 zone 归属
        # 自动纠错，而不是以“可用区与所选区域不一致”422 卡死任务提交。
        zones = [
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
        ]
        runner = self._runner(zones=zones)
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(
                app_module.resolve_youyun_resource(
                    app_module.YouyunResourceSelection(
                        region="cn-bj2-03",
                        zone="cn-wlcb-03",
                        instanceMode="pod",
                        cpuCores=16,
                        memoryGb=64,
                    )
                )
            )
        self.assertEqual(result["region"], "cn-wlcb")
        self.assertEqual(result["zone"], "cn-wlcb-03")

    def test_gpu_specs_uses_config_region_and_container_mode_when_selection_is_empty(self) -> None:
        """Initial load keeps the region but must not put a container image on UHost."""
        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-01", instance_mode="uhost", pod_zone="",
            )

            def describe_support_zones(self, region=None):
                return [
                    {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False, "describe": "North China 2A"},
                    {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "North China 2C"},
                ]

            def describe_instance_types(self, *, instance_type="uhost"):
                return [SimpleNamespace(name="4090", zone="cn-wlcb-03", sold_out=False, description="", graphics_memory_gb=24)]

            def check_resource_capacity(self, **kwargs):
                return SimpleNamespace(
                    specs=[SimpleNamespace(gpu=1, cpu=16, memory_gb=64, resource_enough=True)]
                )

        runner = SimpleNamespace(manager=SimpleNamespace(client=_Client()))
        app_module._YOUYUN_DISCOVERY_CACHE.clear()
        app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()
        try:
            with patch.object(app_module, "build_youyun_runner", return_value=runner):
                result = asyncio.run(
                    app_module.youyun_gpu_specs(instanceMode="pod")
                )
        finally:
            app_module._YOUYUN_DISCOVERY_CACHE.clear()
            app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()

        self.assertEqual(result["region"], "cn-wlcb")
        self.assertEqual(result["zone"], "cn-wlcb-03")
        self.assertEqual(result["instanceMode"], "pod")
        self.assertIs(result["container"], True)
        self.assertEqual(result["gpuType"], "4090")

    def test_gpu_specs_self_heals_zone_shaped_region(self) -> None:
        zones = [
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True, "describe": "华北一C"},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "华北二C"},
        ]

        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-03", instance_mode="pod", pod_zone="cn-wlcb-03",
            )

            def describe_support_zones(self, region=None):
                return zones

            def describe_instance_types(self, *, instance_type="uhost"):
                return [SimpleNamespace(name="4090", zone="cn-wlcb-03", sold_out=False, description="", graphics_memory_gb=24)]

            def check_resource_capacity(self, **kwargs):
                return SimpleNamespace(
                    specs=[SimpleNamespace(gpu=1, cpu=16, memory_gb=64, resource_enough=True)]
                )

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        app_module._YOUYUN_DISCOVERY_CACHE.clear()
        app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()
        try:
            with patch.object(app_module, "build_youyun_runner", return_value=runner):
                result = asyncio.run(
                    app_module.youyun_gpu_specs(region="cn-bj2-03", zone="cn-wlcb-03", instanceMode="pod")
                )
        finally:
            app_module._YOUYUN_DISCOVERY_CACHE.clear()
            app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()
        self.assertEqual(result["region"], "cn-wlcb")
        self.assertEqual(result["zone"], "cn-wlcb-03")
        self.assertEqual(result["gpuType"], "4090")

    def test_gpu_specs_can_list_pod_catalog_before_gpu_selection(self) -> None:
        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-01", instance_mode="pod", pod_zone="cn-wlcb-03",
            )
            instance_type = None
            capacity_checked = False

            def describe_support_zones(self, region=None):
                return [{"region": "cn-sh2", "zone": "cn-sh2-02", "is_pod": True, "describe": ""}]

            def describe_instance_types(self, *, instance_type="uhost"):
                self.instance_type = instance_type
                return [SimpleNamespace(name="4090", zone="cn-sh2-02", sold_out=False, description="", graphics_memory_gb=24)]

            def check_resource_capacity(self, **kwargs):
                self.capacity_checked = True
                self.capacity_kwargs = kwargs
                return SimpleNamespace(
                    specs=[SimpleNamespace(gpu=1, cpu=16, memory_gb=64, resource_enough=True)]
                )

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        def build_runner(*, region=None, zone=None):
            client.config.region = region or client.config.region
            client.config.zone = zone or client.config.zone
            return runner

        with patch.object(app_module, "build_youyun_runner", side_effect=build_runner):
            result = asyncio.run(app_module.youyun_gpu_specs(region="cn-sh2", zone="cn-sh2-02", instanceMode="pod"))
        self.assertEqual(result["region"], "cn-sh2")
        self.assertEqual(result["instanceMode"], "pod")
        self.assertEqual(client.instance_type, "uhost")
        self.assertTrue(client.capacity_checked)
        self.assertEqual(client.capacity_kwargs["gpu_type"], "4090")
        self.assertEqual(client.capacity_kwargs["zone"], "cn-sh2-02")
        self.assertEqual(result["gpuType"], "4090")
        self.assertEqual(result["capacity"], [{"gpu": 1, "cpu": 16, "memoryGb": 64, "available": True}])

    def test_gpu_specs_falls_back_when_region_default_gpu_is_not_in_pod_zone(self) -> None:
        from backend.providers.youyun.models import YouyunApiError

        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-03", instance_mode="pod", pod_zone="cn-wlcb-03",
            )
            capacity_gpu_types = []

            def describe_support_zones(self, region=None):
                return [{"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "华北二C"}]

            def describe_instance_types(self, *, instance_type="uhost"):
                return [
                    SimpleNamespace(name="3080Ti", zone="cn-wlcb-02", sold_out=False, description="", graphics_memory_gb=12),
                    SimpleNamespace(name="4090", zone="cn-wlcb-03", sold_out=False, description="", graphics_memory_gb=24),
                ]

            def check_resource_capacity(self, **kwargs):
                self.capacity_gpu_types.append(kwargs["gpu_type"])
                if kwargs["gpu_type"] == "3080Ti":
                    raise YouyunApiError("CheckCompShareResourceCapacity", 230, "Params [GpuType] not available")
                return SimpleNamespace(specs=[SimpleNamespace(gpu=1, cpu=16, memory_gb=64, resource_enough=True)])

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        app_module._YOUYUN_DISCOVERY_CACHE.clear()
        app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()
        try:
            with patch.object(app_module, "build_youyun_runner", return_value=runner):
                result = asyncio.run(
                    app_module.youyun_gpu_specs(region="cn-wlcb", zone="cn-wlcb-03", instanceMode="pod")
                )
        finally:
            app_module._YOUYUN_DISCOVERY_CACHE.clear()
            app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()

        self.assertEqual(result["gpuType"], "4090")
        self.assertEqual(client.capacity_gpu_types, ["4090"])
        self.assertTrue(result["capacity"][0]["available"])

    def test_gpu_specs_reuses_discovery_snapshot_but_refreshes_capacity(self) -> None:
        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-03", instance_mode="pod", pod_zone="cn-wlcb-03",
                api_base_url="https://api.example", public_key="test-key",
            )
            calls = {"zones": 0, "images": 0, "types": 0, "capacity": 0}

            def describe_support_zones(self, region=None):
                self.calls["zones"] += 1
                return [{"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "华北二C"}]

            def describe_images(self, **kwargs):
                self.calls["images"] += 1
                return []

            def describe_instance_types(self, *, instance_type="uhost"):
                self.calls["types"] += 1
                return [SimpleNamespace(name="4090", zone="cn-wlcb-03", sold_out=False, description="", graphics_memory_gb=24)]

            def check_resource_capacity(self, **kwargs):
                self.calls["capacity"] += 1
                return SimpleNamespace(specs=[SimpleNamespace(gpu=1, cpu=16, memory_gb=64, resource_enough=True)])

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        app_module._YOUYUN_DISCOVERY_CACHE.clear()
        app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()
        try:
            with patch.object(app_module, "build_youyun_runner", return_value=runner):
                first = asyncio.run(
                    app_module.youyun_gpu_specs(region="cn-wlcb", zone="cn-wlcb-03", instanceMode="pod")
                )
                second = asyncio.run(
                    app_module.youyun_gpu_specs(
                        gpuType="4090", region="cn-wlcb", zone="cn-wlcb-03", instanceMode="pod"
                    )
                )
        finally:
            app_module._YOUYUN_DISCOVERY_CACHE.clear()
            app_module._YOUYUN_DISCOVERY_INFLIGHT.clear()

        self.assertEqual(first["gpuType"], "4090")
        self.assertEqual(second["gpuType"], "4090")
        self.assertEqual(client.calls, {"zones": 1, "images": 1, "types": 1, "capacity": 2})

    def test_gpu_specs_filters_to_actual_zone_and_marks_current_stock(self) -> None:
        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-03", instance_mode="pod", pod_zone="cn-wlcb-03",
            )
            probed: list[str] = []
            active_probes = 0
            max_active_probes = 0
            probe_lock = threading.Lock()

            def describe_support_zones(self, region=None):
                return [
                    {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "华北二C"},
                    {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False, "describe": "华北二A"},
                ]

            def describe_instance_types(self, *, instance_type="uhost"):
                return [
                    # Region response includes unrelated families that must not
                    # be shown for the concrete cn-wlcb-03 placement Zone.
                    SimpleNamespace(name="3080Ti", zone="cn-wlcb-01", sold_out=False, description="Ampere", graphics_memory_gb=12),
                    SimpleNamespace(name="3090", zone="cn-wlcb-01", sold_out=False, description="Ampere", graphics_memory_gb=24),
                    SimpleNamespace(name="4090", zone="cn-wlcb-03", sold_out=False, description="Ada", graphics_memory_gb=24),
                    SimpleNamespace(name="5090", zone="cn-wlcb-03", sold_out=False, description="Blackwell", graphics_memory_gb=32),
                    SimpleNamespace(name="A100", zone="", sold_out=False, description="Ampere", graphics_memory_gb=80),
                ]

            def check_resource_capacity(self, **kwargs):
                self.probed.append(kwargs["gpu_type"])
                with self.probe_lock:
                    self.active_probes += 1
                    self.max_active_probes = max(self.max_active_probes, self.active_probes)
                time.sleep(0.03)
                with self.probe_lock:
                    self.active_probes -= 1
                if kwargs["gpu_type"] == "4090":
                    return SimpleNamespace(specs=[SimpleNamespace(gpu=1, cpu=14, memory_gb=64, resource_enough=True)])
                return SimpleNamespace(specs=[SimpleNamespace(gpu=1, cpu=14, memory_gb=64, resource_enough=False)])

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            result = asyncio.run(
                app_module.youyun_gpu_specs(region="cn-wlcb", zone="cn-wlcb-03", instanceMode="pod")
            )

        catalog = {item["name"]: item for item in result["machineTypes"]}
        self.assertEqual(set(catalog), {"4090", "5090"})
        self.assertEqual(catalog["4090"]["inventoryStatus"], "available")
        self.assertFalse(catalog["4090"]["soldOut"])
        self.assertEqual(catalog["5090"]["inventoryStatus"], "sold_out")
        self.assertTrue(catalog["5090"]["soldOut"])
        self.assertEqual(result["gpuType"], "4090")
        self.assertEqual(result["capacity"], [{"gpu": 1, "cpu": 14, "memoryGb": 64, "available": True}])
        self.assertCountEqual(client.probed, ["4090", "5090"])
        self.assertEqual(client.max_active_probes, 2)

    def test_gpu_specs_remaps_incompatible_placement_zone_before_capacity(self) -> None:
        class _Client:
            config = SimpleNamespace(
                training_image_id="train-img", image_id="", gpu_type="4090", charge_type="Postpay",
                region="cn-wlcb", zone="cn-wlcb-01", instance_mode="pod", pod_zone="cn-wlcb-03",
            )
            capacity_kwargs = None

            def describe_support_zones(self, region=None):
                return [
                    {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True, "describe": "华北二C"},
                    {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False, "describe": "华北二A"},
                ]

            def describe_instance_types(self, *, instance_type="uhost"):
                return [SimpleNamespace(name="3080Ti", zone="cn-wlcb-02", sold_out=False, description="", graphics_memory_gb=12)]

            def check_resource_capacity(self, **kwargs):
                self.capacity_kwargs = kwargs
                return SimpleNamespace(specs=[])

        client = _Client()
        runner = SimpleNamespace(manager=SimpleNamespace(client=client))

        def build_runner(*, region=None, zone=None):
            client.config.region = region or client.config.region
            client.config.zone = zone or client.config.zone
            return runner

        with patch.object(app_module, "build_youyun_runner", side_effect=build_runner):
            result = asyncio.run(
                app_module.youyun_gpu_specs(
                    gpuType="3080Ti", region="cn-wlcb", zone="cn-wlcb-01", instanceMode="pod"
                )
            )

        self.assertEqual(result["zone"], "cn-wlcb-03")
        self.assertEqual(result["instanceMode"], "pod")
        self.assertEqual(client.capacity_kwargs["zone"], "cn-wlcb-03")
        self.assertEqual(client.capacity_kwargs["gpu_type"], "3080Ti")

    def test_incompatible_exact_zone_is_rejected(self) -> None:
        runner = self._runner(zones=[{"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False}])
        with patch.object(app_module, "build_youyun_runner", return_value=runner):
            with self.assertRaises(app_module.HTTPException) as caught:
                asyncio.run(
                    app_module.resolve_youyun_resource(
                        app_module.YouyunResourceSelection(
                            instanceMode="pod", zone="cn-wlcb-01", cpuCores=16, memoryGb=64,
                        )
                    )
                )
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("Pod", caught.exception.detail)

    def test_task_create_validates_nested_youyun_resource(self) -> None:
        payload = {
            "name": "远程优云测试",
            "executionMode": "remote",
            "provider": "youyun",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "datasetId": "dataset-1",
            "datasetName": "数据集",
            "outputName": "remote-test",
            "parameters": {
                "epochs": 1,
                "learningRate": "1e-4",
                "batchSize": 1,
                "gradientAccumulation": 1,
                "validationSteps": 10,
                "maxSequenceLength": 512,
                "loraRank": 8,
                "loraAlpha": 16,
                "loraDropout": 0.1,
                "lrScheduler": "cosine",
                "loraPlusRatio": 1,
                "targetModules": "all",
                "optimizer": "adamw_torch",
                "warmupRatio": 0.03,
                "weightDecay": 0,
                "maxGradNorm": 1,
                "seed": 42,
                "loggingSteps": 1,
                "packing": False,
                "gradientCheckpointing": True,
            },
            "youyunResource": {
                "mode": "manual",
                "gpuType": "4090",
                "gpuCount": 1,
                "cpuCores": 16,
                "memoryGb": 64,
            },
        }
        task = app_module.TaskCreate.model_validate(payload)
        self.assertEqual(task.youyunResource.cpuCores, 16)
        self.assertEqual(task.youyunResource.memoryGb, 64)


class GpuInfoTest(unittest.TestCase):
    def test_missing_nvidia_smi_uses_friendly_cpu_node_message(self) -> None:
        missing = FileNotFoundError(
            2, "No such file or directory", "nvidia-smi"
        )
        with patch("backend.app.subprocess.run", side_effect=missing):
            info = app_module.probe_gpu_info()

        self.assertEqual(
            info,
            {
                "available": False,
                "reason": "当前节点未检测到可用 GPU",
            },
        )

    def test_other_gpu_probe_errors_keep_diagnostic_detail(self) -> None:
        with patch(
            "backend.app.subprocess.run",
            side_effect=PermissionError("permission denied"),
        ):
            info = app_module.probe_gpu_info()

        self.assertFalse(info["available"])
        self.assertEqual(info["reason"], "GPU 状态读取失败：permission denied")


class CliTaskApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_cli_submission_is_parsed_into_a_queued_task(self) -> None:
        task = await app_module.create_cli_task(
            app_module.CliTaskCreate(
                command=(
                    "python scripts/train_cli.py submit --name 'api cli check' "
                    "--model Qwen/Qwen3-1.7B --dataset-id qwenruozhi_v1 "
                    "--dataset-name '平台演示 SFT v1' --output-name check "
                    "--validation-split 0 --early-stopping --trainer-param tf32=false"
                )
            )
        )
        self.assertEqual(task["state"], "queued")
        self.assertEqual(task["validationSplit"], 0)
        self.assertTrue(task["earlyStopping"])
        self.assertFalse(task["extraParameters"]["tf32"])
        self.assertEqual(task["dataset"], "平台演示 SFT v1")

    async def test_cli_submission_rejects_shell_syntax(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await app_module.create_cli_task(
                app_module.CliTaskCreate(
                    command=(
                        "python scripts/train_cli.py submit --name x --model m "
                        "--dataset-id qwenruozhi_v1 --dataset-name n "
                        "--output-name o; whoami"
                    )
                )
            )
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("Shell", str(raised.exception.detail))

    async def test_cli_submission_rejects_unknown_dataset_id(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await app_module.create_cli_task(
                app_module.CliTaskCreate(
                    command=(
                        "submit --name x --model m --dataset-id missing-dataset "
                        "--dataset-name n --output-name o"
                    )
                )
            )
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("数据集不存在", str(raised.exception.detail))

    async def test_cli_submission_rejects_unpublished_dataset(self) -> None:
        app_module.store.upsert_dataset(
            {
                "id": "ready-dataset",
                "name": "待发布数据集",
                "type": "文本生成",
                "structure": "SFT",
                "state": "ready",
                "sampleCount": 1,
                "paths": {"trainPath": str(TEST_ROOT / "ready.jsonl")},
                "owner": "tester",
            }
        )
        with self.assertRaises(HTTPException) as raised:
            await app_module.create_cli_task(
                app_module.CliTaskCreate(
                    command=(
                        "submit --name x --model m --dataset-id ready-dataset "
                        "--dataset-name n --output-name o"
                    )
                )
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("已发布", str(raised.exception.detail))


if __name__ == "__main__":
    unittest.main()
