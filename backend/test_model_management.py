from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from backend.test_app_cli import app_module


class ModelManagementTests(unittest.TestCase):
    def test_gpu_platform_support_blocks_v100s_only_for_inference(self):
        support = app_module._youyun_gpu_platform_support("V100S")
        self.assertTrue(support["trainingSupported"])
        self.assertFalse(support["inferenceSupported"])
        self.assertIn("计算能力 7.0", support["supportReason"])

    def test_model_asset_reads_reuse_a_recent_catalog_sync(self):
        store = Mock()
        store.list_model_assets.return_value = []
        with (
            patch.object(app_module, "store", store),
            patch.object(app_module, "model_uploader", None),
            patch.object(app_module, "model_storage_error", "not configured"),
            patch.object(app_module, "_MODEL_ASSET_SYNCED_AT", 0.0),
            patch.object(app_module, "_MODEL_ASSET_SYNCED_STORE_ID", None),
        ):
            asyncio.run(app_module.list_model_assets(page=None))
            asyncio.run(app_module.list_model_assets(page=None))
        store.sync_model_assets.assert_called_once_with(app_module.MODEL_CACHE_ROOT)

    def test_public_metadata_formats_seconds_milliseconds_and_missing_values(self):
        models = [
            {"id": "provider-1", "name": "Qwen", "path": "/models/qwen", "tag": "LLM", "size": None, "createTime": 1704067200},
            {"name": "Qwen2", "path": "/models/qwen2", "size": "  ", "createTime": "1704067200000"},
        ]
        client = SimpleNamespace(describe_model_repository_models=Mock(return_value=models))
        with patch.object(app_module, "build_youyun_runner", return_value=SimpleNamespace(manager=SimpleNamespace(client=client))):
            result = asyncio.run(app_module.list_public_models(q="", page=1, pageSize=5))
        for item in result["items"]:
            self.assertEqual(item["size"], "暂无数据")
            self.assertEqual(item["updatedAt"], "2024-01-01 08:00:00")
        self.assertEqual(result["items"][0]["publicModelId"], "provider-1")
        self.assertEqual(result["items"][0]["tag"], "LLM")
        self.assertIsNone(result["items"][1]["publicModelId"])
        for value in (None, "invalid", "nan", "inf", ""):
            self.assertEqual(app_module._public_model_updated_at(value), "暂无数据")
        self.assertEqual(app_module._public_model_updated_at("2024-01-01T00:00:00Z"), "2024-01-01 08:00:00")
        self.assertEqual(app_module._public_model_size({"size": "", "sizeBytes": 1024 ** 3}), "1.0 GB")

    def test_local_binary_preview_has_checksum_without_content_and_blocks_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            adapter = root / "adapter"
            adapter.mkdir()
            raw = b"adapter-binary"
            (adapter / "adapter_model.safetensors").write_bytes(raw)
            (root / "outside.json").write_text("{}", encoding="utf-8")
            with patch.object(app_module, "_artifact_preview_roots", return_value=(root,)):
                result = asyncio.run(app_module.preview_artifact(app_module.ArtifactPreviewRequest(path=str(adapter), relativePath="adapter_model.safetensors")))
                self.assertEqual(result["kind"], "binary")
                self.assertEqual(result["md5"], hashlib.md5(raw).hexdigest())
                self.assertEqual(result["sizeBytes"], len(raw))
                self.assertEqual(result["status"], "available")
                self.assertNotIn("content", result)
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(app_module.preview_artifact(app_module.ArtifactPreviewRequest(path=str(adapter), relativePath="../outside.json")))
                self.assertEqual(error.exception.status_code, 403)

    def test_local_text_preview_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "config.json").write_text("a" * (512 * 1024 + 30), encoding="utf-8")
            with patch.object(app_module, "_artifact_preview_roots", return_value=(root,)):
                result = asyncio.run(app_module.preview_artifact(app_module.ArtifactPreviewRequest(path=str(root), relativePath="config.json")))
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["content"]), 512 * 1024)

    def test_retry_upload_uses_selected_variant_and_retains_errors(self):
        store = Mock()
        store.get_model_asset.return_value = {"kind": "trained", "taskId": "task-1", "variants": [{"key": "best"}]}
        uploader = AsyncMock(return_value=[])
        with patch.object(app_module, "store", store), patch.object(app_module, "upload_completed_task_artifacts", uploader):
            result = asyncio.run(app_module.upload_model_asset("trained-task-1", "best"))
            self.assertEqual(result["status"], "available")
            uploader.assert_awaited_once_with("task-1", "best")
            uploader.return_value = ["upload interrupted"]
            with self.assertRaises(HTTPException) as error:
                asyncio.run(app_module.upload_model_asset("trained-task-1", "best"))
            self.assertEqual(error.exception.status_code, 502)
            self.assertIn("upload interrupted", error.exception.detail)
            self.assertNotIn("task-1", app_module._adapter_uploads)

    def test_local_completeness_and_backup_status_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            asset = {"kind": "trained", "taskId": "task-1", "variants": [{"key": "best", "path": str(root), "status": "available"}]}
            store = Mock()
            store.list_model_assets.return_value = [asset]
            uploader = Mock()
            uploader.metadata.list.return_value = [{"id": "stored-1", "source_task_id": "task-1", "artifact_variant": "best", "status": "available"}]
            with patch.object(app_module, "store", store), patch.object(app_module, "sync_model_assets"), patch.object(app_module, "model_uploader", uploader):
                result = asyncio.run(app_module.list_model_assets(page=None))
                self.assertEqual(result[0]["variants"][0]["status"], "unavailable")
                self.assertEqual(result[0]["variants"][0]["sharedStorage"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
