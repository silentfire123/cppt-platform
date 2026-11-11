from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from backend.model_storage import ModelArtifactUploader, ModelStorageConfig, _manifest_md5, _md5, adapter_is_complete, iter_model_files


class ModelStorageFileTests(unittest.TestCase):
    def test_model_file_filter_keeps_loadable_files_and_skips_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint-10" / "optimizer.pt").parent.mkdir()
            (root / "checkpoint-10" / "optimizer.pt").write_bytes(b"state")
            (root / "trainer_state.json").write_text("{}", encoding="utf-8")
            (root / "README.md").write_text("ignore", encoding="utf-8")

            files = iter_model_files(root, "base_model")

            self.assertEqual(
                [relative for _, relative in files],
                ["config.json", "model.safetensors", "tokenizer.json"],
            )

    def test_upload_directory_uses_model_object_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"adapter")

            config = ModelStorageConfig(
                postgres_dsn="postgresql://test",
                minio_endpoint="minio:9000",
                minio_access_key="access",
                minio_secret_key="secret",
                minio_bucket="rag-assets",
                minio_secure=False,
            )
            metadata = Mock()
            metadata.find_source.return_value = None
            metadata.get.return_value = {"id": "model-fixed"}
            metadata.list_entries.return_value = []
            objects = Mock()

            with patch("backend.model_storage.PostgresModelMetadataStore", return_value=metadata), patch(
                "backend.model_storage.MinioModelObjectStore", return_value=objects
            ):
                uploader = ModelArtifactUploader(config)
                result = uploader.upload_directory(
                    root,
                    name="adapter",
                    artifact_type="lora_adapter",
                    source_task_id="task-fixed",
                )

            create_metadata = metadata.create_pending.call_args.args[0]
            entries = metadata.create_pending.call_args.args[1]
            expected_prefix = f"model/{create_metadata['id']}"
            self.assertEqual(result["id"], "model-fixed")
            self.assertEqual(create_metadata["minio_object_prefix"], expected_prefix)
            self.assertEqual(entries[0]["minio_object_key"], f"{expected_prefix}/adapter_model.safetensors")
            objects.upload.assert_called_once()

    def test_md5_and_manifest_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            path.write_bytes(b"adapter")
            digest, size = _md5(path)
            self.assertEqual(digest, hashlib.md5(b"adapter").hexdigest())
            self.assertEqual(size, 7)
            manifest = _manifest_md5(
                [
                    {
                        "relative_path": "adapter_model.safetensors",
                        "size_bytes": size,
                        "md5": digest,
                    }
                ]
            )
            self.assertEqual(len(manifest), 32)

    def test_preview_listing_returns_registered_files(self) -> None:
        config = ModelStorageConfig(
            postgres_dsn="postgresql://test",
            minio_endpoint="minio:9000",
            minio_access_key="access",
            minio_secret_key="secret",
            minio_bucket="rag-assets",
            minio_secure=False,
        )
        metadata = Mock()
        metadata.get.return_value = {
            "id": "model-fixed",
            "name": "adapter",
            "status": "available",
            "artifact_type": "lora_adapter",
            "artifact_variant": "best",
            "minio_bucket": "rag-assets",
            "minio_object_prefix": "model/model-fixed",
        }
        metadata.list_entries.return_value = [
            {
                "relative_path": "adapter_config.json",
                "size_bytes": 12,
                "md5": "abc",
                "status": "available",
                "minio_object_key": "model/model-fixed/adapter_config.json",
            }
        ]
        with patch("backend.model_storage.PostgresModelMetadataStore", return_value=metadata), patch(
            "backend.model_storage.MinioModelObjectStore"
        ):
            uploader = ModelArtifactUploader(config)
            result = uploader.preview_listing("model-fixed")

        self.assertEqual(result["objectPrefix"], "model/model-fixed")
        self.assertEqual(result["files"][0]["path"], "adapter_config.json")

    def test_binary_preview_returns_metadata_without_reading_object(self) -> None:
        config = ModelStorageConfig(
            postgres_dsn="postgresql://test",
            minio_endpoint="minio:9000",
            minio_access_key="access",
            minio_secret_key="secret",
            minio_bucket="rag-assets",
            minio_secure=False,
        )
        metadata = Mock()
        metadata.get.return_value = {"id": "model-fixed", "status": "available"}
        metadata.list_entries.return_value = [
            {
                "relative_path": "adapter_model.safetensors",
                "size_bytes": 42,
                "md5": "def",
                "status": "available",
                "minio_object_key": "model/model-fixed/adapter_model.safetensors",
            }
        ]
        objects = Mock()
        with patch("backend.model_storage.PostgresModelMetadataStore", return_value=metadata), patch(
            "backend.model_storage.MinioModelObjectStore", return_value=objects
        ):
            uploader = ModelArtifactUploader(config)
            result = uploader.preview_file("model-fixed", "adapter_model.safetensors")

        self.assertEqual(result["kind"], "binary")
        self.assertFalse(result["previewable"])
        objects.read.assert_not_called()

    def test_text_preview_is_truncated_at_limit(self) -> None:
        config = ModelStorageConfig(
            postgres_dsn="postgresql://test",
            minio_endpoint="minio:9000",
            minio_access_key="access",
            minio_secret_key="secret",
            minio_bucket="rag-assets",
            minio_secure=False,
        )
        metadata = Mock()
        metadata.get.return_value = {"id": "model-fixed", "status": "available"}
        metadata.list_entries.return_value = [
            {
                "relative_path": "config.json",
                "size_bytes": 10,
                "md5": "ghi",
                "status": "available",
                "minio_object_key": "model/model-fixed/config.json",
            }
        ]
        objects = Mock()
        objects.read.return_value = b"abcdef"
        with patch("backend.model_storage.PostgresModelMetadataStore", return_value=metadata), patch(
            "backend.model_storage.MinioModelObjectStore", return_value=objects
        ):
            uploader = ModelArtifactUploader(config)
            result = uploader.preview_file("model-fixed", "config.json", limit=5)

        self.assertEqual(result["kind"], "text")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["content"], "abcde")


class ModelStorageRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"
        self.contents = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"adapter"}
        self.entries = [
            {"relative_path": name, "minio_object_key": f"model/model-fixed/{name}", "size_bytes": len(data),
             "md5": hashlib.md5(data).hexdigest(), "status": "available"}
            for name, data in sorted(self.contents.items())
        ]
        self.record = {
            "id": "model-fixed", "status": "available", "minio_bucket": "rag-assets",
            "minio_object_prefix": "model/model-fixed", "manifest_md5": _manifest_md5(self.entries),
            "total_size_bytes": sum(len(data) for data in self.contents.values()),
        }
        config = ModelStorageConfig("postgresql://test", "minio:9000", "access", "secret", "rag-assets", False)
        self.metadata, self.objects = Mock(), Mock()
        self.metadata.find_source.return_value = self.record
        self.metadata.list_entries.return_value = self.entries
        self.objects.download.side_effect = lambda key, path, **kwargs: path.write_bytes(self.contents[key.removeprefix("model/model-fixed/")])
        with patch("backend.model_storage.PostgresModelMetadataStore", return_value=self.metadata), patch(
            "backend.model_storage.MinioModelObjectStore", return_value=self.objects,
        ):
            self.uploader = ModelArtifactUploader(config)

    def test_restore_verifies_files_and_reuses_cache_without_download(self) -> None:
        restored = Path(self.uploader.restore_adapter("task-1", "best", self.cache))
        self.assertTrue(adapter_is_complete(restored))
        self.assertTrue(restored.is_relative_to(self.cache.resolve()))
        for name, contents in self.contents.items():
            self.assertEqual((restored / name).read_bytes(), contents)
        self.objects.download.reset_mock()
        self.assertEqual(self.uploader.restore_adapter("task-1", "best", self.cache), str(restored))
        self.objects.download.assert_not_called()

    def test_corrupt_cache_is_verified_and_restored(self) -> None:
        restored = Path(self.uploader.restore_adapter("task-1", "final", self.cache))
        (restored / "adapter_model.safetensors").write_bytes(b"corrupt")
        self.objects.download.reset_mock()
        self.uploader.restore_adapter("task-1", "final", self.cache)
        self.assertEqual(self.objects.download.call_count, 2)
        self.assertEqual((restored / "adapter_model.safetensors").read_bytes(), b"adapter")

    def test_checksum_or_size_failure_never_publishes_partial_cache(self) -> None:
        for damaged in (b"corrupt", b"short"):
            with self.subTest(damaged=damaged):
                self.contents["adapter_model.safetensors"] = damaged
                with self.assertRaisesRegex(ValueError, "校验失败"):
                    self.uploader.restore_adapter("task-1", "best", self.cache)
                self.assertEqual(list(self.cache.iterdir()), [])

    def test_missing_object_never_publishes_partial_cache(self) -> None:
        self.objects.download.side_effect = FileNotFoundError("missing object")
        with self.assertRaises(FileNotFoundError):
            self.uploader.restore_adapter("task-1", "best", self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_unavailable_or_incomplete_metadata_is_rejected(self) -> None:
        for field, value in (("status", "uploading"), ("manifest_md5", "0" * 32), ("total_size_bytes", 999), ("minio_bucket", "other")):
            with self.subTest(field=field), patch.dict(self.record, {field: value}):
                with self.assertRaises(ValueError):
                    self.uploader.restore_adapter("task-1", "best", self.cache)
        self.entries[0]["status"] = "failed"
        with self.assertRaises(ValueError):
            self.uploader.restore_adapter("task-1", "best", self.cache)
        self.objects.download.assert_not_called()

    def test_manifest_paths_cannot_escape_cache_or_object_prefix(self) -> None:
        for unsafe in ("../outside", "/absolute", "C:/outside", "..\\outside", "a//b"):
            with self.subTest(unsafe=unsafe), patch.dict(self.entries[0], {"relative_path": unsafe}):
                with self.assertRaisesRegex(ValueError, "不安全"):
                    self.uploader.restore_adapter("task-1", "best", self.cache)
        self.entries[0]["minio_object_key"] = "other/config.json"
        with self.assertRaisesRegex(ValueError, "越界"):
            self.uploader.restore_adapter("task-1", "best", self.cache)
        self.objects.download.assert_not_called()

    def test_missing_adapter_configuration_is_rejected(self) -> None:
        self.entries.pop(0)
        self.record.update(manifest_md5=_manifest_md5(self.entries), total_size_bytes=7)
        with self.assertRaisesRegex(ValueError, "缺少完整"):
            self.uploader.restore_adapter("task-1", "best", self.cache)
        self.objects.download.assert_not_called()

    def test_complete_adapter_requires_config_and_all_index_shards(self) -> None:
        root = Path(self.directory.name)
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        (root / "adapter_model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}), encoding="utf-8")
        (root / "part-1.safetensors").write_bytes(b"one")
        self.assertFalse(adapter_is_complete(root))
        (root / "part-2.safetensors").write_bytes(b"two")
        self.assertTrue(adapter_is_complete(root))
        (root / "adapter_config.json").write_text("broken", encoding="utf-8")
        self.assertFalse(adapter_is_complete(root))

    def test_minio_only_adapter_reaches_inference_launch_as_verified_local_path(self) -> None:
        from backend.inference_service import InferenceCreate, InferenceRuntime, InferenceStore

        async def scenario() -> None:
            task = {"id": "ft-backup", "state": "completed", "model": "Qwen/Qwen2.5-0.5B-Instruct", "artifacts": []}
            runtime = InferenceRuntime(
                InferenceStore(Path(self.directory.name) / "services.db"), SimpleNamespace(get=lambda task_id: task),
                port=8011, training_busy=lambda: False,
                artifact_restorer=lambda task_id, variant: self.uploader.restore_adapter(task_id, variant, self.cache),
            )
            service = await runtime.create(InferenceCreate(name="MinIO recovery", taskId="ft-backup"))
            process = SimpleNamespace(returncode=None)
            with patch.object(runtime, "_validate_vllm_runtime", return_value="test"), patch.object(
                runtime, "_watch_process", new_callable=AsyncMock,
            ), patch.object(runtime, "_wait_until_ready", new_callable=AsyncMock), patch.object(
                runtime, "refresh_metrics", new_callable=AsyncMock,
            ), patch("backend.inference_service.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process) as launch:
                result = await runtime.start(service["id"])
                await asyncio.gather(*runtime.background_tasks)
            self.assertEqual(result["state"], "running")
            self.assertTrue(adapter_is_complete(Path(result["adapter"])))
            self.assertTrue(any(str(arg).endswith("=" + result["adapter"]) for arg in launch.call_args.args))
            self.assertEqual(self.objects.download.call_count, 2)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
