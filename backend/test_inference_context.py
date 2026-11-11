from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from backend.inference_service import (
    InferenceChatMessage,
    InferenceChatRequest,
    InferenceCreate,
    InferenceRuntime,
    InferenceStore,
    read_model_context,
    read_task_model_context,
)


class _TaskStore:
    def __init__(self, task: dict) -> None:
        self.task = task

    def get(self, task_id: str) -> dict:
        return self.task if self.task["id"] == task_id else None  # type: ignore[return-value]


class InferenceContextTests(unittest.TestCase):
    def test_reads_model_position_limit_before_tokenizer_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                '{"max_position_embeddings": 8192}', encoding="utf-8"
            )
            (root / "tokenizer_config.json").write_text(
                '{"model_max_length": 1000000000000000019884624838656}',
                encoding="utf-8",
            )
            metadata = read_model_context(root)
            self.assertEqual(metadata["maxContextLength"], 8192)
            self.assertEqual(metadata["source"], "config.json:max_position_embeddings")

    def test_chat_request_uses_service_context_as_output_limit(self) -> None:
        request = InferenceChatRequest(
            messages=[InferenceChatMessage(role="user", content="hello")],
            maxTokens=1024,
        )
        body = InferenceRuntime._chat_body(
            {
                "id": "svc-1",
                "config": {"maxTokens": 512, "modelContextLength": 4096},
            },
            request,
        )
        self.assertEqual(body["max_tokens"], 1024)

        service_cap = request.model_copy(update={"maxTokens": 2048})
        with self.assertRaisesRegex(ValueError, "服务上限 1024"):
            InferenceRuntime._chat_body(
                {
                    "id": "svc-1",
                    "config": {"modelContextLength": 4096, "maxModelLength": 1024},
                },
                service_cap,
            )

        too_large = request.model_copy(update={"maxTokens": 4097})
        with self.assertRaisesRegex(ValueError, "服务上限 4096"):
            InferenceRuntime._chat_body(
                {"id": "svc-1", "config": {"modelContextLength": 4096}},
                too_large,
            )

    def test_create_rejects_context_longer_than_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                '{"max_position_embeddings": 2048}', encoding="utf-8"
            )
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "inference.db"),
                _TaskStore(
                    {
                        "id": "ft-context",
                        "state": "completed",
                        "model": str(model),
                        "artifacts": [
                            {"type": "adapter", "variant": "best", "path": str(adapter)}
                        ],
                    }
                ),
                port=8011,
                training_busy=lambda: False,
            )
            with self.assertRaisesRegex(ValueError, "不能超过基模支持"):
                asyncio.run(
                    runtime.create(
                        InferenceCreate(
                            name="context",
                            taskId="ft-context",
                            maxModelLength=4096,
                        )
                    )
                )

    def test_create_does_not_persist_request_scoped_generation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                '{"max_position_embeddings": 4096}', encoding="utf-8"
            )
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            runtime = InferenceRuntime(
                InferenceStore(Path(directory) / "inference.db"),
                _TaskStore(
                    {
                        "id": "ft-generation",
                        "state": "completed",
                        "model": str(model),
                        "artifacts": [
                            {"type": "adapter", "variant": "best", "path": str(adapter)}
                        ],
                    }
                ),
                port=8011,
                training_busy=lambda: False,
            )
            service = asyncio.run(
                runtime.create(
                    InferenceCreate(
                        name="generation",
                        taskId="ft-generation",
                        maxTokens=1024,
                        temperature=0.4,
                        topP=0.8,
                        stream=False,
                    )
                )
            )
            for key in ("maxTokens", "temperature", "topP", "stream"):
                self.assertNotIn(key, service["config"])

    def test_task_metadata_is_preferred_for_remote_base_model(self) -> None:
        metadata = read_task_model_context(
            {
                "id": "ft-remote",
                "model": "Qwen/remote-model",
                "baseModelMetadata": {"maxContextLength": 32768},
            },
            "Qwen/remote-model",
        )
        self.assertEqual(metadata["maxContextLength"], 32768)
        self.assertEqual(metadata["source"], "task:baseModelMetadata")
        self.assertFalse(metadata["isFallback"])

    def test_adapter_config_can_trace_base_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            base = root / "base"
            base.mkdir()
            (adapter / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "base-ref"}', encoding="utf-8"
            )
            (base / "config.json").write_text(
                '{"max_position_embeddings": 16384}', encoding="utf-8"
            )
            metadata = read_task_model_context(
                {"id": "ft", "model": "unknown", "artifacts": [{"type": "adapter", "path": str(adapter)}]},
                "unknown",
                adapter,
                resolve_model=lambda value: str(base) if value == "base-ref" else value,
            )
            self.assertEqual(metadata["maxContextLength"], 16384)
            self.assertEqual(metadata["source"], "config.json:max_position_embeddings")

    def test_task_logs_recover_context_when_remote_path_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "/root/remote-cache/model"}',
                encoding="utf-8",
            )
            metadata = read_task_model_context(
                {
                    "id": "ft-legacy-remote",
                    "model": "Qwen/legacy-model",
                    "artifacts": [{"type": "adapter", "path": str(adapter)}],
                    "logs": ['"max_position_embeddings": 32768,'],
                },
                "Qwen/legacy-model",
                adapter,
                resolve_model=lambda value: (_ for _ in ()).throw(
                    PermissionError(f"unreadable: {value}")
                ),
            )
            self.assertEqual(metadata["maxContextLength"], 32768)
            self.assertEqual(metadata["source"], "task:logs:max_position_embeddings")
            self.assertFalse(metadata["isFallback"])


if __name__ == "__main__":
    unittest.main()
