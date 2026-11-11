from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import chat_inference


class ChatInferenceLoaderTest(unittest.TestCase):
    def test_modelscope_failure_falls_back_to_same_huggingface_id(self):
        ms = Mock(side_effect=RuntimeError("not found"))
        hf = Mock(return_value="/remote/hf/original")
        with patch.dict(sys.modules, {"modelscope": SimpleNamespace(snapshot_download=ms), "huggingface_hub": SimpleNamespace(snapshot_download=hf)}):
            self.assertEqual(chat_inference.resolve_model("org/original", "modelscope"), "/remote/hf/original")
        hf.assert_called_once_with(repo_id="org/original")

    def test_cpu_without_unsloth_uses_transformers_fallback(self) -> None:
        torch = Mock()
        torch.cuda.is_available.return_value = False
        expected = (object(), object(), object(), "transformers")
        with patch.object(chat_inference, "_load_model_with_transformers", return_value=expected) as fallback:
            result = chat_inference._load_chat_model(
                "Qwen/Qwen2.5-0.5B-Instruct",
                "modelscope",
                "/workspace/training-work/task/output",
                True,
                torch,
            )

        self.assertIs(result, expected)
        fallback.assert_called_once_with(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "modelscope",
            "/workspace/training-work/task/output",
            torch,
        )

    def test_chat_template_falls_back_for_old_tokenizer(self) -> None:
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = [
            TypeError("unexpected keyword argument 'enable_thinking'"),
            "prompt",
        ]

        result = chat_inference._apply_chat_template(
            tokenizer, [{"role": "user", "content": "hello"}]
        )

        self.assertEqual(result, "prompt")
        self.assertEqual(tokenizer.apply_chat_template.call_count, 2)


if __name__ == "__main__":
    unittest.main()
