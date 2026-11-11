from __future__ import annotations

import unittest

from backend.train_sft import dpo_length_kwargs, normalize_dpo_batch


class NormalizeDpoBatchTest(unittest.TestCase):
    def test_dpo_length_uses_selected_final_sequence_length(self) -> None:
        self.assertEqual(
            dpo_length_kwargs(1024),
            {
                "max_prompt_length": 1024,
                "max_completion_length": 1024,
                "max_length": 1024,
            },
        )

    def test_conversational_shape_keeps_only_trl_preference_fields(self) -> None:
        result = normalize_dpo_batch(
            {
                "messages": [[{"role": "user", "content": "问题"}]],
                "prompt": [None],
                "chosen": [{"role": "assistant", "content": "好回答"}],
                "rejected": [{"role": "assistant", "content": "差回答"}],
            }
        )

        self.assertEqual(set(result), {"prompt", "chosen", "rejected"})
        self.assertEqual(result["prompt"][0][0]["content"], "问题")
        self.assertEqual(result["chosen"][0][0]["content"], "好回答")

    def test_legacy_string_shape_is_converted_to_messages(self) -> None:
        result = normalize_dpo_batch(
            {
                "messages": [None],
                "prompt": ["问题"],
                "chosen": ["好回答"],
                "rejected": ["差回答"],
            }
        )

        self.assertEqual(result["prompt"][0], [{"role": "user", "content": "问题"}])
        self.assertEqual(
            result["rejected"][0],
            [{"role": "assistant", "content": "差回答"}],
        )


if __name__ == "__main__":
    unittest.main()
