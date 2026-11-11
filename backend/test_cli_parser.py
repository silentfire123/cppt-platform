from __future__ import annotations

import unittest

from backend.cli_parser import CliValidationError, parse_cli_command


class CliParserTest(unittest.TestCase):
    def test_parses_one_parameter_per_line(self) -> None:
        payload = parse_cli_command(
            """
            python scripts/train_cli.py submit
            --name "Multiline CLI test"
            --model Qwen/Qwen3-1.7B
            --dataset-id ds-1
            --dataset-name data
            --output-name adapter
            --validation-split 0
            --trainer-param tf32=false
            """
        )
        self.assertEqual(payload["name"], "Multiline CLI test")
        self.assertEqual(payload["validationSplit"], 0)
        self.assertEqual(payload["parameters"]["optimizer"], "adamw_torch")
        self.assertFalse(payload["extraParameters"]["tf32"])

    def test_parses_supported_training_command(self) -> None:
        payload = parse_cli_command(
            'python scripts/train_cli.py submit --name "CLI test" '
            '--model Qwen/Qwen3-1.7B --dataset-id ds-1 --dataset-name data '
            '--output-name adapter --tuning-mode qlora --validation-split 0 '
            '--early-stopping --patience 3 --min-delta 0 '
            '--epochs 2 --trainer-param tf32=false'
        )
        self.assertEqual(payload["tuningMode"], "qlora")
        self.assertEqual(payload["validationSplit"], 0)
        self.assertTrue(payload["earlyStopping"])
        self.assertEqual(payload["parameters"]["epochs"], 2)
        self.assertEqual(payload["extraParameters"]["tf32"], False)

    def test_parses_dpo_task_type(self) -> None:
        payload = parse_cli_command(
            "submit --name dpo --model m --task-type DPO "
            "--dataset-id ds --dataset-name data --output-name adapter"
        )
        self.assertEqual(payload["taskType"], "DPO")

    def test_rejects_shell_operators(self) -> None:
        with self.assertRaisesRegex(CliValidationError, "Shell"):
            parse_cli_command(
                "python scripts/train_cli.py submit --name x --model m "
                "--dataset-id d --dataset-name n --output-name o; whoami"
            )

    def test_rejects_unknown_extra_parameter(self) -> None:
        with self.assertRaisesRegex(CliValidationError, "第 2 行.*不支持的高级训练参数"):
            parse_cli_command(
                "submit\n"
                "--name x --model m --dataset-id d --dataset-name n "
                "--output-name o --trainer-param arbitrary_flag=true"
            )

    def test_reports_line_for_shell_syntax(self) -> None:
        with self.assertRaisesRegex(CliValidationError, "第 3 行.*Shell"):
            parse_cli_command(
                "submit\n"
                "--name x\n"
                "--model m --dataset-id d --dataset-name n --output-name o; whoami"
            )


if __name__ == "__main__":
    unittest.main()
