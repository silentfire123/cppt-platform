from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["XUNTUI_STORAGE_BACKEND"] = "sqlite"

from backend.app import repair_text_dataset, validate_text_dataset_report


class DatasetRepairTests(unittest.TestCase):
    def test_report_checks_all_sft_rows_and_returns_structured_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            source.write_text(
                json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]})
                + "\n"
                + json.dumps({"messages": [{"role": "assistant", "content": "wrong"}]})
                + "\n",
                encoding="utf-8",
            )
            report = validate_text_dataset_report(source, "SFT")
            self.assertEqual(report["checkedRows"], 2)
            self.assertEqual(report["invalidRows"], 1)
            self.assertEqual(report["errors"][0]["line"], 2)
            self.assertEqual(report["status"], "failed")

    def test_manual_repair_does_not_change_unedited_rows_or_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            target = Path(directory) / "repaired.jsonl"
            source.write_text(
                "\n" + json.dumps({"messages": [{"role": "human", "content": " hi "}, {"role": "gpt", "content": " answer "}]}) + "\n",
                encoding="utf-8",
            )
            stats = repair_text_dataset(source, target, "SFT", [{
                "line": 2,
                "record": {"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "answer"},
                ]},
            }])
            self.assertEqual(stats["removedBlankLines"], 0)
            self.assertEqual(stats["repairedRows"], 1)
            lines = target.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "")
            self.assertEqual(json.loads(lines[1])["messages"][0]["role"], "user")

    def test_common_human_gpt_roles_require_explicit_manual_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            target = Path(directory) / "repaired.jsonl"
            source.write_text(json.dumps({"messages": [
                {"role": "human", "content": "hi"},
                {"role": "gpt", "content": "hello"},
            ]}) + "\n", encoding="utf-8")
            stats = repair_text_dataset(source, target, "SFT", [{
                "line": 1,
                "record": {"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]},
            }])
            report = validate_text_dataset_report(target, "SFT")
            self.assertEqual(stats["repairedRows"], 1)
            self.assertFalse(report["errors"])

    def test_manual_line_edit_keeps_other_lines_and_changes_only_selected_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            target = Path(directory) / "repaired.jsonl"
            source.write_text(
                json.dumps({"messages": [{"role": "human", "content": "hi"}, {"role": "gpt", "content": "hello"}]}) + "\n"
                + json.dumps({"messages": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "ok"}]}) + "\n",
                encoding="utf-8",
            )
            stats = repair_text_dataset(source, target, "SFT", [{
                "line": 1,
                "record": {"messages": [{"role": "user", "content": "edited"}, {"role": "assistant", "content": "answer"}]},
            }])
            records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["repairedRows"], 1)
            self.assertEqual(records[0]["messages"][0]["content"], "edited")
            self.assertEqual(records[1]["messages"][0]["content"], "two")


if __name__ == "__main__":
    unittest.main()
