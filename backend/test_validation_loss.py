"""Validation scheduling and metric ingestion regressions (no GPU required)."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import yaml

from backend.local_llamafactory_executor import LocalLlamaFactoryExecutor
from backend.providers.youyun.remote_training import _LogParser
from backend.store import TaskStore, parse_trainer_values
from backend.test_youyun_remote_training import RunnerTestBase


# Same formats and final step as the affected 897-step production run.
FINAL_LOG = [
    "Total optimization steps = 897",
    "Num Epochs = 3",
    "{'loss': '0.8433', 'epoch': '2.993'}",
    "{'train_loss': '0.995', 'epoch': '3'}",
    "***** eval metrics *****",
    "  eval_loss               =      1.075",
]


class ValidationMetricParserTest(unittest.TestCase):
    def test_final_eval_summary_is_parsed(self):
        for line in ("eval_loss = 1.075", "  eval_loss   = 1.075e+00  ", "eval_loss = 0"):
            with self.subTest(line=line):
                self.assertEqual(parse_trainer_values(line), {"eval_loss": float(line.split("=")[1])})

    def test_unrelated_or_invalid_summaries_are_not_metrics(self):
        for line in ("metric_for_best_model = eval_loss", "warning: eval_loss = 1.075",
                     "eval_loss = nan", "eval_loss = inf", "eval_loss = 1e999",
                     "eval_loss = broken", "eval_loss = 1.0 trailing", "eval_runtime = 3.0"):
            with self.subTest(line=line):
                self.assertIsNone(parse_trainer_values(line))

    def test_periodic_eval_dictionary_is_still_parsed(self):
        self.assertEqual(parse_trainer_values("{'eval_loss': 1.2, 'epoch': 0.5}"),
                         {"eval_loss": 1.2, "epoch": 0.5})


class RemoteValidationLossTest(RunnerTestBase):
    def test_final_eval_summary_reaches_loss_history_at_actual_final_step(self):
        store = TaskStore(self.mirror / "metrics.db")
        task = store.create({"name": "eval regression", "model": "tiny", "datasetId": "tiny", "datasetName": "tiny",
                             "parameters": {"loggingSteps": 5}})
        runner = self.make_runner(store)
        parser = _LogParser(5)
        # Poll boundaries must not prevent pairing final eval with final train step.
        runner._ingest_chunk(task["id"], parser, "\n".join(FINAL_LOG[:-1]))
        runner._ingest_chunk(task["id"], parser, FINAL_LOG[-1])
        history = store.metrics(task["id"])
        eval_points = [item for item in history if item.get("evalLoss") is not None]
        self.assertEqual(len(eval_points), 1, "final eval loss was dropped by ingestion")
        self.assertEqual(eval_points[0]["step"], 897)
        self.assertAlmostEqual(eval_points[0]["evalLoss"], 1.075)
        self.assertAlmostEqual(eval_points[0]["trainLoss"], 0.995)
        # A structured event followed by its text summary must merge, not duplicate.
        runner._ingest_chunk(task["id"], parser, "{'eval_loss': 1.075, 'epoch': 3.0}")
        runner._ingest_chunk(task["id"], parser, FINAL_LOG[-1])
        self.assertEqual(len([x for x in store.metrics(task["id"]) if x.get("evalLoss") is not None]), 1)


class LocalValidationLossTest(unittest.IsolatedAsyncioTestCase):
    async def test_final_eval_summary_is_emitted_as_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalLlamaFactoryExecutor(work_root=Path(directory))
            stream = asyncio.StreamReader()
            stream.feed_data(("\n".join(FINAL_LOG) + "\n").encode())
            stream.feed_eof()
            executor._handles["eval-test"] = SimpleNamespace(
                process=SimpleNamespace(stdout=stream, wait=AsyncMock(return_value=0)))
            events = [event async for event in executor.events({"id": "eval-test", "parameters": {"loggingSteps": 5}})]
            metrics = [event for event in events if event.type == "metric" and "eval_loss" in event.payload]
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0].payload["step"], 897)
            self.assertAlmostEqual(metrics[0].payload["eval_loss"], 1.075)

    async def test_ui_validation_interval_reaches_written_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            validation.write_text("{}\n", encoding="utf-8")
            executor = LocalLlamaFactoryExecutor(work_root=root / "work")
            for task_type in ("SFT", "DPO"):
                with self.subTest(task_type=task_type):
                    task = {"id": f"eval-{task_type}", "model": "tiny", "datasetId": "tiny",
                            "taskType": task_type, "validationSplit": 10,
                            "dataset": {"paths": {"trainPath": str(train), "validationPath": str(validation)}},
                            "parameters": {"validationSteps": 25, "loggingSteps": 5}}
                    handle = await executor.prepare(task)
                    config = yaml.safe_load(Path(handle.config_path).read_text(encoding="utf-8"))
                    self.assertTrue(config["do_eval"])
                    self.assertEqual(config["eval_dataset"], "tiny_validation")
                    self.assertEqual(config.get("eval_strategy"), "steps")
                    self.assertEqual(config.get("eval_steps"), 25)
                    self.assertNotIn("validationSteps", config)
                    self.assertNotIn("val_size", config)

    async def test_no_validation_does_not_enable_periodic_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            executor = LocalLlamaFactoryExecutor(work_root=root / "work")
            handle = await executor.prepare({"id": "no-eval", "model": "tiny", "datasetId": "tiny",
                "validationSplit": 0, "dataset": {"paths": {"trainPath": str(train)}},
                "parameters": {"validationSteps": 25}})
            config = yaml.safe_load(Path(handle.config_path).read_text(encoding="utf-8"))
            self.assertFalse(config["do_eval"])
            self.assertNotIn("eval_strategy", config)
