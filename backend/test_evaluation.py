from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from backend import app as app_module
from backend.app import normalize_evaluation_record, validate_evaluation_dataset_report
from backend.evaluation_store import EvaluationStore


class EvaluationDatasetTests(unittest.TestCase):
    def test_normalizes_prompt_completion_messages_and_dpo(self) -> None:
        record, errors = normalize_evaluation_record({"prompt": "question", "completion": "answer"}, "sample.jsonl", 1)
        self.assertFalse(errors)
        self.assertEqual(record["prompt"], "question")
        self.assertEqual(record["completion"], "answer")
        record, errors = normalize_evaluation_record({"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}, "sample.jsonl", 2)
        self.assertFalse(errors)
        self.assertEqual(record["system"], "sys")
        record, errors = normalize_evaluation_record({"prompt": "question", "chosen": "best", "rejected": "bad"}, "sample.jsonl", 3)
        self.assertFalse(errors)
        self.assertEqual(record["completion"], "best")

    def test_report_rejects_missing_reference_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            path.write_text("\n".join([
                json.dumps({"id": "same", "prompt": "q", "completion": "a"}),
                json.dumps({"id": "same", "prompt": "q2", "completion": "a2"}),
                json.dumps({"prompt": "q3"}),
            ]), encoding="utf-8")
            report = validate_evaluation_dataset_report(path)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any(error["code"] == "DUPLICATE_EVALUATION_ID" for error in report["errors"]))
            self.assertTrue(any(error["code"] == "EVALUATION_REFERENCE_REQUIRED" for error in report["errors"]))


class EvaluationApiContractTests(unittest.TestCase):
    def test_retry_route_is_registered(self) -> None:
        paths = {route.path for route in app_module.app.routes if hasattr(route, "path")}
        self.assertIn("/api/evaluations/{evaluation_id}/retry", paths)

    def test_teacher_request_sets_explicit_user_agent(self) -> None:
        captured = {}

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": json.dumps({"score": 5, "dimensions": {}, "reason": "ok"})}}]}).encode()

        def fake_urlopen(request, timeout):
            captured["userAgent"] = request.get_header("User-agent")
            captured["timeout"] = timeout
            return Response()

        with patch.dict(os.environ, {"XUNTUI_EVALUATION_JUDGE_API_KEY": "secret"}, clear=False), patch.object(app_module.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = app_module._evaluation_teacher({"baseUrl": "https://judge.example/v1", "model": "judge"}, "q", "a", "a", ["correctness"])
        self.assertEqual(result["score"], 5)
        self.assertEqual(captured["userAgent"], "Xuntui-Evaluation/1.0")

    def test_health_exposes_release_and_redacted_provider_readiness(self) -> None:
        values = {
            "XUNTUI_RELEASE_VERSION": "v-test",
            "XUNTUI_EVALUATION_JUDGE_BASE_URL": "https://judge.example/v1",
            "XUNTUI_EVALUATION_JUDGE_API_KEY": "secret",
            "XUNTUI_EVALUATION_JUDGE_MODEL": "judge",
            "XUNTUI_EVALUATION_EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "XUNTUI_EVALUATION_EMBEDDING_API_KEY": "secret",
            "XUNTUI_EVALUATION_EMBEDDING_MODEL": "embedding",
        }
        with patch.dict(os.environ, values, clear=False):
            health = app_module.health_snapshot()
        self.assertEqual(health["releaseVersion"], "v-test")
        self.assertEqual(health["evaluationProviders"], {"judgeConfigured": True, "embeddingConfigured": True})



class EvaluationStoreTests(unittest.TestCase):
    def test_human_summary_only_uses_reviewed_human_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore("sqlite", Path(directory) / "evaluation.db")
            task = store.create_task({"name": "eval", "datasetId": "dataset", "datasetVersion": 1, "targetTaskId": "ft-1", "judgeConfig": {}, "dimensions": []}, [
                {"prompt": "q1", "referenceAnswer": "a1", "sourceFile": "x", "sourceLine": 1},
                {"prompt": "q2", "referenceAnswer": "a2", "sourceFile": "x", "sourceLine": 2},
            ])
            items = store.list_items(task["id"])
            store.update_item(items[0]["id"], teacherResult={"score": 1}, similarityResult={"score": .1}, humanResult={"score": 4}, attribution="acceptable_different", status="reviewed")
            store.update_item(items[1]["id"], teacherResult={"score": 3}, similarityResult={"score": .8}, status="generated")
            summary = store.summary(task["id"])
            self.assertEqual(summary["reviewed"], 1)
            self.assertEqual(summary["human"]["mean"], 4)
            self.assertEqual(summary["human"]["variance"], 0)
            self.assertEqual(summary["teacher"]["mean"], 2)
            self.assertEqual(summary["similarity"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
