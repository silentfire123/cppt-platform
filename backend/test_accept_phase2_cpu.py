from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from scripts.accept_phase2_cpu import verify_postgres_and_minio


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Connection:
    def __init__(self, *, prefix: str = "model/model-fixed"):
        self.prefix = prefix
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params):
        normalized = " ".join(str(query).split())
        self.queries.append(normalized)
        if "FROM training_tasks" in normalized:
            return _Result({"state": "completed", "artifacts": []})
        if "FROM training_task_logs" in normalized:
            return _Result({"count": 1})
        if "FROM training_task_metrics" in normalized:
            return _Result({"count": 2})
        if "FROM training_task_events" in normalized:
            return _Result({"count": 3})
        if "FROM model_files" in normalized:
            return _Result(
                {
                    "id": "model-fixed",
                    "status": "available",
                    "error_message": None,
                    "minio_bucket": "rag-assets",
                    "minio_object_prefix": self.prefix,
                }
            )
        if "FROM model_file_entries" in normalized:
            return _Result({"count": 6})
        raise AssertionError(f"unexpected SQL: {normalized}")


def _fake_psycopg(connection: _Connection):
    psycopg = types.ModuleType("psycopg")
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = object()
    psycopg.rows = rows
    psycopg.connect = lambda dsn, row_factory: connection
    return {"psycopg": psycopg, "psycopg.rows": rows}


class Phase2AcceptanceStorageTests(unittest.TestCase):
    def test_accepts_available_storage_records_and_uses_available_entry_status(self):
        connection = _Connection()
        with patch.dict(sys.modules, _fake_psycopg(connection)), patch.dict(
            os.environ, {"POSTGRES_DSN": "postgresql://placeholder"}, clear=False
        ):
            result = verify_postgres_and_minio("ft-test", skip_minio=True)

        self.assertEqual(result["modelFileId"], "model-fixed")
        self.assertEqual(result["objectPrefix"], "model/model-fixed")
        self.assertEqual(result["availableEntries"], 6)
        entry_query = next(query for query in connection.queries if "FROM model_file_entries" in query)
        self.assertIn("status='available'", entry_query)
        self.assertNotIn("status='uploaded'", entry_query)

    def test_rejects_noncanonical_model_object_prefix(self):
        connection = _Connection(prefix="models/model-fixed")
        with patch.dict(sys.modules, _fake_psycopg(connection)), patch.dict(
            os.environ, {"POSTGRES_DSN": "postgresql://placeholder"}, clear=False
        ):
            with self.assertRaisesRegex(AssertionError, "MinIO prefix mismatch"):
                verify_postgres_and_minio("ft-test", skip_minio=True)


if __name__ == "__main__":
    unittest.main()
