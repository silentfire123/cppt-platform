from __future__ import annotations

import sqlite3
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.inference_service import InferenceCreate
from backend.postgres_storage import PostgresInferenceStore


def _postgres_datetime(value: Any) -> Any:
    """Make SQLite timestamp values look like psycopg TIMESTAMPTZ results."""
    if not isinstance(value, str):
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


class _FakePostgresCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self.cursor = cursor

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("created_at", "updated_at"):
            if key in result:
                result[key] = _postgres_datetime(result[key])
        return result

    def fetchone(self) -> dict[str, Any] | None:
        return self._row(self.cursor.fetchone())

    def fetchall(self) -> list[dict[str, Any]]:
        return [self._row(row) for row in self.cursor.fetchall()]  # type: ignore[misc]


class _FakePostgresConnection:
    """Small DB-API adapter that exercises the PostgreSQL store without a server."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def execute(self, sql: str, params: Any = ()) -> _FakePostgresCursor:
        sqlite_sql = sql.replace("?::jsonb", "?").replace("EXCLUDED.", "excluded.")
        return _FakePostgresCursor(self.connection.execute(sqlite_sql, params))


class FakePostgresInferenceStore(PostgresInferenceStore):
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE inference_services (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                task_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                config TEXT NOT NULL,
                state TEXT NOT NULL,
                exposure TEXT NOT NULL DEFAULT 'test',
                gpu TEXT NOT NULL DEFAULT '未占用',
                memory TEXT NOT NULL DEFAULT '0 / 0 GB',
                endpoint TEXT NOT NULL,
                recovery_attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_id TEXT NOT NULL REFERENCES inference_services(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE TABLE inference_metrics (
                service_id TEXT PRIMARY KEY REFERENCES inference_services(id) ON DELETE CASCADE,
                metrics TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE inference_command_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_id TEXT NOT NULL REFERENCES inference_services(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                flags TEXT NOT NULL DEFAULT '[]',
                success INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );
            """
        )

    @contextmanager
    def _connect(self) -> Iterator[_FakePostgresConnection]:
        try:
            yield _FakePostgresConnection(self.connection)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        self.connection.close()


class PostgresInferenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakePostgresInferenceStore()
        self.store.create(
            service_id="svc-pg",
            payload=InferenceCreate(name="PostgreSQL 推理", taskId="ft-pg"),
            model="Qwen/Qwen2.5-0.5B-Instruct",
            adapter="/workspace/training-work/ft-pg/output",
            endpoint="/v1/chat/completions",
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_datetime_logs_metrics_and_command_usage_are_serializable(self) -> None:
        service = self.store.get("svc-pg")
        self.assertIsNotNone(service)
        self.assertIsInstance(service["createdAt"], str)  # type: ignore[index]
        self.assertIn("+00:00", service["createdAt"])  # type: ignore[index]

        self.store.append_log("svc-pg", "vLLM import failed")
        logs = self.store.logs("svc-pg")
        self.assertTrue(any("vLLM import failed" in line for line in logs))
        self.assertTrue(all(line.startswith("[") and "] " in line for line in logs))

        self.store.set_metrics("svc-pg", {"source": "prometheus", "requestCount": 3})
        metrics = self.store.metrics("svc-pg")
        self.assertEqual(metrics["requestCount"], 3)
        self.assertIsInstance(metrics["updatedAt"], str)
        self.assertIn("+00:00", metrics["updatedAt"])

        self.store.record_command_usage(
            "svc-pg",
            flags=["--max-num-seqs", "--max-num-seqs", "--enable-prefix-caching"],
            success=False,
            error="CPU does not accept GPU-only flags",
        )
        usage = self.store.command_usage("svc-pg")
        self.assertEqual(usage["total"], 1)
        self.assertEqual(usage["failed"], 1)
        self.assertEqual(usage["flagCounts"]["--max-num-seqs"], 1)
        self.assertEqual(usage["recentErrors"], ["CPU does not accept GPU-only flags"])

    def test_remote_runtime_config_update_round_trips_json(self) -> None:
        service = self.store.get("svc-pg")
        assert service is not None
        config = dict(service["config"])
        config["remoteRuntime"] = {
            "instanceId": "uhost-pg",
            "endpoint": "http://10.0.0.8:8011",
            "state": "running",
        }
        updated = self.store.update("svc-pg", config=config, state="running")
        assert updated is not None
        self.assertEqual(
            updated["config"]["remoteRuntime"]["instanceId"], "uhost-pg"
        )

    def test_archive_restore_and_delete_preserve_then_cascade_related_rows(self) -> None:
        self.store.append_log("svc-pg", "before archive")
        self.store.set_metrics("svc-pg", {"source": "prometheus", "requestCount": 1})
        self.store.record_command_usage(
            "svc-pg", flags=["--max-num-seqs"], success=True
        )
        self.store.update("svc-pg", state="stopped")

        archived = self.store.archive("svc-pg")
        self.assertEqual(archived["state"], "archived")
        self.assertEqual(self.store.list(), [])
        self.assertEqual(len(self.store.list(include_archived=True)), 1)
        self.assertTrue(any("before archive" in line for line in self.store.logs("svc-pg")))
        self.assertEqual(self.store.metrics("svc-pg")["requestCount"], 1)
        self.assertEqual(self.store.command_usage("svc-pg")["total"], 1)

        restored = self.store.restore("svc-pg")
        self.assertEqual(restored["state"], "stopped")
        self.assertEqual(restored["exposure"], "offline")

        self.store.archive("svc-pg")
        self.store.delete("svc-pg")
        self.assertIsNone(self.store.get("svc-pg"))
        self.assertEqual(self.store.logs("svc-pg"), [])
        self.assertEqual(
            self.store.metrics("svc-pg"),
            {"source": "not_started", "updatedAt": None},
        )
        self.assertEqual(self.store.command_usage("svc-pg")["total"], 0)


if __name__ == "__main__":
    unittest.main()
