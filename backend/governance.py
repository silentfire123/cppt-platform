from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


SUCCESS_LOG_DAYS = 7
FAILED_LOG_DAYS = 30
AUDIT_LOG_DAYS = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AuditStore:
    """Small adapter for immutable user-action records."""

    def __init__(self, backend: str, sqlite_path: Path, postgres_dsn: str = "") -> None:
        self.backend = backend
        self.sqlite_path = sqlite_path
        self.postgres_dsn = postgres_dsn

    def initialize(self) -> None:
        if self.backend == "postgres":
            with self._pg_connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id BIGSERIAL PRIMARY KEY,
                        occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        actor_id TEXT,
                        actor_name TEXT NOT NULL DEFAULT '',
                        action TEXT NOT NULL,
                        object_type TEXT NOT NULL,
                        object_id TEXT NOT NULL,
                        result TEXT NOT NULL,
                        error TEXT,
                        details JSONB NOT NULL DEFAULT '{}'::jsonb
                    );
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_occurred_at
                        ON audit_logs(occurred_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_object
                        ON audit_logs(object_type, object_id, occurred_at DESC);
                    """
                )
            return
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    actor_id TEXT,
                    actor_name TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    result TEXT NOT NULL,
                    error TEXT,
                    details TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_logs_occurred_at ON audit_logs(occurred_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_logs_object ON audit_logs(object_type, object_id, occurred_at DESC)"
            )

    def record(
        self,
        *,
        actor: dict[str, Any] | None,
        action: str,
        object_type: str,
        object_id: str,
        result: str = "success",
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        actor = actor or {}
        actor_id = str(actor.get("id") or actor.get("sub") or "") or None
        actor_name = str(actor.get("name") or actor.get("display_name") or "")[:200]
        payload = json.dumps(details or {}, ensure_ascii=False, default=str)
        occurred = _now()
        if self.backend == "postgres":
            with self._pg_connect() as connection:
                connection.execute(
                    """
                    INSERT INTO audit_logs(occurred_at, actor_id, actor_name, action, object_type, object_id, result, error, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (occurred, actor_id, actor_name, action, object_type, object_id, result, error, payload),
                )
            return
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO audit_logs(occurred_at, actor_id, actor_name, action, object_type, object_id, result, error, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (occurred.isoformat(), actor_id, actor_name, action, object_type, object_id, result, error, payload),
            )

    def cleanup(self, *, now: datetime | None = None) -> int:
        cutoff = (now or _now()) - timedelta(days=AUDIT_LOG_DAYS)
        if self.backend == "postgres":
            with self._pg_connect() as connection:
                row = connection.execute(
                    "DELETE FROM audit_logs WHERE occurred_at < %s RETURNING id", (cutoff,)
                ).fetchall()
            return len(row)
        with sqlite3.connect(self.sqlite_path) as connection:
            cursor = connection.execute("DELETE FROM audit_logs WHERE occurred_at < ?", (cutoff.isoformat(),))
            return int(cursor.rowcount or 0)

    @contextmanager
    def _pg_connect(self) -> Iterator[Any]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("操作审计需要 psycopg。") from exc
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            connection.execute("SET TIME ZONE 'Asia/Shanghai'")
            yield connection


def cleanup_runtime_logs(
    *, backend: str, sqlite_path: Path, postgres_dsn: str = "", now: datetime | None = None
) -> dict[str, int]:
    """Delete expired runtime logs while retaining business summaries."""
    current = now or _now()
    success_cutoff = current - timedelta(days=SUCCESS_LOG_DAYS)
    failed_cutoff = current - timedelta(days=FAILED_LOG_DAYS)
    if backend == "postgres":
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("日志清理需要 psycopg。") from exc
        with psycopg.connect(postgres_dsn, row_factory=dict_row) as connection:
            task_deleted = connection.execute(
                """
                DELETE FROM training_task_logs l
                USING training_tasks t
                LEFT JOIN training_task_attempts a ON a.id = l.attempt_id
                WHERE t.id = l.task_id
                  AND (((COALESCE(a.state, t.state) = 'completed') AND l.created_at < %s)
                       OR ((COALESCE(a.state, t.state) <> 'completed') AND l.created_at < %s))
                """,
                (success_cutoff, failed_cutoff),
            ).rowcount
            inference_deleted = connection.execute(
                """
                DELETE FROM inference_logs l
                USING inference_services s
                LEFT JOIN inference_service_runs r ON r.id = l.run_id
                WHERE s.id = l.service_id
                  AND ((COALESCE(r.state, s.state) = 'completed' AND l.created_at < %s)
                       OR (COALESCE(r.state, s.state) <> 'completed' AND l.created_at < %s))
                """,
                (success_cutoff, failed_cutoff),
            ).rowcount
        return {"trainingLogs": int(task_deleted or 0), "inferenceLogs": int(inference_deleted or 0)}
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        task_deleted = connection.execute(
            """
            DELETE FROM task_logs
            WHERE rowid IN (
                SELECT l.rowid FROM task_logs l
                JOIN tasks t ON t.id = l.task_id
                LEFT JOIN training_task_attempts a ON a.id = l.attempt_id
                WHERE (COALESCE(a.state, t.state)='completed' AND l.created_at < ?)
                   OR (COALESCE(a.state, t.state)<>'completed' AND l.created_at < ?)
            )
            """,
            (success_cutoff.isoformat(), failed_cutoff.isoformat()),
        ).rowcount
        inference_deleted = connection.execute(
            """
            DELETE FROM inference_logs
            WHERE rowid IN (
                SELECT l.rowid FROM inference_logs l
                JOIN inference_services s ON s.id = l.service_id
                LEFT JOIN inference_service_runs r ON r.id = l.run_id
                WHERE (COALESCE(r.state, s.state)='completed' AND l.created_at < ?)
                   OR (COALESCE(r.state, s.state)<>'completed' AND l.created_at < ?)
            )
            """,
            (success_cutoff.isoformat(), failed_cutoff.isoformat()),
        ).rowcount
    return {"trainingLogs": int(task_deleted or 0), "inferenceLogs": int(inference_deleted or 0)}
