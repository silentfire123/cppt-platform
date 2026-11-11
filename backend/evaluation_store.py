from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class EvaluationStore:
    """Durable evaluation task/item store with SQLite and PostgreSQL backends."""

    def __init__(self, backend: str, sqlite_path: Path, dsn: str = "") -> None:
        self.backend = backend
        self.sqlite_path = sqlite_path
        self.dsn = dsn
        self._lock = threading.RLock()
        self._init_schema()

    def _connect_pg(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self.dsn, row_factory=dict_row)

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.backend == "postgres":
            connection = self._connect_pg()
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        else:
            connection = sqlite3.connect(self.sqlite_path)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            finally:
                connection.close()

    def _init_schema(self) -> None:
        if self.backend == "postgres":
            with self._connect() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_tasks (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        dataset_version INTEGER NOT NULL DEFAULT 1,
                        target_task_id TEXT,
                        target_variant TEXT,
                        judge_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        dimensions JSONB NOT NULL DEFAULT '[]'::jsonb,
                        generation_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
                        state TEXT NOT NULL DEFAULT 'draft',
                        total_items INTEGER NOT NULL DEFAULT 0,
                        generated_items INTEGER NOT NULL DEFAULT 0,
                        reviewed_items INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        completed_at TIMESTAMPTZ
                    );
                    CREATE TABLE IF NOT EXISTS evaluation_items (
                        id TEXT PRIMARY KEY,
                        evaluation_id TEXT NOT NULL REFERENCES evaluation_tasks(id) ON DELETE CASCADE,
                        source_index INTEGER NOT NULL,
                        source_file TEXT,
                        source_line INTEGER,
                        prompt TEXT NOT NULL,
                        reference_answer TEXT,
                        system_prompt TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        model_answer TEXT,
                        teacher_result JSONB,
                        similarity_result JSONB,
                        human_result JSONB,
                        attribution TEXT,
                        review_note TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        error TEXT,
                        reviewed_by TEXT,
                        reviewed_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(evaluation_id, source_index)
                    );
                    CREATE INDEX IF NOT EXISTS idx_evaluation_items_task ON evaluation_items(evaluation_id, source_index);
                """)
        else:
            with self._connect() as connection:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS evaluation_tasks (
                        id TEXT PRIMARY KEY, name TEXT NOT NULL, dataset_id TEXT NOT NULL,
                        dataset_version INTEGER NOT NULL DEFAULT 1, target_task_id TEXT,
                        target_variant TEXT, judge_config TEXT NOT NULL DEFAULT '{}',
                        dimensions TEXT NOT NULL DEFAULT '[]', generation_parameters TEXT NOT NULL DEFAULT '{}',
                        state TEXT NOT NULL DEFAULT 'draft', total_items INTEGER NOT NULL DEFAULT 0,
                        generated_items INTEGER NOT NULL DEFAULT 0, reviewed_items INTEGER NOT NULL DEFAULT 0,
                        error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT
                    );
                    CREATE TABLE IF NOT EXISTS evaluation_items (
                        id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL,
                        source_index INTEGER NOT NULL, source_file TEXT, source_line INTEGER,
                        prompt TEXT NOT NULL, reference_answer TEXT, system_prompt TEXT,
                        metadata TEXT NOT NULL DEFAULT '{}', model_answer TEXT,
                        teacher_result TEXT, similarity_result TEXT, human_result TEXT,
                        attribution TEXT, review_note TEXT, status TEXT NOT NULL DEFAULT 'pending',
                        error TEXT, reviewed_by TEXT, reviewed_at TEXT, created_at TEXT NOT NULL,
                        UNIQUE(evaluation_id, source_index),
                        FOREIGN KEY(evaluation_id) REFERENCES evaluation_tasks(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_evaluation_items_task ON evaluation_items(evaluation_id, source_index);
                """)

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat()

    @staticmethod
    def _json(value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value

    def _task(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        for key in ("judge_config", "dimensions", "generation_parameters"):
            item[key] = self._json(item.get(key), {} if key != "dimensions" else [])
        return {
            "id": item["id"], "name": item["name"], "datasetId": item["dataset_id"],
            "datasetVersion": int(item["dataset_version"]), "targetTaskId": item.get("target_task_id"),
            "targetVariant": item.get("target_variant"), "judgeConfig": item["judge_config"],
            "dimensions": item["dimensions"], "generationParameters": item["generation_parameters"],
            "state": item["state"], "totalItems": int(item["total_items"]),
            "generatedItems": int(item["generated_items"]), "reviewedItems": int(item["reviewed_items"]),
            "error": item.get("error"), "createdAt": str(item["created_at"]),
            "updatedAt": str(item["updated_at"]), "completedAt": item.get("completed_at"),
        }

    def _item(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        for key in ("metadata", "teacher_result", "similarity_result", "human_result"):
            item[key] = self._json(item.get(key), {} if key != "metadata" else {})
        return {
            "id": item["id"], "evaluationId": item["evaluation_id"], "sourceIndex": int(item["source_index"]),
            "sourceFile": item.get("source_file"), "sourceLine": item.get("source_line"),
            "prompt": item["prompt"], "referenceAnswer": item.get("reference_answer"),
            "systemPrompt": item.get("system_prompt"), "metadata": item["metadata"],
            "modelAnswer": item.get("model_answer"), "teacherResult": item["teacher_result"],
            "similarityResult": item["similarity_result"], "humanResult": item["human_result"],
            "attribution": item.get("attribution"), "reviewNote": item.get("review_note"),
            "status": item["status"], "error": item.get("error"), "reviewedBy": item.get("reviewed_by"),
            "reviewedAt": item.get("reviewed_at"),
        }

    def create_task(self, payload: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
        task_id = f"eval-{uuid.uuid4().hex[:12]}"
        now = self._now()
        if self.backend == "postgres":
            task_sql = """INSERT INTO evaluation_tasks
                (id,name,dataset_id,dataset_version,target_task_id,target_variant,judge_config,dimensions,generation_parameters,total_items,created_at,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s)"""
            item_sql = """INSERT INTO evaluation_items
                (id,evaluation_id,source_index,source_file,source_line,prompt,reference_answer,system_prompt,metadata,created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)"""
        else:
            task_sql = """INSERT INTO evaluation_tasks
                (id,name,dataset_id,dataset_version,target_task_id,target_variant,judge_config,dimensions,generation_parameters,total_items,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
            item_sql = """INSERT INTO evaluation_items
                (id,evaluation_id,source_index,source_file,source_line,prompt,reference_answer,system_prompt,metadata,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)"""
        with self._lock, self._connect() as connection:
            vals = (task_id, payload["name"], payload["datasetId"], int(payload.get("datasetVersion", 1)),
                    payload.get("targetTaskId"), payload.get("targetVariant"), json.dumps(payload.get("judgeConfig", {}), ensure_ascii=False),
                    json.dumps(payload.get("dimensions", []), ensure_ascii=False), json.dumps(payload.get("generationParameters", {}), ensure_ascii=False), len(records), now, now)
            connection.execute(task_sql, vals)
            for index, record in enumerate(records, 1):
                connection.execute(item_sql, (f"{task_id}-item-{index:06d}", task_id, index, record.get("sourceFile"), record.get("sourceLine"), record["prompt"], record.get("referenceAnswer"), record.get("systemPrompt"), json.dumps(record.get("metadata", {}), ensure_ascii=False), now))
        return self.get_task(task_id) or {}

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM evaluation_tasks ORDER BY created_at DESC").fetchall()
        return [self._task(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM evaluation_tasks WHERE id = " + ("%s" if self.backend == "postgres" else "?"), (task_id,)).fetchone()
        return self._task(row) if row else None

    def update_task(self, task_id: str, **changes: Any) -> dict[str, Any] | None:
        allowed = {"state", "generated_items", "reviewed_items", "error", "completed_at"}
        fields = [(key, value) for key, value in changes.items() if key in allowed]
        if not fields:
            return self.get_task(task_id)
        fields.append(("updated_at", self._now()))
        assignments = ", ".join(f"{key} = {'%s' if self.backend == 'postgres' else '?'}" for key, _ in fields)
        values = [value for _, value in fields] + [task_id]
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE evaluation_tasks SET " + assignments + " WHERE id = " + ("%s" if self.backend == "postgres" else "?"), values)
        return self.get_task(task_id)

    def list_items(self, task_id: str, *, status: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        where = "evaluation_id = " + ("%s" if self.backend == "postgres" else "?")
        params: list[Any] = [task_id]
        if status:
            where += " AND status = " + ("%s" if self.backend == "postgres" else "?")
            params.append(status)
        params.extend([limit, offset])
        ph = "%s" if self.backend == "postgres" else "?"
        with self._connect() as connection:
            rows = connection.execute(f"SELECT * FROM evaluation_items WHERE {where} ORDER BY source_index LIMIT {ph} OFFSET {ph}", params).fetchall()
        return [self._item(row) for row in rows]

    def update_item(self, item_id: str, **changes: Any) -> dict[str, Any] | None:
        mapping = {"modelAnswer": "model_answer", "teacherResult": "teacher_result", "similarityResult": "similarity_result", "humanResult": "human_result", "attribution": "attribution", "reviewNote": "review_note", "status": "status", "error": "error", "reviewedBy": "reviewed_by", "reviewedAt": "reviewed_at"}
        parts: list[str] = []
        vals: list[Any] = []
        for key, value in changes.items():
            column = mapping.get(key)
            if not column:
                continue
            parts.append(f"{column} = {'%s' if self.backend == 'postgres' else '?'}")
            vals.append(json.dumps(value, ensure_ascii=False) if key in {"teacherResult", "similarityResult", "humanResult"} else value)
        if not parts:
            return None
        vals.append(item_id)
        with self._lock, self._connect() as connection:
            row = connection.execute("UPDATE evaluation_items SET " + ", ".join(parts) + " WHERE id = " + ("%s" if self.backend == "postgres" else "?"), vals)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM evaluation_items WHERE id = " + ("%s" if self.backend == "postgres" else "?"), (item_id,)).fetchone()
        if not row:
            return None
        item = self._item(row)
        self._refresh_review_count(item["evaluationId"])
        return item

    def _refresh_review_count(self, task_id: str) -> None:
        ph = "%s" if self.backend == "postgres" else "?"
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM evaluation_items WHERE evaluation_id = {ph} AND status = 'reviewed'", (task_id,)).fetchone()
            count = int(row["count"])
        task = self.get_task(task_id)
        if task and count >= task["totalItems"] and task["totalItems"] > 0 and task["state"] not in {"failed", "cancelled"}:
            self.update_task(task_id, reviewed_items=count, state="completed", completed_at=self._now())
        else:
            self.update_task(task_id, reviewed_items=count)

    def summary(self, task_id: str) -> dict[str, Any]:
        items = self.list_items(task_id, limit=100000, offset=0)
        reviewed = [item for item in items if item.get("status") == "reviewed" and isinstance(item.get("humanResult"), dict) and isinstance(item["humanResult"].get("score"), (int, float))]
        scores = [float(item["humanResult"]["score"]) for item in reviewed]
        teacher = [float(item["teacherResult"]["score"]) for item in items if isinstance(item.get("teacherResult"), dict) and isinstance(item["teacherResult"].get("score"), (int, float))]
        similarity = [float(item["similarityResult"]["score"]) for item in items if isinstance(item.get("similarityResult"), dict) and isinstance(item["similarityResult"].get("score"), (int, float))]
        def stats(values: list[float]) -> dict[str, Any]:
            if not values: return {"count": 0, "mean": None, "variance": None}
            mean = sum(values) / len(values)
            return {"count": len(values), "mean": mean, "variance": sum((v - mean) ** 2 for v in values) / len(values)}
        attribution: dict[str, int] = {}
        for item in reviewed:
            value = item.get("attribution") or "unclassified"
            attribution[value] = attribution.get(value, 0) + 1
        return {"taskId": task_id, "total": len(items), "reviewed": len(reviewed), "coverage": (len(reviewed) / len(items) if items else 0), "human": stats(scores), "teacher": stats(teacher), "similarity": stats(similarity), "attribution": attribution, "lowScoreCount": sum(1 for v in scores if v <= 2), "lowSimilarityCount": sum(1 for v in similarity if v < 0.5)}
