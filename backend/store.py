from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


def display_timezone():
    """平台展示时区（默认北京时间）。系统缺失 tzdata 时退回本地时区。"""
    name = os.environ.get("XUNTUI_TZ", "Asia/Shanghai")
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return None


TOTAL_STEPS_PATTERN = re.compile(r"Total(?: optimization)? steps\s*=\s*([\d,]+)", re.IGNORECASE)
NUM_EPOCHS_PATTERN = re.compile(r"Num Epochs\s*=\s*([\d.]+)")


def resolve_adapter_artifact(
    artifacts: list[dict[str, Any]], variant: str = "selected"
) -> dict[str, Any] | None:
    adapters = [
        artifact for artifact in artifacts if artifact.get("type") == "adapter"
    ]
    if not adapters:
        return None
    if variant == "selected":
        return next(
            (artifact for artifact in adapters if artifact.get("selected") is True),
            next(
                (artifact for artifact in adapters if artifact.get("variant") == "best"),
                adapters[0],
            ),
        )
    if variant == "best":
        return next(
            (artifact for artifact in adapters if artifact.get("variant") == "best"),
            None,
        )
    if variant == "final":
        return next(
            (artifact for artifact in adapters if artifact.get("variant") == "final"),
            next(
                (
                    artifact
                    for artifact in adapters
                    if artifact.get("variant") not in {"best", "final"}
                ),
                None,
            ),
        )
    raise ValueError(f"未知 Adapter 类型：{variant}")


def resolve_youyun_instance_artifact(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return next(
        (
            artifact
            for artifact in artifacts
            if artifact.get("type") == "youyun-instance"
            and artifact.get("instanceId")
        ),
        None,
    )


def has_unreleased_youyun_instance(artifacts: list[dict[str, Any]]) -> bool:
    marker = resolve_youyun_instance_artifact(artifacts)
    return bool(marker and marker.get("state") != "released")


TASK_CLONE_SUFFIX_PATTERN = re.compile(r"(?:\s+副本(?:\s*\d+)?)+$")


def task_clone_base_name(name: str) -> str:
    """去掉任务名尾部的「副本/副本N」后缀，克隆克隆时避免叠加后缀。"""
    stripped = TASK_CLONE_SUFFIX_PATTERN.sub("", str(name).strip()).strip()
    return stripped or str(name).strip()


def next_task_clone_name(source_name: str, existing_names: Iterable[str]) -> str:
    """返回「基础名 副本N」，N 取未被占用的最小编号。"""
    base_name = task_clone_base_name(source_name)
    used_numbers = {
        int(match.group(1))
        for name in existing_names
        if (match := re.fullmatch(rf"{re.escape(base_name)}\s+副本(\d+)", str(name).strip()))
    }
    number = 1
    while number in used_numbers:
        number += 1
    return f"{base_name} 副本{number}"


def parse_trainer_values(message: str) -> dict[str, float] | None:
    start = message.find("{")
    end = message.rfind("}")
    if start < 0 or end <= start:
        # Trainer's final evaluate() summary is a table, not the dictionary
        # emitted during periodic evaluation. Keep the match anchored so
        # configuration lines and diagnostics cannot become chart points.
        summary = re.fullmatch(
            r"\s*eval_loss\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
            message,
        )
        if summary:
            loss = float(summary.group(1))
            return {"eval_loss": loss} if math.isfinite(loss) else None
        return None
    try:
        raw = ast.literal_eval(message[start : end + 1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(raw, dict) or not ({"loss", "train_loss", "eval_loss"} & raw.keys()):
        return None
    values: dict[str, float] = {}
    for key in ("loss", "train_loss", "eval_loss", "epoch", "learning_rate"):
        if raw.get(key) is None:
            continue
        try:
            destination = "loss" if key == "train_loss" else key
            if destination not in values:
                values[destination] = float(raw[key])
        except (TypeError, ValueError):
            continue
    return values if "loss" in values or "eval_loss" in values else None


def infer_metric_step(
    values: dict[str, float],
    *,
    total_steps: int | None,
    total_epochs: float | None,
    last_train_step: int,
    logging_steps: int,
) -> int:
    epoch = values.get("epoch")
    if epoch is not None and total_steps and total_epochs:
        return max(1, min(total_steps, round(epoch * total_steps / total_epochs)))
    if "eval_loss" in values and last_train_step:
        return last_train_step
    if last_train_step == 0:
        return 1
    return last_train_step + max(1, logging_steps)


class TaskStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()
        self._backfill_metrics_from_logs()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        # Windows can race TemporaryDirectory cleanup while WAL sidecar files
        # are being released; use rollback journaling there, while retaining
        # WAL concurrency on Linux production hosts.
        connection.execute(f"PRAGMA journal_mode={'DELETE' if os.name == 'nt' else 'WAL'}")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_step TEXT NOT NULL,
                    loss REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    artifacts TEXT NOT NULL DEFAULT '[]',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    stop_reason TEXT,
                    deleted_at TEXT,
                    deleted_by TEXT,
                    deletion_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS training_task_attempts (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, attempt_no INTEGER NOT NULL,
                    state TEXT NOT NULL DEFAULT 'queued', started_at TEXT, finished_at TEXT,
                    error TEXT, created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                    UNIQUE(task_id, attempt_no)
                );
                CREATE TABLE IF NOT EXISTS task_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    attempt_id TEXT,
                    level TEXT NOT NULL DEFAULT 'INFO',
                    source TEXT NOT NULL DEFAULT 'backend',
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS task_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    epoch REAL,
                    train_loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    created_at TEXT NOT NULL,
                    attempt_id TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                    UNIQUE(attempt_id, step)
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,
                    purpose TEXT NOT NULL DEFAULT 'training',
                    structure TEXT,
                    state TEXT NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    paths TEXT NOT NULL,
                    validation_message TEXT NOT NULL,
                    validation_report TEXT NOT NULL DEFAULT '{}',
                    repair_history TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 1,
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    owner TEXT NOT NULL,
                    builtin INTEGER NOT NULL DEFAULT 0,
                    source_type TEXT NOT NULL DEFAULT 'local',
                    source_uri TEXT,
                    source_bucket TEXT,
                    source_key TEXT,
                    size_bytes INTEGER,
                    content_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_assets (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    revision TEXT NOT NULL DEFAULT 'main',
                    path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'unavailable',
                    source TEXT NOT NULL,
                    base_model_id TEXT,
                    task_id TEXT,
                    method TEXT,
                    dataset_name TEXT,
                    size_bytes INTEGER,
                    variants TEXT NOT NULL DEFAULT '[]',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_state_created
                    ON tasks(state, created_at);
                CREATE INDEX IF NOT EXISTS idx_logs_task_id
                    ON task_logs(task_id, id);
                CREATE INDEX IF NOT EXISTS idx_metrics_task_step
                    ON task_metrics(task_id, step);
                CREATE INDEX IF NOT EXISTS idx_model_assets_kind_archived
                    ON model_assets(kind, archived, updated_at);
                """
            )
            task_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            dataset_columns_initial = {row["name"] for row in connection.execute("PRAGMA table_info(datasets)").fetchall()}
            if "purpose" not in dataset_columns_initial:
                connection.execute("ALTER TABLE datasets ADD COLUMN purpose TEXT NOT NULL DEFAULT 'training'")
            if "stop_reason" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN stop_reason TEXT")
            for column, declaration in {
                "deleted_at": "TEXT",
                "deleted_by": "TEXT",
                "deletion_reason": "TEXT",
            }.items():
                if column not in task_columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {declaration}")
            log_columns = {row["name"] for row in connection.execute("PRAGMA table_info(task_logs)").fetchall()}
            for column, declaration in {
                "attempt_id": "TEXT",
                "level": "TEXT NOT NULL DEFAULT 'INFO'",
                "source": "TEXT NOT NULL DEFAULT 'backend'",
            }.items():
                if column not in log_columns:
                    connection.execute(f"ALTER TABLE task_logs ADD COLUMN {column} {declaration}")
            metric_columns = {row["name"] for row in connection.execute("PRAGMA table_info(task_metrics)").fetchall()}
            if "attempt_id" not in metric_columns:
                connection.execute("ALTER TABLE task_metrics ADD COLUMN attempt_id TEXT")
            connection.execute("INSERT OR IGNORE INTO training_task_attempts(id, task_id, attempt_no, state, created_at) SELECT 'legacy-' || id, id, 1, state, created_at FROM tasks")
            connection.execute("UPDATE task_logs SET attempt_id = (SELECT a.id FROM training_task_attempts a WHERE a.task_id = task_logs.task_id ORDER BY a.attempt_no DESC LIMIT 1) WHERE attempt_id IS NULL")
            connection.execute("UPDATE task_metrics SET attempt_id = (SELECT a.id FROM training_task_attempts a WHERE a.task_id = task_metrics.task_id ORDER BY a.attempt_no DESC LIMIT 1) WHERE attempt_id IS NULL")
            dataset_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(datasets)").fetchall()
            }
            for column, declaration in {"deleted_at": "TEXT", "deleted_by": "TEXT", "deletion_reason": "TEXT"}.items():
                if column not in dataset_columns:
                    connection.execute(f"ALTER TABLE datasets ADD COLUMN {column} {declaration}")
            for column, declaration in {
                "version": "INTEGER NOT NULL DEFAULT 1",
                "validation_report": "TEXT NOT NULL DEFAULT '{}'",
                "repair_history": "TEXT NOT NULL DEFAULT '[]'",
                "source_type": "TEXT NOT NULL DEFAULT 'local'",
                "source_uri": "TEXT",
                "source_bucket": "TEXT",
                "source_key": "TEXT",
                "size_bytes": "INTEGER",
                "content_sha256": "TEXT",
            }.items():
                if column not in dataset_columns:
                    connection.execute(f"ALTER TABLE datasets ADD COLUMN {column} {declaration}")
            connection.execute(
                """
                INSERT OR IGNORE INTO task_metrics (
                    task_id, step, train_loss, created_at
                )
                SELECT id, 1, loss, updated_at
                FROM tasks
                WHERE loss IS NOT NULL
                """
            )
            connection.execute(
                """
                UPDATE tasks
                SET state = 'failed',
                    current_step = '训练服务重启，任务已中断',
                    error = '训练服务重启导致运行中的进程丢失，请重新运行。',
                    updated_at = ?
                WHERE state = 'running'
                """,
                (self._now_iso(),),
            )

    def _backfill_metrics_from_logs(self) -> None:
        with self._lock, self._connect() as connection:
            tasks = connection.execute("SELECT id, payload FROM tasks").fetchall()
            for task in tasks:
                logs = connection.execute(
                    "SELECT message FROM task_logs WHERE task_id = ? ORDER BY id ASC",
                    (task["id"],),
                ).fetchall()
                total_steps: int | None = None
                total_epochs: float | None = None
                values_by_step: dict[int, dict[str, float]] = {}
                last_train_step = 0
                payload = json.loads(task["payload"])
                logging_steps = max(
                    1, int(payload.get("parameters", {}).get("loggingSteps", 10))
                )
                for log in logs:
                    message = log["message"]
                    steps_match = TOTAL_STEPS_PATTERN.search(message)
                    if steps_match:
                        total_steps = int(steps_match.group(1).replace(",", ""))
                    epochs_match = NUM_EPOCHS_PATTERN.search(message)
                    if epochs_match:
                        total_epochs = float(epochs_match.group(1))
                    values = parse_trainer_values(message)
                    if values is None:
                        continue
                    step = infer_metric_step(
                        values,
                        total_steps=total_steps,
                        total_epochs=total_epochs,
                        last_train_step=last_train_step,
                        logging_steps=logging_steps,
                    )
                    if "loss" in values:
                        last_train_step = step
                    current = values_by_step.setdefault(step, {})
                    current.update(values)
                if len(values_by_step) <= 1:
                    continue
                existing_count = connection.execute(
                    "SELECT COUNT(*) FROM task_metrics WHERE task_id = ?",
                    (task["id"],),
                ).fetchone()[0]
                if existing_count >= len(values_by_step):
                    continue
                connection.execute(
                    "DELETE FROM task_metrics WHERE task_id = ?", (task["id"],)
                )
                for step, values in sorted(values_by_step.items()):
                    connection.execute(
                        """
                        INSERT INTO task_metrics (
                            task_id, step, epoch, train_loss, eval_loss,
                            learning_rate, created_at, attempt_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, (SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1))
                        """,
                        (
                            task["id"],
                            step,
                            values.get("epoch"),
                            values.get("loss"),
                            values.get("eval_loss"),
                            values.get("learning_rate"),
                            self._now_iso(),
                            task["id"],
                        ),
                    )

    @staticmethod
    def _now_iso() -> str:
        zone = display_timezone()
        now = datetime.now(zone) if zone else datetime.now().astimezone()
        return now.isoformat(timespec="seconds")

    @staticmethod
    def _display_time(value: str) -> str:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _new_id() -> str:
        zone = display_timezone()
        now = datetime.now(zone) if zone else datetime.now()
        return f"ft-{now.strftime('%y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = self._new_id()
        now = self._now_iso()
        remote = payload.get("provider") == "youyun" or payload.get("executionMode") == "remote"
        queued_step = "等待优云 GPU 资源" if remote else "等待本地 GPU 资源"
        queued_log = "任务已提交，进入优云 GPU 队列。" if remote else "任务已提交，进入本地单 GPU 队列。"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    id, payload, state, progress, current_step,
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?)
                """,
                (task_id, json.dumps(payload, ensure_ascii=False), queued_step, now, now),
            )
            connection.execute(
                "INSERT INTO training_task_attempts(id, task_id, attempt_no, state, created_at) VALUES (?, ?, 1, 'queued', ?)",
                (f"{task_id}-a1", task_id, now),
            )
        self.append_log(task_id, queued_log)
        task = self.get(task_id)
        assert task is not None
        return task

    def upsert_dataset(self, dataset: dict[str, Any]) -> None:
        now = self._now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    id, name, dataset_type, purpose, structure, state, sample_count,
                    paths, validation_message, validation_report, repair_history,
                    version, source_type, source_uri, source_bucket, source_key,
                    warning_count, owner, builtin, size_bytes, content_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    dataset_type = excluded.dataset_type,
                    purpose = excluded.purpose,
                    structure = excluded.structure,
                    state = excluded.state,
                    paths = excluded.paths,
                    sample_count = excluded.sample_count,
                    validation_message = excluded.validation_message,
                    validation_report = excluded.validation_report,
                    repair_history = excluded.repair_history,
                    version = excluded.version,
                    warning_count = excluded.warning_count,
                    builtin = excluded.builtin,
                    source_type = excluded.source_type,
                    source_uri = excluded.source_uri,
                    source_bucket = excluded.source_bucket,
                    source_key = excluded.source_key,
                    size_bytes = excluded.size_bytes,
                    content_sha256 = excluded.content_sha256,
                    updated_at = excluded.updated_at
                """,
                (
                    dataset["id"],
                    dataset["name"],
                    dataset["type"],
                    dataset.get("purpose", "training"),
                    dataset.get("structure"),
                    dataset["state"],
                    int(dataset.get("sampleCount", 0)),
                    json.dumps(dataset.get("paths", {}), ensure_ascii=False),
                    dataset.get("validationMessage", ""),
                    json.dumps(dataset.get("validationReport", {}), ensure_ascii=False),
                    json.dumps(dataset.get("repairHistory", []), ensure_ascii=False),
                    int(dataset.get("version", 1)),
                    dataset.get("sourceType") or (dataset.get("source") or {}).get("type") or "local",
                    dataset.get("sourceUri") or (dataset.get("source") or {}).get("uri"),
                    dataset.get("sourceBucket") or (dataset.get("source") or {}).get("bucket"),
                    dataset.get("sourceKey") or (dataset.get("source") or {}).get("key"),
                    int(dataset.get("warningCount", 0)),
                    dataset.get("owner", "平台管理员"),
                    1 if dataset.get("builtin") else 0,
                    (
                        dataset.get("sizeBytes")
                        if dataset.get("sizeBytes") is not None
                        else (dataset.get("source") or {}).get("sizeBytes")
                    ),
                    dataset.get("contentSha256") or (dataset.get("source") or {}).get("sha256"),
                    dataset.get("createdAt", now),
                    now,
                ),
            )

    def create_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        self.upsert_dataset(dataset)
        result = self.get_dataset(dataset["id"])
        assert result is not None
        return result

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets WHERE deleted_at IS NULL ORDER BY created_at DESC"
            ).fetchall()
        return [self._serialize_dataset(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE id = ? AND deleted_at IS NULL", (dataset_id,)
            ).fetchone()
        return self._serialize_dataset(row) if row else None

    def list_dataset_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            return []
        return [
            {
                "dataset_id": dataset["id"],
                "version": dataset.get("version", 1),
                "source_type": dataset.get("sourceType"),
                "source_uri": dataset.get("sourceUri"),
                "source_bucket": dataset.get("sourceBucket"),
                "source_key": dataset.get("sourceKey"),
                "size_bytes": dataset.get("sizeBytes"),
                "content_sha256": dataset.get("contentSha256"),
                "sample_count": dataset["sampleCount"],
                "paths": dataset["paths"],
                "validation_message": dataset["validationMessage"],
            }
        ]

    def update_dataset_state(
        self, dataset_id: str, state: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE datasets SET state = ?, updated_at = ? WHERE id = ?",
                (state, self._now_iso(), dataset_id),
            )
        return self.get_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            return None
        if dataset["builtin"]:
            raise ValueError("内置验证数据集不能删除。")
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE datasets SET deleted_at = ?, deleted_by = ?, deletion_reason = ? WHERE id = ?", (self._now_iso(), "local-admin", "user_request", dataset_id))
        return dataset

    def _serialize_dataset(self, row: sqlite3.Row) -> dict[str, Any]:
        dataset_type = row["dataset_type"]
        sample_count = row["sample_count"]
        return {
            "id": row["id"],
            "name": row["name"],
            "subtitle": (
                f"格式错误：{row['validation_message']}"
                if row["state"] == "failed"
                else (
                    ("已发布，可用于模型评测" if (row["purpose"] if "purpose" in row.keys() else "training") == "evaluation" else "已发布，可用于微调任务")
                    if row["state"] == "published"
                    else "文件内容校验通过，可发布"
                )
            ),
            "type": dataset_type,
            "purpose": row["purpose"] if "purpose" in row.keys() else "training",
            "structure": row["structure"],
            "count": f"{sample_count:,} {'张' if dataset_type == '文生图' else '条'}",
            "sampleCount": sample_count,
            "state": row["state"],
            "validationMessage": row["validation_message"],
            "validationReport": json.loads(row["validation_report"] or "{}"),
            "repairHistory": json.loads(row["repair_history"] or "[]"),
            "version": int(row["version"] or 1),
            "warningCount": row["warning_count"],
            "updatedAt": self._display_time(row["updated_at"]),
            "owner": row["owner"],
            "builtin": bool(row["builtin"]),
            "paths": json.loads(row["paths"]),
            "sourceType": row["source_type"],
            "sourceUri": row["source_uri"],
            "sourceBucket": row["source_bucket"],
            "sourceKey": row["source_key"],
            "sizeBytes": row["size_bytes"],
            "contentSha256": row["content_sha256"],
        }

    @staticmethod
    def _format_model_size(size_bytes: int | None) -> str:
        if not size_bytes or size_bytes <= 0:
            return "—"
        units = ("B", "KB", "MB", "GB", "TB")
        value = float(size_bytes)
        unit = units[0]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                break
            value /= 1024
        precision = 0 if unit == "B" else (1 if value < 10 else 0)
        return f"{value:.{precision}f} {unit}"

    @staticmethod
    def _path_info(path: str) -> tuple[str, int | None]:
        target = Path(path).expanduser()
        if not target.exists():
            return "unavailable", None
        try:
            if target.is_file():
                return "available", target.stat().st_size
            size = sum(
                item.stat().st_size
                for item in target.rglob("*")
                if item.is_file()
            )
            return "available", size
        except OSError:
            return "unavailable", None

    @staticmethod
    def _base_model_id(repo_id: str) -> str:
        digest = hashlib.sha1(repo_id.encode("utf-8")).hexdigest()[:12]
        return f"base-{digest}"

    def _serialize_model_asset(self, row: sqlite3.Row) -> dict[str, Any]:
        variants = json.loads(row["variants"] or "[]")
        result: dict[str, Any] = {
            "id": row["id"],
            "kind": row["kind"],
            "name": row["name"],
            "repo": row["repo_id"],
            "revision": row["revision"],
            "path": row["path"],
            "status": row["status"],
            "source": row["source"],
            "size": self._format_model_size(row["size_bytes"]),
            "updatedAt": self._display_time(row["updated_at"]),
            "archived": bool(row["archived"]),
        }
        if row["kind"] == "trained":
            result.update(
                {
                    "baseModelId": row["base_model_id"],
                    "baseModelName": row["source"],
                    "taskId": row["task_id"],
                    "method": row["method"],
                    "dataset": row["dataset_name"],
                    "variants": variants,
                }
            )
        return result

    def ensure_base_model(
        self,
        repo_id: str,
        *,
        path: str,
        source: str = "ModelScope 缓存",
        name: str | None = None,
        revision: str = "main",
    ) -> dict[str, Any]:
        clean_repo = repo_id.strip()
        if not clean_repo:
            raise ValueError("ModelScope 仓库 ID 不能为空。")
        model_id = self._base_model_id(clean_repo)
        status, size_bytes = self._path_info(path)
        display_name = name or clean_repo.rsplit("/", 1)[-1]
        now = self._now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_assets (
                    id, kind, name, repo_id, revision, path, status, source,
                    size_bytes, created_at, updated_at
                ) VALUES (?, 'base', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    repo_id = excluded.repo_id,
                    revision = excluded.revision,
                    path = CASE
                        WHEN model_assets.source = '手工登记'
                             AND excluded.source = 'ModelScope 缓存'
                        THEN model_assets.path
                        ELSE excluded.path
                    END,
                    status = CASE
                        WHEN model_assets.source = '手工登记'
                             AND excluded.source = 'ModelScope 缓存'
                        THEN model_assets.status
                        ELSE excluded.status
                    END,
                    source = CASE
                        WHEN model_assets.source = '手工登记'
                             AND excluded.source = 'ModelScope 缓存'
                        THEN model_assets.source
                        ELSE excluded.source
                    END,
                    size_bytes = CASE
                        WHEN model_assets.source = '手工登记'
                             AND excluded.source = 'ModelScope 缓存'
                        THEN model_assets.size_bytes
                        ELSE excluded.size_bytes
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    model_id,
                    display_name,
                    clean_repo,
                    revision or "main",
                    path,
                    status,
                    source,
                    size_bytes,
                    now,
                    now,
                ),
            )
        result = self.get_model_asset(model_id)
        assert result is not None
        return result

    def list_model_assets(
        self,
        *,
        include_archived: bool = False,
        archived_only: bool = False,
        kind: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if archived_only:
            clauses.append("archived = 1")
        elif not include_archived:
            clauses.append("archived = 0")
        if kind in {"base", "trained"}:
            clauses.append("kind = ?")
            values.append(kind)
        if query and query.strip():
            clauses.append(
                "(name LIKE ? OR repo_id LIKE ? OR path LIKE ? OR task_id LIKE ? OR dataset_name LIKE ?)"
            )
            needle = f"%{query.strip()}%"
            values.extend([needle] * 5)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM model_assets {where} ORDER BY updated_at DESC",
                values,
            ).fetchall()
        return [self._serialize_model_asset(row) for row in rows]

    def list_model_assets_page(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        include_archived: bool = False,
        archived_only: bool = False,
        kind: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        clauses: list[str] = []
        values: list[Any] = []
        if archived_only:
            clauses.append("archived = 1")
        elif not include_archived:
            clauses.append("archived = 0")
        if kind in {"base", "trained"}:
            clauses.append("kind = ?")
            values.append(kind)
        if query and query.strip():
            clauses.append(
                "(name LIKE ? OR repo_id LIKE ? OR path LIKE ? OR task_id LIKE ? OR dataset_name LIKE ?)"
            )
            needle = f"%{query.strip()}%"
            values.extend([needle] * 5)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM model_assets {where}", values
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"SELECT * FROM model_assets {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "items": [self._serialize_model_asset(row) for row in rows],
            "page": page,
            "pageSize": page_size,
            "total": total,
            "totalPages": total_pages,
        }

    def get_model_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_assets WHERE id = ?", (asset_id,)
            ).fetchone()
        return self._serialize_model_asset(row) if row else None

    def update_model_asset(
        self,
        asset_id: str,
        *,
        name: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None:
        if self.get_model_asset(asset_id) is None:
            return None
        fields = ["updated_at = ?"]
        values: list[Any] = [self._now_iso()]
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("模型显示名称不能为空。")
            fields.append("name = ?")
            values.append(clean_name)
        if archived is not None:
            fields.append("archived = ?")
            values.append(1 if archived else 0)
        values.append(asset_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE model_assets SET {', '.join(fields)} WHERE id = ?",
                values,
            )
        return self.get_model_asset(asset_id)

    def refresh_model_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if row is None:
            return None
        now = self._now_iso()
        if row["kind"] == "base":
            status, size_bytes = self._path_info(row["path"])
            with self._lock, self._connect() as connection:
                connection.execute(
                    "UPDATE model_assets SET status = ?, size_bytes = ?, updated_at = ? WHERE id = ?",
                    (status, size_bytes, now, asset_id),
                )
        else:
            variants = json.loads(row["variants"] or "[]")
            total_size = 0
            for variant in variants:
                variant_status, variant_size = self._path_info(variant["path"])
                variant["status"] = variant_status
                if variant_size:
                    total_size += variant_size
            status = "available" if variants and all(v["status"] == "available" for v in variants) else "unavailable"
            with self._lock, self._connect() as connection:
                connection.execute(
                    "UPDATE model_assets SET status = ?, size_bytes = ?, variants = ?, updated_at = ? WHERE id = ?",
                    (status, total_size or None, json.dumps(variants, ensure_ascii=False), now, asset_id),
                )
        return self.get_model_asset(asset_id)

    def _sync_model_asset_rows(
        self,
        rows: list[Any],
        model_cache_root: Path,
    ) -> None:
        for row in rows:
            payload = json.loads(row["payload"])
            repo_id = str(payload.get("model", "")).strip()
            if not repo_id:
                continue
            cache_path = model_cache_root / repo_id.replace("/", "--")
            self.ensure_base_model(repo_id, path=str(cache_path))
            if row["state"] != "completed":
                continue
            artifacts = json.loads(row["artifacts"] or "[]")
            adapters = [item for item in artifacts if item.get("type") == "adapter"]
            if not adapters:
                continue
            variants: list[dict[str, Any]] = []
            seen: set[str] = set()
            for artifact in adapters:
                key = artifact.get("variant") if artifact.get("variant") in {"best", "final"} else "final"
                if key in seen:
                    continue
                seen.add(key)
                variant_status, _ = self._path_info(str(artifact.get("path", "")))
                variants.append(
                    {
                        "key": key,
                        "label": "最佳 Adapter" if key == "best" else "最终 Adapter",
                        "path": artifact.get("path", ""),
                        "step": artifact.get("checkpointStep"),
                        "evalLoss": artifact.get("evalLoss"),
                        "status": variant_status,
                        "selected": artifact.get("selected") is True,
                    }
                )
            base = self.get_model_asset(self._base_model_id(repo_id))
            if base is None:
                continue
            task_id = row["id"]
            trained_id = f"trained-{task_id}"
            method = "LoRA" if payload.get("tuningMode") == "lora" else "QLoRA"
            name = payload.get("outputName") or payload.get("name") or trained_id
            now = row["updated_at"] or row["created_at"] or self._now_iso()
            total_size = 0
            for variant in variants:
                _, variant_size = self._path_info(variant["path"])
                if variant_size:
                    total_size += variant_size
            # The executor may store outputs under training-work rather than
            # ARTIFACT_ROOT. Keep the catalog path aligned with the actual
            # adapter location instead of manufacturing a stale path.
            asset_path = str(Path(variants[0]["path"]).parent)
            asset_status = "available" if all(v["status"] == "available" for v in variants) else "unavailable"
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_assets (
                        id, kind, name, repo_id, revision, path, status, source,
                        base_model_id, task_id, method, dataset_name, size_bytes,
                        variants, created_at, updated_at
                    ) VALUES (?, 'trained', ?, ?, 'main', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trained_id,
                        name,
                        repo_id,
                        asset_path,
                        asset_status,
                        base["name"],
                        base["id"],
                        task_id,
                        method,
                        payload.get("datasetName", ""),
                        total_size or None,
                        json.dumps(variants, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE model_assets SET
                        repo_id = ?, path = ?, status = ?, source = ?,
                        base_model_id = ?, task_id = ?, method = ?, dataset_name = ?,
                        size_bytes = ?, variants = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        repo_id,
                        asset_path,
                        asset_status,
                        base["name"],
                        base["id"],
                        task_id,
                        method,
                        payload.get("datasetName", ""),
                        total_size or None,
                        json.dumps(variants, ensure_ascii=False),
                        now,
                        trained_id,
                    ),
                )

    def sync_model_assets(self, model_cache_root: Path) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, payload, state, artifacts, created_at, updated_at FROM tasks"
            ).fetchall()
        self._sync_model_asset_rows(rows, model_cache_root)

    def list(self, *, include_details: bool = True) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE deleted_at IS NULL ORDER BY created_at DESC"
            ).fetchall()
        return [
            self._serialize(
                row,
                include_logs=include_details,
                include_metrics=include_details,
            )
            for row in rows
        ]

    def list_page(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        state: str | None = None,
        query: str | None = None,
        include_details: bool = False,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        clauses = ["deleted_at IS NULL"]
        values: list[Any] = []
        if state and state != "all":
            clauses.append("state = ?")
            values.append(state)
        if query and query.strip():
            clauses.append("(id LIKE ? OR payload LIKE ?)")
            needle = f"%{query.strip()}%"
            values.extend([needle, needle])
        where = " AND ".join(clauses)
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM tasks WHERE {where}", values).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM tasks WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "items": [self._serialize(row, include_logs=include_details, include_metrics=include_details) for row in rows],
            "page": page,
            "pageSize": page_size,
            "total": total,
            "totalPages": total_pages,
        }

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ? AND deleted_at IS NULL", (task_id,)
            ).fetchone()
        return self._serialize(row, include_logs=True, include_metrics=True) if row else None

    def get_payload(self, task_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM tasks WHERE id = ? AND deleted_at IS NULL", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return json.loads(row["payload"])

    def select_adapter(self, task_id: str, variant: str) -> dict[str, Any] | None:
        if variant not in {"best", "final"}:
            raise ValueError("Adapter 类型只能是 best 或 final。")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT state, artifacts FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            if row["state"] not in {"completed", "stopped"}:
                raise ValueError("只有已完成或已停止任务才能切换默认 Adapter。")
            artifacts = json.loads(row["artifacts"] or "[]")
            target = next(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.get("type") == "adapter"
                    and artifact.get("variant") == variant
                    and artifact.get("path")
                ),
                None,
            )
            if target is None:
                raise ValueError(f"该任务没有可用的 {variant} Adapter。")
            for artifact in artifacts:
                if artifact.get("type") == "adapter":
                    artifact["selected"] = artifact is target
            connection.execute(
                "UPDATE tasks SET artifacts = ?, updated_at = ? WHERE id = ?",
                (json.dumps(artifacts, ensure_ascii=False), self._now_iso(), task_id),
            )
        return self.get(task_id)

    def claim_next(self, provider: str | None = None) -> dict[str, Any] | None:
        if provider not in {None, "local", "youyun"}:
            raise ValueError("provider must be local, youyun, or None")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE state = 'queued' AND cancel_requested = 0
                ORDER BY created_at ASC
                """
            ).fetchall()
            row = None
            payload = None
            for candidate in rows:
                candidate_payload = json.loads(candidate["payload"])
                remote = (
                    candidate_payload.get("provider") == "youyun"
                    or candidate_payload.get("executionMode") == "remote"
                )
                if provider == "youyun" and not remote:
                    continue
                if provider == "local" and remote:
                    continue
                row = candidate
                payload = candidate_payload
                break
            if row is None:
                connection.commit()
                return None
            now = self._now_iso()
            assert payload is not None
            remote = payload.get("provider") == "youyun" or payload.get("executionMode") == "remote"
            running_step = "启动优云 GPU 训练" if remote else "启动本地 LLaMA-Factory 训练进程"
            connection.execute(
                """
                UPDATE tasks
                SET state = 'running', current_step = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (running_step, now, row["id"]),
            )
            connection.execute("UPDATE training_task_attempts SET state='running', started_at=COALESCE(started_at, ?) WHERE task_id=? AND id=(SELECT id FROM training_task_attempts WHERE task_id=? ORDER BY attempt_no DESC LIMIT 1)", (now, row["id"], row["id"]))
            connection.commit()
        return self.get(row["id"])

    def update(
        self,
        task_id: str,
        *,
        state: str | None = None,
        progress: int | None = None,
        current_step: str | None = None,
        loss: float | None = None,
        error: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        cancel_requested: bool | None = None,
        stop_reason: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        values: list[Any] = [self._now_iso()]
        changes = {
            "state": state,
            "progress": progress,
            "current_step": current_step,
            "loss": loss,
            "error": error,
        }
        for column, value in changes.items():
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        if artifacts is not None:
            fields.append("artifacts = ?")
            values.append(json.dumps(artifacts, ensure_ascii=False))
        if cancel_requested is not None:
            fields.append("cancel_requested = ?")
            values.append(1 if cancel_requested else 0)
        if stop_reason is not None:
            fields.append("stop_reason = ?")
            values.append(stop_reason)
        values.append(task_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", values
            )
            if state is not None:
                terminal = state in {"completed", "failed", "stopped"}
                connection.execute(
                    "UPDATE training_task_attempts SET state = ?, error = COALESCE(?, error), finished_at = CASE WHEN ? THEN COALESCE(finished_at, ?) ELSE finished_at END, started_at = COALESCE(started_at, ?) WHERE task_id = ? AND id = (SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1)",
                    (state, error, terminal, self._now_iso() if terminal else None, self._now_iso(), task_id, task_id),
                )

    def append_log(self, task_id: str, message: str, *, level: str = "INFO", source: str = "backend") -> None:
        clean_message = message.strip()
        if not clean_message:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO task_logs (task_id, created_at, message, attempt_id, level, source) VALUES (?, ?, ?, (SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1), ?, ?)",
                (task_id, self._now_iso(), clean_message[:8000], task_id, level.upper() if level.upper() in {"INFO", "WARN", "ERROR"} else "INFO", source if source in {"backend", "trainer", "vllm", "remote"} else "backend"),
            )

    def logs(self, task_id: str, limit: int = 300, attempt_id: str | None = None) -> list[str]:
        with self._connect() as connection:
            if attempt_id is None:
                row = connection.execute("SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1", (task_id,)).fetchone()
                attempt_id = row[0] if row else None
            clause = " AND attempt_id = ?" if attempt_id else ""
            rows = connection.execute(
                f"""
                SELECT created_at, message FROM task_logs
                WHERE task_id = ?{clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                (task_id, *( [attempt_id] if attempt_id else [] ), limit),
            ).fetchall()
        rows = list(reversed(rows))
        return [
            f"[{datetime.fromisoformat(row['created_at']).strftime('%H:%M:%S')}] {row['message']}"
            for row in rows
        ]

    def list_attempts(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, attempt_no, state, started_at, finished_at, error, created_at FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def log_items(self, task_id: str, *, attempt_id: str | None = None, after_id: int = 0, limit: int = 200) -> dict[str, Any]:
        with self._connect() as connection:
            if attempt_id is None:
                row = connection.execute("SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1", (task_id,)).fetchone()
                attempt_id = row[0] if row else None
            clause = " AND attempt_id = ?" if attempt_id else ""
            params = (task_id, *( [attempt_id] if attempt_id else [] ), after_id, min(max(limit, 1), 1000) + 1)
            rows = connection.execute(f"SELECT id, created_at, level, source, message FROM task_logs WHERE task_id = ?{clause} AND id > ? ORDER BY id ASC LIMIT ?", params).fetchall()
        has_more = len(rows) > min(max(limit, 1), 1000)
        rows = rows[: min(max(limit, 1), 1000)]
        return {"taskId": task_id, "attemptId": attempt_id, "items": [dict(row) for row in rows], "nextAfterId": rows[-1]["id"] if rows else after_id, "hasMore": has_more}

    def append_metric(
        self,
        task_id: str,
        *,
        step: int,
        epoch: float | None = None,
        train_loss: float | None = None,
        eval_loss: float | None = None,
        learning_rate: float | None = None,
    ) -> None:
        if train_loss is None and eval_loss is None:
            return
        with self._lock, self._connect() as connection:
            attempt_row = connection.execute("SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1", (task_id,)).fetchone()
            attempt_id = attempt_row[0] if attempt_row else None
            existing = connection.execute("SELECT id FROM task_metrics WHERE attempt_id IS ? AND step = ?", (attempt_id, step)).fetchone() if attempt_id else None
            if existing:
                connection.execute("UPDATE task_metrics SET epoch=COALESCE(?,epoch), train_loss=COALESCE(?,train_loss), eval_loss=COALESCE(?,eval_loss), learning_rate=COALESCE(?,learning_rate) WHERE id=?", (epoch, train_loss, eval_loss, learning_rate, existing[0]))
                return
            connection.execute(
                """
                INSERT INTO task_metrics (
                    task_id, step, epoch, train_loss, eval_loss,
                    learning_rate, created_at, attempt_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, (SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1))
                """,
                (
                    task_id,
                    step,
                    epoch,
                    train_loss,
                    eval_loss,
                    learning_rate,
                    self._now_iso(),
                    task_id,
                ),
            )

    def metrics(self, task_id: str, attempt_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if attempt_id is None:
                row = connection.execute("SELECT id FROM training_task_attempts WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1", (task_id,)).fetchone()
                attempt_id = row[0] if row else None
            clause = " AND attempt_id = ?" if attempt_id else ""
            rows = connection.execute(
                f"""
                SELECT step, epoch, train_loss, eval_loss, learning_rate
                FROM task_metrics
                WHERE task_id = ?{clause}
                ORDER BY step ASC
                """,
                (task_id, *( [attempt_id] if attempt_id else [] )),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            metric: dict[str, Any] = {"step": row["step"]}
            for key, column in (
                ("epoch", "epoch"),
                ("trainLoss", "train_loss"),
                ("evalLoss", "eval_loss"),
                ("learningRate", "learning_rate"),
            ):
                if row[column] is not None:
                    metric[key] = row[column]
            result.append(metric)
        return result

    def request_stop(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        if task["state"] == "queued":
            self.update(
                task_id,
                state="stopped",
                current_step="由用户停止",
                cancel_requested=True,
                stop_reason="manual",
            )
        elif task["state"] in {"running", "provisioning", "preparing", "collecting", "uploading_artifact"}:
            self.update(
                task_id,
                state="stopping",
                current_step="正在停止训练进程",
                cancel_requested=True,
            )
        self.append_log(task_id, "收到停止任务请求。")
        return self.get(task_id)

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def retry(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        if task["state"] not in {"failed", "stopped"}:
            raise ValueError("只有失败或已停止的任务可以重新运行。")
        if has_unreleased_youyun_instance(task.get("artifacts") or []):
            raise ValueError("优云实例尚未释放，请先结束任务并释放资源或销毁实例。")
        remote = task.get("provider") == "youyun" or task.get("executionMode") == "remote"
        queued_step = "等待优云 GPU 资源" if remote else "等待本地 GPU 资源"
        with self._lock, self._connect() as connection:
            next_attempt = connection.execute("SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM training_task_attempts WHERE task_id = ?", (task_id,)).fetchone()[0]
            attempt_id = f"{task_id}-a{next_attempt}"
            now = self._now_iso()
            connection.execute("INSERT INTO training_task_attempts(id, task_id, attempt_no, state, created_at) VALUES (?, ?, ?, 'queued', ?)", (attempt_id, task_id, next_attempt, now))
            connection.execute(
                """
                UPDATE tasks
                SET state = 'queued', progress = 0,
                    current_step = ?, loss = NULL,
                    error = NULL, artifacts = '[]', cancel_requested = 0,
                    stop_reason = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (queued_step, now, task_id),
            )
        self.append_log(task_id, "任务已重新加入优云 GPU 队列。" if remote else "任务已重新加入本地训练队列。")
        return self.get(task_id)

    def clone(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        payload = self.get_payload(task_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM tasks WHERE deleted_at IS NULL"
            ).fetchall()
        existing_names = [str(json.loads(row["payload"]).get("name") or "") for row in rows]
        payload["name"] = next_task_clone_name(payload["name"], existing_names)
        return self.create(payload)

    def delete(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None:
            return False
        if task["state"] in {"running", "stopping", "provisioning", "preparing", "collecting", "uploading_artifact"}:
            raise ValueError("运行中的任务需要先停止。")
        if has_unreleased_youyun_instance(task.get("artifacts") or []):
            raise ValueError("优云实例尚未释放，请先结束任务并释放资源或销毁实例。")
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE tasks SET deleted_at = ?, deleted_by = ?, deletion_reason = ? WHERE id = ?", (self._now_iso(), "local-admin", "user_request", task_id))
        return True

    def _serialize(
        self,
        row: sqlite3.Row,
        *,
        include_logs: bool,
        include_metrics: bool = True,
    ) -> dict[str, Any]:
        payload = json.loads(row["payload"])
        artifacts = json.loads(row["artifacts"])
        loss_history = self.metrics(row["id"]) if include_metrics else []
        eval_metrics = [
            metric for metric in loss_history if metric.get("evalLoss") is not None
        ]
        best_metric = (
            min(eval_metrics, key=lambda metric: metric["evalLoss"])
            if eval_metrics
            else None
        )
        final_eval_metric = (
            max(eval_metrics, key=lambda metric: metric["step"])
            if eval_metrics
            else None
        )
        adapter_artifacts = [
            artifact for artifact in artifacts if artifact.get("type") == "adapter"
        ]
        selected_adapter = resolve_adapter_artifact(artifacts)
        best_adapter = resolve_adapter_artifact(artifacts, "best")
        final_adapter = resolve_adapter_artifact(artifacts, "final")
        selected_variant = (
            selected_adapter.get("variant")
            if selected_adapter
            and selected_adapter.get("variant") in {"best", "final"}
            else ("final" if selected_adapter else None)
        )
        # New training runs persist the authoritative evaluation metadata on
        # each Adapter. Prefer it over rounded/incomplete log history; legacy
        # tasks continue to derive metrics from their historical log rows.
        best_step = (
            int(best_adapter["checkpointStep"])
            if best_adapter and best_adapter.get("checkpointStep") is not None
            else (best_metric["step"] if best_metric else None)
        )
        best_eval_loss = (
            float(best_adapter["evalLoss"])
            if best_adapter and best_adapter.get("evalLoss") is not None
            else (best_metric["evalLoss"] if best_metric else None)
        )
        final_step = (
            int(final_adapter["checkpointStep"])
            if final_adapter and final_adapter.get("checkpointStep") is not None
            else max(
                (
                    int(artifact["checkpointStep"])
                    for artifact in adapter_artifacts
                    if artifact.get("checkpointStep") is not None
                ),
                default=(max((metric["step"] for metric in loss_history), default=None)),
            )
        )
        final_eval_loss = (
            float(final_adapter["evalLoss"])
            if final_adapter and final_adapter.get("evalLoss") is not None
            else (final_eval_metric["evalLoss"] if final_eval_metric else None)
        )
        selection_metric = (
            str(best_adapter.get("metric"))
            if best_adapter and best_adapter.get("metric")
            else ("eval_loss" if best_eval_loss is not None or final_eval_loss is not None else None)
        )
        best_loss = (
            float(best_adapter["loss"])
            if best_adapter and best_adapter.get("loss") is not None
            else best_eval_loss
        )
        final_loss = (
            float(final_adapter["loss"])
            if final_adapter and final_adapter.get("loss") is not None
            else final_eval_loss
        )
        task = {
            "id": row["id"],
            "name": payload["name"],
            "state": row["state"],
            "model": payload["model"],
            "dataset": payload["datasetName"],
            "datasetId": payload["datasetId"],
            "taskType": payload.get("taskType", "SFT"),
            "executionMode": payload.get("executionMode", "local"),
            "provider": payload.get("provider", "local"),
            "youyunResource": payload.get("youyunResource"),
            "method": (
                f"{payload.get('taskType', 'SFT')} / LoRA"
                if payload.get("tuningMode") == "lora"
                else f"{payload.get('taskType', 'SFT')} / QLoRA"
            ),
            "tuningMode": payload.get("tuningMode", "qlora"),
            "progress": row["progress"],
            "currentStep": row["current_step"],
            "loss": row["loss"],
            "createdAt": self._display_time(row["created_at"]),
            "owner": payload.get("owner", "平台管理员"),
            "mergeModel": payload.get("mergeModel", False),
            "validationSplit": payload.get("validationSplit", 0),
            "outputName": payload.get("outputName"),
            "checkpointLimit": payload.get("checkpointLimit", 2),
            "checkpointStrategy": payload.get("checkpointStrategy", "epoch"),
            "checkpointInterval": payload.get("checkpointInterval", 100),
            "maxSteps": payload.get("maxSteps", -1),
            "earlyStopping": payload.get("earlyStopping", False),
            "patience": payload.get("patience", 3),
            "minDelta": payload.get("minDelta", 0),
            "extraParameters": payload.get("extraParameters", {}),
            "parameters": payload.get("parameters", {}),
            "baselineMessages": payload.get("baselineMessages", []),
            "chatParameters": payload.get(
                "chatParameters",
                {
                    "maxNewTokens": 256,
                    "temperature": 0.2,
                    "topP": 0.9,
                    "seed": 42,
                },
            ),
            "error": row["error"],
            "stopReason": row["stop_reason"],
            "artifacts": artifacts,
            "lossHistory": loss_history,
            "bestStep": best_step,
            "bestEvalLoss": best_eval_loss,
            "finalEvalLoss": final_eval_loss,
            "finalStep": final_step,
            "selectedAdapterPath": (
                selected_adapter.get("path") if selected_adapter else None
            ),
            "selectedAdapterVariant": (
                selected_variant
            ),
            "bestAdapterPath": best_adapter.get("path") if best_adapter else None,
            "finalAdapterPath": final_adapter.get("path") if final_adapter else None,
            "selectionMetric": selection_metric,
            "bestLoss": best_loss,
            "finalLoss": final_loss,
            "selectionReason": (selected_adapter.get("selectionReason") if selected_adapter else None) or (
                (
                    "验证集 Loss 最低"
                    if selection_metric == "eval_loss"
                    else "训练 Loss 最低"
                )
                if selected_variant == "best"
                else (
                    (
                        "旧任务未保留历史最佳权重，使用最终 Adapter"
                        if selected_adapter and best_metric and not best_adapter
                        else (
                            "未产生有效验证 Loss，使用最终 Adapter"
                            if selected_adapter and not best_metric
                            else (
                                "当前选择最终 Adapter"
                                if selected_adapter
                                else None
                            )
                        )
                    )
                )
            ),
        }
        if include_logs:
            task["logs"] = self.logs(row["id"])
        return task
