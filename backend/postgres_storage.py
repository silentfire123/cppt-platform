from __future__ import annotations

"""PostgreSQL persistence for durable training and inference state.

The existing SQLite store remains the local catalog/compatibility cache.  This
module owns the phase-one PostgreSQL migration and the two durable stores used
by production deployments. Connections use a bounded psycopg pool when the
psycopg-pool package is installed; development environments without that
optional package retain the short-lived connection fallback.
"""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.inference_service import InferenceCreate, InferenceStore, now_iso
from backend.store import (
    TaskStore,
    has_unreleased_youyun_instance,
    next_task_clone_name,
    resolve_adapter_artifact,
)


def display_timezone_name() -> str:
    return os.environ.get("XUNTUI_TZ", "Asia/Shanghai")


MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS training_tasks (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            state TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            current_step TEXT NOT NULL DEFAULT '',
            loss DOUBLE PRECISION,
            artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
            cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
            stop_reason TEXT,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS training_task_logs (
            id BIGSERIAL PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            message TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_task_metrics (
            id BIGSERIAL PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            epoch DOUBLE PRECISION,
            train_loss DOUBLE PRECISION,
            eval_loss DOUBLE PRECISION,
            learning_rate DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(task_id, step)
        );
        CREATE TABLE IF NOT EXISTS training_task_events (
            id BIGSERIAL PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
            old_state TEXT,
            new_state TEXT,
            stage TEXT,
            source TEXT NOT NULL DEFAULT 'backend',
            error TEXT,
            remote_instance_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS inference_services (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            task_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            config JSONB NOT NULL,
            state TEXT NOT NULL,
            exposure TEXT NOT NULL DEFAULT 'test',
            gpu TEXT NOT NULL DEFAULT '未占用',
            memory TEXT NOT NULL DEFAULT '0 / 0 GB',
            endpoint TEXT NOT NULL,
            recovery_attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS inference_logs (
            id BIGSERIAL PRIMARY KEY,
            service_id TEXT NOT NULL REFERENCES inference_services(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            message TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inference_metrics (
            service_id TEXT PRIMARY KEY REFERENCES inference_services(id) ON DELETE CASCADE,
            metrics JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS inference_command_usage (
            id BIGSERIAL PRIMARY KEY,
            service_id TEXT NOT NULL REFERENCES inference_services(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            flags JSONB NOT NULL DEFAULT '[]'::jsonb,
            success BOOLEAN NOT NULL DEFAULT FALSE,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_training_tasks_state_created
            ON training_tasks(state, created_at);
        CREATE INDEX IF NOT EXISTS idx_training_task_logs_task
            ON training_task_logs(task_id, id);
        CREATE INDEX IF NOT EXISTS idx_training_task_metrics_task_step
            ON training_task_metrics(task_id, step);
        CREATE INDEX IF NOT EXISTS idx_training_task_events_task
            ON training_task_events(task_id, id);
        CREATE INDEX IF NOT EXISTS idx_inference_services_state
            ON inference_services(state, updated_at);
        CREATE INDEX IF NOT EXISTS idx_inference_logs_service
            ON inference_logs(service_id, id);
        CREATE INDEX IF NOT EXISTS idx_inference_command_usage_service
            ON inference_command_usage(service_id, created_at);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            purpose TEXT NOT NULL DEFAULT 'training',
            structure TEXT,
            state TEXT NOT NULL,
            sample_count INTEGER NOT NULL DEFAULT 0,
            validation_message TEXT NOT NULL DEFAULT '',
            warning_count INTEGER NOT NULL DEFAULT 0,
            owner TEXT NOT NULL DEFAULT '平台管理员',
            builtin BOOLEAN NOT NULL DEFAULT FALSE,
            current_version_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            version INTEGER NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'local',
            source_uri TEXT,
            source_bucket TEXT,
            source_key TEXT,
            local_path TEXT,
            size_bytes BIGINT,
            content_sha256 TEXT,
            sample_count INTEGER NOT NULL DEFAULT 0,
            paths JSONB NOT NULL DEFAULT '{}'::jsonb,
            validation_message TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(dataset_id, version)
        );
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'datasets_current_version_fk'
            ) THEN
                ALTER TABLE datasets ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'training';
                ALTER TABLE datasets
                    ADD CONSTRAINT datasets_current_version_fk
                    FOREIGN KEY(current_version_id) REFERENCES dataset_versions(id)
                    ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;
            END IF;
        END $$;
        CREATE INDEX IF NOT EXISTS idx_datasets_state_updated
            ON datasets(state, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset
            ON dataset_versions(dataset_id, version DESC);
        """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS training_task_attempts (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES training_tasks(id) ON DELETE CASCADE,
            attempt_no INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued',
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(task_id, attempt_no)
        );
        CREATE INDEX IF NOT EXISTS idx_training_task_attempts_task
            ON training_task_attempts(task_id, attempt_no DESC);
        CREATE TABLE IF NOT EXISTS inference_service_runs (
            id TEXT PRIMARY KEY,
            service_id TEXT NOT NULL REFERENCES inference_services(id) ON DELETE CASCADE,
            run_no INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'starting',
            config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(service_id, run_no)
        );
        CREATE INDEX IF NOT EXISTS idx_inference_service_runs_service
            ON inference_service_runs(service_id, run_no DESC);
        CREATE TABLE IF NOT EXISTS inference_run_metrics (
            run_id TEXT PRIMARY KEY REFERENCES inference_service_runs(id) ON DELETE CASCADE,
            metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
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
        ALTER TABLE training_task_logs ADD COLUMN IF NOT EXISTS attempt_id TEXT;
        ALTER TABLE training_task_logs ADD COLUMN IF NOT EXISTS level TEXT NOT NULL DEFAULT 'INFO';
        ALTER TABLE training_task_logs ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'backend';
        ALTER TABLE training_task_metrics ADD COLUMN IF NOT EXISTS attempt_id TEXT;
        ALTER TABLE training_task_events ADD COLUMN IF NOT EXISTS attempt_id TEXT;
        ALTER TABLE inference_logs ADD COLUMN IF NOT EXISTS run_id TEXT;
        ALTER TABLE inference_logs ADD COLUMN IF NOT EXISTS level TEXT NOT NULL DEFAULT 'INFO';
        ALTER TABLE inference_logs ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'backend';
        CREATE INDEX IF NOT EXISTS idx_training_task_logs_attempt
            ON training_task_logs(attempt_id, id);
        CREATE INDEX IF NOT EXISTS idx_training_task_metrics_attempt
            ON training_task_metrics(attempt_id, step);
        ALTER TABLE training_task_metrics DROP CONSTRAINT IF EXISTS training_task_metrics_task_id_step_key;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_training_task_metrics_attempt_step
            ON training_task_metrics(attempt_id, step) WHERE attempt_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_inference_logs_run
            ON inference_logs(run_id, id);
        ALTER TABLE training_tasks ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
        ALTER TABLE training_tasks ADD COLUMN IF NOT EXISTS deleted_by TEXT;
        ALTER TABLE training_tasks ADD COLUMN IF NOT EXISTS deletion_reason TEXT;
        ALTER TABLE datasets ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
        ALTER TABLE datasets ADD COLUMN IF NOT EXISTS deleted_by TEXT;
        ALTER TABLE datasets ADD COLUMN IF NOT EXISTS deletion_reason TEXT;
        ALTER TABLE inference_services ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
        ALTER TABLE inference_services ADD COLUMN IF NOT EXISTS deleted_by TEXT;
        ALTER TABLE inference_services ADD COLUMN IF NOT EXISTS deletion_reason TEXT;
        CREATE INDEX IF NOT EXISTS idx_training_tasks_deleted ON training_tasks(deleted_at);
        CREATE INDEX IF NOT EXISTS idx_datasets_deleted ON datasets(deleted_at);
        CREATE INDEX IF NOT EXISTS idx_inference_services_deleted ON inference_services(deleted_at);
        DO $$ BEGIN
            IF to_regclass('public.model_files') IS NOT NULL THEN
                ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_dataset_id TEXT;
                ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_dataset_name TEXT;
                ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_dataset_sha256 TEXT;
                ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_task_name TEXT;
                ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;
            END IF;
            IF to_regclass('public.training_tasks') IS NOT NULL THEN COMMENT ON TABLE training_tasks IS '训练任务当前状态、配置和产物摘要'; END IF;
            IF to_regclass('public.training_task_logs') IS NOT NULL THEN COMMENT ON TABLE training_task_logs IS '训练过程运行日志，按任务关联'; END IF;
            IF to_regclass('public.training_task_metrics') IS NOT NULL THEN COMMENT ON TABLE training_task_metrics IS '训练过程结构化指标'; END IF;
            IF to_regclass('public.training_task_events') IS NOT NULL THEN COMMENT ON TABLE training_task_events IS '训练任务生命周期状态事件'; END IF;
            IF to_regclass('public.training_task_attempts') IS NOT NULL THEN COMMENT ON TABLE training_task_attempts IS '训练任务每次实际执行尝试'; END IF;
            IF to_regclass('public.inference_services') IS NOT NULL THEN COMMENT ON TABLE inference_services IS '推理服务配置与当前状态'; END IF;
            IF to_regclass('public.inference_logs') IS NOT NULL THEN COMMENT ON TABLE inference_logs IS '推理服务运行日志'; END IF;
            IF to_regclass('public.inference_metrics') IS NOT NULL THEN COMMENT ON TABLE inference_metrics IS '推理服务当前指标快照'; END IF;
            IF to_regclass('public.inference_service_runs') IS NOT NULL THEN COMMENT ON TABLE inference_service_runs IS '推理服务每次启动或重试运行'; END IF;
            IF to_regclass('public.datasets') IS NOT NULL THEN COMMENT ON TABLE datasets IS '数据集主记录与当前文件快照'; END IF;
            IF to_regclass('public.dataset_versions') IS NOT NULL THEN COMMENT ON TABLE dataset_versions IS '兼容性数据集文件快照记录'; END IF;
            IF to_regclass('public.model_files') IS NOT NULL THEN COMMENT ON TABLE model_files IS '基模或 Adapter 的共享存储逻辑对象'; END IF;
            IF to_regclass('public.model_file_entries') IS NOT NULL THEN COMMENT ON TABLE model_file_entries IS '共享模型对象的物理文件清单'; END IF;
            IF to_regclass('public.cppt_users') IS NOT NULL THEN COMMENT ON TABLE cppt_users IS 'OAuth 登录用户映射与资料快照'; END IF;
            IF to_regclass('public.audit_logs') IS NOT NULL THEN COMMENT ON TABLE audit_logs IS '用户操作审计记录'; END IF;
            IF to_regclass('public.training_tasks') IS NOT NULL THEN
                COMMENT ON COLUMN training_tasks.payload IS '训练提交参数 JSON，保留完整复现配置';
                COMMENT ON COLUMN training_tasks.state IS '任务当前状态：queued/running/completed/failed/stopped';
                COMMENT ON COLUMN training_tasks.deleted_at IS '软删除时间，非空时默认业务查询隐藏';
            END IF;
            IF to_regclass('public.training_task_logs') IS NOT NULL THEN
                COMMENT ON COLUMN training_task_logs.attempt_id IS '本条日志所属训练尝试';
                COMMENT ON COLUMN training_task_logs.level IS '日志级别 INFO/WARN/ERROR';
                COMMENT ON COLUMN training_task_logs.source IS '日志来源 backend/trainer/vllm/remote';
            END IF;
            IF to_regclass('public.inference_logs') IS NOT NULL THEN
                COMMENT ON COLUMN inference_logs.run_id IS '本条日志所属推理运行';
                COMMENT ON COLUMN inference_logs.level IS '日志级别 INFO/WARN/ERROR';
                COMMENT ON COLUMN inference_logs.source IS '日志来源 backend/trainer/vllm/remote';
            END IF;
            IF to_regclass('public.model_files') IS NOT NULL THEN
                COMMENT ON COLUMN model_files.source_dataset_name IS '训练时来源数据集名称快照';
                COMMENT ON COLUMN model_files.source_dataset_sha256 IS '训练时来源数据内容 SHA256';
                COMMENT ON COLUMN model_files.source_snapshot IS '训练来源脱敏快照 JSON';
            END IF;
        END $$;
        """,
    ),
    (
        4,
        """
        ALTER TABLE datasets ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'training';
        """,
    ),
)


def _psycopg() -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError(
            "PostgreSQL 存储需要 psycopg，请安装 requirements-storage.txt。"
        ) from exc
    return psycopg, dict_row


_PG_POOLS: dict[str, Any] = {}
_PG_POOL_LOCK = threading.RLock()


def _pg_pool(dsn: str) -> Any | None:
    """Use a bounded pool when psycopg_pool is installed; keep direct fallback for dev/tests."""
    try:
        from psycopg_pool import ConnectionPool
    except ImportError:  # pragma: no cover - optional dependency in local compatibility mode
        return None
    with _PG_POOL_LOCK:
        pool = _PG_POOLS.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=max(1, int(os.environ.get("XUNTUI_PG_POOL_MIN", "2"))),
                max_size=max(2, int(os.environ.get("XUNTUI_PG_POOL_MAX", "10"))),
                timeout=float(os.environ.get("XUNTUI_PG_POOL_TIMEOUT", "3")),
                open=True,
            )
            _PG_POOLS[dsn] = pool
        return pool


def close_pg_pools() -> None:
    """Close every process-local pool during application shutdown."""
    with _PG_POOL_LOCK:
        pools = list(_PG_POOLS.values())
        _PG_POOLS.clear()
    for pool in pools:
        try:
            pool.close()
        except Exception:
            pass


@contextmanager
def pg_connect(dsn: str) -> Iterator[Any]:
    psycopg, dict_row = _psycopg()
    pool = _pg_pool(dsn)
    if pool is not None:
        with pool.connection() as connection:
            connection.row_factory = dict_row
            try:
                connection.execute(
                    "SELECT set_config('TimeZone', %s, false)", (display_timezone_name(),)
                )
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return
    connection = psycopg.connect(dsn, row_factory=dict_row)
    try:
        connection.execute(
            "SELECT set_config('TimeZone', %s, false)", (display_timezone_name(),)
        )
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def run_postgres_migrations(dsn: str) -> None:
    """Apply idempotent numbered migrations and record their versions."""
    if not dsn.strip():
        raise RuntimeError("XUNTUI_STORAGE_BACKEND=postgres 时必须配置 POSTGRES_DSN。")
    with pg_connect(dsn) as connection:
        # Serialize startup migrations when more than one API/worker process
        # starts at the same time.  Every migration is idempotent, but the
        # version insert must also be protected from a duplicate-key race.
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext('xuntui-platform-schema'))"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        applied = {
            row["version"]
            for row in connection.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            connection.execute(sql)
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (%s)", (version,)
            )


class PostgresTaskStore:
    """Durable task/log/metric and dataset store.

    SQLite remains the compatibility catalog for model assets and local-only
    deployments. Dataset records are authoritative in PostgreSQL whenever
    this store is selected, with an immutable version row for every create.
    """

    _now_iso = staticmethod(TaskStore._now_iso)
    _display_time = staticmethod(TaskStore._display_time)

    def __init__(self, dsn: str, catalog: TaskStore) -> None:
        self.dsn = dsn
        self.catalog = catalog
        self._lock = threading.RLock()
        run_postgres_migrations(dsn)

    def __getattr__(self, name: str) -> Any:
        # Model-management tables remain in the SQLite compatibility catalog.
        return getattr(self.catalog, name)

    @staticmethod
    def _dataset_query() -> str:
        return """
            SELECT d.id, d.name, d.dataset_type, d.purpose, d.structure, d.state,
                   d.sample_count, d.validation_message, d.warning_count,
                   d.owner, d.builtin, d.current_version_id,
                   d.created_at::text AS created_at, d.updated_at::text AS updated_at,
                   d.deleted_at::text AS deleted_at, d.deleted_by, d.deletion_reason,
                   v.version, v.source_type, v.source_uri, v.source_bucket,
                   v.source_key, v.local_path, v.size_bytes, v.content_sha256,
                   v.paths::text AS paths, v.validation_message AS version_validation_message
            FROM datasets d
            LEFT JOIN dataset_versions v ON v.id = d.current_version_id
        """

    @classmethod
    def _serialize_dataset(cls, row: Any) -> dict[str, Any]:
        sample_count = int(row["sample_count"] or 0)
        state = row["state"]
        paths = row.get("paths") or "{}"
        if isinstance(paths, str):
            paths = json.loads(paths)
        result = {
            "id": row["id"],
            "name": row["name"],
            "subtitle": (
                f"格式错误：{row['validation_message']}"
                if state == "failed"
                else ("已发布，可用于模型评测" if (row.get("purpose") or "training") == "evaluation" else "已发布，可用于微调任务") if state == "published"
                else "文件内容校验通过，可发布"
            ),
            "type": row["dataset_type"],
            "purpose": row.get("purpose") or "training",
            "structure": row["structure"],
            "count": f"{sample_count:,} {'张' if row['dataset_type'] == '文生图' else '条'}",
            "sampleCount": sample_count,
            "state": state,
            "validationMessage": row["validation_message"],
            "warningCount": int(row["warning_count"] or 0),
            "updatedAt": cls._display_time(str(row["updated_at"])),
            "owner": row["owner"],
            "builtin": bool(row["builtin"]),
            "paths": paths,
            "version": row.get("version"),
            "sourceType": row.get("source_type"),
            "sourceUri": row.get("source_uri"),
            "sourceBucket": row.get("source_bucket"),
            "sourceKey": row.get("source_key"),
            "sizeBytes": row.get("size_bytes"),
            "contentSha256": row.get("content_sha256"),
            "deletedAt": row.get("deleted_at"),
        }
        return result

    @staticmethod
    def _version_payload(dataset: dict[str, Any]) -> dict[str, Any]:
        source = dataset.get("source") or {}
        return {
            "source_type": dataset.get("sourceType") or source.get("type") or "local",
            "source_uri": dataset.get("sourceUri") or source.get("uri"),
            "source_bucket": dataset.get("sourceBucket") or source.get("bucket"),
            "source_key": dataset.get("sourceKey") or source.get("key"),
            "local_path": (dataset.get("paths") or {}).get("trainPath"),
            "size_bytes": (
                dataset.get("sizeBytes")
                if dataset.get("sizeBytes") is not None
                else source.get("sizeBytes")
            ),
            "content_sha256": dataset.get("contentSha256") or source.get("sha256"),
            "sample_count": int(dataset.get("sampleCount", 0)),
            "paths": json.dumps(dataset.get("paths", {}), ensure_ascii=False),
            "validation_message": dataset.get("validationMessage", ""),
        }

    def upsert_dataset(self, dataset: dict[str, Any]) -> None:
        """Persist a dataset and create its first version when needed."""
        now = self._now_iso()
        version_id = str(dataset.get("versionId") or f"dsv-{uuid.uuid4().hex[:16]}")
        version = int(dataset.get("version") or 1)
        payload = self._version_payload(dataset)
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT current_version_id FROM datasets WHERE id = %s FOR UPDATE",
                (dataset["id"],),
            ).fetchone()
            if existing and existing["current_version_id"]:
                version_id = existing["current_version_id"]
                version = int(connection.execute(
                    "SELECT version FROM dataset_versions WHERE id = %s", (version_id,)
                ).fetchone()["version"])
            connection.execute(
                """
                INSERT INTO datasets(
                    id, name, dataset_type, purpose, structure, state, sample_count,
                    validation_message, warning_count, owner, builtin,
                    current_version_id, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    name = EXCLUDED.name, dataset_type = EXCLUDED.dataset_type,
                    purpose = EXCLUDED.purpose, structure = EXCLUDED.structure, state = EXCLUDED.state,
                    sample_count = EXCLUDED.sample_count,
                    validation_message = EXCLUDED.validation_message,
                    warning_count = EXCLUDED.warning_count, owner = EXCLUDED.owner,
                    current_version_id = COALESCE(datasets.current_version_id, EXCLUDED.current_version_id),
                    updated_at = EXCLUDED.updated_at
                """,
                (dataset["id"], dataset["name"], dataset["type"], dataset.get("purpose", "training"), dataset.get("structure"),
                 dataset["state"], int(dataset.get("sampleCount", 0)),
                 dataset.get("validationMessage", ""), int(dataset.get("warningCount", 0)),
                 dataset.get("owner", "平台管理员"), bool(dataset.get("builtin")),
                 version_id, dataset.get("createdAt", now), now),
            )
            connection.execute(
                """
                INSERT INTO dataset_versions(
                    id, dataset_id, version, source_type, source_uri, source_bucket,
                    source_key, local_path, size_bytes, content_sha256, sample_count,
                    paths, validation_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT(dataset_id, version) DO UPDATE SET
                    source_type = EXCLUDED.source_type, source_uri = EXCLUDED.source_uri,
                    source_bucket = EXCLUDED.source_bucket, source_key = EXCLUDED.source_key,
                    local_path = EXCLUDED.local_path, size_bytes = EXCLUDED.size_bytes,
                    content_sha256 = EXCLUDED.content_sha256, sample_count = EXCLUDED.sample_count,
                    paths = EXCLUDED.paths, validation_message = EXCLUDED.validation_message
                """,
                (version_id, dataset["id"], version, payload["source_type"], payload["source_uri"],
                 payload["source_bucket"], payload["source_key"], payload["local_path"],
                 payload["size_bytes"], payload["content_sha256"], payload["sample_count"],
                 payload["paths"], payload["validation_message"]),
            )

    def create_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        self.upsert_dataset(dataset)
        result = self.get_dataset(dataset["id"])
        assert result is not None
        return result

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                self._dataset_query() + " WHERE d.deleted_at IS NULL ORDER BY d.created_at DESC"
            ).fetchall()
        return [self._serialize_dataset(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                self._dataset_query() + " WHERE d.id = %s AND d.deleted_at IS NULL", (dataset_id,)
            ).fetchone()
        return self._serialize_dataset(row) if row else None

    def update_dataset_state(self, dataset_id: str, state: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE datasets SET state = %s, updated_at = %s WHERE id = %s",
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
            connection.execute("UPDATE datasets SET deleted_at = %s, deleted_by = %s, deletion_reason = %s WHERE id = %s", (self._now_iso(), "local-admin", "user_request", dataset_id))
        return dataset

    def list_dataset_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id = %s ORDER BY version DESC",
                (dataset_id,),
            ).fetchall()
        result = []
        for row in rows:
            paths = row["paths"]
            if isinstance(paths, str):
                paths = json.loads(paths)
            result.append({**dict(row), "paths": paths})
        return result

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        with pg_connect(self.dsn) as connection:
            yield connection

    @staticmethod
    def _task_query() -> str:
        return """
            SELECT id, payload::text AS payload, state, progress, current_step,
                   loss, artifacts::text AS artifacts, cancel_requested,
                   stop_reason, error, created_at::text AS created_at,
                   updated_at::text AS updated_at
            FROM training_tasks
        """

    def sync_model_assets(self, model_cache_root: Path) -> None:
        # Training tasks live in PostgreSQL in production. The model catalog
        # remains the SQLite compatibility catalog, so the inherited method
        # cannot read the durable PostgreSQL task rows through __getattr__.
        # Mirror the completed-task rows into the catalog synchronizer instead.
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, payload::text AS payload, state,
                       artifacts::text AS artifacts,
                       created_at::text AS created_at,
                       updated_at::text AS updated_at
                FROM training_tasks
                """
            ).fetchall()
        self.catalog._sync_model_asset_rows(rows, model_cache_root)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = f"ft-{datetime.now().strftime('%y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        timestamp = self._now_iso()
        remote = payload.get("provider") == "youyun" or payload.get("executionMode") == "remote"
        queued_step = "等待优云 GPU 资源" if remote else "等待本地训练资源"
        queued_log = "任务已创建，等待优云 GPU 资源。" if remote else "任务已创建，等待本地训练资源。"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_tasks(
                    id, payload, state, progress, current_step, artifacts,
                    cancel_requested, created_at, updated_at
                ) VALUES (%s, %s::jsonb, 'queued', 0, %s,
                          '[]'::jsonb, FALSE, %s, %s)
                """,
                (task_id, json.dumps(payload, ensure_ascii=False), queued_step, timestamp, timestamp),
            )
            connection.execute(
                "INSERT INTO training_task_attempts(id, task_id, attempt_no, state, created_at) VALUES (%s, %s, 1, 'queued', %s)",
                (f"{task_id}-a1", task_id, timestamp),
            )
            self._event(connection, task_id, None, "queued", "queued")
        self.append_log(task_id, queued_log)
        return self.get(task_id)  # type: ignore[return-value]

    def list(self, *, include_details: bool = True) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                self._task_query() + " WHERE deleted_at IS NULL ORDER BY created_at DESC"
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
        """Paginate durable PostgreSQL tasks instead of the SQLite compatibility catalog."""
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        clauses = ["deleted_at IS NULL"]
        values: list[Any] = []
        if state and state != "all":
            clauses.append("state = %s")
            values.append(state)
        if query and query.strip():
            clauses.append("(id ILIKE %s OR payload::text ILIKE %s)")
            needle = f"%{query.strip()}%"
            values.extend([needle, needle])
        where = " AND ".join(clauses)
        with self._connect() as connection:
            total_row = connection.execute(
                f"SELECT COUNT(*) AS total_count FROM training_tasks WHERE {where}",
                tuple(values),
            ).fetchone()
            total = int(total_row["total_count"] if total_row else 0)
            rows = connection.execute(
                self._task_query()
                + f" WHERE {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (*values, page_size, (page - 1) * page_size),
            ).fetchall()
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "items": [
                self._serialize(
                    row,
                    include_logs=include_details,
                    include_metrics=include_details,
                )
                for row in rows
            ],
            "page": page,
            "pageSize": page_size,
            "total": total,
            "totalPages": total_pages,
        }

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
            self._task_query() + " WHERE id = %s AND deleted_at IS NULL", (task_id,)
            ).fetchone()
        return self._serialize(row, include_logs=True, include_metrics=True) if row else None

    def get_payload(self, task_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload::text AS payload FROM training_tasks WHERE id = %s AND deleted_at IS NULL",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return json.loads(row["payload"])

    def select_adapter(self, task_id: str, variant: str) -> dict[str, Any] | None:
        if variant not in {"best", "final"}:
            raise ValueError("Adapter 类型只能是 best 或 final。")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT state, artifacts::text AS artifacts FROM training_tasks WHERE id = %s FOR UPDATE",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            if row["state"] not in {"completed", "stopped"}:
                raise ValueError("只有已完成或已停止任务才能切换默认 Adapter。")
            artifacts = json.loads(row["artifacts"] or "[]")
            target = next(
                (item for item in artifacts if item.get("type") == "adapter"
                 and item.get("variant") == variant and item.get("path")),
                None,
            )
            if target is None:
                raise ValueError(f"该任务没有可用的 {variant} Adapter。")
            for artifact in artifacts:
                if artifact.get("type") == "adapter":
                    artifact["selected"] = artifact is target
            connection.execute(
                "UPDATE training_tasks SET artifacts = %s::jsonb, updated_at = %s WHERE id = %s",
                (json.dumps(artifacts, ensure_ascii=False), self._now_iso(), task_id),
            )
        return self.get(task_id)

    def claim_next(self, provider: str | None = None) -> dict[str, Any] | None:
        if provider not in {None, "local", "youyun"}:
            raise ValueError("provider must be local, youyun, or None")
        provider_filter = ""
        if provider == "youyun":
            provider_filter = (
                "AND (COALESCE(payload->>'provider', 'local') = 'youyun' "
                "OR COALESCE(payload->>'executionMode', 'local') = 'remote')"
            )
        elif provider == "local":
            provider_filter = (
                "AND NOT (COALESCE(payload->>'provider', 'local') = 'youyun' "
                "OR COALESCE(payload->>'executionMode', 'local') = 'remote')"
            )
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT id, state, payload::text AS payload FROM training_tasks
                WHERE state = 'queued' AND cancel_requested = FALSE
                {provider_filter}
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            now = self._now_iso()
            payload = json.loads(row["payload"])
            remote = payload.get("provider") == "youyun" or payload.get("executionMode") == "remote"
            running_step = "启动优云 GPU 训练" if remote else "启动本地 LLaMA-Factory 训练"
            connection.execute(
                """
                UPDATE training_tasks
                SET state = 'running', current_step = %s, updated_at = %s
                WHERE id = %s
                """,
                (running_step, now, row["id"]),
            )
            connection.execute("UPDATE training_task_attempts SET state='running', started_at=COALESCE(started_at, %s) WHERE task_id=%s AND id=(SELECT id FROM training_task_attempts WHERE task_id=%s ORDER BY attempt_no DESC LIMIT 1)", (now, row["id"], row["id"]))
            self._event(connection, row["id"], row["state"], "running", "worker")
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
        fields = ["updated_at = %s"]
        values: list[Any] = [self._now_iso()]
        for column, value in {
            "state": state,
            "progress": progress,
            "current_step": current_step,
            "loss": loss,
            "error": error,
        }.items():
            if value is not None:
                fields.append(f"{column} = %s")
                values.append(value)
        if artifacts is not None:
            fields.append("artifacts = %s::jsonb")
            values.append(json.dumps(artifacts, ensure_ascii=False))
        if cancel_requested is not None:
            fields.append("cancel_requested = %s")
            values.append(cancel_requested)
        if stop_reason is not None:
            fields.append("stop_reason = %s")
            values.append(stop_reason)
        values.append(task_id)
        with self._lock, self._connect() as connection:
            previous = connection.execute(
                "SELECT state FROM training_tasks WHERE id = %s FOR UPDATE", (task_id,)
            ).fetchone()
            connection.execute(
                f"UPDATE training_tasks SET {', '.join(fields)} WHERE id = %s", values
            )
            if previous and state and previous["state"] != state:
                self._event(connection, task_id, previous["state"], state, state, error)
            if state is not None:
                terminal = state in {"completed", "failed", "stopped"}
                connection.execute(
                    "UPDATE training_task_attempts SET state = %s, error = COALESCE(%s, error), finished_at = CASE WHEN %s THEN COALESCE(finished_at, %s) ELSE finished_at END, started_at = COALESCE(started_at, %s) WHERE task_id = %s AND id = (SELECT id FROM training_task_attempts WHERE task_id = %s ORDER BY attempt_no DESC LIMIT 1)",
                    (state, error, terminal, self._now_iso() if terminal else None, self._now_iso(), task_id, task_id),
                )

    def append_log(self, task_id: str, message: str, *, level: str = "INFO", source: str = "backend") -> None:
        clean = message.strip()
        if not clean:
            return
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO training_task_logs(task_id, message, attempt_id, level, source) VALUES (%s, %s, (SELECT id FROM training_task_attempts WHERE task_id=%s ORDER BY attempt_no DESC LIMIT 1), %s, %s)",
                (task_id, clean[:8000], task_id, level.upper() if level.upper() in {"INFO", "WARN", "ERROR"} else "INFO", source if source in {"backend", "trainer", "vllm", "remote"} else "backend"),
            )

    def logs(self, task_id: str, limit: int = 300, attempt_id: str | None = None) -> list[str]:
        with self._connect() as connection:
            if attempt_id is None:
                row = connection.execute("SELECT id FROM training_task_attempts WHERE task_id = %s ORDER BY attempt_no DESC LIMIT 1", (task_id,)).fetchone()
                attempt_id = row["id"] if row else None
            clause = " AND attempt_id = %s" if attempt_id else ""
            rows = connection.execute(
                f"""
                SELECT created_at::text AS created_at, message
                FROM training_task_logs WHERE task_id = %s
                {clause}
                ORDER BY id DESC LIMIT %s
                """, (task_id, *( [attempt_id] if attempt_id else [] ), limit)
            ).fetchall()
        rows = list(reversed(rows))
        return [
            f"[{datetime.fromisoformat(row['created_at']).strftime('%H:%M:%S')}] {row['message']}"
            for row in rows
        ]

    def list_attempts(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, attempt_no, state, started_at::text AS started_at, finished_at::text AS finished_at, error, created_at::text AS created_at FROM training_task_attempts WHERE task_id = %s ORDER BY attempt_no DESC", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def log_items(self, task_id: str, *, attempt_id: str | None = None, after_id: int = 0, limit: int = 200) -> dict[str, Any]:
        with self._connect() as connection:
            if attempt_id is None:
                row = connection.execute("SELECT id FROM training_task_attempts WHERE task_id = %s ORDER BY attempt_no DESC LIMIT 1", (task_id,)).fetchone()
                attempt_id = row["id"] if row else None
            clause = " AND attempt_id = %s" if attempt_id else ""
            params = (task_id, *( [attempt_id] if attempt_id else [] ), after_id, min(max(limit, 1), 1000) + 1)
            rows = connection.execute(f"SELECT id, created_at::text AS created_at, level, source, message FROM training_task_logs WHERE task_id = %s{clause} AND id > %s ORDER BY id ASC LIMIT %s", params).fetchall()
        size = min(max(limit, 1), 1000)
        has_more = len(rows) > size
        rows = rows[:size]
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
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_task_metrics(
                    task_id, step, epoch, train_loss, eval_loss, learning_rate, attempt_id
                ) VALUES (%s, %s, %s, %s, %s, %s, (SELECT id FROM training_task_attempts WHERE task_id=%s ORDER BY attempt_no DESC LIMIT 1))
                ON CONFLICT (attempt_id, step) WHERE attempt_id IS NOT NULL DO UPDATE SET
                    epoch = COALESCE(EXCLUDED.epoch, training_task_metrics.epoch),
                    train_loss = COALESCE(EXCLUDED.train_loss, training_task_metrics.train_loss),
                    eval_loss = COALESCE(EXCLUDED.eval_loss, training_task_metrics.eval_loss),
                    learning_rate = COALESCE(EXCLUDED.learning_rate, training_task_metrics.learning_rate)
                """,
                (task_id, step, epoch, train_loss, eval_loss, learning_rate, task_id),
            )

    def metrics(self, task_id: str, attempt_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            clause = " AND attempt_id = %s" if attempt_id else ""
            rows = connection.execute(
                f"""
                SELECT step, epoch, train_loss, eval_loss, learning_rate
                FROM training_task_metrics WHERE task_id = %s{clause} ORDER BY step ASC
                """, (task_id, *( [attempt_id] if attempt_id else [] ))
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            metric: dict[str, Any] = {"step": row["step"]}
            for key, column in (("epoch", "epoch"), ("trainLoss", "train_loss"),
                                ("evalLoss", "eval_loss"), ("learningRate", "learning_rate")):
                if row[column] is not None:
                    metric[key] = row[column]
            result.append(metric)
        return result

    def request_stop(self, task_id: str) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        if task["state"] == "queued":
            self.update(task_id, state="stopped", current_step="由用户停止",
                        cancel_requested=True, stop_reason="manual")
        elif task["state"] in {"running", "provisioning", "preparing", "collecting", "uploading_artifact"}:
            self.update(task_id, state="stopping", current_step="正在停止训练进程", cancel_requested=True)
        self.append_log(task_id, "收到停止任务请求。")
        return self.get(task_id)

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM training_tasks WHERE id = %s", (task_id,)
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
        queued_step = "等待优云 GPU 资源" if remote else "等待本地训练资源"
        with self._connect() as connection:
            next_attempt = connection.execute("SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_no FROM training_task_attempts WHERE task_id = %s", (task_id,)).fetchone()["next_no"]
            attempt_id = f"{task_id}-a{next_attempt}"
            now = self._now_iso()
            connection.execute("INSERT INTO training_task_attempts(id, task_id, attempt_no, state, created_at) VALUES (%s, %s, %s, 'queued', %s)", (attempt_id, task_id, next_attempt, now))
            connection.execute(
                """
                UPDATE training_tasks SET state = 'queued', progress = 0,
                    current_step = %s, loss = NULL, error = NULL,
                    artifacts = '[]'::jsonb, cancel_requested = FALSE,
                    stop_reason = NULL, updated_at = %s WHERE id = %s
                """, (queued_step, now, task_id)
            )
            self._event(connection, task_id, task["state"], "queued", "retry")
        self.append_log(task_id, "任务已重新加入优云 GPU 队列。" if remote else "任务已重新加入本地训练队列。")
        return self.get(task_id)

    def clone(self, task_id: str) -> dict[str, Any] | None:
        if self.get(task_id) is None:
            return None
        payload = self.get_payload(task_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload->>'name' AS name FROM training_tasks WHERE deleted_at IS NULL"
            ).fetchall()
        existing_names = [str(row["name"] or "") for row in rows]
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
        with self._connect() as connection:
            connection.execute("UPDATE training_tasks SET deleted_at = %s, deleted_by = %s, deletion_reason = %s WHERE id = %s", (self._now_iso(), "local-admin", "user_request", task_id))
        return True

    def _event(
        self,
        connection: Any,
        task_id: str,
        old_state: str | None,
        new_state: str | None,
        stage: str,
        error: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO training_task_events(task_id, old_state, new_state, stage, source, error)
            VALUES (%s, %s, %s, %s, 'backend', %s)
            """, (task_id, old_state, new_state, stage, error)
        )

    def _serialize(
        self,
        row: Any,
        *,
        include_logs: bool,
        include_metrics: bool = True,
    ) -> dict[str, Any]:
        # Reuse the mature API serializer while the durable SQL schema evolves.
        return TaskStore._serialize(
            self,
            row,
            include_logs=include_logs,
            include_metrics=include_metrics,
        )

    def reconcile_interrupted_tasks(self) -> list[str]:
        """Mark active PostgreSQL tasks as failed after a backend restart."""
        active_states = (
            "running",
            "provisioning",
            "preparing",
            "stopping",
            "collecting",
            "uploading_artifact",
        )
        placeholders = ", ".join(["%s"] * len(active_states))
        message = "训练服务重启导致运行中的进程丢失，请重新运行。"
        recovered: list[str] = []
        now = self._now_iso()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, state FROM training_tasks WHERE state IN ({placeholders})",
                active_states,
            ).fetchall()
            for row in rows:
                task_id = row["id"]
                connection.execute(
                    """
                    UPDATE training_tasks
                    SET state = 'failed', current_step = '训练服务重启，任务已中断',
                        error = %s, cancel_requested = FALSE, updated_at = %s
                    WHERE id = %s
                    """,
                    (message, now, task_id),
                )
                connection.execute(
                    "UPDATE training_task_attempts SET state = 'failed', error = %s, finished_at = COALESCE(finished_at, %s) WHERE task_id = %s AND state IN ('queued', 'running')",
                    (message, now, task_id),
                )
                self._event(connection, task_id, row["state"], "failed", "restart", message)
                recovered.append(task_id)
        for task_id in recovered:
            self.append_log(task_id, f"{message} 当前任务已标记为失败。")
        return recovered


class _JsonRow(dict[str, Any]):
    """Make JSONB values look like SQLite JSON text to inherited serializers."""

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if key in {"config", "flags", "metrics"} and not isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return value


class _PgCursor:
    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor

    def fetchone(self) -> Any:
        row = self.cursor.fetchone()
        return _JsonRow(row) if row is not None else None

    def fetchall(self) -> list[Any]:
        return [_JsonRow(row) for row in self.cursor.fetchall()]


class _PgConnectionAdapter:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    @staticmethod
    def _sql(sql: str) -> str:
        return sql.replace("?", "%s")

    def execute(self, sql: str, params: Any = ()) -> _PgCursor:
        return _PgCursor(self.connection.execute(self._sql(sql), params))


class PostgresInferenceStore(InferenceStore):
    """Drop-in InferenceStore implementation backed by the phase-one schema."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        run_postgres_migrations(dsn)

    @contextmanager
    def _connect(self) -> Iterator[_PgConnectionAdapter]:
        with pg_connect(self.dsn) as connection:
            yield _PgConnectionAdapter(connection)

    def create_run(self, service_id: str, config: dict[str, Any] | None = None) -> str:
        run_id = f"{service_id}-r{uuid.uuid4().hex[:10]}"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(run_no), 0) + 1 AS next_no FROM inference_service_runs WHERE service_id = ?",
                (service_id,),
            ).fetchone()
            run_no = int(row["next_no"] if row else 1)
            connection.execute(
                "INSERT INTO inference_service_runs(id, service_id, run_no, state, config_snapshot, started_at, created_at) VALUES (?, ?, ?, 'starting', ?::jsonb, ?, ?)",
                (run_id, service_id, run_no, json.dumps(config or {}, ensure_ascii=False), now_iso(), now_iso()),
            )
        return run_id

    def update(self, service_id: str, **changes: Any) -> dict[str, Any] | None:
        config = changes.pop("config", None)
        if config is not None:
            if isinstance(config, str):
                config = json.loads(config)
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE inference_services
                    SET config = ?::jsonb, updated_at = ?
                    WHERE id = ?
                    """,
                    (json.dumps(config, ensure_ascii=False), now_iso(), service_id),
                )
        if changes:
            return super().update(service_id, **changes)
        return self.get(service_id)

    def create(
        self,
        *,
        service_id: str,
        payload: InferenceCreate,
        model: str,
        adapter: str,
        endpoint: str,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        config = payload.model_dump()
        for key in ("maxTokens", "temperature", "topP", "stream"):
            config.pop(key, None)
        config.update({"model": model, "adapter": adapter})
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_services(
                    id, name, description, task_id, owner, config, state,
                    exposure, endpoint, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?::jsonb, 'created', 'test', ?, ?, ?)
                """,
                (service_id, payload.name, payload.description, payload.taskId,
                 payload.owner, json.dumps(config, ensure_ascii=False), endpoint,
                 timestamp, timestamp),
            )
        self.append_log(service_id, "服务配置已创建，等待手动启动。")
        return self.get(service_id)  # type: ignore[return-value]

    def record_command_usage(
        self, service_id: str, *, flags: list[str], success: bool, error: str | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO inference_command_usage(service_id, flags, success, error) VALUES (?, ?::jsonb, ?, ?)",
                (service_id, json.dumps(sorted(set(flags)), ensure_ascii=False), success, error),
            )

    def set_metrics(self, service_id: str, metrics: dict[str, Any]) -> None:
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_metrics(service_id, metrics, updated_at)
                VALUES (?, ?::jsonb, ?)
                ON CONFLICT(service_id) DO UPDATE SET metrics = EXCLUDED.metrics,
                    updated_at = EXCLUDED.updated_at
                """,
                (service_id, json.dumps(metrics, ensure_ascii=False), timestamp),
            )
            try:
                run = connection.execute(
                    "SELECT id FROM inference_service_runs WHERE service_id = ? ORDER BY run_no DESC LIMIT 1",
                    (service_id,),
                ).fetchone()
                if run:
                    connection.execute(
                        "INSERT INTO inference_run_metrics(run_id, metrics, updated_at) VALUES (?, ?::jsonb, ?) ON CONFLICT(run_id) DO UPDATE SET metrics = EXCLUDED.metrics, updated_at = EXCLUDED.updated_at",
                        (run["id"], json.dumps(metrics, ensure_ascii=False), timestamp),
                    )
            except Exception as exc:
                # Keep compatibility with databases created before the run
                # migration; the service-level snapshot is still durable.
                if "inference_service_runs" not in str(exc) and "inference_run_metrics" not in str(exc):
                    raise
