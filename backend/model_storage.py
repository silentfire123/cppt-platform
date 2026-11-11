from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MODEL_FILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_files (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK (artifact_type IN ('base_model', 'lora_adapter')),
    artifact_variant TEXT NOT NULL DEFAULT 'base' CHECK (artifact_variant IN ('base', 'best', 'final')),
    training_type TEXT CHECK (training_type IN ('SFT', 'DPO')),
    tuning_method TEXT CHECK (tuning_method IN ('LoRA', 'QLoRA')),
    base_model_id TEXT,
    base_model_name TEXT,
    base_model_repo TEXT,
    base_model_revision TEXT,
    base_model_local_path TEXT,
    dataset_id TEXT,
    source_dataset_id TEXT,
    source_dataset_name TEXT,
    source_dataset_sha256 TEXT,
    source_platform TEXT,
    source_uri TEXT,
    source_task_id TEXT,
    uploaded_by TEXT NOT NULL DEFAULT 'local-admin',
    status TEXT NOT NULL DEFAULT 'pending_upload',
    total_size_bytes BIGINT NOT NULL DEFAULT 0,
    manifest_md5 TEXT,
    minio_bucket TEXT,
    minio_object_prefix TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    upload_started_at TIMESTAMPTZ,
    uploaded_at TIMESTAMPTZ,
    error_message TEXT,
    extra_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS model_file_entries (
    id BIGSERIAL PRIMARY KEY,
    model_file_id TEXT NOT NULL REFERENCES model_files(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    minio_object_key TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    md5 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_upload',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    uploaded_at TIMESTAMPTZ,
    error_message TEXT,
    UNIQUE(model_file_id, relative_path),
    UNIQUE(minio_object_key)
);

CREATE INDEX IF NOT EXISTS idx_model_files_status_created
    ON model_files(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_files_source_task
    ON model_files(source_task_id);
CREATE INDEX IF NOT EXISTS idx_model_file_entries_model
    ON model_file_entries(model_file_id);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


@dataclass(frozen=True)
class ModelStorageConfig:
    postgres_dsn: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "ModelStorageConfig":
        _load_dotenv(dotenv_path or Path(__file__).resolve().parent.parent / ".env")
        return cls(
            postgres_dsn=os.environ.get("POSTGRES_DSN", "").strip(),
            minio_endpoint=os.environ.get("MINIO_ENDPOINT", "").strip().removeprefix("http://").removeprefix("https://").rstrip("/"),
            minio_access_key=os.environ.get("MINIO_ACCESS_KEY", "").strip(),
            minio_secret_key=os.environ.get("MINIO_SECRET_KEY", "").strip(),
            minio_bucket=os.environ.get("MINIO_BUCKET", "rag-assets").strip(),
            minio_secure=_env_bool(os.environ.get("MINIO_SECURE"), False),
        )

    def validate(self) -> None:
        missing = [
            key
            for key, value in {
                "POSTGRES_DSN": self.postgres_dsn,
                "MINIO_ENDPOINT": self.minio_endpoint,
                "MINIO_ACCESS_KEY": self.minio_access_key,
                "MINIO_SECRET_KEY": self.minio_secret_key,
                "MINIO_BUCKET": self.minio_bucket,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"模型对象存储配置缺少：{', '.join(missing)}")


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.md5()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _manifest_md5(entries: Iterable[dict[str, Any]]) -> str:
    payload = [
        {
            "relativePath": entry["relative_path"],
            "sizeBytes": entry["size_bytes"],
            "md5": entry["md5"],
        }
        for entry in entries
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or "\\" in value or ":" in value or path.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError("模型文件清单包含不安全的相对路径。")
    return value


def adapter_is_complete(root: Path) -> bool:
    """Check the loadable Adapter configuration and all declared weight shards."""
    try:
        root = root.resolve(strict=True)
        config = root / "adapter_config.json"
        if not config.resolve().is_relative_to(root) or not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
            return False
        for filename in ("adapter_model.safetensors", "adapter_model.bin"):
            weight = root / filename
            if weight.is_file() and weight.resolve().is_relative_to(root) and weight.stat().st_size > 0:
                return True
        for filename in ("adapter_model.safetensors.index.json", "adapter_model.bin.index.json"):
            index = root / filename
            if not index.is_file() or not index.resolve().is_relative_to(root):
                continue
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                continue
            shards = [root / _safe_relative_path(str(name)) for name in mapping.values()]
            if all(path.is_file() and path.resolve().is_relative_to(root) and path.stat().st_size > 0 for path in shards):
                return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return False


def iter_model_files(root: Path, artifact_type: str) -> list[tuple[Path, str]]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{root}")
    if artifact_type not in {"base_model", "lora_adapter"}:
        raise ValueError("不支持的模型文件类型。")
    allowed_base = {".safetensors", ".bin", ".json", ".model", ".txt", ".py", ".yaml", ".yml", ".tiktoken"}
    allowed_adapter = {".safetensors", ".bin", ".json", ".model", ".txt", ".py", ".yaml", ".yml", ".tiktoken"}
    allowed = allowed_base if artifact_type == "base_model" else allowed_adapter
    result: list[tuple[Path, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        relative_parts = Path(relative).parts
        if any(part.startswith(".") or part.startswith("checkpoint-") for part in relative_parts):
            continue
        if path.suffix.lower() not in allowed:
            continue
        if any(token in relative.lower() for token in ("optimizer", "scheduler", "trainer_state", "training_args", "runs/", "logs/")):
            continue
        result.append((path, relative))
    if not result:
        raise ValueError(f"目录中没有找到可上传的模型文件：{root}")
    return result


class PostgresModelMetadataStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("缺少 psycopg 依赖，请安装 requirements-storage.txt。") from exc
        self._psycopg = psycopg
        self._dict_row = dict_row

    def connect(self):
        return self._psycopg.connect(self.dsn, row_factory=self._dict_row)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(MODEL_FILE_SCHEMA)
            connection.execute(
                "ALTER TABLE model_files ADD COLUMN IF NOT EXISTS artifact_variant TEXT NOT NULL DEFAULT 'base'"
            )
            for statement in (
                "ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_dataset_id TEXT",
                "ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_dataset_name TEXT",
                "ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_dataset_sha256 TEXT",
                "ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_task_name TEXT",
                "ALTER TABLE model_files ADD COLUMN IF NOT EXISTS source_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
            ):
                connection.execute(statement)
            connection.execute(
                """
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'model_files_artifact_variant_check'
                    ) THEN
                        ALTER TABLE model_files
                            ADD CONSTRAINT model_files_artifact_variant_check
                            CHECK (artifact_variant IN ('base', 'best', 'final'));
                    END IF;
                END $$
                """
            )

    def create_pending(self, metadata: dict[str, Any], entries: list[dict[str, Any]]) -> str:
        model_id = str(metadata.get("id") or f"model-{uuid.uuid4().hex[:16]}")
        manifest = _manifest_md5(entries)
        now = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_files (
                    id, name, artifact_type, training_type, tuning_method,
                    artifact_variant,
                    base_model_id, base_model_name, base_model_repo, base_model_revision,
                    base_model_local_path, dataset_id, source_dataset_id, source_dataset_name, source_dataset_sha256, source_task_name, source_snapshot, source_platform, source_uri,
                    source_task_id, uploaded_by, status, total_size_bytes, manifest_md5,
                    minio_bucket, minio_object_prefix, created_at, extra_metadata
                ) VALUES (
                    %(id)s, %(name)s, %(artifact_type)s, %(training_type)s, %(tuning_method)s,
                    %(artifact_variant)s,
                    %(base_model_id)s, %(base_model_name)s, %(base_model_repo)s, %(base_model_revision)s,
                    %(base_model_local_path)s, %(dataset_id)s, %(source_dataset_id)s, %(source_dataset_name)s, %(source_dataset_sha256)s, %(source_task_name)s, %(source_snapshot)s::jsonb, %(source_platform)s, %(source_uri)s,
                    %(source_task_id)s, %(uploaded_by)s, 'pending_upload', %(total_size_bytes)s, %(manifest_md5)s,
                    %(minio_bucket)s, %(minio_object_prefix)s, %(created_at)s, %(extra_metadata)s::jsonb
                )
                """,
                {**metadata, "id": model_id, "manifest_md5": manifest, "created_at": now, "extra_metadata": json.dumps(metadata.get("extra_metadata") or {}, ensure_ascii=False), "source_snapshot": json.dumps(metadata.get("source_snapshot") or {}, ensure_ascii=False)},
            )
            for entry in entries:
                connection.execute(
                    """
                    INSERT INTO model_file_entries (
                        model_file_id, relative_path, minio_object_key, size_bytes, md5
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (model_id, entry["relative_path"], entry["minio_object_key"], entry["size_bytes"], entry["md5"]),
                )
        return model_id

    def reset_pending(self, metadata: dict[str, Any], entries: list[dict[str, Any]]) -> str:
        model_id = str(metadata["id"])
        manifest = _manifest_md5(entries)
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE model_files SET
                    name=%(name)s,
                    artifact_type=%(artifact_type)s,
                    artifact_variant=%(artifact_variant)s,
                    training_type=%(training_type)s,
                    tuning_method=%(tuning_method)s,
                    base_model_id=%(base_model_id)s,
                    base_model_name=%(base_model_name)s,
                    base_model_repo=%(base_model_repo)s,
                    base_model_revision=%(base_model_revision)s,
                    base_model_local_path=%(base_model_local_path)s,
                    dataset_id=%(dataset_id)s,
                    source_dataset_id=%(source_dataset_id)s,
                    source_dataset_name=%(source_dataset_name)s,
                    source_dataset_sha256=%(source_dataset_sha256)s,
                    source_task_name=%(source_task_name)s,
                    source_snapshot=%(source_snapshot)s::jsonb,
                    source_platform=%(source_platform)s,
                    source_uri=%(source_uri)s,
                    source_task_id=%(source_task_id)s,
                    uploaded_by=%(uploaded_by)s,
                    status='pending_upload',
                    total_size_bytes=%(total_size_bytes)s,
                    manifest_md5=%(manifest_md5)s,
                    minio_bucket=%(minio_bucket)s,
                    minio_object_prefix=%(minio_object_prefix)s,
                    upload_started_at=NULL,
                    uploaded_at=NULL,
                    error_message=NULL,
                    extra_metadata=%(extra_metadata)s::jsonb
                WHERE id=%(id)s
                """,
                {
                    **metadata,
                    "manifest_md5": manifest,
                    "extra_metadata": json.dumps(metadata.get("extra_metadata") or {}, ensure_ascii=False),
                    "source_snapshot": json.dumps(metadata.get("source_snapshot") or {}, ensure_ascii=False),
                },
            )
            connection.execute("DELETE FROM model_file_entries WHERE model_file_id=%s", (model_id,))
            for entry in entries:
                connection.execute(
                    """
                    INSERT INTO model_file_entries (
                        model_file_id, relative_path, minio_object_key, size_bytes, md5
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (model_id, entry["relative_path"], entry["minio_object_key"], entry["size_bytes"], entry["md5"]),
                )
        return model_id

    def mark_upload_started(self, model_id: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE model_files SET status='uploading', upload_started_at=%s, error_message=NULL WHERE id=%s", (_now(), model_id))

    def mark_entry(self, model_id: str, relative_path: str, *, status: str, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE model_file_entries SET status=%s, uploaded_at=%s, error_message=%s WHERE model_file_id=%s AND relative_path=%s", (status, _now() if status == "available" else None, error, model_id, relative_path))

    def mark_available(self, model_id: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE model_files SET status='available', uploaded_at=%s, error_message=NULL WHERE id=%s", (_now(), model_id))

    def mark_failed(self, model_id: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE model_files SET status='failed', error_message=%s WHERE id=%s", (error[:4000], model_id))

    def get(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM model_files WHERE id=%s", (model_id,)).fetchone()

    def find_source(
        self,
        *,
        artifact_type: str,
        artifact_variant: str,
        source_task_id: str | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["artifact_type=%s", "artifact_variant=%s"]
        values: list[Any] = [artifact_type, artifact_variant]
        if source_task_id:
            clauses.append("source_task_id=%s")
            values.append(source_task_id)
        if source_uri:
            clauses.append("source_uri=%s")
            values.append(source_uri)
        if len(clauses) == 2:
            return None
        with self.connect() as connection:
            return connection.execute(
                f"SELECT * FROM model_files WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 1",
                values,
            ).fetchone()

    def list(self, *, source_task_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM model_files"
        values: tuple[Any, ...] = ()
        if source_task_id:
            query += " WHERE source_task_id=%s"
            values = (source_task_id,)
        query += " ORDER BY created_at DESC"
        with self.connect() as connection:
            return list(connection.execute(query, values).fetchall())

    def list_entries(self, model_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM model_file_entries WHERE model_file_id=%s ORDER BY relative_path", (model_id,)).fetchall())


class MinioModelObjectStore:
    def __init__(self, config: ModelStorageConfig) -> None:
        try:
            from minio import Minio
        except ImportError as exc:
            raise RuntimeError("缺少 minio 依赖，请安装 requirements-storage.txt。") from exc
        self.client = Minio(config.minio_endpoint, access_key=config.minio_access_key, secret_key=config.minio_secret_key, secure=config.minio_secure)
        self.bucket = config.minio_bucket

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload(self, path: Path, object_key: str, *, md5: str, size: int) -> None:
        self.client.fput_object(self.bucket, object_key, str(path), content_type="application/octet-stream", metadata={"x-amz-meta-md5": md5, "x-amz-meta-uploaded-at": _now().isoformat()})

    def read(self, object_key: str, *, limit: int) -> bytes:
        response = self.client.get_object(self.bucket, object_key)
        try:
            return response.read(limit + 1)
        finally:
            response.close()
            response.release_conn()

    def download(self, object_key: str, target: Path, *, size: int) -> None:
        response = self.client.get_object(self.bucket, object_key)
        try:
            remaining = size
            with target.open("xb") as destination:
                while remaining:
                    chunk = response.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("MinIO 权重文件大小与清单不一致。")
                    destination.write(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError("MinIO 权重文件大小与清单不一致。")
        finally:
            response.close()
            response.release_conn()


class ModelArtifactUploader:
    def __init__(self, config: ModelStorageConfig) -> None:
        config.validate()
        self.config = config
        self.metadata = PostgresModelMetadataStore(config.postgres_dsn)
        self.objects = MinioModelObjectStore(config)
        self._restore_lock = threading.Lock()

    def restore_adapter(self, task_id: str, artifact_variant: str, cache_root: Path) -> str:
        if artifact_variant not in {"best", "final"}:
            raise ValueError("只能恢复 Best 或 Final Adapter。")
        record = self.metadata.find_source(
            artifact_type="lora_adapter", artifact_variant=artifact_variant, source_task_id=task_id,
        )
        if not record or record.get("status") != "available":
            raise ValueError("本地完整权重不可用，且没有上传成功的 MinIO 备份，无法部署。")
        entries = self.metadata.list_entries(record["id"])
        if not entries or any(entry.get("status") != "available" for entry in entries):
            raise ValueError("MinIO 权重文件清单不完整，无法部署。")
        prefix = str(record.get("minio_object_prefix") or "")
        _safe_relative_path(prefix)
        if record.get("minio_bucket") != self.config.minio_bucket:
            raise ValueError("MinIO 权重不属于当前配置的存储桶。")
        names: set[str] = set()
        for entry in entries:
            relative = _safe_relative_path(str(entry["relative_path"]))
            if relative.casefold() in names or entry.get("minio_object_key") != f"{prefix}/{relative}":
                raise ValueError("MinIO 权重文件清单存在重复路径或对象越界。")
            names.add(relative.casefold())
            if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{32}", str(entry.get("md5") or "")):
                raise ValueError("MinIO 权重文件缺少有效的大小或 MD5。")
        entries = sorted(entries, key=lambda entry: entry["relative_path"])
        manifest = _manifest_md5(entries)
        if manifest != record.get("manifest_md5") or sum(entry["size_bytes"] for entry in entries) != record.get("total_size_bytes"):
            raise ValueError("MinIO 权重清单校验失败。")
        if "adapter_config.json" not in names or not names.intersection({
            "adapter_model.safetensors", "adapter_model.bin", "adapter_model.safetensors.index.json", "adapter_model.bin.index.json",
        }):
            raise ValueError("MinIO 备份缺少完整 Adapter 配置或权重。")

        def valid_cache(directory: Path) -> bool:
            try:
                if directory.is_symlink() or not directory.is_dir():
                    return False
                files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
                if files != {entry["relative_path"] for entry in entries}:
                    return False
                for entry in entries:
                    path = directory / entry["relative_path"]
                    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or _md5(path) != (entry["md5"], entry["size_bytes"]):
                        return False
                return adapter_is_complete(directory)
            except OSError:
                return False

        with self._restore_lock:
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_root = cache_root.resolve(strict=True)
            cache_key = hashlib.sha256(f"{record['id']}:{manifest}".encode()).hexdigest()
            target = cache_root / cache_key
            if target.is_symlink() or not target.resolve().is_relative_to(cache_root):
                raise ValueError("权重缓存路径越界。")
            if valid_cache(target):
                return str(target)
            with tempfile.TemporaryDirectory(prefix=".restore-", dir=cache_root) as staging:
                staged = Path(staging) / "adapter"
                staged.mkdir()
                for entry in entries:
                    destination = staged / entry["relative_path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self.objects.download(entry["minio_object_key"], destination, size=entry["size_bytes"])
                if not valid_cache(staged):
                    raise ValueError("MinIO 恢复后的权重大小、MD5 或 Adapter 完整性校验失败。")
                if target.exists():
                    shutil.rmtree(target)
                staged.replace(target)
            return str(target)

    def initialize(self) -> None:
        self.metadata.initialize()
        self.objects.ensure_bucket()

    def preview_listing(self, model_id: str) -> dict[str, Any] | None:
        record = self.metadata.get(model_id)
        if not record:
            return None
        entries = self.metadata.list_entries(model_id)
        return {
            "id": record.get("id"),
            "name": record.get("name"),
            "status": record.get("status"),
            "artifactType": record.get("artifact_type"),
            "artifactVariant": record.get("artifact_variant"),
            "bucket": record.get("minio_bucket"),
            "objectPrefix": record.get("minio_object_prefix"),
            "files": [
                {
                    "path": entry.get("relative_path"),
                    "sizeBytes": entry.get("size_bytes"),
                    "md5": entry.get("md5"),
                    "status": entry.get("status"),
                }
                for entry in entries
            ],
        }

    def preview_file(self, model_id: str, relative_path: str, *, limit: int = 512 * 1024) -> dict[str, Any] | None:
        record = self.metadata.get(model_id)
        if not record:
            return None
        if record.get("status") != "available":
            return {"path": relative_path, "status": record.get("status")}
        entry = next(
            (item for item in self.metadata.list_entries(model_id) if item.get("relative_path") == relative_path),
            None,
        )
        if not entry:
            raise FileNotFoundError("模型文件不存在。")
        suffix = Path(relative_path).suffix.lower()
        text_suffixes = {".json", ".jsonl", ".txt", ".md", ".py", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".xml", ".csv", ".tsv"}
        payload: dict[str, Any] = {
            "path": relative_path,
            "sizeBytes": entry.get("size_bytes"),
            "md5": entry.get("md5"),
            "status": entry.get("status"),
            "previewable": suffix in text_suffixes,
        }
        if suffix not in text_suffixes:
            payload["kind"] = "binary"
            payload["message"] = "这是二进制权重或模型文件，平台不直接展示原始内容。"
            return payload
        raw = self.objects.read(str(entry["minio_object_key"]), limit=limit)
        payload["kind"] = "text"
        payload["truncated"] = len(raw) > limit
        payload["content"] = raw[:limit].decode("utf-8", errors="replace")
        return payload

    def upload_directory(self, root: Path, *, name: str, artifact_type: str, uploaded_by: str = "local-admin", **metadata: Any) -> dict[str, Any]:
        artifact_variant = metadata.get("artifact_variant") or ("base" if artifact_type == "base_model" else "final")
        existing = self.metadata.find_source(
            artifact_type=artifact_type,
            artifact_variant=artifact_variant,
            source_task_id=metadata.get("source_task_id"),
            source_uri=metadata.get("source_uri"),
        )
        if existing and existing.get("status") == "available":
            existing["entries"] = self.metadata.list_entries(existing["id"])
            return existing
        files = iter_model_files(root, artifact_type)
        model_id = str(existing["id"]) if existing else f"model-{uuid.uuid4().hex[:16]}"
        prefix = f"model/{model_id}"
        entries: list[dict[str, Any]] = []
        for path, relative in files:
            digest, size = _md5(path)
            entries.append({"relative_path": relative, "minio_object_key": f"{prefix}/{relative}", "size_bytes": size, "md5": digest})
        total = sum(item["size_bytes"] for item in entries)
        record_metadata = {
            "id": model_id,
            "name": name,
            "artifact_type": artifact_type,
            "artifact_variant": artifact_variant,
            "training_type": metadata.get("training_type"),
            "tuning_method": metadata.get("tuning_method"),
            "base_model_id": metadata.get("base_model_id"),
            "base_model_name": metadata.get("base_model_name"),
            "base_model_repo": metadata.get("base_model_repo"),
            "base_model_revision": metadata.get("base_model_revision"),
            "base_model_local_path": metadata.get("base_model_local_path"),
            "dataset_id": metadata.get("dataset_id"),
            "source_dataset_id": metadata.get("source_dataset_id") or metadata.get("dataset_id"),
            "source_dataset_name": metadata.get("source_dataset_name"),
            "source_dataset_sha256": metadata.get("source_dataset_sha256"),
            "source_task_name": metadata.get("source_task_name"),
            "source_snapshot": metadata.get("source_snapshot") or {},
            "source_platform": metadata.get("source_platform"),
            "source_uri": metadata.get("source_uri"),
            "source_task_id": metadata.get("source_task_id"),
            "uploaded_by": uploaded_by,
            "total_size_bytes": total,
            "minio_bucket": self.config.minio_bucket,
            "minio_object_prefix": prefix,
            "extra_metadata": metadata.get("extra_metadata") or {},
        }
        if existing:
            self.metadata.reset_pending(record_metadata, entries)
        else:
            self.metadata.create_pending(record_metadata, entries)
        try:
            self.metadata.mark_upload_started(model_id)
            self.objects.ensure_bucket()
            for path, relative in files:
                entry = next(item for item in entries if item["relative_path"] == relative)
                self.objects.upload(path, entry["minio_object_key"], md5=entry["md5"], size=entry["size_bytes"])
                self.metadata.mark_entry(model_id, relative, status="available")
            self.metadata.mark_available(model_id)
        except Exception as exc:
            self.metadata.mark_failed(model_id, str(exc))
            raise
        result = self.metadata.get(model_id) or {}
        result["entries"] = self.metadata.list_entries(model_id)
        return result
