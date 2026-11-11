from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shlex
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Literal, Mapping

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.capabilities import require_provider
from backend.store import TaskStore, resolve_adapter_artifact
from backend.model_storage import adapter_is_complete


def _clean_remote_runtime_line(line: str) -> str | None:
    """统一清洗远端日志，避免启动后 watcher 与启动阶段规则不一致。"""
    from backend.providers.youyun.inference import _clean_remote_log_line

    return _clean_remote_log_line(line)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_timestamp(value: Any) -> str:
    """Return the same ISO timestamp shape for SQLite strings and PostgreSQL datetimes."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def format_log_timestamp(value: Any) -> str:
    """Format SQLite/PostgreSQL timestamps for the inference log console."""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")[:19]
    return str(value).replace("T", " ")[:19]


_MODEL_CONTEXT_KEYS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_sequence_length",
    "seq_length",
    "max_context_length",
)

WEBSOCKET_HEARTBEAT_INTERVAL_SECONDS = 15.0


def _valid_context_length(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if 128 <= number <= 1_048_576 else None


def _context_metadata(value: Any, *, source: str, source_type: str = "task-metadata") -> dict[str, Any] | None:
    """Normalize persisted Base Model metadata into the runtime response shape."""
    if not isinstance(value, dict):
        return None
    for key in (
        "maxContextLength",
        "max_context_length",
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
    ):
        length = _valid_context_length(value.get(key))
        if length is not None:
            return {
                "maxContextLength": length,
                "maxOutputTokens": length,
                "source": source,
                "sourceType": source_type,
                "isFallback": False,
                "configPath": value.get("configPath"),
            }
    return None


def _context_metadata_from_task_logs(task: dict[str, Any]) -> dict[str, Any] | None:
    """Recover Base Model context for legacy tasks that logged their loaded config."""
    logs = task.get("logs")
    if not isinstance(logs, list):
        return None
    for entry in reversed(logs):
        if not isinstance(entry, str):
            continue
        for key in _MODEL_CONTEXT_KEYS:
            match = re.search(rf'["\']?{re.escape(key)}["\']?\s*:\s*(\d+)', entry)
            if match is None:
                continue
            length = _valid_context_length(match.group(1))
            if length is not None:
                return {
                    "maxContextLength": length,
                    "maxOutputTokens": length,
                    "source": f"task:logs:{key}",
                    "sourceType": "training-log-recovery",
                    "isFallback": False,
                    "configPath": None,
                }
    return None


def read_model_context(model_path: str | Path) -> dict[str, Any]:
    path = Path(str(model_path)).expanduser()
    fallback = {
        "maxContextLength": 4096,
        "maxOutputTokens": 4096,
        "source": "platform-default",
        "sourceType": "platform-fallback",
        "isFallback": True,
        "configPath": None,
    }
    try:
        is_directory = path.is_dir()
    except OSError:
        return fallback
    if not is_directory:
        return fallback
    tokenizer_fallback: tuple[int, Path] | None = None
    for filename in ("config.json", "tokenizer_config.json"):
        config_path = path / filename
        try:
            is_file = config_path.is_file()
        except OSError:
            continue
        if not is_file:
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        sections: list[tuple[str, dict[str, Any]]] = [("", data)]
        for section_name in ("text_config", "llm_config", "language_config"):
            section = data.get(section_name)
            if isinstance(section, dict):
                sections.append((f"{section_name}.", section))
        for prefix, section in sections:
            for key in _MODEL_CONTEXT_KEYS:
                value = _valid_context_length(section.get(key))
                if value is not None:
                    return {
                        "maxContextLength": value,
                        "maxOutputTokens": value,
                        "source": f"{filename}:{prefix}{key}",
                        "sourceType": "model-config",
                        "isFallback": False,
                        "configPath": str(config_path),
                    }
        tokenizer_value = _valid_context_length(data.get("model_max_length"))
        if tokenizer_value is not None and tokenizer_fallback is None:
            tokenizer_fallback = (tokenizer_value, config_path)
    if tokenizer_fallback is None:
        return fallback
    value, config_path = tokenizer_fallback
    return {
        "maxContextLength": value,
        "maxOutputTokens": value,
        "source": f"{config_path.name}:model_max_length",
        "sourceType": "tokenizer-fallback",
        "isFallback": True,
        "configPath": str(config_path),
    }


def read_task_model_context(
    task: dict[str, Any],
    model: str,
    adapter_path: str | Path | None = None,
    *,
    resolve_model: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Resolve Base Model context from task metadata, adapter metadata, or local files.

    Training outputs are LoRA weights and do not contain the Base Model config. The
    task therefore gets first priority, followed by an adapter's base-model
    reference, and finally the resolved model directory. The platform fallback is
    retained for legacy local services; callers deploying remote services should
    reject ``isFallback`` explicitly.
    """
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else task
    for key in ("baseModelMetadata", "base_model_metadata", "modelMetadata", "model_metadata"):
        metadata = _context_metadata(payload.get(key), source=f"task:{key}")
        if metadata:
            return metadata
    for key in ("modelContextLength", "maxContextLength", "max_context_length"):
        metadata = _context_metadata({key: payload.get(key)}, source=f"task:{key}")
        if metadata:
            return metadata
    resolver = resolve_model or (lambda value: value)
    candidate_paths: list[str] = []
    if adapter_path:
        adapter_root = Path(str(adapter_path)).expanduser()
        config_file = adapter_root / "adapter_config.json"
        if config_file.is_file():
            try:
                adapter_config = json.loads(config_file.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                adapter_config = {}
            if isinstance(adapter_config, dict):
                base_ref = adapter_config.get("base_model_name_or_path") or adapter_config.get("baseModel")
                if isinstance(base_ref, str) and base_ref.strip():
                    candidate_paths.append(base_ref.strip())
        for key in ("baseModelPath", "base_model_local_path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                candidate_paths.append(value.strip())
        artifacts = task.get("artifacts") if isinstance(task.get("artifacts"), list) else []
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("type") != "adapter":
                continue
            if artifact.get("path") and str(Path(str(artifact["path"])).expanduser()) != str(Path(str(adapter_path)).expanduser()):
                continue
            for key in ("baseModelPath", "base_model_local_path"):
                value = artifact.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_paths.append(value.strip())
            for key in ("baseModelMetadata", "base_model_metadata"):
                metadata = _context_metadata(artifact.get(key), source=f"artifact:{key}")
                if metadata:
                    return metadata
    recovered = _context_metadata_from_task_logs(task)
    if recovered:
        return recovered
    candidate_paths.append(model)
    seen: set[str] = set()
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            resolved = resolver(candidate)
        except OSError:
            continue
        metadata = read_model_context(resolved)
        if not metadata.get("isFallback"):
            return metadata
    try:
        return read_model_context(resolver(model))
    except OSError:
        return read_model_context("")


class InferenceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    taskId: str = Field(min_length=1)
    adapterVariant: Literal["best", "final"] = "best"
    owner: str = "平台管理员"
    maxModelLength: int = Field(default=4096, ge=128, le=131072)
    # Kept as optional compatibility fields for older API clients. Generation
    # settings are request-scoped now and are intentionally not persisted for
    # newly created services.
    maxTokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0, le=2)
    topP: float | None = Field(default=None, gt=0, le=1)
    stream: bool | None = None
    dtype: Literal["auto", "float16", "bfloat16"] = "auto"
    quantization: Literal["none", "bitsandbytes"] = "none"
    gpuMemoryUtilization: float = Field(default=0.75, gt=0, le=0.99)
    kvCacheMode: Literal["auto", "manual"] = "auto"
    kvCacheMemoryGiB: float | None = Field(default=None, gt=0, le=80)
    kvCacheDtype: Literal["auto", "fp8"] = "auto"
    maxNumSeqs: int = Field(default=4, ge=1, le=1024)
    maxNumBatchedTokens: int = Field(default=8192, ge=128, le=131072)
    enablePrefixCaching: bool = True
    enableChunkedPrefill: bool = True
    tensorParallelSize: int = Field(default=1, ge=1, le=8)
    loadTimeout: int = Field(default=600, ge=30, le=3600)
    deploymentMode: Literal["managed", "custom"] = "managed"
    customCommand: str | None = Field(default=None, max_length=20000)
    provider: Literal["local", "youyun"] = "local"
    youyunResource: dict[str, Any] | None = None
    modelContextLength: int | None = Field(default=None, ge=128, le=1_048_576)
    modelContextSource: str | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("服务名称不能为空。")
        return name

    @field_validator("customCommand")
    @classmethod
    def validate_custom_command(cls, value: str | None) -> str | None:
        if value is None:
            return None
        command = value.strip()
        return command or None



class InferenceCommandValidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taskId: str = Field(min_length=1)
    adapterVariant: Literal["best", "final"] = "best"
    command: str = Field(min_length=1, max_length=20000)


class InferenceChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class InferenceChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[InferenceChatMessage] = Field(min_length=1, max_length=60)
    maxTokens: int = Field(default=512, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0, le=2)
    topP: float = Field(default=0.9, gt=0, le=1)
    stream: bool = True


class ExposureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exposure: Literal["test", "online", "offline"]


class InferenceRetryRequest(BaseModel):
    """Optional resource override for a failed Youyun service retry."""

    model_config = ConfigDict(extra="forbid")

    youyunResource: dict[str, Any] | None = None


SHELL_CONTROL_TOKENS = ("&&", "||", ";", "|", ">", "<", "`")
PROTECTED_COMMAND_OPTIONS = {
    "--model",
    "--served-model-name",
    "--lora-modules",
    "--enable-lora",
    "--host",
    "--port",
}


def _parse_vllm_command(command: str) -> tuple[list[str], list[str]]:
    normalized = command.replace("\\\r\n", " ").replace("\\\n", " ")
    if "\n" in normalized or "\r" in normalized:
        raise ValueError("命令包含未转义换行，请使用反斜杠续行或粘贴单行命令。")
    if any(token in normalized for token in SHELL_CONTROL_TOKENS) or re.search(
        r"\$\(|\$\{|\b(source|exec|eval)\b", normalized
    ):
        raise ValueError("只允许 vLLM 启动命令，不允许 Shell 管道、重定向或脚本语法。")
    try:
        lexer = shlex.shlex(normalized, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        if os.name == "nt":
            lexer.escape = ""
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(f"无法解析 vLLM 命令：{exc}") from exc
    if not tokens:
        raise ValueError("请粘贴 vLLM 启动命令。")

    executable = Path(tokens[0]).name.lower()
    if executable in {"vllm", "vllm.exe"} and tokens[1:2] == ["serve"]:
        args = tokens[2:]
    elif executable.startswith("python") and tokens[1:3] == [
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]:
        args = tokens[3:]
    else:
        raise ValueError(
            "命令必须以 `vllm serve` 或 `python -m vllm.entrypoints.openai.api_server` 开头。"
        )
    if any(token in {"-c", "--command", "--shell"} for token in args):
        raise ValueError("不允许通过参数嵌套执行其他命令。")

    flags = sorted(
        {
            token.split("=", 1)[0]
            for token in args
            if token.startswith("--")
            and token.split("=", 1)[0] not in PROTECTED_COMMAND_OPTIONS
        }
    )
    return args, flags


def _read_command_option(args: list[str], option: str) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == option:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"参数 {option} 缺少值。")
            values.append(args[index + 1])
            index += 2
            continue
        if token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ValueError(f"参数 {option} 只能出现一次。")
    return values[0] if values else None


def _read_positional_model(args: list[str]) -> str | None:
    """Read the normal first positional model used by `vllm serve MODEL`."""
    if args and not args[0].startswith("-"):
        return args[0]
    return None


def _replace_command_option(args: list[str], option: str, value: str) -> list[str]:
    result: list[str] = []
    replaced = False
    index = 0
    while index < len(args):
        token = args[index]
        if token == option:
            index += 2
            if not replaced:
                result.extend([option, value])
                replaced = True
            continue
        if token.startswith(option + "="):
            index += 1
            if not replaced:
                result.extend([option, value])
                replaced = True
            continue
        result.append(token)
        index += 1
    if not replaced:
        result.extend([option, value])
    return result


def _ensure_command_flag(args: list[str], flag: str) -> list[str]:
    if flag in args or any(token.startswith(flag + "=") for token in args):
        return args
    return [*args, flag]


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return os.path.normpath(left) == os.path.normpath(right)


def _row_value(row: Any, key: str, index: int = 0, default: Any = None) -> Any:
    """Read a DB row from sqlite3.Row, psycopg mappings, or tuple rows."""
    if row is None:
        return default
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        try:
            return row[index]
        except (KeyError, TypeError, IndexError):
            return default


class InferenceStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

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
                CREATE TABLE IF NOT EXISTS inference_services (
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
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    deleted_by TEXT,
                    deletion_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS inference_service_runs (
                    id TEXT PRIMARY KEY, service_id TEXT NOT NULL, run_no INTEGER NOT NULL,
                    state TEXT NOT NULL DEFAULT 'starting', config_snapshot TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT, finished_at TEXT, error TEXT, created_at TEXT NOT NULL,
                    FOREIGN KEY(service_id) REFERENCES inference_services(id) ON DELETE CASCADE,
                    UNIQUE(service_id, run_no)
                );
                CREATE TABLE IF NOT EXISTS inference_run_metrics (
                    run_id TEXT PRIMARY KEY, metrics TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES inference_service_runs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    run_id TEXT,
                    level TEXT NOT NULL DEFAULT 'INFO',
                    source TEXT NOT NULL DEFAULT 'backend',
                    FOREIGN KEY(service_id) REFERENCES inference_services(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    service_id TEXT PRIMARY KEY,
                    metrics TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(service_id) REFERENCES inference_services(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS inference_command_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    flags TEXT NOT NULL DEFAULT '[]',
                    success INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    FOREIGN KEY(service_id) REFERENCES inference_services(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_inference_services_state
                    ON inference_services(state, updated_at);
                CREATE INDEX IF NOT EXISTS idx_inference_logs_service
                    ON inference_logs(service_id, id);
                CREATE INDEX IF NOT EXISTS idx_inference_command_usage_service
                    ON inference_command_usage(service_id, created_at);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(inference_services)").fetchall()}
            for column, declaration in {"deleted_at": "TEXT", "deleted_by": "TEXT", "deletion_reason": "TEXT"}.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE inference_services ADD COLUMN {column} {declaration}")
            log_columns = {row["name"] for row in connection.execute("PRAGMA table_info(inference_logs)").fetchall()}
            for column, declaration in {"run_id": "TEXT", "level": "TEXT NOT NULL DEFAULT 'INFO'", "source": "TEXT NOT NULL DEFAULT 'backend'"}.items():
                if column not in log_columns:
                    connection.execute(f"ALTER TABLE inference_logs ADD COLUMN {column} {declaration}")

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        config = json.loads(row["config"])
        startup = config.get("startup") or {}
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "taskId": row["task_id"],
            "owner": row["owner"],
            "config": config,
            "state": row["state"],
            "exposure": row["exposure"],
            "gpu": row["gpu"],
            "memory": row["memory"],
            "endpoint": row["endpoint"],
            "recoveryAttempts": row["recovery_attempts"],
            "error": row["error"],
            "createdAt": serialize_timestamp(row["created_at"]),
            "updatedAt": serialize_timestamp(row["updated_at"]),
            "model": config.get("model", ""),
            "adapter": config.get("adapter", ""),
            "startup": config.get("startup") or None,
            "startupStage": startup.get("stage"),
            "startupProgress": startup.get("progress"),
            "startupStartedAt": startup.get("startedAt"),
            "startupElapsedSeconds": startup.get("elapsedSeconds"),
            "remoteRuntime": config.get("remoteRuntime") or None,
        }

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
                INSERT INTO inference_services (
                    id, name, description, task_id, owner, config, state,
                    exposure, endpoint, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'created', 'test', ?, ?, ?)
                """,
                (
                    service_id,
                    payload.name,
                    payload.description,
                    payload.taskId,
                    payload.owner,
                    json.dumps(config, ensure_ascii=False),
                    endpoint,
                    timestamp,
                    timestamp,
                ),
            )
        self.append_log(service_id, "服务配置已创建，等待手动启动。")
        return self.get(service_id)  # type: ignore[return-value]

    def create_run(self, service_id: str, config: dict[str, Any] | None = None) -> str:
        run_id = f"{service_id}-r{uuid.uuid4().hex[:10]}"
        with self._connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(run_no), 0) + 1 AS next_run_no FROM inference_service_runs WHERE service_id = ?", (service_id,)).fetchone()
            run_no = int(_row_value(row, "next_run_no", default=1) or 1)
            connection.execute("INSERT INTO inference_service_runs(id, service_id, run_no, state, config_snapshot, started_at, created_at) VALUES (?, ?, ?, 'starting', ?, ?, ?)", (run_id, service_id, run_no, json.dumps(config or {}, ensure_ascii=False), now_iso(), now_iso()))
        return run_id

    def update_run(self, service_id: str, state: str, *, error: str | None = None) -> None:
        terminal = state in {"completed", "failed", "stopped"}
        with self._connect() as connection:
            connection.execute(
                "UPDATE inference_service_runs SET state = ?, error = COALESCE(?, error), finished_at = CASE WHEN ? THEN COALESCE(finished_at, ?) ELSE finished_at END WHERE service_id = ? AND run_no = (SELECT MAX(run_no) FROM inference_service_runs WHERE service_id = ?)",
                (state, error, terminal, now_iso() if terminal else None, service_id, service_id),
            )

    def get(self, service_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inference_services WHERE id = ?", (service_id,)
            ).fetchone()
        if row is not None and "deleted_at" in row.keys() and row["deleted_at"]:
            return None
        return self._serialize(row) if row else None

    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        with self._connect() as connection:
            where = "" if include_archived else "WHERE state != 'archived'"
            rows = connection.execute(
                f"SELECT * FROM inference_services {where} ORDER BY updated_at DESC"
            ).fetchall()
        return [self._serialize(row) for row in rows if "deleted_at" not in row.keys() or not row["deleted_at"]]

    def list_page(
        self,
        *,
        include_archived: bool = False,
        page: int = 1,
        page_size: int = 20,
        state: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        # PostgreSQL stores deleted_at as TIMESTAMPTZ; comparing it to an empty
        # string raises InvalidDatetimeFormat. SQLite also treats NULL as the
        # canonical non-deleted value, so the portable predicate is IS NULL.
        clauses: list[str] = ["deleted_at IS NULL"]
        values: list[Any] = []
        if not include_archived:
            clauses.append("state != 'archived'")
        if state and state != "all":
            clauses.append("state = ?")
            values.append(state)
        if query and query.strip():
            clauses.append("(id LIKE ? OR name LIKE ? OR task_id LIKE ? OR config LIKE ?)")
            needle = f"%{query.strip()}%"
            values.extend([needle] * 4)
        where = " AND ".join(clauses)
        with self._connect() as connection:
            count_row = connection.execute(
                f"SELECT COUNT(*) AS total_count FROM inference_services WHERE {where}",
                values,
            ).fetchone()
            if count_row is None:
                total = 0
            else:
                try:
                    total = int(count_row["total_count"])
                except (KeyError, TypeError, IndexError):
                    total = int(_row_value(count_row, "total_count") or 0)
            rows = connection.execute(
                f"SELECT * FROM inference_services WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "items": [self._serialize(row) for row in rows],
            "page": page,
            "pageSize": page_size,
            "total": total,
            "totalPages": total_pages,
        }

    def archive(self, service_id: str) -> dict[str, Any]:
        service = self.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        if service["state"] in {"starting", "running", "stopping", "recovering"}:
            raise RuntimeError("运行中的服务不能归档，请先停止服务。")
        if service["state"] == "archived":
            return service
        updated = self.update(
            service_id,
            state="archived",
            exposure="offline",
            gpu="未占用",
            memory="0 / 0 GB",
        )
        self.append_log(service_id, "服务已归档，配置、日志和指标仍然保留。")
        return updated  # type: ignore[return-value]

    def restore(self, service_id: str) -> dict[str, Any]:
        service = self.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        if service["state"] != "archived":
            raise RuntimeError("只有已归档服务可以恢复。")
        updated = self.update(
            service_id,
            state="stopped",
            exposure="offline",
            gpu="未占用",
            memory="0 / 0 GB",
            error=None,
        )
        self.append_log(service_id, "服务已恢复为已停止状态，不会自动启动。")
        return updated  # type: ignore[return-value]

    def delete(self, service_id: str) -> None:
        service = self.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        if service["state"] != "archived":
            raise RuntimeError("只有已归档服务可以永久删除。")
        with self._connect() as connection:
            try:
                connection.execute(
                    "UPDATE inference_services SET deleted_at = ?, deleted_by = ?, deletion_reason = ? WHERE id = ?", (now_iso(), "local-admin", "user_request", service_id)
                )
            except sqlite3.OperationalError as exc:
                if "deleted_at" not in str(exc):
                    raise
                connection.execute("DELETE FROM inference_services WHERE id = ?", (service_id,))

    def update(self, service_id: str, **changes: Any) -> dict[str, Any] | None:
        allowed = {
            "state", "exposure", "gpu", "memory", "error",
            "recovery_attempts", "updated_at", "config",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return self.get(service_id)
        if "config" in values and not isinstance(values["config"], str):
            values["config"] = json.dumps(values["config"], ensure_ascii=False)
        values.setdefault("updated_at", now_iso())
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE inference_services SET {assignments} WHERE id = ?",
                (*values.values(), service_id),
            )
        return self.get(service_id)

    def append_log(self, service_id: str, message: str, *, level: str = "INFO", source: str = "backend") -> None:
        clean = message.strip()
        if not clean:
            return
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO inference_logs(service_id, created_at, message, run_id, level, source) VALUES (?, ?, ?, (SELECT id FROM inference_service_runs WHERE service_id = ? ORDER BY run_no DESC LIMIT 1), ?, ?)",
                    (service_id, now_iso(), clean, service_id, level.upper() if level.upper() in {"INFO", "WARN", "ERROR"} else "INFO", source if source in {"backend", "trainer", "vllm", "remote"} else "backend"),
                )
            except sqlite3.OperationalError as exc:
                if "run_id" not in str(exc):
                    raise
                connection.execute("INSERT INTO inference_logs(service_id, created_at, message) VALUES (?, ?, ?)", (service_id, now_iso(), clean))

    def record_command_usage(
        self,
        service_id: str,
        *,
        flags: list[str],
        success: bool,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_command_usage(
                    service_id, created_at, flags, success, error
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    service_id,
                    now_iso(),
                    json.dumps(sorted(set(flags)), ensure_ascii=False),
                    1 if success else 0,
                    error,
                ),
            )

    def command_usage(self, service_id: str | None = None) -> dict[str, Any]:
        where = "WHERE service_id = ?" if service_id else ""
        args: tuple[Any, ...] = (service_id,) if service_id else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT flags, success, error FROM inference_command_usage {where}",
                args,
            ).fetchall()
        flag_counts: dict[str, int] = {}
        total = 0
        success = 0
        failures: list[str] = []
        for row in rows:
            total += 1
            if row["success"]:
                success += 1
            if row["error"]:
                failures.append(str(row["error"]))
            for flag in json.loads(row["flags"] or "[]"):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        return {
            "serviceId": service_id,
            "total": total,
            "success": success,
            "failed": total - success,
            "flagCounts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
            "recentErrors": failures[-10:],
        }

    def logs(self, service_id: str, limit: int = 200) -> list[str]:
        with self._connect() as connection:
            service = connection.execute("SELECT * FROM inference_services WHERE id = ?", (service_id,)).fetchone()
        if service is None or ("deleted_at" in service.keys() and service["deleted_at"]):
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, message FROM inference_logs
                WHERE service_id = ? ORDER BY id DESC LIMIT ?
                """,
                (service_id, min(max(limit, 1), 1000)),
            ).fetchall()
        return [
            f"[{format_log_timestamp(row['created_at'])}] {row['message']}"
            for row in reversed(rows)
        ]

    def list_runs(self, service_id: str) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                rows = connection.execute("SELECT id, run_no, state, started_at, finished_at, error, created_at FROM inference_service_runs WHERE service_id = ? ORDER BY run_no DESC", (service_id,)).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.OperationalError:
            return []

    def log_items(
        self,
        service_id: str,
        *,
        run_id: str | None = None,
        after_id: int = 0,
        before_id: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            if run_id is None:
                row = connection.execute("SELECT id AS run_id FROM inference_service_runs WHERE service_id = ? ORDER BY run_no DESC LIMIT 1", (service_id,)).fetchone()
                run_id = _row_value(row, "run_id")
            clause = " AND run_id = ?" if run_id else ""
            size = min(max(limit, 1), 1000)
            if before_id > 0:
                params = (service_id, *( [run_id] if run_id else [] ), before_id, size + 1)
                rows = connection.execute(
                    f"SELECT id, created_at, level, source, message FROM inference_logs WHERE service_id = ?{clause} AND id < ? ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall()
                has_older = len(rows) > size
                rows = list(reversed(rows[:size]))
                has_more = False
            elif after_id <= 0:
                # The detail page is a tail view. Start with the newest window
                # so a long-running service does not pin the UI to its first logs.
                params = (service_id, *( [run_id] if run_id else [] ), size + 1)
                rows = connection.execute(f"SELECT id, created_at, level, source, message FROM inference_logs WHERE service_id = ?{clause} ORDER BY id DESC LIMIT ?", params).fetchall()
                has_older = len(rows) > size
                rows = list(reversed(rows[:size]))
                has_more = False
            else:
                params = (service_id, *( [run_id] if run_id else [] ), after_id, size + 1)
                rows = connection.execute(f"SELECT id, created_at, level, source, message FROM inference_logs WHERE service_id = ?{clause} AND id > ? ORDER BY id ASC LIMIT ?", params).fetchall()
                has_more = len(rows) > size
                rows = rows[:size]
                has_older = False
        return {
            "serviceId": service_id,
            "runId": run_id,
            "items": [dict(row) for row in rows],
            "firstItemId": rows[0]["id"] if rows else before_id,
            "nextAfterId": rows[-1]["id"] if rows and before_id <= 0 else after_id,
            "hasOlder": has_older,
            "hasMore": has_more,
        }

    def export_logs(self, service_id: str, *, run_id: str | None = None) -> list[str]:
        """Return the complete persisted log history in display format."""
        with self._connect() as connection:
            clause = " AND run_id = ?" if run_id else ""
            params = (service_id, *( [run_id] if run_id else [] ))
            rows = connection.execute(
                f"SELECT created_at, message FROM inference_logs WHERE service_id = ?{clause} ORDER BY id ASC",
                params,
            ).fetchall()
        return [
            f"[{format_log_timestamp(row['created_at'])}] {row['message']}"
            for row in rows
        ]

    def set_metrics(self, service_id: str, metrics: dict[str, Any]) -> None:
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_metrics(service_id, metrics, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(service_id) DO UPDATE SET
                    metrics = excluded.metrics, updated_at = excluded.updated_at
                """,
                (service_id, json.dumps(metrics, ensure_ascii=False), timestamp),
            )
            run = connection.execute(
                "SELECT id FROM inference_service_runs WHERE service_id = ? ORDER BY run_no DESC LIMIT 1",
                (service_id,),
            ).fetchone()
            if run:
                connection.execute(
                    "INSERT INTO inference_run_metrics(run_id, metrics, updated_at) VALUES (?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET metrics = excluded.metrics, updated_at = excluded.updated_at",
                    (_row_value(run, "run_id"), json.dumps(metrics, ensure_ascii=False), timestamp),
                )

    def metrics(self, service_id: str, run_id: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            service = connection.execute("SELECT * FROM inference_services WHERE id = ?", (service_id,)).fetchone()
        if service is None or ("deleted_at" in service.keys() and service["deleted_at"]):
            return {"source": "not_started", "updatedAt": None}
        with self._connect() as connection:
            if run_id:
                row = connection.execute(
                    "SELECT metrics, updated_at FROM inference_run_metrics WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT metrics, updated_at FROM inference_metrics WHERE service_id = ?",
                    (service_id,),
                ).fetchone()
        if not row:
            return {"source": "not_started", "updatedAt": None}
        return {**json.loads(row["metrics"]), "updatedAt": serialize_timestamp(row["updated_at"])}


class InferenceRuntime:
    def __init__(
        self,
        store: InferenceStore,
        task_store: TaskStore,
        *,
        port: int,
        training_busy: Callable[[], bool],
        remote_provider_factory: Callable[[], Any] | None = None,
        youyun_resource_resolver: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        artifact_restorer: Callable[[str, str], str] | None = None,
    ) -> None:
        self.store = store
        self.task_store = task_store
        self.port = port
        self.training_busy = training_busy
        self.remote_provider_factory = remote_provider_factory
        self.youyun_resource_resolver = youyun_resource_resolver
        self.artifact_restorer = artifact_restorer
        self.process: asyncio.subprocess.Process | None = None
        self.service_id: str | None = None
        self.remote_providers: dict[str, Any] = {}
        self.remote_handles: dict[str, Any] = {}
        self.remote_endpoints: dict[str, str] = {}
        self.remote_start_locks: dict[str, asyncio.Lock] = {}
        self.lock = asyncio.Lock()
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.start_tasks: dict[str, asyncio.Task[Any]] = {}
        self.cancel_events: dict[str, asyncio.Event] = {}

    def _update_run(self, service_id: str, state: str, *, error: str | None = None) -> None:
        updater = getattr(self.store, "update_run", None)
        if callable(updater):
            try:
                updater(service_id, state, error=error)
            except Exception:
                # 运行记录不能阻断服务状态更新，尤其是旧数据库尚未迁移时。
                pass

    def _set_terminal_state(
        self,
        service_id: str,
        state: Literal["failed", "stopped"],
        *,
        error: str | None = None,
        cleanup_pending: bool = False,
    ) -> dict[str, Any] | None:
        """Persist a terminal lifecycle state with consistent exposure/resource labels."""
        return self.store.update(
            service_id,
            state=state,
            exposure="offline",
            error=error,
            gpu="优云实例待清理" if cleanup_pending else "未占用",
            memory="清理失败" if cleanup_pending else "0 / 0 GB",
        )

    def _update_startup(self, service_id: str, stage: str, message: str, **details: Any) -> None:
        """Persist observable remote startup stage in service config and log it."""
        service = self.store.get(service_id)
        if service is None:
            return
        previous = dict(service["config"].get("startup") or {})
        started_at = previous.get("startedAt") or now_iso()
        progress_by_stage = {
            "preparing": 5,
            "memory_preflight": 10,
            "creating_instance": 15,
            "instance_running": 30,
            "ssh_connecting": 38,
            "ssh_connected": 45,
            "diagnosing_vllm_runtime": 47,
            "checking_vllm_runtime": 48,
            "installing_vllm": 52,
            "vllm_runtime_ready": 54,
            "preparing_model": 56,
            "uploading_model": 58,
            "downloading_model": 60,
            "model_ready": 64,
            "uploading_adapter": 68,
            "adapter_uploaded": 74,
            "vllm_starting": 80,
            "loading_model": 88,
            "health_check_passed": 100,
            "running": 100,
            "health_check_failed": 100,
            "failed": 100,
        }
        progress = details.pop("progress", progress_by_stage.get(stage))
        startup = {
            **previous,
            "stage": stage,
            "message": message,
            "updatedAt": now_iso(),
            "startedAt": started_at,
            "progress": progress,
            **details,
        }
        self._store_config_fragment(service_id, "startup", startup)
        self.store.append_log(service_id, message)

    def _store_config_fragment(self, service_id: str, key: str, value: Any | None) -> None:
        service = self.store.get(service_id)
        if service is None:
            return
        config = dict(service["config"])
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
        self.store.update(service_id, config=config)

    async def start_background(self, service_id: str, *, recovering: bool = False) -> dict[str, Any]:
        """Queue startup and return immediately; lifecycle errors are persisted in the store."""
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        require_provider(service["config"].get("provider"))
        if service["state"] in {"starting", "recovering", "running", "stopping"}:
            raise RuntimeError("服务正在启动、重试、运行或停止中，请等待当前操作结束。")
        is_remote = service["config"].get("provider", "local") == "youyun"
        if not is_remote:
            return await self.start(service_id, recovering=recovering)
        stage = "recovering" if recovering else "starting"
        self.store.update(
            service_id,
            state=stage,
            exposure="test",
            gpu="优云 GPU · 准备中",
            memory="加载中",
            error=None,
        )
        self._update_startup(service_id, stage, "已提交优云推理启动任务。")
        cancel_event = asyncio.Event()
        self.cancel_events[service_id] = cancel_event
        task = asyncio.create_task(self.start(service_id, recovering=recovering, cancel_event=cancel_event))
        self.start_tasks[service_id] = task
        self.background_tasks.add(task)
        def cleanup_start(done: asyncio.Task[Any]) -> None:
            self.background_tasks.discard(done)
            if self.start_tasks.get(service_id) is done:
                self.start_tasks.pop(service_id, None)
                self.cancel_events.pop(service_id, None)
        task.add_done_callback(cleanup_start)
        # Prevent unhandled-task warnings while retaining the detailed error in the store.
        def consume_done(done: asyncio.Task[Any]) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - state/log already persisted by start
                pass
        task.add_done_callback(consume_done)
        return self.store.get(service_id)  # type: ignore[return-value]

    async def startup(self) -> None:
        # Inference services are started explicitly by the user. Backend
        # startup must not allocate GPU or create remote Youyun instances.
        return

    async def shutdown(self) -> None:
        for task in list(self.background_tasks):
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        if self.process is not None:
            await self._terminate_process()
        remote_items = list(self.remote_handles.items())
        for service_id, handle in remote_items:
            provider = self.remote_providers.get(service_id) or self._make_remote_provider(service_id)
            report = await provider.stop(handle)
            if report.get("terminated"):
                self.store.append_log(service_id, "后端关闭时已释放优云推理实例，服务保持停止，请手动重试。")
                self._store_remote_runtime(service_id, None)
            else:
                self.store.append_log(
                    service_id,
                    f"后端关闭时优云推理实例释放失败：{report.get('error')}",
                )
            self._clear_remote_runtime(service_id)

    @property
    def active_service_id(self) -> str | None:
        return self.service_id or next(iter(self.remote_handles), None)

    @property
    def local_service_active(self) -> bool:
        return self.service_id is not None and self.process is not None and self.process.returncode is None

    def _clear_remote_runtime(self, service_id: str) -> None:
        self.remote_providers.pop(service_id, None)
        self.remote_handles.pop(service_id, None)
        self.remote_endpoints.pop(service_id, None)

    def _remote_provider_for(self, service_id: str) -> Any | None:
        return self.remote_providers.get(service_id)

    def _remote_handle_for(self, service_id: str) -> Any | None:
        return self.remote_handles.get(service_id)

    def _store_remote_runtime(self, service_id: str, value: dict[str, Any] | None) -> None:
        service = self.store.get(service_id)
        if service is None:
            return
        config = dict(service["config"])
        if value is None:
            config.pop("remoteRuntime", None)
        else:
            config["remoteRuntime"] = value
        self.store.update(service_id, config=config)

    def _make_remote_provider(self, service_id: str | None = None) -> Any:
        if self.remote_provider_factory is not None:
            return self.remote_provider_factory()
        from backend.providers.youyun.client import YouyunClient
        from backend.providers.youyun.inference import YouyunInferenceProvider
        from backend.providers.youyun.instance_manager import YouyunInstanceManager
        from backend.providers.youyun.models import YouyunConfig

        # 实例的创建与释放必须使用服务创建时选定的 Region/Zone 组合：
        # 客户端只透传 zone 而区域停留在服务端默认值时，优云会以
        # RetCode=230 (Params [Zone] not available) 拒绝请求。
        resource: dict[str, Any] = {}
        if service_id:
            service = self.store.get(service_id)
            resource = dict(((service or {}).get("config") or {}).get("youyunResource") or {})
        base_config = YouyunConfig.from_env()
        client = YouyunClient(config=base_config.with_overrides(region=resource.get("region"), zone=resource.get("zone")))
        return YouyunInferenceProvider(
            YouyunInstanceManager(client),
            remote_python=os.environ.get("XUNTUI_REMOTE_VLLM_PYTHON", "python3"),
            port=int(os.environ.get("XUNTUI_REMOTE_INFERENCE_PORT", str(self.port))),
        )

    def _artifact_for_task(
        self,
        task_id: str,
        adapter_variant: Literal["best", "final"] = "best",
    ) -> tuple[dict[str, Any], str, str]:
        task = self.task_store.get(task_id)
        if task is None:
            raise ValueError("训练任务不存在。")
        if task.get("state") not in {"completed", "stopped"}:
            raise ValueError("只有已完成或已手动/自动早停的训练任务可以部署。")
        artifact = resolve_adapter_artifact(task.get("artifacts", []), adapter_variant)
        variant_label = "Best" if adapter_variant == "best" else "Final"
        if (artifact is None or not artifact.get("path")) and self.artifact_restorer is None:
            raise ValueError(f"该训练任务没有可部署的 {variant_label} Adapter。")
        adapter_path = self._resolve_adapter_path(task_id, adapter_variant, str((artifact or {}).get("path") or ""))
        model = str(task.get("model") or task.get("payload", {}).get("model") or "")
        if not model:
            raise ValueError("训练任务缺少 Base Model 信息。")
        return task, model, str(adapter_path)

    def _resolve_adapter_path(self, task_id: str, variant: str, local_path: str) -> str:
        if local_path and adapter_is_complete(Path(local_path)):
            return local_path
        if self.artifact_restorer is None:
            raise ValueError(f"所选 {variant.title()} Adapter 路径不存在或权重不完整，且 MinIO 恢复未配置，无法部署。")
        try:
            restored = str(self.artifact_restorer(task_id, variant))
        except Exception as exc:
            raise ValueError(f"本地完整权重不可用，MinIO 恢复失败：{exc}") from exc
        if "://" in restored or not adapter_is_complete(Path(restored)):
            raise ValueError("MinIO 恢复未生成完整的本地 Adapter，无法部署。")
        return restored

    def validate_task(
        self,
        task_id: str,
        adapter_variant: Literal["best", "final"] = "best",
    ) -> tuple[dict[str, Any], str, str]:
        return self._artifact_for_task(task_id, adapter_variant)

    @staticmethod
    def _vllm_python() -> str:
        return os.environ.get("XUNTUI_VLLM_PYTHON", sys.executable)

    @staticmethod
    def _cpu_inference() -> bool:
        return os.environ.get("XUNTUI_INFERENCE_DEVICE", "auto").strip().lower() == "cpu"

    def _vllm_env(self) -> dict[str, str]:
        environment = os.environ.copy()
        preload = os.environ.get("XUNTUI_VLLM_LD_PRELOAD", "").strip()
        if preload:
            environment["LD_PRELOAD"] = preload
        return environment

    @staticmethod
    def _cpu_flags() -> set[str]:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return set()
        flags: set[str] = set()
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"flags", "features"}:
                flags.update(value.strip().lower().split())
        return flags

    def _validate_cpu_vllm_isa(self) -> None:
        if not self._cpu_inference():
            return
        machine = platform.machine().strip().lower()
        if machine not in {"x86_64", "amd64", "i386", "i686", "x86"}:
            return
        flags = self._cpu_flags()
        if flags and "avx2" not in flags:
            raise RuntimeError(
                "当前 x86 CPU 缺少 AVX2，无法运行官方 vLLM CPU x86 构建；"
                "请迁移到支持 AVX2/AVX512 的 CPU 或 GPU 节点。"
            )

    def _validate_vllm_runtime(self) -> str:
        executable = Path(self._vllm_python())
        if not executable.is_file():
            raise RuntimeError(
                f"vLLM Python 不存在：{executable}。请检查 XUNTUI_VLLM_PYTHON 和 backend 镜像。"
            )
        self._validate_cpu_vllm_isa()
        try:
            result = subprocess.run(
                [str(executable), "-c", "import vllm; print(vllm.__version__)"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=self._vllm_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"无法检查 vLLM 运行环境：{exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown import error").strip()[-1000:]
            raise RuntimeError(
                f"vLLM 运行环境不可用（Python: {executable}）：{detail}"
            )
        return (result.stdout.strip().splitlines() or ["unknown"])[-1]

    def _command(self, service: dict[str, Any]) -> list[str]:
        config = service["config"]
        if config.get("deploymentMode") == "custom":
            return self._custom_command(service)
        model = self._resolve_model_path(str(config["model"]))
        command = [
            self._vllm_python(),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--served-model-name",
            service["id"],
            "--enable-lora",
            "--lora-modules",
            f"{service['id']}={config['adapter']}",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--max-model-len",
            str(config["maxModelLength"]),
            "--max-num-seqs",
            str(config["maxNumSeqs"]),
            "--max-num-batched-tokens",
            str(config["maxNumBatchedTokens"]),
            "--gpu-memory-utilization",
            str(config["gpuMemoryUtilization"]),
            "--dtype",
            config["dtype"],
            "--tensor-parallel-size",
            str(config["tensorParallelSize"]),
            "--no-enable-log-requests",
        ]
        if self._cpu_inference():
            gpu_option = command.index("--gpu-memory-utilization")
            del command[gpu_option : gpu_option + 2]
            if config.get("quantization") == "bitsandbytes":
                raise ValueError("CPU 冒烟推理不支持 bitsandbytes 量化，请选择 none。")
            if config.get("kvCacheDtype") == "fp8":
                raise ValueError("CPU 冒烟推理不支持 FP8 KV Cache，请选择 auto。")
        elif config.get("quantization") == "bitsandbytes":
            command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
        if config.get("enablePrefixCaching"):
            command.append("--enable-prefix-caching")
        if config.get("enableChunkedPrefill"):
            command.append("--enable-chunked-prefill")
        if config.get("kvCacheDtype") == "fp8" and not self._cpu_inference():
            command.extend(["--kv-cache-dtype", "fp8"])
        if config.get("kvCacheMode") == "manual" and config.get("kvCacheMemoryGiB"):
            cache_bytes = int(float(config["kvCacheMemoryGiB"]) * 1024**3)
            command.extend(["--kv-cache-memory-bytes", str(cache_bytes)])
        return command

    @staticmethod
    def _remote_managed_vllm_args(
        config: Mapping[str, Any],
        resource: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Build the managed vLLM options shared with the remote provider."""
        dtype = str(config["dtype"])
        gpu_type = str((resource or {}).get("gpuType") or "")
        normalized_gpu = gpu_type.replace(" ", "").replace("_", "").replace("-", "").lower()
        if dtype == "auto" and normalized_gpu.startswith("v100"):
            dtype = "float16"
        args = [
            "--max-num-seqs",
            str(config["maxNumSeqs"]),
            "--max-num-batched-tokens",
            str(config["maxNumBatchedTokens"]),
            "--gpu-memory-utilization",
            str(config["gpuMemoryUtilization"]),
            "--dtype",
            dtype,
            "--tensor-parallel-size",
            str(config["tensorParallelSize"]),
        ]
        if config.get("quantization") == "bitsandbytes":
            args.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
        if config.get("enablePrefixCaching"):
            args.append("--enable-prefix-caching")
        if config.get("enableChunkedPrefill"):
            args.append("--enable-chunked-prefill")
        if config.get("kvCacheDtype") == "fp8":
            args.extend(["--kv-cache-dtype", "fp8"])
        if config.get("kvCacheMode") == "manual" and config.get("kvCacheMemoryGiB"):
            cache_bytes = int(float(config["kvCacheMemoryGiB"]) * 1024**3)
            args.extend(["--kv-cache-memory-bytes", str(cache_bytes)])
        return args

    @staticmethod
    def _remote_memory_preflight(
        resource: Mapping[str, Any], config: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        gpu_type = str(resource.get("gpuType") or "优云 GPU")
        normalized_gpu = gpu_type.replace(" ", "").replace("_", "").lower()
        known_vram_gib = {
            "2080": 8,
            "2080ti": 11,
            "3080ti": 12,
            "3090": 24,
            "4090": 24,
            "409048g": 48,
            "5090": 32,
            "p40": 24,
            "v100s": 32,
        }.get(normalized_gpu)
        max_num_seqs = int(config.get("maxNumSeqs") or 4)
        max_num_batched_tokens = int(config.get("maxNumBatchedTokens") or 8192)
        gpu_memory_utilization = float(config.get("gpuMemoryUtilization") or 0.75)
        max_model_length = int(config.get("maxModelLength") or 4096)
        details = {
            "gpuType": gpu_type,
            "gpuMemoryGiB": known_vram_gib,
            "maxNumSeqs": max_num_seqs,
            "maxNumBatchedTokens": max_num_batched_tokens,
            "gpuMemoryUtilization": gpu_memory_utilization,
            "maxModelLength": max_model_length,
        }
        if known_vram_gib is None:
            return (
                "优云显存配置预检："
                f"max_num_seqs={max_num_seqs}，"
                f"gpu_memory_utilization={gpu_memory_utilization:g}。",
                details,
            )
        risky = (
            max_num_seqs > 4
            or max_num_batched_tokens > 8192
            or gpu_memory_utilization > 0.8
            or max_model_length > 8192
        )
        if risky:
            return (
                f"小显存风险提示：{gpu_type} 约 {known_vram_gib}GB；当前 "
                f"max_num_seqs={max_num_seqs}、"
                f"gpu_memory_utilization={gpu_memory_utilization:g}。"
                "如启动阶段显存不足，建议将并发降到 4、显存预算降到 0.80 以下，"
                "并将上下文长度控制在 8192 以内。",
                {**details, "risk": True},
            )
        return (
            f"小显存配置预检通过：{gpu_type} 约 {known_vram_gib}GB，"
            f"将使用 max_num_seqs={max_num_seqs}、"
            f"gpu_memory_utilization={gpu_memory_utilization:g}。",
            {**details, "risk": False},
        )

    def _normalized_custom_args(
        self,
        command: str,
        *,
        model: str,
        adapter: str,
        service_id: str,
        port: int,
    ) -> tuple[list[str], list[str]]:
        args, flags = _parse_vllm_command(command)
        expected_model = self._resolve_model_path(model)
        supplied_model = _read_command_option(args, "--model")
        positional_model = _read_positional_model(args)
        if supplied_model and positional_model:
            raise ValueError("自定义命令只能通过位置参数或 --model 指定 Base Model，不能重复指定。")
        supplied_model = supplied_model or positional_model
        if not supplied_model:
            raise ValueError("自定义命令必须提供所选训练任务的 Base Model。")
        if self._resolve_model_path(supplied_model) != expected_model:
            raise ValueError("自定义命令中的 --model 必须与所选训练任务的 Base Model 一致。")

        supplied_lora = _read_command_option(args, "--lora-modules")
        if supplied_lora:
            try:
                _, supplied_adapter = supplied_lora.rsplit("=", 1)
            except ValueError as exc:
                raise ValueError("--lora-modules 必须使用 adapter-name=/path/to/adapter 格式。") from exc
            if not _same_path(supplied_adapter, adapter):
                raise ValueError("自定义命令中的 --lora-modules 必须指向所选训练任务的 Adapter。")

        supplied_port = _read_command_option(args, "--port")
        if supplied_port is not None:
            try:
                if int(supplied_port) != port:
                    raise ValueError(f"--port 必须使用平台分配的端口 {port}。")
            except ValueError as exc:
                if str(exc).startswith("--port 必须"):
                    raise
                raise ValueError("--port 必须是数字。") from exc

        if self._cpu_inference():
            for option in ("--gpu-memory-utilization", "--quantization", "--kv-cache-dtype"):
                if _read_command_option(args, option) is not None:
                    raise ValueError(f"CPU 冒烟推理不支持 {option}，请从自定义命令中移除。")

        supplied_host = _read_command_option(args, "--host")
        if supplied_host and supplied_host not in {"0.0.0.0", "127.0.0.1", "localhost"}:
            raise ValueError("--host 只能使用本机或 0.0.0.0。")

        if positional_model:
            # The direct api_server entrypoint is launched by the platform;
            # convert `vllm serve MODEL`'s positional model to its explicit flag.
            args = ["--model", expected_model, *args[1:]]
        else:
            args = _replace_command_option(args, "--model", expected_model)
        args = _replace_command_option(args, "--served-model-name", service_id)
        args = _replace_command_option(args, "--lora-modules", f"{service_id}={adapter}")
        args = _replace_command_option(args, "--host", "0.0.0.0")
        args = _replace_command_option(args, "--port", str(port))
        args = _ensure_command_flag(args, "--enable-lora")
        return args, flags

    def _custom_command(self, service: dict[str, Any]) -> list[str]:
        config = service["config"]
        raw_command = str(config.get("customCommand") or "").strip()
        if not raw_command:
            raise ValueError("自定义命令行服务缺少 vLLM 启动命令。")
        args, _ = self._normalized_custom_args(
            raw_command,
            model=str(config["model"]),
            adapter=str(config["adapter"]),
            service_id=str(service["id"]),
            port=self.port,
        )
        return [
            self._vllm_python(),
            "-m",
            "vllm.entrypoints.openai.api_server",
            *args,
        ]

    def preview_custom_command(
        self,
        task_id: str,
        command: str,
        adapter_variant: Literal["best", "final"] = "best",
    ) -> dict[str, Any]:
        _, model, adapter = self._artifact_for_task(task_id, adapter_variant)
        args, flags = self._normalized_custom_args(
            command,
            model=model,
            adapter=adapter,
            service_id="<service-id>",
            port=self.port,
        )
        executable = self._vllm_python()
        normalized = shlex.join(
            [executable, "-m", "vllm.entrypoints.openai.api_server", *args]
        )
        return {
            "valid": True,
            "command": normalized,
            "flags": flags,
            "model": self._resolve_model_path(model),
            "adapter": adapter,
            "port": self.port,
            "note": "服务创建后会将 <service-id> 替换为真实服务 ID。",
        }

    @staticmethod
    def _resolve_model_path(model: str) -> str:
        direct_path = Path(model)
        try:
            direct_exists = direct_path.exists()
        except OSError:
            direct_exists = False
        if direct_exists:
            return str(direct_path)
        cache_root = Path(
            os.environ.get("XUNTUI_MODELSCOPE_CACHE", "/root/.cache/modelscope")
        ) / "models" / model.replace("/", "--") / "snapshots"
        try:
            is_cache_directory = cache_root.is_dir()
        except OSError:
            is_cache_directory = False
        if is_cache_directory:
            try:
                snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
            except OSError:
                snapshots = []
            if snapshots:
                preferred = next((path for path in snapshots if path.name == "master"), snapshots[0])
                return str(preferred)
        return model

    def model_context(
        self,
        model: str,
        *,
        task: dict[str, Any] | None = None,
        adapter_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if task is None:
            metadata = read_model_context(self._resolve_model_path(model))
        else:
            metadata = read_task_model_context(
                task,
                model,
                adapter_path,
                resolve_model=self._resolve_model_path,
            )
        return {"model": model, **metadata}

    async def create(self, payload: InferenceCreate) -> dict[str, Any]:
        require_provider(payload.provider)
        task, model, adapter_path = await asyncio.to_thread(
            self._artifact_for_task, payload.taskId, payload.adapterVariant
        )
        context = self.model_context(model, task=task, adapter_path=adapter_path)
        model_context = int(context["maxContextLength"])
        if payload.provider == "youyun" and context.get("isFallback"):
            raise ValueError(
                "无法确认所选 Base Model 的上下文上限。请先同步 Base Model 配置，"
                "或在训练任务中保存基模元数据后再创建优云推理服务。"
            )
        if payload.maxModelLength > model_context:
            raise ValueError(
                f"最大上下文长度不能超过基模支持的 {model_context} tokens "
                f"（来源：{context['source']}）。"
            )
        payload = payload.model_copy(
            update={
                "modelContextLength": model_context,
                "modelContextSource": context["source"],
            }
        )
        if payload.deploymentMode == "custom":
            if not payload.customCommand:
                raise ValueError("请选择自定义命令行后粘贴 vLLM 启动命令。")
            self._normalized_custom_args(
                payload.customCommand,
                model=model,
                adapter=adapter_path,
                service_id="<service-id>",
                port=self.port,
            )
        elif payload.customCommand:
            raise ValueError("标准配置模式不应包含自定义 vLLM 命令。")
        if payload.provider == "youyun" and not payload.youyunResource:
            raise ValueError("优云推理服务必须提供 youyunResource 资源配置。")
        if payload.provider == "youyun" and payload.deploymentMode == "custom":
            raise ValueError("优云推理当前只支持标准配置模式，不支持自定义命令行。")
        if payload.provider == "youyun":
            resource = dict(payload.youyunResource or {})
            training_resource = task.get("youyunResource") or {}
            # Inference serves this Adapter's original base, retaining its exact
            # catalog path while letting users select new compute resources.
            public_path = training_resource.get("publicModelPath")
            if (
                not public_path
                and task.get("modelSource") == "public"
                and str(model).startswith(("/models/", "/model/"))
            ):
                public_path = model
            if public_path:
                resource["publicModelPath"] = public_path
            else:
                resource.pop("publicModelPath", None)
            payload = payload.model_copy(update={"youyunResource": resource})
        if payload.provider == "youyun" and self.youyun_resource_resolver is not None:
            payload = payload.model_copy(
                update={"youyunResource": await self.youyun_resource_resolver(payload.youyunResource or {})}
            )
        service_id = "svc-" + uuid.uuid4().hex[:12]
        return self.store.create(
            service_id=service_id,
            payload=payload,
            model=model,
            adapter=adapter_path,
            endpoint="/v1/chat/completions",
        )

    async def start(
        self,
        service_id: str,
        *,
        recovering: bool = False,
        cancel_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        cancel_event = cancel_event or self.cancel_events.get(service_id)
        initial = self.store.get(service_id)
        if initial is None:
            raise KeyError("推理服务不存在。")
        require_provider(initial["config"].get("provider"))
        is_remote = initial["config"].get("provider", "local") == "youyun"
        if cancel_event is None and initial["state"] in {"starting", "recovering", "running", "stopping"}:
            raise RuntimeError("服务正在启动、重试、运行或停止中，请等待当前操作结束。")
        runtime_lock = self.remote_start_locks.setdefault(service_id, asyncio.Lock()) if is_remote else self.lock
        async with runtime_lock:
            service = self.store.get(service_id)
            if service is None:
                raise KeyError("推理服务不存在。")
            if service["state"] == "archived":
                raise RuntimeError("已归档服务不能启动，请先恢复服务。")
            if not is_remote and self.training_busy():
                raise RuntimeError("GPU 正在执行训练任务，请等待训练完成。")
            if not is_remote and self.process is not None and self.process.returncode is None:
                if self.service_id == service_id:
                    return service
                raise RuntimeError("当前已有其他推理服务占用 GPU。")
            try:
                self.store.create_run(service_id, service.get("config") or {})
            except Exception:
                # 兼容尚未应用 run 迁移的旧 SQLite/测试数据库。
                pass
            custom_flags: list[str] = []
            try:
                current_adapter = str(service["config"].get("adapter") or "")
                if not current_adapter or not adapter_is_complete(Path(current_adapter)):
                    self.store.append_log(service_id, "本地完整权重不可用，正在检查并恢复 MinIO 备份。")
                    adapter = await asyncio.to_thread(
                        self._resolve_adapter_path,
                        str(service["config"].get("taskId") or service.get("taskId") or ""),
                        str(service["config"].get("adapterVariant") or "best"),
                        current_adapter,
                    )
                    config = {**service["config"], "adapter": adapter}
                    if config.get("deploymentMode") == "custom" and current_adapter:
                        argv = shlex.split(str(config.get("customCommand") or ""))
                        config["customCommand"] = shlex.join([
                            item[:-len(current_adapter)] + adapter if item.endswith("=" + current_adapter) else item
                            for item in argv
                        ])
                    service = self.store.update(service_id, config=config) or service
                    self.store.append_log(service_id, f"权重恢复校验完成，本次运行路径：{adapter}")
                    if cancel_event is not None and cancel_event.is_set():
                        self._set_terminal_state(service_id, "stopped")
                        self.store.append_log(service_id, "权重恢复完成，服务启动已取消。")
                        return self.store.get(service_id) or service
                if is_remote:
                    if not service["config"].get("youyunResource"):
                        raise ValueError("优云推理服务缺少 youyunResource 资源配置。")
                    vllm_version = "remote"
                    launch_command: list[str] = []
                else:
                    vllm_version = await asyncio.to_thread(self._validate_vllm_runtime)
                    launch_command = self._command(service)
                    if service["config"].get("deploymentMode") == "custom":
                        _, custom_flags = _parse_vllm_command(str(service["config"].get("customCommand") or ""))
            except (ValueError, RuntimeError) as exc:
                self.store.update(
                    service_id,
                    state="failed",
                    exposure="offline",
                    error=str(exc),
                    gpu="未占用",
                    memory="0 / 0 GB",
                )
                self._update_run(service_id, "failed", error=str(exc))
                self.store.append_log(service_id, f"vLLM 启动前检查失败：{exc}")
                if service["config"].get("deploymentMode") == "custom":
                    self.store.record_command_usage(
                        service_id,
                        flags=custom_flags,
                        success=False,
                        error=str(exc),
                    )
                raise RuntimeError(str(exc)) from exc
            if recovering:
                self.store.update(service_id, state="recovering", gpu="优云 GPU · 重试中" if is_remote else ("CPU" if self._cpu_inference() else "GPU 0 · 重试中"), memory="加载中")
            else:
                self.store.update(service_id, state="starting", gpu="优云 GPU · 分配中" if is_remote else ("CPU" if self._cpu_inference() else "GPU 0 · 分配中"), memory="加载中")
            device_label = "优云 GPU" if is_remote else ("CPU" if self._cpu_inference() else "自动检测")
            if not is_remote:
                self.store.append_log(service_id, f"Inference Python: {self._vllm_python()}")
            if is_remote:
                resource = dict(service["config"].get("youyunResource") or {})
                self.store.append_log(
                    service_id,
                    "优云 GPU 选择："
                    f"{resource.get('gpuType') or '未知型号'} · "
                    f"显存 {resource.get('gpuMemoryGb') or resource.get('gpuMemoryGiB') or '?'}GB · "
                    f"{resource.get('gpuCount') or 1} 卡 · "
                    f"{resource.get('cpuCores') or '?'} 核 / "
                    f"{resource.get('memoryGb') or '?'}GB 内存 · "
                    f"镜像 {resource.get('imageName') or resource.get('imageId') or '?'} · "
                    f"ID {resource.get('imageId') or '?'}"
                    f"（{resource.get('imageVersion') or '优云未提供版本信息'}）",
                )
            self.store.append_log(service_id, f"vLLM version: {vllm_version}; device: {device_label}")
            self.store.append_log(service_id, f"Base Model: {service['model']}")
            self.store.append_log(service_id, f"Adapter: {service['adapter']}")
            if not is_remote:
                self.store.append_log(service_id, "开始启动 vLLM 服务。")
            process: asyncio.subprocess.Process | None = None
            try:
                if is_remote:
                    from backend.providers.youyun.inference import YouyunInferenceHandle

                    # Reuse the old owner while releasing its instance; replacing
                    # the provider first would strand its dedicated log reader.
                    previous_provider = self._remote_provider_for(service_id)
                    provider = self._make_remote_provider(service_id)
                    previous_runtime = service["config"].get("remoteRuntime")
                    if recovering and previous_runtime:
                        previous_handle = YouyunInferenceHandle.restore(previous_runtime)
                        self.store.append_log(
                            service_id,
                            f"重试前先清理旧优云实例 {previous_handle.instance_id}。",
                        )
                        report = await (previous_provider or provider).stop(previous_handle)
                        if not report.get("terminated"):
                            raise RuntimeError(
                                f"旧优云实例 {previous_handle.instance_id} 清理失败：{report.get('error')}"
                            )
                        self._store_remote_runtime(service_id, None)
                    self.remote_providers[service_id] = provider

                    def persist_instance(handle: Any) -> None:
                        self.remote_handles[service_id] = handle
                        self.remote_endpoints[service_id] = handle.endpoint
                        self._store_remote_runtime(
                            service_id,
                            {**handle.persisted(), "state": "starting", "logOffset": 0},
                        )
                        self.store.append_log(
                            service_id,
                            f"优云推理实例已创建：{handle.instance_id}。",
                        )

                    def on_stage(stage: str, message: str, details: dict[str, Any] | None = None) -> None:
                        self._update_startup(service_id, stage, message, **dict(details or {}))

                    def on_log(line: str) -> None:
                        self.store.append_log(service_id, f"[remote] {line}")

                    preflight_message, preflight_details = self._remote_memory_preflight(
                        resource,
                        service["config"],
                    )
                    self._update_startup(
                        service_id,
                        "memory_preflight",
                        preflight_message,
                        **preflight_details,
                    )
                    handle = await provider.start(
                        service_id,
                        model=str(service["config"]["model"]),
                        adapter=str(service["config"]["adapter"]),
                        resource=dict(service["config"].get("youyunResource") or {}),
                        max_model_length=int(service["config"].get("maxModelLength") or 4096),
                        extra_args=self._remote_managed_vllm_args(service["config"], resource),
                        on_instance=persist_instance,
                        on_stage=on_stage,
                        on_log=on_log,
                        on_log_cursor=lambda cursor: self._checkpoint_remote_logs(service_id, cursor),
                        cancel_requested=(cancel_event.is_set if cancel_event is not None else None),
                    )
                    if cancel_event is not None and cancel_event.is_set():
                        report = await provider.stop(handle)
                        if report.get("terminated"):
                            self._store_remote_runtime(service_id, None)
                            self._clear_remote_runtime(service_id)
                        raise RuntimeError("服务启动已取消")
                    self.remote_providers[service_id] = provider
                    self.remote_handles[service_id] = handle
                    self.remote_endpoints[service_id] = handle.endpoint
                    self._store_remote_runtime(
                        service_id,
                        {**handle.persisted(), "state": "running"},
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *launch_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        env=self._vllm_env(),
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                cleanup_report = getattr(exc, "cleanup_report", None)
                cleanup_handle = getattr(exc, "handle", None)
                was_cancelled = bool(cancel_event is not None and cancel_event.is_set())
                if is_remote and cleanup_report is not None and cleanup_handle is not None:
                    if cleanup_report.get("terminated"):
                        self._store_remote_runtime(service_id, None)
                        self._clear_remote_runtime(service_id)
                        self.store.append_log(service_id, "失败后的优云实例已自动清理。")
                    else:
                        cleanup_error = str(cleanup_report.get("error") or "未知清理错误")
                        self._store_remote_runtime(
                            service_id,
                            {
                                **cleanup_handle.persisted(),
                                "state": "cleanup_failed",
                                "cleanupError": cleanup_error,
                            },
                        )
                        self.store.append_log(
                            service_id,
                            f"失败后的优云实例自动清理失败：{cleanup_error}",
                        )
                if was_cancelled:
                    self.store.update(
                        service_id,
                        state="stopped",
                        exposure="offline",
                        error=None,
                        gpu="未占用",
                        memory="0 / 0 GB",
                    )
                    self._update_run(service_id, "stopped")
                    self.store.append_log(service_id, "启动已取消，远端实例与安装进程已释放。")
                else:
                    self._set_terminal_state(
                        service_id,
                        "failed",
                        error=f"无法启动 vLLM：{exc}",
                        cleanup_pending=bool(
                            is_remote
                            and cleanup_report is not None
                            and not cleanup_report.get("terminated")
                        ),
                    )
                    self._update_run(service_id, "failed", error=str(exc))
                    self.store.append_log(service_id, f"启动失败：{exc}")
                if is_remote:
                    if was_cancelled:
                        self._update_startup(service_id, "stopped", "优云推理启动已取消。")
                    else:
                        self._update_startup(service_id, "failed", f"优云推理启动失败：{exc}", error=str(exc))
                if service["config"].get("deploymentMode") == "custom":
                    self.store.record_command_usage(
                        service_id,
                        flags=custom_flags,
                        success=False,
                        error=str(exc),
                    )
                if was_cancelled:
                    return self.store.get(service_id)  # type: ignore[return-value]
                raise RuntimeError(f"无法启动 vLLM：{exc}") from exc
            if process is not None:
                self.process = process
                self.service_id = service_id
                watcher = asyncio.create_task(self._watch_process(service_id, process))
            else:
                watcher = asyncio.create_task(
                    self._watch_remote(service_id, provider, handle)
                )
            self.background_tasks.add(watcher)
            watcher.add_done_callback(self.background_tasks.discard)
        try:
            if process is not None:
                await self._wait_until_ready(service_id, int(service["config"]["loadTimeout"]))
        except Exception as exc:
            current = self.store.get(service_id)
            if current and current["state"] == "stopped":
                if service["config"].get("deploymentMode") == "custom":
                    self.store.record_command_usage(
                        service_id,
                        flags=custom_flags,
                        success=False,
                        error="服务启动已取消。",
                    )
                raise RuntimeError("服务启动已取消。") from exc
            await self.stop(service_id, failed=True, error=str(exc))
            if service["config"].get("deploymentMode") == "custom":
                self.store.record_command_usage(
                    service_id,
                    flags=custom_flags,
                    success=False,
                    error=str(exc),
                )
            raise
        self.store.update(
            service_id,
            state="running",
            exposure="test",
            gpu="优云 GPU" if is_remote else ("CPU" if self._cpu_inference() else "GPU 0 · 1 卡"),
            memory="远程实例运行中" if is_remote else "运行中",
        )
        self._update_run(service_id, "running")
        self.store.append_log(service_id, "vLLM 健康检查通过，服务已运行。")
        if service["config"].get("deploymentMode") == "custom" and not is_remote:
            self.store.record_command_usage(service_id, flags=custom_flags, success=True)
        await self.refresh_metrics(service_id)
        return self.store.get(service_id)  # type: ignore[return-value]

    async def _wait_until_ready(self, service_id: str, timeout: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self.process is None or self.process.returncode is not None:
                logs = self.store.logs(service_id, 120)
                detail = None
                for marker in (
                    "ModuleNotFoundError",
                    "ValueError:",
                    "CUDA",
                    "out of memory",
                    "not enough",
                    "failed to start",
                    "RuntimeError:",
                ):
                    detail = next((line for line in reversed(logs) if marker in line), None)
                    if detail:
                        break
                if detail:
                    raise RuntimeError(f"vLLM 进程在健康检查前退出：{detail[-1000:]}")
                raise RuntimeError("vLLM 进程在健康检查前退出。")
            try:
                await asyncio.to_thread(self._health_request)
                return
            except (OSError, urllib.error.URLError):
                await asyncio.sleep(1)
        raise TimeoutError("vLLM 启动超时，未通过 /health 检查。")

    def _health_request(self) -> None:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
            if response.status >= 400:
                raise OSError(f"health status {response.status}")

    async def _watch_process(self, service_id: str, process: asyncio.subprocess.Process) -> None:
        if process.stdout is not None:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.store.append_log(service_id, text[-2000:])
        return_code = await process.wait()
        if self.service_id == service_id and self.process is process:
            self.process = None
            self.service_id = None
            current = self.store.get(service_id)
            if current and current["state"] in {"starting", "running", "recovering"}:
                message = f"vLLM 进程退出，返回码 {return_code}。"
                self._set_terminal_state(service_id, "failed", error=message)
                self._update_run(service_id, "failed", error=message)
                self.store.append_log(service_id, message)

    async def _watch_remote(self, service_id: str, provider: Any, handle: Any) -> None:
        failures = 0
        while self._remote_handle_for(service_id) is handle:
            await asyncio.sleep(5)
            if await provider.health(handle):
                failures = 0
                continue
            failures += 1
            if failures < 3:
                continue
            message = "优云远程 vLLM 连续 3 次健康检查失败。"
            report = await provider.stop(handle)
            if report.get("terminated"):
                self._store_remote_runtime(service_id, None)
                self._clear_remote_runtime(service_id)
                message += " 实例已自动释放。"
            else:
                self._store_remote_runtime(
                    service_id,
                    {**handle.persisted(), "state": "cleanup_failed", "cleanupError": report.get("error")},
                )
                message += f" 实例清理失败：{report.get('error')}。"
            self._set_terminal_state(
                service_id,
                "failed",
                error=message,
                cleanup_pending=not report.get("terminated"),
            )
            self._update_run(service_id, "failed", error=message)
            self.store.append_log(service_id, message)
            return

    async def _terminate_process(self) -> None:
        process = self.process
        self.process = None
        self.service_id = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def stop(self, service_id: str, *, failed: bool = False, error: str | None = None) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        if service["state"] == "archived":
            raise RuntimeError("已归档服务不能停止，请先恢复服务。")
        is_remote = service["config"].get("provider", "local") == "youyun"
        remote_runtime = service["config"].get("remoteRuntime")
        cancel_event = self.cancel_events.get(service_id)
        if cancel_event is not None:
            cancel_event.set()
        if is_remote:
            self.store.update(
                service_id,
                state="stopping",
                exposure="offline",
                gpu="优云实例释放中",
                memory="释放中",
            )
            provider = self._remote_provider_for(service_id)
            handle = self._remote_handle_for(service_id)
            if handle is None and remote_runtime:
                from backend.providers.youyun.inference import YouyunInferenceHandle

                handle = YouyunInferenceHandle.restore(remote_runtime)
            if handle is not None:
                provider = provider or self._make_remote_provider(service_id)
                self.store.append_log(service_id, f"开始停止并释放优云推理实例 {handle.instance_id}。")
                report = await provider.stop(handle)
                if not report.get("terminated"):
                    message = f"优云推理实例 {handle.instance_id} 释放失败：{report.get('error')}"
                    self._store_remote_runtime(
                        service_id,
                        {**handle.persisted(), "state": "cleanup_failed", "cleanupError": report.get("error")},
                    )
                    self._set_terminal_state(
                        service_id,
                        "failed",
                        error=message,
                        cleanup_pending=True,
                    )
                    self._update_run(service_id, "failed", error=message)
                    self.store.append_log(service_id, message)
                    raise RuntimeError(message)
                self._store_remote_runtime(service_id, None)
                self._clear_remote_runtime(service_id)
            self._set_terminal_state(
                service_id,
                "failed" if failed else "stopped",
                error=error if failed else None,
            )
            self._update_run(service_id, "failed" if failed else "stopped", error=error if failed else None)
            self.store.append_log(service_id, error if failed and error else "服务已停止，GPU 资源已释放。")
            return self.store.get(service_id)  # type: ignore[return-value]
        async with self.lock:
            if self.service_id == service_id:
                await self._terminate_process()
            self._set_terminal_state(
                service_id,
                "failed" if failed else "stopped",
                error=error if failed else None,
            )
            self._update_run(service_id, "failed" if failed else "stopped", error=error if failed else None)
            self.store.append_log(service_id, error if failed and error else "服务已停止，GPU 资源已释放。")
            return self.store.get(service_id)  # type: ignore[return-value]

    async def restart(self, service_id: str) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        require_provider(service["config"].get("provider"))
        await self.stop(service_id)
        return await self.start(service_id)

    async def recover(
        self,
        service_id: str,
        *,
        youyun_resource: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        require_provider(service["config"].get("provider"))
        if service["state"] == "archived":
            raise RuntimeError("已归档服务不能重试，请先恢复服务。")
        start_task = self.start_tasks.get(service_id)
        operation_active = service["state"] in {"starting", "recovering", "stopping"} or (
            service["state"] == "running" and self._remote_handle_for(service_id) is not None
        )
        if operation_active or (
            start_task is not None and not start_task.done()
        ):
            raise RuntimeError("服务正在启动、重试、运行或停止中，请等待当前操作结束后再重试。")
        self.store.append_log(service_id, "开始按原配置重试。")
        is_remote = service["config"].get("provider", "local") == "youyun"
        self.store.update(
            service_id,
            state="recovering",
            error=None,
            gpu="优云 GPU · 重试中" if is_remote else ("CPU" if self._cpu_inference() else "GPU 0 · 重试中"),
            memory="加载中",
        )
        if is_remote and self.youyun_resource_resolver is not None:
            resource = youyun_resource or service["config"].get("youyunResource") or {}
            resource = {
                key: resource[key]
                for key in ("mode", "gpuType", "gpuCount", "cpuCores", "memoryGb", "region", "zone", "instanceMode", "imageId", "imageName", "imageVersion", "purpose")
                if key in resource
            }
            try:
                resolved = await self.youyun_resource_resolver(dict(resource))
            except Exception as exc:  # noqa: BLE001 - persist inventory failures for the retry UI
                detail = str(getattr(exc, "detail", "") or exc).strip() or "优云 GPU 库存校验失败。"
                self._set_terminal_state(service_id, "failed", error=detail)
                self.store.append_log(service_id, f"重试前库存校验失败：{detail}")
                raise
            config = dict(service["config"])
            config["youyunResource"] = resolved
            self.store.update(service_id, config=config)
        cancel_event = asyncio.Event()
        current_task = asyncio.current_task()
        if current_task is not None:
            self.start_tasks[service_id] = current_task
        self.cancel_events[service_id] = cancel_event
        try:
            return await self.start(service_id, recovering=True)
        except Exception as exc:
            self._set_terminal_state(service_id, "failed", error=str(exc))
            self.store.append_log(service_id, f"重试失败：{exc}")
            raise
        finally:
            if self.start_tasks.get(service_id) is current_task:
                self.start_tasks.pop(service_id, None)
                self.cancel_events.pop(service_id, None)

    async def archive(self, service_id: str) -> dict[str, Any]:
        async with self.lock:
            service = self.store.get(service_id)
            if service is None:
                raise KeyError("推理服务不存在。")
            if service["state"] in {"starting", "running", "stopping", "recovering"}:
                raise RuntimeError("运行中的服务不能归档，请先停止服务。")
            if self.service_id == service_id and self.process is not None:
                raise RuntimeError("运行中的服务不能归档，请先停止服务。")

            remote_runtime = service["config"].get("remoteRuntime")
            if remote_runtime:
                from backend.providers.youyun.inference import YouyunInferenceHandle

                handle = self._remote_handle_for(service_id) or YouyunInferenceHandle.restore(remote_runtime)
                provider = self._remote_provider_for(service_id) or self._make_remote_provider(service_id)
                self.store.append_log(
                    service_id,
                    f"归档前确认并释放优云推理实例 {handle.instance_id}。",
                )
                report = await provider.stop(handle)
                if not report.get("terminated"):
                    cleanup_error = str(report.get("error") or "未知清理错误")
                    message = f"优云推理实例 {handle.instance_id} 释放失败，不能归档：{cleanup_error}"
                    self._store_remote_runtime(
                        service_id,
                        {
                            **handle.persisted(),
                            "state": "cleanup_failed",
                            "cleanupError": cleanup_error,
                        },
                    )
                    self._set_terminal_state(
                        service_id,
                        "failed",
                        error=message,
                        cleanup_pending=True,
                    )
                    self.store.append_log(service_id, message)
                    raise RuntimeError(message)
                self._store_remote_runtime(service_id, None)
                self._clear_remote_runtime(service_id)
                self.store.append_log(service_id, "归档前已确认优云推理实例释放完成。")
            return self.store.archive(service_id)

    async def restore(self, service_id: str) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        require_provider(service["config"].get("provider"))
        async with self.lock:
            return self.store.restore(service_id)

    async def delete(self, service_id: str) -> None:
        async with self.lock:
            service = self.store.get(service_id)
            if service is None:
                raise KeyError("推理服务不存在。")
            if service["state"] != "archived":
                raise RuntimeError("只有已归档服务可以永久删除。")
            if service["config"].get("remoteRuntime"):
                raise RuntimeError("优云实例尚未释放，不能删除。")
            if (self.service_id == service_id and self.process is not None) or self._remote_handle_for(service_id) is not None:
                raise RuntimeError("服务仍占用运行资源，不能删除。")
            self.store.delete(service_id)

    def _default_metrics(self, service: dict[str, Any], *, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "pagedAttention": True,
            "continuousBatching": True,
            "chunkedPrefill": bool(service["config"].get("enableChunkedPrefill")),
            "prefixCaching": bool(service["config"].get("enablePrefixCaching")),
            "kvCacheBytes": None,
            "kvCacheTokenCapacity": None,
            "kvCacheUsagePercent": None,
            "prefixCacheHitRate": None,
            "runningRequests": 0,
            "waitingRequests": 0,
            "batchSize": 0,
            "ttftMs": None,
            "decodeTokensPerSecond": None,
        }

    async def metrics(self, service_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        current = self.store.metrics(service_id, run_id=run_id) if run_id else self.store.metrics(service_id)
        if service["state"] == "running" and run_id is None:
            current = await self.refresh_metrics(service_id)
        return {**current, "state": service["state"], "serviceId": service_id}

    def _checkpoint_remote_logs(self, service_id: str, cursor: Any) -> None:
        current = self.store.get(service_id)
        runtime = dict((current or {}).get("config", {}).get("remoteRuntime") or {})
        if runtime:
            runtime.update(logOffset=cursor.offset, logFileId=cursor.file_id)
            self._store_remote_runtime(service_id, runtime)

    async def refresh_remote_logs(self, service_id: str) -> None:
        """Idempotently attach a missing lifecycle reader; never poll per request."""
        service = self.store.get(service_id)
        if service is None or service["config"].get("provider", "local") != "youyun":
            return
        if service.get("state") not in {"starting", "running", "recovering"}:
            return
        remote_runtime = service["config"].get("remoteRuntime")
        if not remote_runtime:
            return
        # The startup owner installs the reader once the selected Python and
        # vllm.log are ready. An HTTP request must not race it with defaults.
        start_lock = self.remote_start_locks.get(service_id)
        if service_id in self.start_tasks or (start_lock is not None and start_lock.locked()):
            return
        provider = self._remote_provider_for(service_id)
        if provider is None:
            provider = self._make_remote_provider(service_id)
            self.remote_providers[service_id] = provider
        handle = self._remote_handle_for(service_id)
        if handle is None:
            handle = self._restore_remote_handle(remote_runtime)
            self.remote_handles[service_id] = handle
        ensure = getattr(provider, "ensure_log_collector", None)
        if callable(ensure):
            ensure(handle,
                on_log=lambda line: self.store.append_log(service_id, f"[remote] {line}"),
                on_cursor=lambda cursor: self._checkpoint_remote_logs(service_id, cursor))

    @staticmethod
    def _restore_remote_handle(value: dict[str, Any]) -> Any:
        from backend.providers.youyun.inference import YouyunInferenceHandle

        return YouyunInferenceHandle.restore(value)

    async def refresh_metrics(self, service_id: str) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        base = self._default_metrics(service, source="vllm")
        try:
            observed = await asyncio.to_thread(self._read_prometheus_metrics, service_id)
        except (OSError, urllib.error.URLError, TimeoutError):
            observed = {}
        metrics = {**base, **observed}
        self.store.set_metrics(service_id, metrics)
        return metrics

    def _read_prometheus_metrics(self, service_id: str | None = None) -> dict[str, Any]:
        with urllib.request.urlopen(
            f"{self._inference_endpoint(service_id)}/metrics", timeout=3
        ) as response:
            text = response.read().decode("utf-8", errors="replace")
        values: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.match(
                r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)",
                line,
            )
            if match:
                try:
                    values[match.group(1)] = float(match.group(2))
                except ValueError:
                    continue

        def first(*names: str) -> float | None:
            for name in names:
                if name in values:
                    return values[name]
            return None

        prefix_queries = first("vllm:prefix_cache_queries")
        prefix_hits = first("vllm:prefix_cache_hits")
        prefix_rate = (
            prefix_hits / prefix_queries
            if prefix_queries is not None and prefix_queries > 0 and prefix_hits is not None
            else None
        )

        def average(*prefixes: str) -> float | None:
            for prefix in prefixes:
                total = values.get(prefix + "_sum")
                count = values.get(prefix + "_count")
                if total is not None and count and count > 0:
                    return total / count
            return None

        ttft = average(
            "vllm:time_to_first_token_seconds",
            "vllm:time_to_first_token",
        )
        tpot = average(
            "vllm:time_per_output_token_seconds",
            "vllm:inter_token_latency_seconds",
        )
        return {
            "source": "vllm_prometheus",
            "kvCacheUsagePercent": (
                first("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
            ),
            "runningRequests": first("vllm:num_requests_running"),
            "waitingRequests": first(
                "vllm:num_requests_waiting", "vllm:num_requests_waiting_swapped"
            ),
            "prefixCacheQueries": prefix_queries,
            "prefixCacheHits": prefix_hits,
            "prefixCacheHitRate": prefix_rate,
            "promptTokensTotal": first("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
            "generationTokensTotal": first(
                "vllm:generation_tokens_total", "vllm:generation_tokens"
            ),
            "ttftMs": ttft * 1000 if ttft is not None else None,
            "tpotMs": tpot * 1000 if tpot is not None else None,
        }

    async def proxy_json(self, service_id: str, payload: InferenceChatRequest) -> dict[str, Any]:
        service = self._running_service(service_id)
        body = self._chat_body(service, payload)
        try:
            raw = await asyncio.to_thread(self._post_json, service_id, body)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"推理请求失败：{exc}") from exc
        return json.loads(raw)

    def proxy_stream(self, service_id: str, payload: InferenceChatRequest):
        service = self._running_service(service_id)
        body = self._chat_body(service, payload)

        remote_handle = self._remote_handle_for(service_id)
        remote_ssh = getattr(remote_handle, "ssh", None) if remote_handle is not None else None
        if remote_handle is not None and remote_ssh is not None:
            def remote_iterator():
                command = (
                    f"curl -fsS -N -X POST http://127.0.0.1:{self.port}/v1/chat/completions "
                    f"-H 'Content-Type: application/json' "
                    f"--data-raw {shlex.quote(json.dumps(body, ensure_ascii=False))}"
                )
                yield from remote_ssh.stream(command, timeout=600)

            return remote_iterator()

        def iterator():
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                f"{self._inference_endpoint(service_id)}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                while True:
                    # vLLM emits small SSE lines; read() may wait for the full
                    # buffer before yielding the first token to the browser.
                    line = response.readline()
                    if not line:
                        break
                    yield line

        return iterator()

    def _running_service(self, service_id: str) -> dict[str, Any]:
        service = self.store.get(service_id)
        if service is None:
            raise KeyError("推理服务不存在。")
        require_provider(service["config"].get("provider"))
        is_remote = service["config"].get("provider", "local") == "youyun"
        is_active = (
            self._remote_handle_for(service_id) is not None
            if is_remote
            else self.service_id == service_id
        )
        if service["state"] != "running" or not is_active:
            raise RuntimeError("推理服务当前未运行。")
        return service

    def _inference_endpoint(self, service_id: str | None = None) -> str:
        if service_id and service_id in self.remote_endpoints:
            return self.remote_endpoints[service_id]
        return f"http://127.0.0.1:{self.port}"

    @staticmethod
    def _chat_body(service: dict[str, Any], payload: InferenceChatRequest) -> dict[str, Any]:
        config = service["config"]
        model_context_limit = int(config.get("modelContextLength") or 4096)
        service_context_limit = int(config.get("maxModelLength") or model_context_limit)
        context_limit = min(model_context_limit, service_context_limit)
        # Generation length is request-scoped. Older services may still carry
        # config.maxTokens, but it no longer acts as a service default or cap.
        output_limit = context_limit
        if payload.maxTokens > output_limit:
            raise ValueError(
                f"请求最大输出为 {payload.maxTokens} tokens，超过服务上限 {output_limit} tokens。"
            )
        return {
            "model": service["id"],
            "messages": [message.model_dump() for message in payload.messages],
            "max_tokens": payload.maxTokens,
            "temperature": payload.temperature,
            "top_p": payload.topP,
            "stream": payload.stream,
        }

    def _post_json(self, service_id: str, body: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self._inference_endpoint(service_id)}/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read().decode("utf-8")


def _startup_progress_signature(config: Any) -> str:
    """Fingerprint of the remote startup progress embedded in service config.

    While a remote service boots, ``state`` stays ``starting`` but
    ``config.startup`` (stage/progress/message/elapsed) advances every few
    seconds. The stream endpoint must push those changes or the UI progress
    panel freezes on its first snapshot.
    """
    if not isinstance(config, dict):
        return ""
    startup = config.get("startup")
    if not isinstance(startup, dict):
        return ""
    return json.dumps(
        {
            "stage": startup.get("stage"),
            "progress": startup.get("progress"),
            "message": startup.get("message"),
            "elapsedSeconds": startup.get("elapsedSeconds"),
            "stageElapsedSeconds": startup.get("stageElapsedSeconds"),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def create_inference_router(
    runtime: InferenceRuntime,
    audit_callback: Callable[..., None] | None = None,
    websocket_auth: Callable[[WebSocket], bool] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/inference", tags=["inference"])

    def audit(action: str, service_id: str, **details: Any) -> None:
        if audit_callback is not None:
            audit_callback(action, "inference_service", service_id, details=details)

    def get_service(service_id: str) -> dict[str, Any]:
        service = runtime.store.get(service_id)
        if service is None:
            raise HTTPException(status_code=404, detail="推理服务不存在。")
        return service

    @router.get("/services")
    async def list_inference_services(
        includeArchived: bool = False,
        page: int = Query(default=1, ge=1),
        pageSize: int = Query(default=20, ge=1, le=100),
        state: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        return runtime.store.list_page(
            include_archived=includeArchived,
            page=page,
            page_size=pageSize,
            state=state,
            query=query,
        )

    @router.post("/services", status_code=status.HTTP_201_CREATED)
    async def create_inference_service(payload: InferenceCreate) -> dict[str, Any]:
        try:
            # Cache sizing/dtype and tensor parallelism remain platform-owned
            # until their compatibility matrix is complete.
            payload = payload.model_copy(
                update={
                    "kvCacheMode": "auto",
                    "kvCacheMemoryGiB": None,
                    "kvCacheDtype": "auto",
                    "tensorParallelSize": 1,
                }
            )
            result = await runtime.create(payload)
            audit("create", result["id"], provider=payload.provider)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/commands/validate")
    async def validate_inference_command(payload: InferenceCommandValidate) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                runtime.preview_custom_command, payload.taskId, payload.command, payload.adapterVariant
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _model_context_response(
        model: str,
        taskId: str | None = None,
        adapterVariant: Literal["best", "final"] = "best",
    ) -> dict[str, Any]:
        if taskId:
            try:
                task, task_model, adapter = await asyncio.to_thread(runtime._artifact_for_task, taskId, adapterVariant)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if model and model != task_model:
                raise HTTPException(status_code=400, detail="所选模型与训练任务的 Base Model 不一致。")
            return runtime.model_context(task_model, task=task, adapter_path=adapter)
        return runtime.model_context(model)

    @router.get("/model-context")
    async def inference_model_context(
        model: str,
        taskId: str | None = None,
        adapterVariant: Literal["best", "final"] = "best",
    ) -> dict[str, Any]:
        return await _model_context_response(model, taskId, adapterVariant)

    # Keep the original frontend path working while clients migrate to the
    # canonical ``/model-context`` endpoint.
    @router.get("/models/context")
    async def inference_model_context_compat(
        model: str,
        taskId: str | None = None,
        adapterVariant: Literal["best", "final"] = "best",
    ) -> dict[str, Any]:
        return await _model_context_response(model, taskId, adapterVariant)

    @router.get("/command-usage")
    async def inference_command_usage(serviceId: str | None = None) -> dict[str, Any]:
        return runtime.store.command_usage(serviceId)

    @router.get("/services/{service_id}")
    async def get_inference_service(service_id: str) -> dict[str, Any]:
        return get_service(service_id)

    @router.post("/services/{service_id}/start")
    async def start_inference_service(service_id: str) -> dict[str, Any]:
        service = get_service(service_id)
        try:
            if service["config"].get("provider", "local") == "youyun":
                result = await runtime.start_background(service_id)
            else:
                result = await runtime.start(service_id)
            audit("start", service_id)
            return result
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/services/{service_id}/stop")
    async def stop_inference_service(service_id: str) -> dict[str, Any]:
        try:
            result = await runtime.stop(service_id)
            audit("stop", service_id)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/services/{service_id}/restart")
    async def restart_inference_service(service_id: str) -> dict[str, Any]:
        try:
            result = await runtime.restart(service_id)
            audit("restart", service_id)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/services/{service_id}/recover")
    async def recover_inference_service(
        service_id: str,
        payload: InferenceRetryRequest | None = None,
    ) -> dict[str, Any]:
        try:
            result = await runtime.recover(
                service_id,
                youyun_resource=payload.youyunResource if payload else None,
            )
            audit("retry", service_id)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HTTPException:
            raise
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/services/{service_id}/archive")
    async def archive_inference_service(service_id: str) -> dict[str, Any]:
        try:
            result = await runtime.archive(service_id)
            audit("archive", service_id)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/services/{service_id}/restore")
    async def restore_inference_service(service_id: str) -> dict[str, Any]:
        try:
            result = await runtime.restore(service_id)
            audit("restore", service_id)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.delete("/services/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_inference_service(service_id: str) -> None:
        try:
            await runtime.delete(service_id)
            audit("delete", service_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/services/{service_id}/exposure")
    async def update_exposure(service_id: str, payload: ExposureRequest) -> dict[str, Any]:
        service = get_service(service_id)
        if payload.exposure == "online" and service["state"] != "running":
            raise HTTPException(status_code=409, detail="只有运行中的服务可以上线。")
        if service["state"] in {"failed", "stopped", "archived"} and payload.exposure != "offline":
            raise HTTPException(status_code=409, detail="启动失败、已停止或已归档的服务必须保持已下线。")
        return runtime.store.update(service_id, exposure=payload.exposure)  # type: ignore[return-value]

    @router.get("/services/{service_id}/metrics")
    async def inference_metrics(service_id: str, runId: str | None = None) -> dict[str, Any]:
        get_service(service_id)
        return await runtime.metrics(service_id, run_id=runId)

    @router.get("/services/{service_id}/logs")
    async def inference_logs(
        service_id: str,
        limit: int = Query(default=200, ge=1, le=1000),
        runId: str | None = None,
        afterId: int = Query(default=0, ge=0),
        beforeId: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        get_service(service_id)
        try:
            await runtime.refresh_remote_logs(service_id)
        except Exception as exc:  # noqa: BLE001 - return persisted logs even if SSH is unavailable
            runtime.store.append_log(service_id, f"远端日志刷新失败：{exc}")
        if hasattr(runtime.store, "log_items"):
            return runtime.store.log_items(
                service_id,
                run_id=runId,
                after_id=afterId,
                before_id=beforeId,
                limit=limit,
            )
        return {"serviceId": service_id, "logs": runtime.store.logs(service_id, limit)}

    @router.websocket("/services/{service_id}/stream")
    async def inference_stream(websocket: WebSocket, service_id: str) -> None:
        """Push persisted inference logs plus service state/metrics in real time.

        The stream is deliberately backed by the same durable stores as the REST
        endpoints.  A reconnect can therefore resume from ``afterId`` without
        relying on process-local buffers.
        """
        if websocket_auth is not None and not websocket_auth(websocket):
            await websocket.close(code=4401)
            return
        service = runtime.store.get(service_id)
        if service is None:
            await websocket.close(code=4404)
            return
        try:
            cursor = max(0, int(websocket.query_params.get("afterId", "0")))
        except (TypeError, ValueError):
            cursor = 0
        initial_snapshot = cursor == 0
        await websocket.accept()
        last_state: str | None = None
        last_startup_signature: str | None = None
        last_metrics: str | None = None
        terminal_sent = False
        last_heartbeat = monotonic()
        heartbeat_interval = WEBSOCKET_HEARTBEAT_INTERVAL_SECONDS
        try:
            while True:
                current = runtime.store.get(service_id)
                if current is None:
                    await websocket.close(code=4404)
                    return
                # Remote providers persist newly appended lines through this
                # method; local services simply return without extra work.
                try:
                    await runtime.refresh_remote_logs(service_id)
                except Exception as exc:  # noqa: BLE001 - keep the stream alive
                    logging.getLogger(__name__).debug(
                        "inference websocket remote log refresh failed: %s", exc
                    )
                if hasattr(runtime.store, "log_items"):
                    while True:
                        payload = runtime.store.log_items(
                            service_id, after_id=cursor, limit=500
                        )
                        previous = cursor
                        for item in payload.get("items", []):
                            item_id = int(item.get("id") or 0)
                            if item_id <= cursor:
                                continue
                            cursor = item_id
                            await websocket.send_json({
                                "type": "log",
                                "serviceId": service_id,
                                "id": item_id,
                                "runId": item.get("run_id") or payload.get("runId"),
                                "level": item.get("level"),
                                "source": item.get("source"),
                                "createdAt": serialize_timestamp(item.get("created_at"))
                                if item.get("created_at") is not None
                                else None,
                                "snapshot": initial_snapshot,
                                "message": item.get("message") or "",
                            })
                        if not payload.get("hasMore") or cursor == previous:
                            break
                state = str(current.get("state") or "created")
                startup_signature = _startup_progress_signature(current.get("config"))
                if state != last_state or startup_signature != last_startup_signature:
                    await websocket.send_json({
                        "type": "state",
                        "serviceId": service_id,
                        "state": state,
                        "service": current,
                        "snapshot": initial_snapshot,
                    })
                    last_state = state
                    last_startup_signature = startup_signature
                try:
                    metrics = await runtime.metrics(service_id)
                except Exception as exc:  # noqa: BLE001 - metrics must not terminate log streaming
                    logging.getLogger(__name__).warning(
                        "inference websocket metrics refresh failed: %s", exc
                    )
                else:
                    metrics_key = json.dumps(metrics, sort_keys=True, ensure_ascii=False, default=str)
                    if metrics_key != last_metrics:
                        await websocket.send_json({
                            "type": "metrics",
                            "serviceId": service_id,
                            "metrics": metrics,
                        })
                        last_metrics = metrics_key
                if state in {"failed", "stopped", "archived"}:
                    if not terminal_sent:
                        await websocket.send_json({
                            "type": "terminal",
                            "serviceId": service_id,
                            "state": state,
                            "service": current,
                        })
                    terminal_sent = True
                    return
                now = monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "serviceId": service_id,
                        "timestamp": now_iso(),
                    })
                    last_heartbeat = now
                initial_snapshot = False
                await asyncio.sleep(0.8)
        except WebSocketDisconnect:
            return

    @router.get("/services/{service_id}/runs")
    async def inference_runs(service_id: str) -> list[dict[str, Any]]:
        get_service(service_id)
        return runtime.store.list_runs(service_id) if hasattr(runtime.store, "list_runs") else []

    @router.get("/services/{service_id}/logs/download")
    async def download_inference_logs(service_id: str, runId: str | None = None) -> StreamingResponse:
        get_service(service_id)
        try:
            await runtime.refresh_remote_logs(service_id)
        except Exception as exc:  # noqa: BLE001 - export persisted logs even if SSH is unavailable
            runtime.store.append_log(service_id, f"远端日志刷新失败：{exc}")
        if hasattr(runtime.store, "export_logs"):
            lines = runtime.store.export_logs(service_id, run_id=runId)
        else:
            lines = runtime.store.logs(service_id, limit=1000)
        safe_service_id = re.sub(r"[^A-Za-z0-9._-]+", "_", service_id) or "inference-service"
        return StreamingResponse(
            (f"{line}\n" for line in lines),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe_service_id}-runtime.log"'},
        )

    @router.post("/services/{service_id}/chat")
    async def inference_chat(service_id: str, payload: InferenceChatRequest):
        get_service(service_id)
        try:
            if payload.stream:
                return StreamingResponse(
                    runtime.proxy_stream(service_id, payload),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return await runtime.proxy_json(service_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError, OSError, urllib.error.URLError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router
