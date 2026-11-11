from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator

from backend.inference_service import InferenceStore, _row_value


class _Cursor:
    def __init__(self, *, count: int | None = None) -> None:
        self.count = count

    def fetchone(self) -> dict[str, int] | None:
        return {"total_count": self.count} if self.count is not None else None

    def fetchall(self) -> list[Any]:
        return []


class _TimestampStrictConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str, params: Any = ()) -> _Cursor:
        self.statements.append(sql)
        if "deleted_at = ''" in sql:
            raise AssertionError("TIMESTAMPTZ must not be compared with an empty string")
        return _Cursor(count=0) if "COUNT(*)" in sql else _Cursor()


class _RowsCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def fetchone(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _LogConnection:
    def execute(self, sql: str, params: Any = ()) -> _RowsCursor:
        if "SELECT id AS run_id" in sql:
            return _RowsCursor([{"run_id": "run-1"}])
        if "SELECT id, created_at, level, source, message" in sql:
            return _RowsCursor(
                [{"id": 1, "created_at": "2026-08-28T12:00:00+00:00", "level": "INFO", "source": "backend", "message": "ok"}]
            )
        return _RowsCursor([])


class _TimestampStrictStore(InferenceStore):
    def __init__(self) -> None:
        self.connection = _TimestampStrictConnection()

    @contextmanager
    def _connect(self) -> Iterator[_TimestampStrictConnection]:
        yield self.connection


class _LogStore(InferenceStore):
    def __init__(self) -> None:
        self.connection = _LogConnection()

    @contextmanager
    def _connect(self) -> Iterator[_LogConnection]:
        yield self.connection


class InferencePostgresPaginationTests(unittest.TestCase):
    def test_list_page_uses_timestamp_safe_soft_delete_predicate(self) -> None:
        store = _TimestampStrictStore()

        result = store.list_page(include_archived=True)

        self.assertEqual(result["total"], 0)
        self.assertTrue(store.connection.statements)
        self.assertTrue(
            all("deleted_at IS NULL" in sql for sql in store.connection.statements)
        )

    def test_log_items_reads_postgres_mapping_rows_by_name(self) -> None:
        result = _LogStore().log_items("svc-1")

        self.assertEqual(result["runId"], "run-1")
        self.assertEqual(result["items"][0]["message"], "ok")

    def test_row_value_supports_mapping_and_tuple_rows(self) -> None:
        self.assertEqual(_row_value({"run_id": "run-1"}, "run_id"), "run-1")
        self.assertEqual(_row_value(("run-2",), "run_id"), "run-2")


if __name__ == "__main__":
    unittest.main()
