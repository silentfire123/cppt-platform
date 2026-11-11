from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.governance import AuditStore, cleanup_runtime_logs


def main() -> int:
    parser = argparse.ArgumentParser(description="清理训推平台过期运行日志和操作审计")
    parser.add_argument("--dry-run", action="store_true", help="仅输出策略，不执行删除")
    args = parser.parse_args()
    backend = os.environ.get("XUNTUI_STORAGE_BACKEND", "").strip().lower()
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not backend:
        backend = "postgres" if dsn else "sqlite"
    database = Path(os.environ.get("XUNTUI_DATABASE_PATH", "runtime/training.db"))
    if args.dry_run:
        print("dry-run: 成功日志 7 天，失败/停止日志 30 天，操作审计 30 天")
        return 0
    result = cleanup_runtime_logs(backend=backend, sqlite_path=database, postgres_dsn=dsn)
    audit = AuditStore(backend, database, dsn)
    audit.initialize()
    result["auditLogs"] = audit.cleanup()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
