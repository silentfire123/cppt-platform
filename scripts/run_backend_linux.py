from __future__ import annotations

import os
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Linux/CPU-server FastAPI entrypoint; no Windows or WSL dependency."""
    host = os.environ.get("XUNTUI_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("XUNTUI_BACKEND_PORT", "8010"))
    log_level = os.environ.get("XUNTUI_LOG_LEVEL", "info")
    uvicorn.run(
        "backend.app:app",
        app_dir=str(PROJECT_ROOT),
        host=host,
        port=port,
        log_level=log_level,
        proxy_headers=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
