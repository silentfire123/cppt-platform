from __future__ import annotations

import os
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Linux/CPU-server FastAPI entrypoint; no Windows or WSL dependency."""
    # Uvicorn chooses its bind address before importing backend.app, so load
    # the project configuration here as well without importing the API.
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
    host = os.environ.get("XUNTUI_BACKEND_HOST", "127.0.0.1")
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
