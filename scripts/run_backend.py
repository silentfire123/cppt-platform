from __future__ import annotations

import os
import select
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path, PureWindowsPath


DISTRO = os.environ.get("XUNTUI_WSL_DISTRO", "Ubuntu-24.04")
WSL_PYTHON = os.environ.get(
    "XUNTUI_WSL_PYTHON", "/root/miniconda3/envs/unsloth_env/bin/python"
)
VLLM_PYTHON = os.environ.get(
    "XUNTUI_VLLM_PYTHON", "/root/miniconda3/envs/vllm_env/bin/python"
)
BACKEND_PORT = int(os.environ.get("XUNTUI_BACKEND_PORT", "8010"))


def backend_environment_args() -> list[str]:
    values = [f"XUNTUI_VLLM_PYTHON={VLLM_PYTHON}"]
    storage_backend = os.environ.get("XUNTUI_STORAGE_BACKEND", "").strip()
    if storage_backend:
        values.append(f"XUNTUI_STORAGE_BACKEND={storage_backend}")
    return values


def windows_to_wsl(path: Path) -> str:
    windows_path = PureWindowsPath(path.resolve())
    drive = windows_path.drive.rstrip(":").lower()
    tail = "/".join(windows_path.parts[1:])
    return f"/mnt/{drive}/{tail}"


def wsl_ip() -> str:
    result = subprocess.run(
        ["wsl.exe", "-d", DISTRO, "--", "hostname", "-I"],
        check=True,
        capture_output=True,
        text=True,
    )
    addresses = result.stdout.strip().split()
    if not addresses:
        raise RuntimeError(f"Unable to determine the IP address of {DISTRO}.")
    return addresses[0]


def start_localhost_proxy(upstream_host: str) -> socketserver.ThreadingTCPServer:
    class ProxyHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                upstream = socket.create_connection(
                    (upstream_host, BACKEND_PORT), timeout=5
                )
            except OSError:
                return
            with upstream:
                sockets = [self.request, upstream]
                while True:
                    readable, _, _ = select.select(sockets, [], [], 30)
                    if not readable:
                        continue
                    for source in readable:
                        try:
                            data = source.recv(65536)
                        except OSError:
                            return
                        if not data:
                            return
                        target = upstream if source is self.request else self.request
                        try:
                            target.sendall(data)
                        except OSError:
                            return

    class ProxyServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ProxyServer(("127.0.0.1", BACKEND_PORT), ProxyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    wsl_root = windows_to_wsl(project_root)
    upstream_host = wsl_ip()
    proxy = start_localhost_proxy(upstream_host)
    command = [
        "wsl.exe",
        "-d",
        DISTRO,
        "--",
        "env",
        *backend_environment_args(),
        WSL_PYTHON,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--app-dir",
        wsl_root,
        "--host",
        "0.0.0.0",
        "--port",
        str(BACKEND_PORT),
        "--no-access-log",
    ]
    print(
        f"Starting backend in {DISTRO} ({upstream_host}) and forwarding "
        f"http://127.0.0.1:{BACKEND_PORT}",
        flush=True,
    )
    try:
        return subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        print("wsl.exe not found. Enable WSL before starting the backend.", file=sys.stderr)
        return 1
    finally:
        proxy.shutdown()
        proxy.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
