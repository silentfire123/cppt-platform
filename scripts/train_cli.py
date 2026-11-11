from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> Any:
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接训练服务 {url}: {exc.reason}") from exc


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def task_url(api: str, task_id: str) -> str:
    return f"{api.rstrip('/')}/tasks/{task_id}"


def command_submit(args: argparse.Namespace) -> int:
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        task = request_json(
            f"{args.api.rstrip('/')}/tasks",
            method="POST",
            payload=config,
            timeout=60,
        )
    else:
        command = "python scripts/train_cli.py submit " + shlex.join(args.command_args)
        task = request_json(
            f"{args.api.rstrip('/')}/tasks/cli",
            method="POST",
            payload={"command": command},
            timeout=60,
        )
    print_json(task)
    return 0


def command_status(args: argparse.Namespace) -> int:
    task = request_json(task_url(args.api, args.task_id))
    print_json(task)
    return 0


def command_watch(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout_seconds
    last_marker = ""
    while time.monotonic() < deadline:
        task = request_json(task_url(args.api, args.task_id))
        marker = "{} {}% {} loss={} best={}".format(
            task["state"],
            task.get("progress", 0),
            task.get("currentStep", ""),
            task.get("loss"),
            task.get("bestLoss"),
        )
        if marker != last_marker:
            print(marker, flush=True)
            last_marker = marker
        if task["state"] in {"completed", "failed", "stopped"}:
            print_json(task)
            return 0 if task["state"] in {"completed", "stopped"} else 1
        time.sleep(args.poll_seconds)
    raise TimeoutError(f"任务 {args.task_id} 在超时时间内未结束")


def command_action(args: argparse.Namespace, action: str) -> int:
    task = request_json(task_url(args.api, args.task_id) + f"/{action}", method="POST")
    print_json(task)
    return 0


def command_artifacts(args: argparse.Namespace) -> int:
    task = request_json(task_url(args.api, args.task_id))
    print_json(
        {
            "taskId": task["id"],
            "state": task["state"],
            "selectionMetric": task.get("selectionMetric"),
            "bestLoss": task.get("bestLoss"),
            "bestAdapterPath": task.get("bestAdapterPath"),
            "finalLoss": task.get("finalLoss"),
            "finalAdapterPath": task.get("finalAdapterPath"),
            "artifacts": task.get("artifacts", []),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训推平台微调任务命令行客户端")
    parser.add_argument("--api", default="http://127.0.0.1:8010/api")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="提交受控训练命令")
    submit.add_argument("--config", type=Path)
    submit.set_defaults(command_args=[])
    submit.set_defaults(handler=command_submit)

    status = subparsers.add_parser("status", help="查询任务状态")
    status.add_argument("task_id")
    status.set_defaults(handler=command_status)

    watch = subparsers.add_parser("watch", help="持续查看任务进度")
    watch.add_argument("task_id")
    watch.add_argument("--poll-seconds", type=float, default=2)
    watch.add_argument("--timeout-seconds", type=int, default=1800)
    watch.set_defaults(handler=command_watch)

    stop = subparsers.add_parser("stop", help="请求安全停止并保存最佳 Adapter")
    stop.add_argument("task_id")
    stop.set_defaults(handler=lambda args: command_action(args, "stop"))

    retry = subparsers.add_parser("retry", help="重新排队失败或已停止任务")
    retry.add_argument("task_id")
    retry.set_defaults(handler=lambda args: command_action(args, "retry"))

    artifacts = subparsers.add_parser("artifacts", help="查看 Adapter 和指标产物")
    artifacts.add_argument("task_id")
    artifacts.set_defaults(handler=command_artifacts)
    return parser


def main() -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if args.command == "submit":
        args.command_args = unknown
    elif unknown:
        parser.error(f"未识别参数：{' '.join(unknown)}")
    try:
        return int(args.handler(args))
    except (RuntimeError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
