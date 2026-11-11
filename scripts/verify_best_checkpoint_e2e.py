from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    timeout: float = 300,
) -> Any:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


def post_json(url: str, payload: dict[str, Any], *, timeout: float = 300) -> Any:
    return request_json(
        url,
        method="POST",
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8010/api")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("test-data/best-checkpoint-e2e.jsonl"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    query = urllib.parse.urlencode(
        {
            "name": f"最佳检查点 E2E {time.strftime('%y%m%d-%H%M%S')}",
            "structure": "SFT",
            "filename": dataset_path.name,
        }
    )
    uploaded = request_json(
        f"{args.api}/datasets/upload?{query}",
        method="POST",
        body=dataset_path.read_bytes(),
    )
    if uploaded["state"] != "ready":
        raise RuntimeError(f"dataset validation failed: {uploaded}")
    published = post_json(f"{args.api}/datasets/{uploaded['id']}/publish", {})
    if published["state"] != "published":
        raise RuntimeError(f"dataset publish failed: {published}")

    task = post_json(
        f"{args.api}/tasks",
        {
            "name": f"最佳 Adapter E2E {time.strftime('%y%m%d-%H%M%S')}",
            "model": args.model,
            "modelSource": "modelscope",
            "tuningMode": "qlora",
            "datasetId": uploaded["id"],
            "datasetName": uploaded["name"],
            "owner": "平台管理员",
            "mergeModel": False,
            "validationSplit": 25,
            "outputName": "best-checkpoint-e2e-qlora",
            "checkpointLimit": 1,
            "checkpointStrategy": "steps",
            "checkpointInterval": 1,
            "maxSteps": 3,
            "parameters": {
                "epochs": 1,
                "learningRate": "2e-4",
                "batchSize": 1,
                "gradientAccumulation": 1,
                "validationSteps": 1,
                "maxSequenceLength": 256,
                "loraRank": 8,
                "loraAlpha": 16,
                "loraDropout": 0,
                "lrScheduler": "linear",
                "loraPlusRatio": 16,
                "targetModules": "attention",
                "optimizer": "adamw_torch",
                "warmupRatio": 0,
                "weightDecay": 0.01,
                "maxGradNorm": 1,
                "seed": 42,
                "loggingSteps": 1,
                "packing": False,
                "gradientCheckpointing": True,
            },
            "chatParameters": {
                "maxNewTokens": 48,
                "temperature": 0.2,
                "topP": 0.9,
                "seed": 42,
            },
        },
        timeout=30,
    )

    deadline = time.monotonic() + args.timeout_seconds
    last_state = ""
    while time.monotonic() < deadline:
        task = request_json(f"{args.api}/tasks/{task['id']}", timeout=30)
        marker = f"{task['state']} {task['progress']}% {task['currentStep']}"
        if marker != last_state:
            print(marker, flush=True)
            last_state = marker
        if task["state"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError(f"training task {task['id']} did not finish")

    if task["state"] != "completed":
        raise RuntimeError(
            json.dumps(
                {"taskId": task["id"], "state": task["state"], "error": task["error"]},
                ensure_ascii=False,
            )
        )
    if task["selectedAdapterVariant"] != "best":
        raise RuntimeError(f"best adapter was not selected: {task}")
    if not task["bestAdapterPath"] or not task["finalAdapterPath"]:
        raise RuntimeError(f"best/final adapters were not both retained: {task}")

    required_metrics = ("bestStep", "bestEvalLoss", "finalEvalLoss", "finalStep")
    missing_metrics = [name for name in required_metrics if task.get(name) is None]
    if missing_metrics:
        raise RuntimeError(f"best/final checkpoint metrics are missing: {missing_metrics}: {task}")
    for variant, path_value in (("best", task["bestAdapterPath"]), ("final", task["finalAdapterPath"])):
        adapter_path = Path(path_value)
        if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
            raise RuntimeError(f"{variant} adapter directory is incomplete: {path_value}")

    idle_deadline = time.monotonic() + 60
    while time.monotonic() < idle_deadline:
        health = request_json(f"{args.api}/health", timeout=30)
        if health.get("worker") == "idle":
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("training task completed but GPU worker did not become idle")
    chat = post_json(
        f"{args.api}/chat",
        {
            "mode": "adapter",
            "adapterVariant": "selected",
            "taskId": task["id"],
            "model": task["model"],
            "modelSource": "modelscope",
            "messages": [
                {
                    "role": "user",
                    "content": "聊天默认加载哪个 Adapter？请简短回答。",
                }
            ],
            "parameters": {
                "maxNewTokens": 48,
                "temperature": 0.2,
                "topP": 0.9,
                "seed": 42,
            },
        },
        timeout=300,
    )
    if chat.get("adapterVariant") != "best":
        raise RuntimeError(f"chat did not load best adapter: {chat}")
    if chat.get("adapterStep") != task["bestStep"]:
        raise RuntimeError(f"chat loaded the wrong adapter step: {chat}")
    if Path(chat.get("adapterPath", "")).resolve() != Path(task["bestAdapterPath"]).resolve():
        raise RuntimeError(f"chat loaded the wrong adapter path: {chat}")

    print(
        json.dumps(
            {
                "datasetId": uploaded["id"],
                "datasetState": published["state"],
                "taskId": task["id"],
                "taskState": task["state"],
                "bestStep": task["bestStep"],
                "bestEvalLoss": task["bestEvalLoss"],
                "finalEvalLoss": task["finalEvalLoss"],
                "finalStep": task["finalStep"],
                "selectedAdapterVariant": task["selectedAdapterVariant"],
                "bestAdapterPath": task["bestAdapterPath"],
                "finalAdapterPath": task["finalAdapterPath"],
                "chatAdapterVariant": chat["adapterVariant"],
                "chatAdapterStep": chat["adapterStep"],
                "chatText": chat["text"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
