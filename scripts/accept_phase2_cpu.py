#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"completed", "failed", "stopped"}


def request_json(base: str, path: str, *, method: str = "GET", body: Any = None, raw: bytes | None = None) -> Any:
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    headers: dict[str, str] = {}
    data = raw
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail[:500]}") from exc


def upload_smoke_dataset(api: str, source: Path) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "name": f"Phase 2 CPU acceptance {int(time.time())}",
            "structure": "SFT",
            "filename": source.name,
        }
    )
    dataset = request_json(api, f"datasets/upload?{query}", method="POST", raw=source.read_bytes())
    if dataset.get("state") == "failed":
        raise RuntimeError(f"smoke dataset validation failed: {dataset.get('validationMessage')}")
    return request_json(api, f"datasets/{dataset['id']}/publish", method="POST")


def create_smoke_task(api: str, dataset: dict[str, Any], model: str) -> dict[str, Any]:
    payload = {
        "name": "Phase 2 real CPU smoke (1 step)",
        "framework": "llamafactory",
        "executionMode": "local",
        "provider": "local",
        "model": model,
        "modelSource": "local" if Path(model).is_absolute() else "modelscope",
        "taskType": "SFT",
        "tuningMode": "lora",
        "datasetId": dataset["id"],
        "datasetName": dataset["name"],
        "owner": "phase2-acceptance",
        "mergeModel": False,
        "validationSplit": 0,
        "outputName": "qwen2.5-0.5b-cpu-smoke-lora",
        "checkpointLimit": 1,
        "checkpointStrategy": "steps",
        "checkpointInterval": 1,
        "maxSteps": 1,
        "earlyStopping": False,
        "patience": 3,
        "minDelta": 0,
        "extraParameters": {},
        "parameters": {
            "epochs": 1,
            "learningRate": "2e-4",
            "batchSize": 1,
            "gradientAccumulation": 1,
            "validationSteps": 1,
            "maxSequenceLength": 64,
            "loraRank": 4,
            "loraAlpha": 8,
            "loraDropout": 0,
            "lrScheduler": "linear",
            "loraPlusRatio": 16,
            "targetModules": "attention",
            "optimizer": "adamw_torch",
            "warmupRatio": 0,
            "weightDecay": 0,
            "maxGradNorm": 1,
            "seed": 42,
            "loggingSteps": 1,
            "packing": False,
            "gradientCheckpointing": False,
        },
        "baselineMessages": [],
        "chatParameters": {"maxNewTokens": 64, "temperature": 0.2, "topP": 0.9, "seed": 42},
    }
    return request_json(api, "tasks", method="POST", body=payload)


def wait_for_terminal(api: str, task_id: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        task = request_json(api, f"tasks/{task_id}")
        state = task.get("state")
        if state != last_state:
            print(f"task {task_id}: {state} - {task.get('currentStep', '')}", flush=True)
            last_state = state
        if state in TERMINAL_STATES:
            return task
        time.sleep(2)
    raise TimeoutError(f"task {task_id} did not finish within {timeout} seconds")


def verify_websocket(api: str, task_id: str) -> dict[str, int]:
    from websockets.sync.client import connect

    parsed = urllib.parse.urlsplit(api)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/tasks/{task_id}/stream"
    uri = urllib.parse.urlunsplit((scheme, parsed.netloc, path, "", ""))
    counts = {"log": 0, "state": 0}
    terminal_seen = False
    ssl_context = ssl.create_default_context() if scheme == "wss" else None
    with connect(uri, ssl=ssl_context, open_timeout=10, close_timeout=5) as websocket:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                event = json.loads(websocket.recv(timeout=5))
            except TimeoutError:
                continue
            event_type = str(event.get("type"))
            if event_type in counts:
                counts[event_type] += 1
            if event_type == "state" and event.get("state") in TERMINAL_STATES:
                terminal_seen = True
                break
    if not terminal_seen or counts["log"] == 0:
        raise AssertionError(f"WebSocket acceptance failed: {counts}, terminal={terminal_seen}")
    return counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(task: dict[str, Any]) -> dict[str, Any]:
    artifacts = list(task.get("artifacts") or [])
    adapters = [item for item in artifacts if item.get("type") == "adapter"]
    manifests = [item for item in artifacts if item.get("type") == "manifest"]
    if not adapters or not manifests:
        raise AssertionError("task is missing Adapter or manifest artifacts")
    if any(item.get("variant") == "best" for item in adapters):
        raise AssertionError("validationSplit=0 must not fabricate a Best Adapter")
    final = next((item for item in adapters if item.get("variant") == "final"), None)
    if final is None or task.get("selectedAdapterVariant") != "final":
        raise AssertionError("Final Adapter was not generated and selected")

    manifest_artifact = manifests[-1]
    manifest_path = Path(str(manifest_artifact["path"])).resolve()
    if not manifest_path.is_file():
        raise AssertionError(f"manifest does not exist: {manifest_path}")
    manifest_digest = sha256(manifest_path)
    if manifest_digest != manifest_artifact.get("sha256"):
        raise AssertionError("manifest artifact SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("taskId") != task.get("id"):
        raise AssertionError("manifest taskId does not match task")
    for entry in manifest.get("files", []):
        file_path = manifest_path.parent / entry["relativePath"]
        if not file_path.is_file() or sha256(file_path) != entry["sha256"]:
            raise AssertionError(f"manifest entry mismatch: {entry['relativePath']}")

    train_results = manifest_path.parent / "output" / "train_results.json"
    if not train_results.is_file():
        raise AssertionError("real Hugging Face train_results.json is missing; a command test double is not accepted")
    results = json.loads(train_results.read_text(encoding="utf-8"))
    if not any(key in results for key in ("train_runtime", "train_loss", "train_samples_per_second")):
        raise AssertionError("train_results.json does not contain real trainer metrics")
    return {
        "manifest": str(manifest_path),
        "manifestSha256": manifest_digest,
        "manifestFiles": len(manifest.get("files", [])),
        "finalAdapter": str(final.get("path")),
    }


def verify_postgres_and_minio(task_id: str, *, skip_minio: bool) -> dict[str, Any]:
    import psycopg
    from psycopg.rows import dict_row

    dsn = os.environ.get("POSTGRES_DSN", "").strip()
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is required for production acceptance")
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        task = connection.execute(
            "SELECT state, artifacts FROM training_tasks WHERE id=%s", (task_id,)
        ).fetchone()
        if not task or task["state"] != "completed":
            raise AssertionError("PostgreSQL training_tasks does not contain the completed task")
        counts = {}
        for table in ("training_task_logs", "training_task_metrics", "training_task_events"):
            counts[table] = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE task_id=%s", (task_id,)
            ).fetchone()["count"]
            if counts[table] <= 0:
                raise AssertionError(f"PostgreSQL {table} has no rows for the task")

        deadline = time.monotonic() + 180
        record = None
        while time.monotonic() < deadline:
            record = connection.execute(
                """
                SELECT * FROM model_files
                WHERE source_task_id=%s AND artifact_variant='final'
                ORDER BY created_at DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if record and record["status"] == "available":
                break
            time.sleep(3)
        if not record or record["status"] != "available":
            status = record["status"] if record else "missing"
            error = record.get("error_message") if record else None
            raise AssertionError(f"Final Adapter shared-storage record is not available: {status}, {error or ''}")
        entry_count = connection.execute(
            "SELECT COUNT(*) AS count FROM model_file_entries WHERE model_file_id=%s AND status='available'",
            (record["id"],),
        ).fetchone()["count"]
        if entry_count <= 0:
            raise AssertionError("model_file_entries has no available files")
        expected_prefix = f"model/{record['id']}"
        if record["minio_object_prefix"] != expected_prefix:
            raise AssertionError(
                f"model_files MinIO prefix mismatch: {record['minio_object_prefix']!r} != {expected_prefix!r}"
            )

    result = {
        "postgresRows": counts,
        "modelFileId": record["id"],
        "bucket": record["minio_bucket"],
        "objectPrefix": record["minio_object_prefix"],
        "availableEntries": entry_count,
    }
    if skip_minio:
        result["minio"] = "skipped"
        return result

    from minio import Minio

    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required")
    endpoint = endpoint.removeprefix("http://").removeprefix("https://").rstrip("/")
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    objects = list(client.list_objects(record["minio_bucket"], prefix=f"{record['minio_object_prefix']}/", recursive=True))
    if len(objects) != entry_count or sum(item.size or 0 for item in objects) <= 0:
        raise AssertionError(f"MinIO object count mismatch: objects={len(objects)}, entries={entry_count}")
    result["minioObjects"] = len(objects)
    result["minioBytes"] = sum(item.size or 0 for item in objects)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-two real CPU and production persistence acceptance.")
    parser.add_argument("--api", default="http://127.0.0.1:8010/api")
    parser.add_argument("--task-id")
    parser.add_argument("--run-real-smoke", action="store_true", help="Explicitly create a dataset and a max_steps=1 task.")
    parser.add_argument("--model", default=os.environ.get("XUNTUI_CPU_SMOKE_MODEL", ""))
    parser.add_argument("--dataset", type=Path, default=Path("data/qwenruozhi_v1/smoke-test.jsonl"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-minio", action="store_true")
    args = parser.parse_args()

    if bool(args.task_id) == bool(args.run_real_smoke):
        parser.error("choose exactly one of --task-id or --run-real-smoke")
    health = request_json(args.api, "health")
    if health.get("status") != "ok" or health.get("storageBackend") != "postgres":
        raise AssertionError(f"production health preflight failed: {health}")

    if args.run_real_smoke:
        if not args.model:
            parser.error("--model or XUNTUI_CPU_SMOKE_MODEL is required for real smoke")
        if Path(args.model).is_absolute() and not Path(args.model).exists():
            parser.error(f"model path does not exist inside the backend runtime: {args.model}")
        dataset = upload_smoke_dataset(args.api, args.dataset.resolve())
        created = create_smoke_task(args.api, dataset, args.model)
        task_id = str(created["id"])
        print(f"created real CPU acceptance task: {task_id}", flush=True)
    else:
        task_id = str(args.task_id)

    task = wait_for_terminal(args.api, task_id, args.timeout)
    if task.get("state") != "completed":
        raise AssertionError(f"real CPU task did not complete: {task.get('state')} {task.get('error') or ''}")
    if task.get("maxSteps") != 1:
        raise AssertionError("accepted task is not max_steps=1")

    logs = request_json(args.api, f"tasks/{task_id}/logs?limit=5000").get("logs", [])
    metrics = request_json(args.api, f"tasks/{task_id}/metrics").get("metrics", [])
    if not logs or not metrics:
        raise AssertionError("REST logs or metrics are empty")
    websocket = verify_websocket(args.api, task_id)
    artifact_result = verify_artifacts(task)
    persistence_result = verify_postgres_and_minio(task_id, skip_minio=args.skip_minio)

    summary = {
        "accepted": True,
        "taskId": task_id,
        "state": task["state"],
        "restLogs": len(logs),
        "metrics": len(metrics),
        "websocket": websocket,
        "artifacts": artifact_result,
        "persistence": persistence_result,
        "note": "This result is valid only when run against the real llamafactory-cli CPU image, not the command-level smoke double.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
