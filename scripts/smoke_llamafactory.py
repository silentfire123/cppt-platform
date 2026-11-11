from __future__ import annotations

"""Run the phase-two local smoke path without requiring a model download.

Use ``--real`` with an installed LLaMA-Factory command to run the same executor
against a real tiny model/data set. The default fake command validates config,
argv safety, event handling, artifacts and manifest generation.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.local_llamafactory_executor import LocalLlamaFactoryExecutor


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--command", default="llamafactory-cli")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="xuntui-llamafactory-smoke-"))
    train = root / "train.jsonl"
    source = PROJECT_ROOT / "data" / "qwenruozhi_v1" / "smoke-test.jsonl"
    train.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    command = args.command if args.real else [sys.executable, str(Path(__file__).resolve().parent / "llamafactory_smoke_cli.py")]
    executor = LocalLlamaFactoryExecutor(work_root=root / "work", command=command)
    task = {
        "id": "ft-cli-smoke",
        "model": "Qwen/Qwen2.5-0.5B-Instruct" if args.real else "tiny-model",
        "datasetId": "qwenruozhi_smoke",
        "taskType": "SFT",
        "tuningMode": "lora",
        "outputName": "llamafactory-smoke",
        "parameters": {"epochs": 1, "batchSize": 1, "maxSequenceLength": 128, "useCpu": True, "maxSteps": 1},
        "dataset": {"paths": {"trainPath": str(train)}},
    }
    await executor.start(task)
    async for event in executor.events(task):
        print(json.dumps({"type": event.type, "message": event.message, **event.payload}, ensure_ascii=False), flush=True)
    result = await executor.status(task)
    if result.get("returncode", 0) != 0:
        raise SystemExit(f"smoke failed: {result}")
    print(json.dumps({"artifacts": await executor.collect_artifacts(task), "workdir": str(root / "work")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
