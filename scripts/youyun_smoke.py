"""优云远程训练真实冒烟：python scripts/youyun_smoke.py

流程：创建一次性 4090 实例 → SSH 安装 LLaMA-Factory → 上传小数据集与 YAML →
Qwen2.5-0.5B + max_steps=6 远程训练 → 产物回传本地镜像 → 删除实例。
仅用于阶段五验收联调，产物落在 runtime/remote-jobs/smoke-<date>/。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.local_llamafactory_executor import LocalLlamaFactoryExecutor
from backend.providers.youyun.client import YouyunClient
from backend.providers.youyun.instance_manager import YouyunInstanceManager
from backend.providers.youyun.remote_training import YouyunTrainingRunner

TASK_ID = f"smoke-{time.strftime('%m%d%H%M')}"
WORK_ROOT = PROJECT_ROOT / "runtime" / "remote-jobs"


class PrintSink:
    def __init__(self) -> None:
        self.metric_count = 0

    def append_log(self, task_id: str, message: str) -> None:
        print(f"[log] {message}")

    def append_metric(self, task_id: str, **kw) -> None:
        self.metric_count += 1
        print(f"[metric] {kw}")

    def update(self, task_id: str, **kw) -> None:
        step = kw.get("current_step")
        if step:
            print(f"[step] {step}")

    def is_cancel_requested(self, task_id: str) -> bool:
        return False

    def metrics(self, task_id: str):
        return [{"step": i} for i in range(self.metric_count)]


def build_smoke_dataset(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    samples = [
        {"messages": [
            {"role": "user", "content": f"把这句话翻译成英文：测试样本 {i}。"},
            {"role": "assistant", "content": f"This is test sample {i}."},
        ]}
        for i in range(1, 13)
    ]
    train_path = directory / "train.jsonl"
    train_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8")
    return {"trainPath": str(train_path)}


def make_task(dataset_paths: dict[str, str]) -> dict:
    return {
        "taskId": TASK_ID,
        "id": TASK_ID,
        "name": "优云冒烟",
        "framework": "llamafactory",
        "provider": "youyun",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "modelSource": "modelscope",
        "taskType": "SFT",
        "tuningMode": "qlora",
        "datasetId": "smoke",
        "datasetName": "优云冒烟数据集",
        "outputName": "youyun-smoke-adapter",
        "mergeModel": False,
        "validationSplit": 0,
        "maxSteps": 6,
        "checkpointLimit": 1,
        "checkpointStrategy": "epoch",
        "checkpointInterval": 1,
        "earlyStopping": False,
        "patience": 3,
        "minDelta": 0,
        "extraParameters": {},
        "baselineMessages": [],
        "parameters": {
            "epochs": 1,
            "learningRate": 5e-5,
            "batchSize": 1,
            "gradientAccumulation": 2,
            "maxSequenceLength": 512,
            "loraRank": 8,
            "loraAlpha": 16,
            "loraDropout": 0.05,
            "loggingSteps": 1,
            "warmupRatio": 0,
            "weightDecay": 0,
            "quantizationBit": 4,
            "useCpu": False,
        },
        "dataset": {"paths": dataset_paths, "template": "qwen", "formatting": "sharegpt", "columns": {"messages": "messages"}},
    }


def main() -> int:
    dataset_paths = build_smoke_dataset(WORK_ROOT / TASK_ID / "source-dataset")
    mirror_executor = LocalLlamaFactoryExecutor(work_root=WORK_ROOT / TASK_ID)
    task = make_task(dataset_paths)

    client = YouyunClient()
    manager = YouyunInstanceManager(client)
    runner = YouyunTrainingRunner(
        manager=manager,
        sink=PrintSink(),
        mirror_root=WORK_ROOT / TASK_ID,
        prepare_local=mirror_executor.prepare,
        collect_local=mirror_executor.collect_artifacts,
    )
    import asyncio

    outcome = asyncio.run(runner.run(TASK_ID, task))
    print(f"\n=== 冒烟结果：{outcome} ===")
    artifacts_dir = WORK_ROOT / TASK_ID / TASK_ID / "output"
    print(f"产物目录：{artifacts_dir}（存在：{artifacts_dir.is_dir()}）")
    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
