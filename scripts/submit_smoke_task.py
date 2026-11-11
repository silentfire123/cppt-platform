from __future__ import annotations

import json
import urllib.request


API_URL = "http://127.0.0.1:8010/api/tasks"


def main() -> None:
    payload = {
        "name": "Qwen3-1.7B 本地链路冒烟测试（1 step）",
        "model": "Qwen/Qwen3-1.7B",
        "modelSource": "modelscope",
        "datasetId": "qwenruozhi_v1",
        "datasetName": "平台演示 SFT v1",
        "owner": "平台管理员",
        "mergeModel": False,
        "validationSplit": 0,
        "outputName": "qwen3-1.7b-smoke-lora",
        "checkpointLimit": 1,
        "checkpointStrategy": "steps",
        "checkpointInterval": 1,
        "maxSteps": 1,
        "parameters": {
            "epochs": 1,
            "learningRate": "2e-4",
            "batchSize": 1,
            "gradientAccumulation": 1,
            "validationSteps": 50,
            "maxSequenceLength": 512,
            "loraRank": 8,
            "loraAlpha": 16,
            "loraDropout": 0,
            "lrScheduler": "linear",
            "loraPlusRatio": 16,
            "targetModules": "auto",
            "optimizer": "adamw_torch",
            "warmupRatio": 0.1,
            "weightDecay": 0.01,
            "maxGradNorm": 1,
            "seed": 42,
            "loggingSteps": 1,
            "packing": False,
            "gradientCheckpointing": True,
        },
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        task = json.load(response)
    print(json.dumps(task, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
