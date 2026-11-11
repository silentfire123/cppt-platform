from __future__ import annotations

"""Tiny command-compatible test double for the local LLaMA-Factory smoke path.

It intentionally mimics only the process contract (``train <yaml>``), not
model training. Production uses the real ``llamafactory-cli`` executable.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "train":
        print("usage: llamafactory_smoke_cli.py train CONFIG", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[2]).resolve()
    config = config_path.read_text(encoding="utf-8")
    output_dir = None
    for line in config.splitlines():
        if line.startswith("output_dir:"):
            output_dir = line.split(":", 1)[1].strip().strip('"')
            break
    if not output_dir:
        print("missing output_dir", file=sys.stderr)
        return 3
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint-1"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2}), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"smoke-adapter")
    (output / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2}), encoding="utf-8"
    )
    (output / "adapter_model.safetensors").write_bytes(b"smoke-final-adapter")
    (output / "trainer_state.json").write_text(
        json.dumps(
            {
                "best_model_checkpoint": str(checkpoint),
                "log_history": [
                    {"loss": 1.5, "learning_rate": 0.0002, "epoch": 0.5, "step": 1},
                    {"eval_loss": 1.1, "epoch": 1.0, "step": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    print("__XUNTUI_EVENT__" + json.dumps({"type": "status", "message": "smoke model loaded"}), flush=True)
    print("__XUNTUI_EVENT__" + json.dumps({"type": "metric", "step": 1, "loss": 1.5, "learning_rate": 0.0002, "progress": 90}), flush=True)
    print("__XUNTUI_EVENT__" + json.dumps({"type": "completed", "message": "smoke training complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
