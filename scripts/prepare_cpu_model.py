#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the fixed tiny CPU acceptance model from ModelScope.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="/workspace/model-cache/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    from modelscope import snapshot_download

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(args.model, local_dir=str(output))
    print(f"model ready: {Path(resolved).resolve()}")


if __name__ == "__main__":
    main()
