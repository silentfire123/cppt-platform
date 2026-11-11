"""Convert HuggingFace or local JSONL datasets into the platform's dataset formats.

The platform gates uploads by structure (backend/app.py):

- SFT  ``{"messages": [{role, content}, ...]}`` - optional leading system, starts
  with user, ends with assistant, strict alternation (``_validate_sft_record``).
- DPO  ``{"prompt": str, "chosen": str, "rejected": str}`` - the string contract
  of ``_validate_dpo_record``.
- Evaluation ``{"prompt": str, "reference": str}`` - accepted by
  ``normalize_evaluation_record`` (prompt/question + reference/completion).

Examples::

    python scripts/convert_hf_dataset.py dpo --source llamafactory/dpo_zh_demo \
        --limit 400 --seed 42 --output test-data/practice-dpo.jsonl
    python scripts/convert_hf_dataset.py sft --source llamafactory/alpaca_gpt4_zh \
        --limit 400 --system "你是一个乐于助人的助手。" --output test-data/practice-sft.jsonl
    python scripts/convert_hf_dataset.py evaluation --source test-data/practice-dpo.jsonl \
        --limit 40 --output test-data/practice-eval.jsonl

``--source`` accepts a HuggingFace dataset id (requires the ``datasets`` package
and network access) or a local ``.json``/``.jsonl`` file. Record fields are
auto-detected (alpaca instruction/input/output, prompt/chosen/rejected,
messages conversations, question/answer, query/response).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator


def _as_text(value: Any) -> str:
    """Flatten a string or a message list into plain text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                role = str(item.get("role") or item.get("from") or "").strip()
                content = str(item.get("content") or item.get("value") or "").strip()
                parts.append(f"{role}: {content}" if role and role not in {"user", "assistant"} else content)
        return "\n".join(p for p in parts if p)
    return ""


def _last_assistant(value: Any) -> str:
    """Extract the response text: the last assistant message of a list, or the string itself."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") in ("assistant", "gpt"):
                text = str(item.get("content") or item.get("value") or "").strip()
                if text:
                    return text
        return _as_text(value)
    return ""


def _first_user(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict) and item.get("role") in ("user", "human"):
                text = str(item.get("content") or item.get("value") or "").strip()
                if text:
                    return text
    return ""


def convert_record(record: dict[str, Any], fmt: str, system: str) -> dict[str, Any] | None:
    """Map one raw record into the target platform format, or None if unconvertible."""
    if fmt == "dpo":
        prompt = ""
        for key in ("prompt", "instruction", "question", "query", "conversations"):
            if record.get(key):
                prompt = _as_text(record[key]) if key != "conversations" else _first_user(record)
                if prompt:
                    break
        prompt = prompt or _first_user(record)
        chosen = _last_assistant(record["chosen"]) if record.get("chosen") else ""
        rejected = _last_assistant(record["rejected"]) if record.get("rejected") else ""
        if not (prompt and chosen and rejected) or chosen == rejected:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    if fmt == "sft":
        user = ""
        for key in ("instruction", "prompt", "question", "query"):
            if record.get(key):
                user = str(record[key]).strip()
                if key in ("instruction", "query") and record.get("input"):
                    user = f"{user}\n\n{str(record['input']).strip()}"
                break
        assistant = ""
        for key in ("output", "response", "answer", "chosen"):
            if record.get(key):
                assistant = _last_assistant(record[key])
                break
        if not user and record.get("messages"):
            user = _first_user(record)
            assistant = _last_assistant(record["messages"])
        if not (user and assistant):
            return None
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
        return {"messages": messages}

    if fmt == "evaluation":
        prompt = ""
        for key in ("prompt", "question", "instruction", "query"):
            if record.get(key):
                prompt = str(record[key]).strip()
                break
        reference = ""
        for key in ("reference", "completion", "answer", "output", "chosen"):
            if record.get(key):
                reference = _last_assistant(record[key])
                break
        if not prompt and record.get("messages"):
            prompt = _first_user(record)
        if not reference and record.get("messages"):
            reference = _last_assistant(record["messages"])
        if not (prompt and reference):
            return None
        out: dict[str, Any] = {"prompt": prompt, "reference": reference}
        if system:
            out["system"] = system
        return out

    raise ValueError(f"unknown format: {fmt}")


def iter_local(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        first = source.read(1)
        source.seek(0)
        if first == "[":
            for record in json.load(source):
                if isinstance(record, dict):
                    yield record
            return
        for line in source:
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict):
                    yield record


def iter_hf(dataset_id: str, split: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The `datasets` package is required for HuggingFace sources. "
            "Install it with `pip install datasets`, or download the dataset "
            "manually and pass a local .jsonl file via --source."
        ) from exc
    dataset = load_dataset(dataset_id, split=split) if split else load_dataset(dataset_id)
    if not split:
        dataset = next(iter(dataset.values()))
    for record in dataset:
        yield dict(record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("format", choices=("dpo", "sft", "evaluation"))
    parser.add_argument("--source", required=True, help="HuggingFace dataset id or local .json/.jsonl path")
    parser.add_argument("--split", default="train", help="HuggingFace split (default: train)")
    parser.add_argument("--limit", type=int, default=0, help="max records to write (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed before sampling --limit")
    parser.add_argument("--system", default="", help="optional system prompt (sft/evaluation)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = Path(args.source)
    records = iter_local(source) if source.exists() else iter_hf(args.source, args.split)

    rng = random.Random(args.seed)
    pool: list[dict[str, Any]] = []
    read_count = 0
    for record in records:
        read_count += 1
        converted = convert_record(record, args.format, args.system)
        if converted is not None:
            pool.append(converted)
        if args.limit and len(pool) >= args.limit * 3:
            break  # keep a conversion surplus so sampling after dedup still fills the limit

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in pool:
        if args.format == "sft":
            dedup_source = next(m["content"] for m in item["messages"] if m["role"] == "user")
        else:
            dedup_source = item["prompt"]
        digest = hashlib.sha256(dedup_source.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(item)

    if args.limit:
        rng.shuffle(unique)
        unique = unique[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for item in unique:
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")

    lengths = [
        len(item["prompt"] if args.format != "sft" else item["messages"][-1]["content"])
        for item in unique
    ]
    print(f"source            : {args.source}")
    print(f"format            : {args.format}")
    print(f"records read      : {read_count}")
    print(f"records written   : {len(unique)} (skipped/unconvertible: {read_count - len(unique)})")
    if lengths:
        print(f"avg target length : {sum(lengths) // len(lengths)} chars")
    print(f"output            : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
