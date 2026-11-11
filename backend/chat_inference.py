from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


EVENT_PREFIX = "__XUNTUI_CHAT__"


def emit(event_type: str, **payload: Any) -> None:
    print(
        EVENT_PREFIX
        + json.dumps({"type": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def resolve_model(model_id: str, source: str) -> str:
    if source != "modelscope":
        return model_id
    cache_dir = os.environ.get("XUNTUI_MODELSCOPE_CACHE") or os.environ.get("MODELSCOPE_CACHE", "/root/.cache/modelscope")
    try:
        from modelscope import snapshot_download
        return snapshot_download(model_id, cache_dir=cache_dir)
    except Exception as modelscope_error:
        print("ModelScope download failed; trying the same repository ID on Hugging Face.", file=sys.stderr, flush=True)
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            return hf_snapshot_download(repo_id=model_id)
        except Exception as hf_error:
            raise RuntimeError(f"ModelScope and Hugging Face downloads failed for {model_id} ({type(modelscope_error).__name__}, {type(hf_error).__name__}).") from hf_error


def clean_response(value: str) -> str:
    without_thinking = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.S)
    return without_thinking.strip()


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older Transformers/tokenizers do not expose enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _load_model_with_transformers(
    model_id: str,
    source: str,
    adapter_path: str | None,
    torch: Any,
) -> tuple[Any, Any, Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = resolve_model(model_id, source)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device)
    model.eval()
    return model, tokenizer, device, "transformers"


def _load_chat_model(
    model_id: str,
    source: str,
    adapter_path: str | None,
    load_in_4bit: bool,
    torch: Any,
) -> tuple[Any, Any, Any, str]:
    # Unsloth remains the preferred path for the GPU environment. The CPU
    # deployment intentionally does not install it, so comparison/chat must
    # fall back to the standard Transformers + PEFT loader instead of failing
    # with ``No module named 'unsloth'``.
    if torch.cuda.is_available():
        try:
            from unsloth import FastLanguageModel

            model_path = adapter_path or resolve_model(model_id, source)
            model_dtype = (
                None
                if load_in_4bit
                else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=model_dtype,
                load_in_4bit=load_in_4bit,
                trust_remote_code=False,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer, torch.device("cuda"), "unsloth"
        except ImportError:
            pass
    return _load_model_with_transformers(model_id, source, adapter_path, torch)


def run(
    model_id: str,
    source: str,
    adapter_path: str | None,
    load_in_4bit: bool,
) -> None:
    import torch

    model, tokenizer, device, loader = _load_chat_model(
        model_id, source, adapter_path, load_in_4bit, torch
    )
    emit("ready", mode="adapter" if adapter_path else "base", loader=loader, device=str(device))

    for raw_line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw_line)
            request_id = str(request["requestId"])
            messages = request["messages"]
            parameters = request["parameters"]
            adapter_enabled = bool(request.get("adapterEnabled", adapter_path is not None))
            prompt = _apply_chat_template(tokenizer, messages)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device)
            seed = int(parameters["seed"])
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            temperature = float(parameters["temperature"])
            generate_options: dict[str, Any] = {
                "max_new_tokens": int(parameters["maxNewTokens"]),
                "do_sample": temperature > 0,
                "use_cache": True,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }
            if temperature > 0:
                generate_options.update(
                    temperature=temperature,
                    top_p=float(parameters["topP"]),
                )

            started = time.perf_counter()
            adapter_context = (
                nullcontext()
                if adapter_path is None or adapter_enabled
                else model.disable_adapter()
            )
            with adapter_context:
                with torch.inference_mode():
                    output = model.generate(**inputs, **generate_options)
            elapsed = time.perf_counter() - started
            generated = output[0, inputs["input_ids"].shape[-1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            emit(
                "response",
                requestId=request_id,
                text=clean_response(text),
                elapsedSeconds=round(elapsed, 3),
                outputTokens=int(generated.shape[-1]),
            )
        except Exception as exc:
            emit(
                "error",
                requestId=request.get("requestId"),
                message=str(exc),
                traceback=traceback.format_exc(),
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="modelscope")
    parser.add_argument("--adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    try:
        run(args.model, args.source, args.adapter, args.load_in_4bit)
        return 0
    except Exception as exc:
        emit("fatal", message=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
