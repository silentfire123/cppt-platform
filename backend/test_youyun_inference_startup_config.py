from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from backend.test_remote_log_follow import MemoryLogSSH
from unittest.mock import AsyncMock

from backend.inference_service import InferenceCreate, InferenceRuntime, InferenceStore
from backend.providers.youyun.inference import YouyunInferenceProvider


class _TaskStore:
    def __init__(self, task: dict) -> None:
        self.task = task

    def get(self, task_id: str) -> dict | None:
        return self.task if task_id == self.task["id"] else None


class _Manager:
    def __init__(self) -> None:
        self.client = SimpleNamespace()

    def create_one_shot(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(uhost_id="uhost-config", ip="10.0.0.2")

    def cleanup_after_task(
        self, instance_id: str, stop_first: bool = True
    ) -> dict[str, object]:
        return {"terminated": True, "error": None}


class _SSH:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def mkdirs(self, path: str) -> None:
        return None

    def run(self, command: str, timeout: int = 120, check: bool = True):
        self.commands.append(command)
        return 0, ""

    def close(self) -> None:
        return None


class YouyunStartupConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_v100s_auto_dtype_is_resolved_to_float16_before_launch(self) -> None:
        config = {
            "maxNumSeqs": 4,
            "maxNumBatchedTokens": 8192,
            "gpuMemoryUtilization": 0.75,
            "dtype": "auto",
            "tensorParallelSize": 1,
            "quantization": "none",
            "enablePrefixCaching": False,
            "enableChunkedPrefill": False,
            "kvCacheDtype": "auto",
            "kvCacheMode": "auto",
        }
        args = InferenceRuntime._remote_managed_vllm_args(
            config,
            {"gpuType": "V100S"},
        )
        self.assertEqual(args[args.index("--dtype") + 1], "float16")

    async def test_saved_managed_settings_reach_remote_vllm_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            task = {
                "id": "ft-9527",
                "state": "completed",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "baseModelMetadata": {"maxContextLength": 4096},
                "artifacts": [
                    {
                        "type": "adapter",
                        "variant": "best",
                        "selected": True,
                        "path": str(adapter),
                    }
                ],
            }
            ssh = _SSH()
            provider = YouyunInferenceProvider(
                _Manager(),  # type: ignore[arg-type]
                poll_interval=0,
                ssh_factory=MemoryLogSSH,
            )
            provider._connect = lambda _instance: ssh  # type: ignore[method-assign]
            provider._diagnose_remote_runtime = AsyncMock(return_value="")  # type: ignore[method-assign]
            provider._prepare_vllm_runtime_async = AsyncMock(return_value="python3")  # type: ignore[method-assign]
            provider._prepare_remote_model = AsyncMock(return_value="/models/base")  # type: ignore[method-assign]
            provider._upload_tree = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            provider.process_alive = AsyncMock(return_value=True)  # type: ignore[method-assign]
            provider.health = AsyncMock(return_value=True)  # type: ignore[method-assign]

            runtime = InferenceRuntime(
                InferenceStore(root / "inference.db"),
                _TaskStore(task),  # type: ignore[arg-type]
                port=8011,
                training_busy=lambda: False,
                remote_provider_factory=lambda: provider,
            )
            runtime.refresh_metrics = AsyncMock(return_value={})  # type: ignore[method-assign]
            service = await runtime.create(
                InferenceCreate(
                    name="9527",
                    taskId=task["id"],
                    provider="youyun",
                    youyunResource={
                        "mode": "manual",
                        "gpuType": "3080Ti",
                        "gpuCount": 1,
                        "cpuCores": 16,
                        "memoryGb": 64,
                    },
                    maxModelLength=4096,
                    maxNumSeqs=4,
                    maxNumBatchedTokens=8192,
                    gpuMemoryUtilization=0.75,
                    dtype="auto",
                    tensorParallelSize=1,
                    enablePrefixCaching=True,
                    enableChunkedPrefill=True,
                )
            )

            started = await runtime.start(service["id"])
            try:
                launch = next(command for command in ssh.commands if "nohup" in command)
                self.assertIn("--max-num-seqs 4", launch)
                self.assertIn("--max-num-batched-tokens 8192", launch)
                self.assertIn("--gpu-memory-utilization 0.75", launch)
                self.assertIn("--dtype auto", launch)
                self.assertIn("--tensor-parallel-size 1", launch)
                self.assertIn("--enable-prefix-caching", launch)
                self.assertIn("--enable-chunked-prefill", launch)
                current = runtime.store.get(service["id"])
                assert current is not None
                self.assertTrue(
                    any("小显存" in line for line in runtime.store.logs(service["id"]))
                )
                self.assertIn("stageElapsedSeconds", current["config"]["startup"])
            finally:
                await runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
