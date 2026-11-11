from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.test_remote_log_follow import MemoryLogSSH

from backend.inference_service import InferenceCreate, InferenceRuntime, InferenceStore
from backend.providers.youyun.inference import (
    YouyunInferenceHandle,
    YouyunInferenceProvider,
    YouyunInferenceStartError,
    _clean_remote_log_line,
)
from backend.providers.youyun.models import YouyunConfig


REALISTIC_VLLM_OOM_LOG = """\
\x1b[0;36m(EngineCore_DP0 pid=440)\x1b[0;0m ERROR 08-28 01:48:04 [core.py:843] EngineCore failed to start.
\x1b[0;36m(EngineCore_DP0 pid=440)\x1b[0;0m torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 446.00 MiB. GPU 0 has a total capacity of 11.63 GiB of which 313.50 MiB is free.
\x1b[0;36m(EngineCore_DP0 pid=440)\x1b[0;0m RuntimeError: CUDA out of memory occurred when warming up sampler with 256 dummy requests. Please try lowering `max_num_seqs` or `gpu_memory_utilization` when initializing the engine.
\x1b[0;36m(APIServer pid=382)\x1b[0;0m RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


class _Client:
    def __init__(self, instance: object) -> None:
        self.instance = instance

    def describe_instance(self, instance_id: str) -> object:
        return self.instance


class _Manager:
    def __init__(self) -> None:
        self.client = _Client(object())
        self.cleaned: list[str] = []
        self.cleanup_error: str | None = None

    def ssh_credentials(self, instance: object) -> dict[str, object]:
        return {"host": "1.2.3.4", "port": 22, "user": "root", "password": "pass"}

    def cleanup_after_task(self, instance_id: str, stop_first: bool = True) -> dict[str, object]:
        self.cleaned.append(instance_id)
        return {"terminated": self.cleanup_error is None, "error": self.cleanup_error}


class _FailingSSH:
    def connect(self, host: str, port: int, user: str, password: str) -> None:
        return None

    def run(self, command: str, timeout: int = 120, check: bool = True):
        raise OSError("SSH connection lost")

    def close(self) -> None:
        return None


class ProviderCleanupTests(unittest.IsolatedAsyncioTestCase):
    def test_adapter_upload_reports_aggregate_progress(self) -> None:
        class SSH:
            def mkdirs(self, path: str) -> None:
                return None

            def put_file(self, local_path: Path, remote_path: str, callback=None) -> None:
                size = Path(local_path).stat().st_size
                if callback is not None:
                    callback(size // 2, size)
                    callback(size, size)

        progress: list[tuple[int, int, str]] = []
        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"a" * 10)
            (adapter / "adapter_config.json").write_bytes(b"b" * 2)

            provider._upload_tree(
                SSH(),  # type: ignore[arg-type]
                adapter,
                "/remote/adapter",
                on_progress=lambda transferred, total, path: progress.append(
                    (transferred, total, path)
                ),
            )

        self.assertTrue(progress)
        self.assertEqual(progress[-1][:2], (12, 12))
        self.assertEqual(
            {item[2] for item in progress},
            {"adapter_config.json", "adapter_model.safetensors"},
        )
        self.assertEqual([item[0] for item in progress], sorted(item[0] for item in progress))

    async def test_start_emits_adapter_upload_progress_stage(self) -> None:
        class Manager(_Manager):
            def create_one_shot(self, *_args: object, **_kwargs: object) -> object:
                return SimpleNamespace(uhost_id="uhost-progress", ip="10.0.0.2")

        class SSH:
            def mkdirs(self, path: str) -> None:
                return None

            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 0, ""

        provider = YouyunInferenceProvider(Manager(), poll_interval=0, ssh_factory=MemoryLogSSH)  # type: ignore[arg-type]
        provider._connect = lambda _instance: SSH()  # type: ignore[method-assign]
        provider._diagnose_remote_runtime = AsyncMock(return_value="")  # type: ignore[method-assign]
        provider._prepare_vllm_runtime_async = AsyncMock(return_value="python3")  # type: ignore[method-assign]
        provider._validate_remote_gpu = AsyncMock(return_value=None)  # type: ignore[method-assign]
        provider._prepare_remote_model = AsyncMock(return_value="/tmp/base-model")  # type: ignore[method-assign]
        provider._upload_tree = (  # type: ignore[method-assign]
            lambda *_args, on_progress, **_kwargs: on_progress(
                5,
                10,
                "adapter_model.safetensors",
            )
        )
        provider.process_alive = AsyncMock(return_value=True)  # type: ignore[method-assign]
        provider.health = AsyncMock(return_value=True)  # type: ignore[method-assign]
        stages: list[tuple[str, str, dict[str, object]]] = []

        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter"
            adapter.mkdir()
            await provider.start(
                "svc-progress",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter=str(adapter),
                resource={"gpuType": "4090", "gpuCount": 1},
                max_model_length=4096,
                on_stage=lambda stage, message, details: stages.append(
                    (stage, message, details)
                ),
            )

        progress_stage = next(item for item in stages if "Adapter 上传中：50%" in item[1])
        self.assertEqual(progress_stage[0], "uploading_adapter")
        self.assertEqual(progress_stage[2]["progress"], 71)

    async def test_start_summarizes_real_oom_log_without_hiding_raw_logs_or_cleanup(self) -> None:
        class Manager(_Manager):
            def create_one_shot(self, *_args: object, **_kwargs: object) -> object:
                return SimpleNamespace(uhost_id="uhost-oom", ip="10.0.0.2")

        class SSH:
            def mkdirs(self, path: str) -> None:
                return None

            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 0, ""

        manager = Manager()
        provider = YouyunInferenceProvider(manager, poll_interval=0, ssh_factory=MemoryLogSSH)  # type: ignore[arg-type]
        ssh = SSH()
        provider._connect = lambda _instance: ssh  # type: ignore[method-assign]
        provider._diagnose_remote_runtime = AsyncMock(return_value="")  # type: ignore[method-assign]
        provider._prepare_vllm_runtime_async = AsyncMock(return_value="python3")  # type: ignore[method-assign]
        provider._prepare_remote_model = AsyncMock(return_value="/tmp/base-model")  # type: ignore[method-assign]
        provider._upload_tree = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        provider.ssh_factory = lambda: MemoryLogSSH(lambda: REALISTIC_VLLM_OOM_LOG.encode())
        provider.process_alive = AsyncMock(return_value=False)  # type: ignore[method-assign]
        provider.read_log_tail = AsyncMock(return_value=REALISTIC_VLLM_OOM_LOG)  # type: ignore[method-assign]
        provider.stop = AsyncMock(  # type: ignore[method-assign]
            return_value={"terminated": False, "error": "delete timeout"}
        )
        raw_logs: list[str] = []
        stages: list[str] = []

        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter"
            adapter.mkdir()
            with self.assertRaises(YouyunInferenceStartError) as raised:
                await provider.start(
                    "svc-oom",
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    adapter=str(adapter),
                    resource={"gpuType": "3080Ti", "gpuCount": 1},
                    max_model_length=4096,
                    on_log=raw_logs.append,
                    on_stage=lambda stage, *_args, **_kwargs: stages.append(stage),
                )

        message = str(raised.exception)
        self.assertIn("显存不足", message)
        self.assertIn("256", message)
        self.assertIn("11.63 GiB", message)
        self.assertIn("313.50 MiB", message)
        self.assertIn("446.00 MiB", message)
        self.assertNotIn("\x1b", message)
        self.assertNotIn("Engine core initialization failed", message)
        self.assertNotIn("自动清理失败", message)
        self.assertEqual(raised.exception.cleanup_report["error"], "delete timeout")
        # 入库日志保留完整错误内容但剥离 ANSI；健康检查访问行不入库。
        self.assertTrue(any("torch.OutOfMemoryError" in line for line in raw_logs))
        self.assertTrue(all("\x1b" not in line for line in raw_logs))
        self.assertTrue(all("GET /health" not in line for line in raw_logs))
        self.assertLess(stages.index("ssh_connected"), stages.index("vllm_starting"))

    async def test_remote_model_preparation_returns_weight_directory(self) -> None:
        class SSH:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str, timeout: int = 120, check: bool = True):
                self.commands.append(command)
                return 0, "MODEL_READY:/root/cppt/modelscope-cache/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/rev1\n"

        provider = YouyunInferenceProvider(_Manager(), install_vllm=False)
        remote_model = await provider._prepare_remote_model(
            SSH(),
            model="Qwen/Qwen2.5-0.5B-Instruct",
            workdir="/root/cppt/inference/svc-test",
        )
        self.assertEqual(
            remote_model,
            "/root/cppt/modelscope-cache/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/rev1",
        )

    async def test_remote_model_download_prefers_modelscope_and_keeps_huggingface_fallback(self) -> None:
        class SSH:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.status_checks = 0

            def run(self, command: str, timeout: int = 120, check: bool = True):
                self.commands.append(command)
                if "MODEL_READY:" in command and "MODEL_MISS" in command:
                    return 0, "MODEL_MISS\n"
                if "nohup bash -c" in command:
                    return 0, ""
                if "echo status:" in command:
                    self.status_checks += 1
                    if self.status_checks == 1:
                        return 0, "status:pending\n"
                    return 0, (
                        "status:ok\n"
                        "result:/root/cppt/inference/svc-test/base-model\n"
                    )
                return 0, ""

        ssh = SSH()
        provider = YouyunInferenceProvider(
            _Manager(),
            install_vllm=False,
            poll_interval=0,
            model_prepare_timeout=1,
        )
        remote_model = await provider._prepare_remote_model(
            ssh,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            workdir="/root/cppt/inference/svc-test",
        )

        self.assertEqual(remote_model, "/root/cppt/inference/svc-test/base-model")
        launch = next(command for command in ssh.commands if "nohup bash -c" in command)
        self.assertLess(launch.index("modelscope"), launch.index("huggingface_hub"))
        self.assertIn("pip install --no-cache-dir modelscope", launch)
        self.assertIn("pip install --no-cache-dir huggingface_hub", launch)
        self.assertIn("from modelscope import snapshot_download", launch)
        self.assertIn("from huggingface_hub import snapshot_download", launch)

    def test_observability_callback_failures_do_not_abort_startup(self) -> None:
        def fail(*_args: object) -> None:
            raise TimeoutError()

        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        with self.assertLogs("backend.providers.youyun.inference", level="ERROR"):
            provider._stage(fail, "loading_model", "仍在加载")
            provider._emit_log(fail, "remote line")

    def test_background_launch_only_detaches_vllm_process(self) -> None:
        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        launch = provider._background_launch_command(
            ["python", "-m", "vllm.entrypoints.openai.api_server"],
            workdir="/root/cppt/inference/svc-1",
            pid_path="/root/cppt/inference/svc-1/vllm.pid",
            log_path="/root/cppt/inference/svc-1/vllm.log",
        )
        self.assertIn("</dev/null & pid=$!", launch)
        self.assertIn("printf '%s\\n' \"$pid\"", launch)
        self.assertNotIn("&& nohup", launch)

    async def test_runtime_diagnostics_emit_python_torch_cuda_and_vllm_lines(self) -> None:
        class SSH:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str, timeout: int = 120, check: bool = True):
                self.commands.append(command)
                return 0, (
                    "runtime.diagnostic.begin\n"
                    "python.path:/opt/conda/bin/python\n"
                    "python.version:Python 3.11.9\n"
                    "torch.version:2.7.1+cu128\n"
                    "torch.cuda.version:12.8\n"
                    "torch.cuda.available:True\n"
                    "torch.cuda.device_count:1\n"
                    "vllm.version:0.10.2\n"
                    "nvidia-smi:NVIDIA A10, 550.54\n"
                    "runtime.diagnostic.end\n"
                )

        logs: list[str] = []
        stages: list[str] = []
        provider = YouyunInferenceProvider(_Manager(), ssh_factory=SSH)
        result = await provider._diagnose_remote_runtime(
            SSH(),
            on_stage=lambda stage, *_args: stages.append(stage),
            on_log=logs.append,
        )
        self.assertIn("torch.cuda.available:True", result)
        self.assertIn("vllm.version:0.10.2", "\n".join(logs))
        self.assertIn("diagnosing_vllm_runtime", stages)

    async def test_gpu_preflight_rejects_nvidia_smi_failure(self) -> None:
        class SSH:
            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 42, (
                    "gpu.preflight.begin\n"
                    "nvidia-smi:Failed to initialize NVML: Unknown Error\n"
                )

        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            RuntimeError,
            "优云实例 GPU 驱动不可用，请更换实例或镜像",
        ) as raised:
            await provider._validate_remote_gpu(SSH())  # type: ignore[arg-type]

        self.assertIn("Failed to initialize NVML", str(raised.exception))

    async def test_gpu_preflight_rejects_torch_without_cuda_device(self) -> None:
        class SSH:
            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 43, (
                    "gpu.preflight.begin\n"
                    "nvidia-smi:NVIDIA GeForce RTX 4090, 595.80\n"
                    "torch.cuda.available:False\n"
                    "torch.cuda.device_count:0\n"
                )

        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "PyTorch 未检测到可用 CUDA 设备"):
            await provider._validate_remote_gpu(SSH())  # type: ignore[arg-type]

    async def test_start_gpu_preflight_failure_cleans_instance_before_model_transfer(self) -> None:
        class Manager(_Manager):
            def create_one_shot(self, *_args: object, **_kwargs: object) -> object:
                return SimpleNamespace(uhost_id="uhost-gpu-fail", ip="10.0.0.2")

        class SSH:
            def mkdirs(self, path: str) -> None:
                return None

            def run(self, command: str, timeout: int = 120, check: bool = True):
                if "gpu.preflight.begin" in command:
                    return 42, "nvidia-smi:Failed to initialize NVML: Unknown Error\n"
                return 0, ""

            def close(self) -> None:
                return None

        manager = Manager()
        provider = YouyunInferenceProvider(manager, poll_interval=0, ssh_factory=MemoryLogSSH)  # type: ignore[arg-type]
        ssh = SSH()
        provider._connect = lambda _instance: ssh  # type: ignore[method-assign]
        provider._diagnose_remote_runtime = AsyncMock(return_value="")  # type: ignore[method-assign]
        provider._prepare_vllm_runtime_async = AsyncMock(return_value="python3")  # type: ignore[method-assign]
        provider._prepare_remote_model = AsyncMock(return_value="/tmp/base-model")  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter"
            adapter.mkdir()
            with self.assertRaisesRegex(
                YouyunInferenceStartError,
                "优云实例 GPU 驱动不可用，请更换实例或镜像",
            ):
                await provider.start(
                    "svc-gpu-fail",
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    adapter=str(adapter),
                    resource={"gpuType": "4090", "gpuCount": 1},
                    max_model_length=4096,
                )

        provider._prepare_remote_model.assert_not_awaited()
        self.assertEqual(manager.cleaned, ["uhost-gpu-fail"])

    def test_remote_vllm_install_defaults_to_disabled(self) -> None:
        provider = YouyunInferenceProvider(_Manager())
        self.assertFalse(provider.install_vllm)

    async def test_async_runtime_install_can_be_cancelled(self) -> None:
        class SSH:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str, timeout: int = 120, check: bool = True):
                self.commands.append(command)
                if "import vllm" in command:
                    return 42, ""
                if "echo status:" in command:
                    return 0, "status:pending\n"
                return 0, ""

        ssh = SSH()
        cancelled = [False]
        original_run = ssh.run
        def run(command: str, timeout: int = 120, check: bool = True):
            result = original_run(command, timeout, check)
            if "echo status:" in command:
                cancelled[0] = True
            return result
        ssh.run = run  # type: ignore[method-assign]
        provider = YouyunInferenceProvider(
            _Manager(),  # type: ignore[arg-type]
            ssh_factory=SSH,
            poll_interval=0,
            install_vllm=True,
        )
        with self.assertRaisesRegex(RuntimeError, "服务启动已取消"):
            await provider._prepare_vllm_runtime_async(
                ssh,  # type: ignore[arg-type]
                workdir="/tmp/service",
                cancel_requested=lambda: cancelled[0],
            )
        self.assertTrue(any("kill -TERM" in command for command in ssh.commands))

    async def test_stop_still_deletes_instance_when_ssh_kill_fails(self) -> None:
        manager = _Manager()
        provider = YouyunInferenceProvider(
            manager,  # type: ignore[arg-type]
            ssh_factory=_FailingSSH,
        )
        report = await provider.stop(
            YouyunInferenceHandle(
                instance_id="uhost-1",
                endpoint="http://1.2.3.4:8011",
                pid_path="/tmp/vllm.pid",
                workdir="/tmp/service",
            )
        )
        self.assertTrue(report["terminated"])
        self.assertEqual(manager.cleaned, ["uhost-1"])
        self.assertIn("SSH connection lost", str(report["processError"]))

    def test_connect_refreshes_forwarded_ssh_port_and_retries(self) -> None:
        class Instance:
            def __init__(self, port: int) -> None:
                self.uhost_id = "uhost-1"
                self.port = port

        class RefreshingManager(_Manager):
            def __init__(self) -> None:
                super().__init__()
                self.instance = Instance(23)
                self.refreshed = Instance(60022)
                self.lookups = 0

            def ssh_credentials(self, instance: Instance) -> dict[str, object]:
                return {"host": "1.2.3.4", "port": instance.port, "user": "root", "password": "pass"}

            def get(self, instance_id: str) -> Instance:
                self.lookups += 1
                return self.instance if self.lookups == 1 else self.refreshed

        class RetryingSSH:
            calls: list[int] = []

            def connect(self, host: str, port: int, user: str, password: str) -> None:
                self.calls.append(port)
                if port == 23:
                    raise OSError("Unable to connect to port 23")

            def close(self) -> None:
                return None

        manager = RefreshingManager()
        provider = YouyunInferenceProvider(
            manager,  # type: ignore[arg-type]
            ssh_factory=RetryingSSH,
            poll_interval=0,
            ssh_ready_timeout=1,
        )
        ssh = provider._connect(manager.instance)
        self.assertIsInstance(ssh, RetryingSSH)
        self.assertEqual(RetryingSSH.calls, [23, 60022])

    def test_prepare_vllm_runtime_fails_fast_when_module_is_missing(self) -> None:
        class SSH:
            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 42, ""

        provider = YouyunInferenceProvider(
            _Manager(),  # type: ignore[arg-type]
            install_vllm=False,
        )
        with self.assertRaisesRegex(RuntimeError, "未安装 vLLM"):
            provider._prepare_vllm_runtime(SSH())  # type: ignore[arg-type]

    def test_prepare_vllm_runtime_selects_verified_python(self) -> None:
        class SSH:
            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 0, "/opt/conda/bin/python\n"

        provider = YouyunInferenceProvider(
            _Manager(),  # type: ignore[arg-type]
            install_vllm=False,
        )
        selected = provider._prepare_vllm_runtime(SSH())  # type: ignore[arg-type]
        self.assertEqual(selected, "/opt/conda/bin/python")
        self.assertEqual(provider.remote_python, selected)

    def test_ensure_remote_ninja_skips_install_when_present(self) -> None:
        class SSH:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str, timeout: int = 120, check: bool = True):
                self.commands.append(command)
                return 0, "ninja:present\nnvcc:present"

        provider = YouyunInferenceProvider(_Manager(), install_vllm=False)  # type: ignore[arg-type]
        ssh = SSH()
        asyncio.run(provider._ensure_remote_ninja(ssh))  # type: ignore[arg-type]
        self.assertEqual(len(ssh.commands), 1)
        self.assertNotIn("pip install", ssh.commands[0])

    def test_ensure_remote_ninja_installs_when_missing(self) -> None:
        class SSH:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str, timeout: int = 120, check: bool = True):
                self.commands.append(command)
                if "pip install" in command:
                    return 0, "ninja:installed"
                return 0, "ninja:missing\nnvcc:present"

        provider = YouyunInferenceProvider(_Manager(), install_vllm=False)  # type: ignore[arg-type]
        ssh = SSH()
        stages: list[tuple[str, str]] = []
        logs: list[str] = []
        asyncio.run(
            provider._ensure_remote_ninja(
                ssh,  # type: ignore[arg-type]
                on_stage=lambda stage, message, _details=None: stages.append((stage, message)),
                on_log=logs.append,
            )
        )
        self.assertEqual(len(ssh.commands), 2)
        self.assertIn("pip install --no-cache-dir ninja", ssh.commands[1])
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0][0], "installing_vllm")
        self.assertIn("ninja", stages[0][1])
        self.assertTrue(logs and "ninja" in logs[-1])

    def test_ensure_remote_ninja_install_failure_is_not_fatal(self) -> None:
        class SSH:
            def run(self, command: str, timeout: int = 120, check: bool = True):
                if "pip install" in command:
                    return 1, "ninja:install-failed"
                return 0, "ninja:missing\nnvcc:missing"

        provider = YouyunInferenceProvider(_Manager(), install_vllm=False)  # type: ignore[arg-type]
        asyncio.run(provider._ensure_remote_ninja(SSH()))  # type: ignore[arg-type]

    async def test_process_alive_reports_remote_pid_exit(self) -> None:
        class SSH:
            def run(self, command: str, timeout: int = 120, check: bool = True):
                return 1, ""

        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        handle = YouyunInferenceHandle(
            instance_id="uhost-1",
            endpoint="http://1.2.3.4:8011",
            pid_path="/tmp/vllm.pid",
            workdir="/tmp/service",
            ssh=SSH(),  # type: ignore[arg-type]
        )
        self.assertFalse(await provider.process_alive(handle))

    async def test_process_alive_treats_ssh_timeout_as_unknown(self) -> None:
        class SSH:
            closed = False

            def run(self, command: str, timeout: int = 120, check: bool = True):
                raise TimeoutError()

            def close(self) -> None:
                self.closed = True

        ssh = SSH()
        provider = YouyunInferenceProvider(_Manager())  # type: ignore[arg-type]
        handle = YouyunInferenceHandle(
            instance_id="uhost-1",
            endpoint="http://1.2.3.4:8011",
            pid_path="/tmp/vllm.pid",
            workdir="/tmp/service",
            ssh=ssh,  # type: ignore[arg-type]
        )
        self.assertIsNone(await provider.process_alive(handle))
        self.assertTrue(ssh.closed)
        self.assertIsNone(handle.ssh)
        self.assertEqual(handle.last_probe_error, "TimeoutError")

    async def test_log_refresh_tolerates_ssh_reconnect_timeout(self) -> None:
        class TimeoutClient:
            def describe_instance(self, instance_id: str) -> object:
                raise TimeoutError()

        manager = _Manager()
        manager.client = TimeoutClient()  # type: ignore[assignment]
        provider = YouyunInferenceProvider(manager)  # type: ignore[arg-type]
        handle = YouyunInferenceHandle(
            instance_id="uhost-1",
            endpoint="http://1.2.3.4:8011",
            pid_path="/tmp/vllm.pid",
            workdir="/tmp/service",
        )
        chunk, offset = await provider.read_logs(handle, 123)
        self.assertEqual((chunk, offset), ("", 123))
        self.assertEqual(handle.last_probe_error, "TimeoutError")


class _TaskStore:
    def __init__(self, task: dict) -> None:
        self.task = task

    def get(self, task_id: str) -> dict | None:
        return self.task if task_id == self.task["id"] else None


class _FakeRemoteProvider:
    def __init__(self, stop_results: list[dict[str, object]] | None = None) -> None:
        self.stop_results = list(stop_results or [])
        self.started: list[str] = []
        self.stopped: list[str] = []

    async def start(self, service_id: str, **kwargs: object) -> YouyunInferenceHandle:
        handle = YouyunInferenceHandle(
            instance_id=f"new-{len(self.started) + 1}",
            endpoint="http://10.0.0.2:8011",
            pid_path=f"/tmp/{service_id}.pid",
            workdir=f"/tmp/{service_id}",
        )
        callback = kwargs.get("on_instance")
        if callable(callback):
            callback(handle)
        self.started.append(service_id)
        return handle

    async def stop(self, handle: YouyunInferenceHandle) -> dict[str, object]:
        self.stopped.append(handle.instance_id)
        if self.stop_results:
            return self.stop_results.pop(0)
        return {"terminated": True, "error": None}

    async def health(self, handle: YouyunInferenceHandle) -> bool:
        return True


class RuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.runtimes: list[InferenceRuntime] = []
        root = Path(self.temporary.name)
        self.adapter = root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        self.task = {
            "id": "ft-1",
            "state": "completed",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "baseModelMetadata": {"maxContextLength": 4096},
            "artifacts": [
                {
                    "type": "adapter",
                    "variant": "best",
                    "selected": True,
                    "path": str(self.adapter),
                }
            ],
        }
        self.store = InferenceStore(root / "inference.db")

    async def asyncTearDown(self) -> None:
        for runtime in self.runtimes:
            await runtime.shutdown()
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.temporary.cleanup()

    def runtime(
        self,
        provider: _FakeRemoteProvider,
        *,
        training_busy: bool = False,
        provider_factory=None,
        resource_resolver=None,
    ) -> InferenceRuntime:
        runtime = InferenceRuntime(
            self.store,
            _TaskStore(self.task),  # type: ignore[arg-type]
            port=8011,
            training_busy=lambda: training_busy,
            remote_provider_factory=provider_factory or (lambda: provider),
            youyun_resource_resolver=resource_resolver,
        )
        runtime.refresh_metrics = AsyncMock(return_value={})  # type: ignore[method-assign]
        self.runtimes.append(runtime)
        return runtime

    async def create_service(self, runtime: InferenceRuntime, *, name: str = "remote") -> dict:
        return await runtime.create(
            InferenceCreate(
                name=name,
                taskId="ft-1",
                provider="youyun",
                youyunResource={
                    "gpuType": "4090",
                    "gpuCount": 1,
                    "cpuCores": 16,
                    "memoryGb": 64,
                },
            )
        )

    async def test_single_reader_spans_startup_running_and_many_ui_requests(self) -> None:
        from backend.test_remote_log_follow import eventually
        data = bytearray(b"loading model\n")
        connections = []

        class Manager(_Manager):
            def create_one_shot(self, *args, **kwargs):
                return SimpleNamespace(uhost_id="stream-one", ip="10.0.0.2")

            def cleanup_after_task(inner, *args, **kwargs):
                self.assertTrue(all(connection.closed for connection in connections))
                return super().cleanup_after_task(*args, **kwargs)

        class ControlSSH:
            def mkdirs(self, path):
                pass
            def run(self, *args):
                return 0, ""
            def close(self):
                pass

        def factory(**kwargs):
            connection = MemoryLogSSH(lambda: bytes(data))
            connections.append(connection)
            return connection

        provider = YouyunInferenceProvider(Manager(), ssh_factory=factory, poll_interval=.01)
        provider._connect = lambda _: ControlSSH()
        provider._diagnose_remote_runtime = AsyncMock(return_value="")
        provider._prepare_vllm_runtime_async = AsyncMock(return_value="python3")
        provider._validate_remote_gpu = AsyncMock(return_value=None)
        provider._prepare_remote_model = AsyncMock(return_value="/models/base")
        provider._upload_tree = lambda *args, **kwargs: None
        provider.process_alive = AsyncMock(return_value=True)
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        service_id = service["id"]

        async def healthy(_):
            # Startup cannot complete until its live logs are already persisted.
            await eventually(lambda: any("loading model" in line for line in self.store.logs(service_id)))
            return True

        provider.health = healthy
        started = await runtime.start(service_id)
        self.assertEqual(started["state"], "running")
        self.assertEqual(started["config"]["remoteRuntime"]["logOffset"], len(data))
        await asyncio.gather(*(runtime.refresh_remote_logs(service_id) for _ in range(12)))
        self.assertEqual(len(connections), 1)
        data.extend(b"ready for inference\n")
        await eventually(lambda: any("ready for inference" in line for line in self.store.logs(service_id)))
        # Without any browser connection, the lifecycle reader keeps working.
        data.extend(b"final partial")
        await runtime.stop(service_id)
        logs = self.store.logs(service_id)
        self.assertEqual(sum("loading model" in line for line in logs), 1)
        self.assertEqual(sum("ready for inference" in line for line in logs), 1)
        self.assertEqual(sum("final partial" in line for line in logs), 1)
        await runtime.refresh_remote_logs(service_id)
        self.assertEqual(len(connections), 2)  # one live reader and one final snapshot
        self.assertFalse(provider._log_collectors)

    async def test_make_remote_provider_uses_service_region_and_zone(self) -> None:
        # 回归：启动/释放实例必须使用服务创建时选定的 Region/Zone，
        # 否则优云以 RetCode=230 (Params [Zone] not available) 拒绝请求。
        runtime = InferenceRuntime(
            self.store,
            _TaskStore(self.task),  # type: ignore[arg-type]
            port=8011,
            training_busy=lambda: False,
        )
        self.runtimes.append(runtime)
        service = await runtime.create(
            InferenceCreate(
                name="zone-check",
                taskId="ft-1",
                provider="youyun",
                youyunResource={
                    "region": "cn-bj2",
                    "zone": "cn-bj2-03",
                    "instanceMode": "pod",
                    "gpuType": "4090",
                    "gpuCount": 1,
                    "cpuCores": 14,
                    "memoryGb": 32,
                },
            )
        )
        captured: dict[str, YouyunConfig] = {}

        class _CapturingClient:
            def __init__(self, config: YouyunConfig | None = None, require_full: bool = True) -> None:
                captured["config"] = config  # type: ignore[assignment]

        with patch("backend.providers.youyun.client.YouyunClient", _CapturingClient):
            runtime._make_remote_provider(service["id"])
        self.assertEqual(captured["config"].region, "cn-bj2")
        self.assertEqual(captured["config"].zone, "cn-bj2-03")

    async def test_youyun_start_ignores_local_training_lease_but_local_start_does_not(self) -> None:
        provider = _FakeRemoteProvider()
        runtime = self.runtime(provider, training_busy=True)
        remote = await self.create_service(runtime)
        started = await runtime.start(remote["id"])
        self.assertEqual(started["state"], "running")

        local = await runtime.create(InferenceCreate(name="local", taskId="ft-1"))
        with self.assertRaisesRegex(RuntimeError, "训练任务"):
            await runtime.start(local["id"])
        await runtime.stop(remote["id"])
        await runtime.shutdown()

    async def test_two_youyun_services_run_and_stop_independently(self) -> None:
        providers: list[_FakeRemoteProvider] = []

        def provider_factory() -> _FakeRemoteProvider:
            provider = _FakeRemoteProvider()
            providers.append(provider)
            return provider

        runtime = self.runtime(_FakeRemoteProvider(), provider_factory=provider_factory)
        first = await self.create_service(runtime, name="remote-one")
        second = await self.create_service(runtime, name="remote-two")
        first_started, second_started = await asyncio.gather(
            runtime.start(first["id"]),
            runtime.start(second["id"]),
        )
        self.assertEqual(first_started["state"], "running")
        self.assertEqual(second_started["state"], "running")

        await runtime.stop(first["id"])
        current_second = self.store.get(second["id"])
        assert current_second is not None
        self.assertEqual(current_second["state"], "running")
        self.assertIn(second["id"], runtime.remote_handles)
        self.assertNotIn(first["id"], runtime.remote_handles)

        await runtime.stop(second["id"])
        self.assertEqual([provider.stopped for provider in providers], [["new-1"], ["new-1"]])
        await runtime.shutdown()

    async def test_recover_rejects_duplicate_until_previous_start_finishes_and_has_no_limit(self) -> None:
        class RecoverProvider(_FakeRemoteProvider):
            def __init__(self) -> None:
                super().__init__()
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def start(self, service_id: str, **kwargs: object) -> YouyunInferenceHandle:
                self.entered.set()
                await self.release.wait()
                return await super().start(service_id, **kwargs)

        providers = [RecoverProvider(), RecoverProvider()]
        resolver_entered = asyncio.Event()
        resolver_release = asyncio.Event()
        resolve_calls = 0

        async def resolve(resource: dict[str, object]) -> dict[str, object]:
            nonlocal resolve_calls
            resolve_calls += 1
            # Service creation performs the first stock resolution. Hold the
            # first retry resolution so a duplicate retry can race it.
            if resolve_calls == 2:
                resolver_entered.set()
                await resolver_release.wait()
            return resource

        runtime = self.runtime(
            providers[0],
            provider_factory=lambda: providers.pop(0),
            resource_resolver=resolve,
        )
        service = await self.create_service(runtime)
        self.store.update(service["id"], state="failed", recovery_attempts=99)

        first_provider = providers[0]
        first_recover = asyncio.create_task(runtime.recover(service["id"]))
        await resolver_entered.wait()
        with self.assertRaisesRegex(RuntimeError, "重试|启动"):
            await runtime.recover(service["id"])
        resolver_release.set()
        await first_provider.entered.wait()
        first_provider.release.set()
        recovered = await first_recover
        self.assertEqual(recovered["state"], "running")
        await runtime.stop(service["id"])

        self.store.update(service["id"], state="failed")
        second_provider = providers[0]
        second_recover = asyncio.create_task(runtime.recover(service["id"]))
        await second_provider.entered.wait()
        second_provider.release.set()
        recovered_again = await second_recover
        self.assertEqual(recovered_again["state"], "running")
        self.assertEqual(recovered_again["recoveryAttempts"], 99)
        await runtime.stop(service["id"])
        await runtime.shutdown()

    async def test_recover_passes_region_and_instance_mode_to_resource_resolver(self) -> None:
        # 回归：重试前库存校验必须带上服务创建时选定的 region 与 instanceMode，
        # 否则 zone 单独传入会与默认区域错配（RetCode=230 Params [Zone] not available）。
        captured: list[dict] = []

        async def resolve(resource: dict[str, object]) -> dict[str, object]:
            captured.append(dict(resource))
            return dict(resource)

        runtime = self.runtime(_FakeRemoteProvider(), resource_resolver=resolve)
        service = await runtime.create(
            InferenceCreate(
                name="retry-region",
                taskId="ft-1",
                provider="youyun",
                youyunResource={
                    "region": "cn-bj2",
                    "zone": "cn-bj2-03",
                    "instanceMode": "pod",
                    "gpuType": "4090",
                    "gpuCount": 1,
                    "cpuCores": 14,
                    "memoryGb": 32,
                },
            )
        )
        self.store.update(service["id"], state="failed")
        recovered = await runtime.recover(service["id"])
        self.assertEqual(recovered["state"], "running")
        self.assertTrue(captured)
        self.assertEqual(captured[0].get("region"), "cn-bj2")
        self.assertEqual(captured[0].get("zone"), "cn-bj2-03")
        self.assertEqual(captured[0].get("instanceMode"), "pod")
        await runtime.stop(service["id"])

    async def test_stop_reaches_provider_without_waiting_for_slow_start(self) -> None:
        class SlowStartProvider(_FakeRemoteProvider):
            def __init__(self) -> None:
                super().__init__()
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def start(self, service_id: str, **kwargs: object) -> YouyunInferenceHandle:
                handle = YouyunInferenceHandle(
                    instance_id="starting-slow",
                    endpoint="http://10.0.0.2:8011",
                    pid_path=f"/tmp/{service_id}.pid",
                    workdir=f"/tmp/{service_id}",
                )
                callback = kwargs.get("on_instance")
                if callable(callback):
                    callback(handle)
                self.entered.set()
                await self.release.wait()
                if kwargs["cancel_requested"]():
                    raise RuntimeError("服务启动已取消")
                return handle

            async def stop(self, handle: YouyunInferenceHandle) -> dict[str, object]:
                self.stopped.append(handle.instance_id)
                self.release.set()
                return {"terminated": True, "error": None}

        provider = SlowStartProvider()
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        await runtime.start_background(service["id"])
        await provider.entered.wait()

        stopped = await asyncio.wait_for(runtime.stop(service["id"]), timeout=0.2)
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(provider.stopped, ["starting-slow"])
        self.assertNotIn("remoteRuntime", stopped["config"])
        await runtime.shutdown()

    async def test_cleanup_failure_is_persisted_and_stop_can_retry(self) -> None:
        provider = _FakeRemoteProvider(
            stop_results=[
                {"terminated": False, "error": "billing lock"},
                {"terminated": True, "error": None},
            ]
        )
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        await runtime.start(service["id"])
        with self.assertRaisesRegex(RuntimeError, "billing lock"):
            await runtime.stop(service["id"])
        failed = self.store.get(service["id"])
        assert failed is not None
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["config"]["remoteRuntime"]["state"], "cleanup_failed")
        stopped = await runtime.stop(service["id"])
        self.assertEqual(stopped["state"], "stopped")
        self.assertNotIn("remoteRuntime", stopped["config"])
        await runtime.shutdown()

    async def test_archive_reconciles_stale_remote_runtime_before_archiving(self) -> None:
        provider = _FakeRemoteProvider()
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        config = dict(service["config"])
        config["remoteRuntime"] = {
            "instanceId": "already-gone-1",
            "endpoint": "http://10.0.0.1:8011",
            "pidPath": "/tmp/old.pid",
            "workdir": "/tmp/old",
            "state": "starting",
        }
        self.store.update(service["id"], config=config, state="failed")

        archived = await runtime.archive(service["id"])

        self.assertEqual(provider.stopped, ["already-gone-1"])
        self.assertEqual(archived["state"], "archived")
        self.assertNotIn("remoteRuntime", archived["config"])

    async def test_archive_keeps_blocking_when_remote_cleanup_fails(self) -> None:
        provider = _FakeRemoteProvider(
            stop_results=[{"terminated": False, "error": "billing lock"}]
        )
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        config = dict(service["config"])
        config["remoteRuntime"] = {
            "instanceId": "still-running-1",
            "endpoint": "http://10.0.0.1:8011",
            "pidPath": "/tmp/old.pid",
            "workdir": "/tmp/old",
            "state": "cleanup_failed",
        }
        self.store.update(service["id"], config=config, state="failed")

        with self.assertRaisesRegex(RuntimeError, "billing lock"):
            await runtime.archive(service["id"])

        current = self.store.get(service["id"])
        assert current is not None
        self.assertEqual(current["state"], "failed")
        self.assertEqual(current["config"]["remoteRuntime"]["instanceId"], "still-running-1")

    async def test_start_failure_keeps_raw_log_and_records_cleanup_separately(self) -> None:
        class FailingProvider(_FakeRemoteProvider):
            async def start(self, service_id: str, **kwargs: object) -> YouyunInferenceHandle:
                handle = YouyunInferenceHandle(
                    instance_id="failed-start-1",
                    endpoint="http://10.0.0.2:8011",
                    pid_path=f"/tmp/{service_id}.pid",
                    workdir=f"/tmp/{service_id}",
                )
                on_instance = kwargs.get("on_instance")
                if callable(on_instance):
                    on_instance(handle)
                on_log = kwargs.get("on_log")
                if callable(on_log):
                    on_log(REALISTIC_VLLM_OOM_LOG)
                raise YouyunInferenceStartError(
                    "远端 vLLM 进程已退出：远端 vLLM 显存不足。请降低最大并发数或 GPU 显存利用率。",
                    handle,
                    {"terminated": False, "error": "delete timeout"},
                )

        provider = FailingProvider()
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)

        with self.assertRaisesRegex(RuntimeError, "显存不足"):
            await runtime.start(service["id"])

        failed = self.store.get(service["id"])
        assert failed is not None
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["exposure"], "offline")
        self.assertEqual(failed["gpu"], "优云实例待清理")
        self.assertIn("显存不足", failed["error"])
        self.assertNotIn("delete timeout", failed["error"])
        self.assertEqual(failed["config"]["remoteRuntime"]["state"], "cleanup_failed")
        self.assertEqual(failed["config"]["remoteRuntime"]["cleanupError"], "delete timeout")
        logs = "\n".join(self.store.logs(service["id"], 100))
        self.assertIn("\x1b[0;36m", logs)
        self.assertIn("自动清理失败：delete timeout", logs)
        await runtime.shutdown()

    async def test_stop_cancels_background_remote_start_without_error(self) -> None:
        class CancellableProvider(_FakeRemoteProvider):
            async def start(self, service_id: str, **kwargs: object) -> YouyunInferenceHandle:
                handle = YouyunInferenceHandle(
                    instance_id="starting-1",
                    endpoint="http://10.0.0.2:8011",
                    pid_path=f"/tmp/{service_id}.pid",
                    workdir=f"/tmp/{service_id}",
                )
                callback = kwargs.get("on_instance")
                if callable(callback):
                    callback(handle)
                requested = kwargs["cancel_requested"]
                while not requested():
                    await asyncio.sleep(0)
                raise RuntimeError("服务启动已取消")

        provider = CancellableProvider()
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        await runtime.start_background(service["id"])
        await asyncio.sleep(0)
        stopped = await runtime.stop(service["id"])
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(provider.stopped, ["starting-1"])
        self.assertNotIn("remoteRuntime", stopped["config"])
        await runtime.shutdown()

    async def test_recovery_cleans_persisted_instance_before_replacement(self) -> None:
        provider = _FakeRemoteProvider()
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        config = dict(service["config"])
        config["remoteRuntime"] = {
            "instanceId": "orphan-1",
            "endpoint": "http://10.0.0.1:8011",
            "pidPath": "/tmp/old.pid",
            "workdir": "/tmp/old",
            "state": "running",
        }
        self.store.update(service["id"], config=config, state="running")
        recovered = await runtime.recover(service["id"])
        self.assertEqual(provider.stopped[0], "orphan-1")
        self.assertEqual(provider.started, [service["id"]])
        self.assertEqual(recovered["state"], "running")
        self.assertEqual(recovered["recoveryAttempts"], 0)
        await runtime.stop(service["id"])
        await runtime.shutdown()

    async def test_recovery_does_not_create_second_instance_when_orphan_cleanup_fails(self) -> None:
        provider = _FakeRemoteProvider(
            stop_results=[{"terminated": False, "error": "delete denied"}]
        )
        runtime = self.runtime(provider)
        service = await self.create_service(runtime)
        config = dict(service["config"])
        config["remoteRuntime"] = {
            "instanceId": "orphan-2",
            "endpoint": "http://10.0.0.1:8011",
            "pidPath": "/tmp/old.pid",
            "workdir": "/tmp/old",
            "state": "running",
        }
        self.store.update(service["id"], config=config, state="running")
        with self.assertRaisesRegex(RuntimeError, "delete denied"):
            await runtime.recover(service["id"])
        self.assertEqual(provider.started, [])
        current = self.store.get(service["id"])
        assert current is not None
        self.assertEqual(current["state"], "failed")
        self.assertEqual(current["config"]["remoteRuntime"]["instanceId"], "orphan-2")


class RemoteLogNoiseFilterTest(unittest.TestCase):
    def test_keeps_access_log_and_progress_lines_while_dropping_blank_lines(self) -> None:
        self.assertEqual(
            _clean_remote_log_line('(APIServer pid=701) INFO:     127.0.0.1:47594 - "GET /health HTTP/1.1" 200 OK'),
            'INFO:     127.0.0.1:47594 - "GET /health HTTP/1.1" 200 OK',
        )
        self.assertEqual(
            _clean_remote_log_line("(EngineCore_DP0 pid=754) GET /metrics"),
            "GET /metrics",
        )
        self.assertEqual(
            _clean_remote_log_line("Capturing CUDA graphs (PIECEWISE):   0%|          | 0/8 [00:00<?, ?it/s]"),
            "Capturing CUDA graphs (PIECEWISE):   0%|          | 0/8 [00:00<?, ?it/s]",
        )
        self.assertIsNone(_clean_remote_log_line("   "))

    def test_keeps_meaningful_lines_and_strips_ansi(self) -> None:
        self.assertEqual(
            _clean_remote_log_line("\x1b[0;36m(EngineCore_DP0 pid=754)\x1b[0;0m Initializing a V1 LLM engine (v0.12.0)"),
            "Initializing a V1 LLM engine (v0.12.0)",
        )
        self.assertEqual(
            _clean_remote_log_line("Loaded new LoRA adapter: name 'svc-1'"),
            "Loaded new LoRA adapter: name 'svc-1'",
        )


if __name__ == "__main__":
    unittest.main()
