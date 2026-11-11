"""YouyunTrainingRunner 端到端单元测试：伪造 SSH/实例管理/store，覆盖成功、失败、停止路径。"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.providers.youyun.models import YouyunInstance
from backend.providers.youyun.remote_training import YouyunTrainingRunner, _INSTALL_COMMAND, _gpu_supports_bf16

LOG_LINES = [
    "***** Running training *****",
    "Total optimization steps 20",
    "{'loss': 2.31, 'learning_rate': 5e-05, 'epoch': 0.05}",
    "{'loss': 1.82, 'learning_rate': 5e-05, 'epoch': 0.10}",
]


class FakeSSH:
    def __init__(self, *, install_polls: int = 1, install_fail: bool = False) -> None:
        self.install_polls = install_polls
        self.install_fail = install_fail
        self.install_poll_count = 0
        self.train_poll_count = 0
        self.uploaded: dict[str, str] = {}
        self.mkdir_calls: list[str] = []
        self.get_tree_calls: list[str] = []
        self.commands: list[str] = []

    def log_connection(self):
        from backend.test_remote_log_follow import MemoryLogSSH
        return MemoryLogSSH(lambda: ("\n".join(LOG_LINES) + "\ntrain finished\n").encode())

    def connect(self, host, port, user, password) -> None:
        return None

    def close(self) -> None:
        return None

    def run(self, command, timeout=120, check=True):
        self.commands.append(command)
        if "nvidia-smi" in command:
            return 0, "NVIDIA GeForce RTX 4090, 24564 MiB\n"
        if "env-install.done" in command and "cat" in command:
            self.install_poll_count += 1
            if self.install_poll_count <= self.install_polls:
                return 0, "---\nstill installing\n"
            marker = "fail" if self.install_fail else "ok"
            return 0, f"{marker}\n---\nlast install line\n"
        if "nohup" in command and "env-install" in command:
            return 0, ""
        if "MODEL_MISS" in command:
            return 0, "MODEL_READY:/root/cppt/modelscope-cache/models/demo\n"
        if "train.yaml" in command and "nohup" in command:
            return 0, ""
        if command.startswith("echo code:"):
            self.train_poll_count += 1
            if self.train_poll_count <= len(LOG_LINES):
                line = LOG_LINES[self.train_poll_count - 1]
                return 0, f"code:none\n{line}\n"
            return 0, "code:0\ntrain finished\n"
        return 0, ""

    def mkdirs(self, remote_dir: str) -> None:
        self.mkdir_calls.append(remote_dir)

    def put_file(self, local_path, remote_path) -> None:
        self.uploaded[remote_path] = Path(local_path).read_text(encoding="utf-8")

    def put_text(self, text: str, remote_path: str) -> None:
        self.uploaded[remote_path] = text

    def get_tree(self, remote_dir: str, local_dir) -> list[str]:
        self.get_tree_calls.append(remote_dir)
        output = Path(local_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "/root/cppt/modelscope-cache/models/demo"}',
            encoding="utf-8",
        )
        (output / "trainer_state.json").write_text(json.dumps({"log_history": []}), encoding="utf-8")
        return [str(output / "adapter_config.json")]

    def get_file(self, remote_path: str, local_path) -> None:
        self.commands.append(f"get_file:{remote_path}")
        Path(local_path).write_text(
            '{"max_position_embeddings": 32768}', encoding="utf-8"
        )


class FakeManager:
    def __init__(self) -> None:
        self.cleaned: list[str] = []
        self.cleanup_error: str | None = None

    def create_one_shot(self, task_id, **kwargs):
        return YouyunInstance(
            uhost_id="uhost-9", name="cppt-t", state="Running", region="cn-wlcb", zone="cn-wlcb-01",
            gpu=1, gpu_type="4090", cpu=16, memory_mb=65536, ip="1.2.3.4", ssh_port=60022,
            ssh_user="root", password_b64="cGFzcw==", image_id="img", os_type="Linux",
            charge_type="Postpay", instance_type="UHost", raw={},
        )

    def ssh_credentials(self, instance):
        return {"host": instance.ip, "port": instance.ssh_port, "user": instance.ssh_user, "password": "pass"}

    def cleanup_after_task(self, uhost_id, stop_first=True):
        self.cleaned.append(uhost_id)
        return {"terminated": not self.cleanup_error, "error": self.cleanup_error}


class FakeSink:
    def __init__(self, cancel_after_logs: int | None = None) -> None:
        self.logs: list[str] = []
        self.metrics: list[dict] = []
        self.updates: list[dict] = []
        self.artifacts: list[dict] = []
        self._cancel_after = cancel_after_logs
        self._log_count = 0

    def append_log(self, task_id, message) -> None:
        self.logs.append(message)
        self._log_count += 1

    def append_metric(self, task_id, **kw) -> None:
        self.metrics.append(kw)

    def update(self, task_id, **kw) -> None:
        self.updates.append(kw)
        if "artifacts" in kw:
            self.artifacts = kw["artifacts"]

    def get(self, task_id):
        return {"artifacts": self.artifacts}

    def is_cancel_requested(self, task_id) -> bool:
        return self._cancel_after is not None and self._log_count >= self._cancel_after

    def metrics_(self, task_id) -> list[dict]:
        return self.metrics


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


class RunnerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mirror = Path(self.tmp.name)
        self.ssh = FakeSSH()
        self.manager = FakeManager()

    def make_runner(self, sink: FakeSink, on_retained=None) -> YouyunTrainingRunner:
        async def prepare_local(task):
            workdir = self.mirror / task["taskId"]
            (workdir / "dataset").mkdir(parents=True, exist_ok=True)
            (workdir / "dataset" / "train.jsonl").write_text("{}\n", encoding="utf-8")
            (workdir / "dataset" / "dataset_info.json").write_text("{}", encoding="utf-8")
            (workdir / "train.yaml").write_text("dataset_dir: /local/dataset\noutput_dir: /local/output\n", encoding="utf-8")
            return SimpleNamespace(workdir=str(workdir), config_path=str(workdir / "train.yaml"), prepared=True)

        async def collect_local(task):
            workdir = self.mirror / (task.get("taskId") or task.get("id"))
            adapter = workdir / "output" / "adapter_config.json"
            assert adapter.is_file(), "collect 前产物应已下载到本地镜像"
            return [{"type": "adapter", "name": "final", "variant": "final", "path": str(adapter)}]

        return YouyunTrainingRunner(
            manager=self.manager,
            sink=sink,
            ssh_factory=lambda: self.ssh,
            mirror_root=self.mirror,
            prepare_local=prepare_local,
            collect_local=collect_local,
            sleep=lambda s: None,
            clock=_FakeClock(),
            on_retained=on_retained,
        )


class HappyPathTest(RunnerTestBase):
    def test_completed(self) -> None:
        sink = FakeSink()
        outcome = asyncio.run(self.make_runner(sink).run("t1", {"taskId": "t1", "parameters": {"loggingSteps": 5}}))
        self.assertEqual(outcome, "completed")
        self.assertIn("优云实例已就绪", "\n".join(sink.logs))
        self.assertTrue(any(log.startswith("GPU（探测到 1 张）：NVIDIA GeForce RTX 4090") for log in sink.logs))
        # 两条训练指标被解析入库
        self.assertEqual(len(sink.metrics), 2)
        self.assertAlmostEqual(sink.metrics[0]["train_loss"], 2.31)
        # 产物收集 + 上传内容检查
        self.assertEqual(self.ssh.get_tree_calls, ["/root/cppt/t1/output"])
        remote_yaml = self.ssh.uploaded["/root/cppt/t1/train.yaml"]
        self.assertIn("/root/cppt/t1/dataset", remote_yaml)
        self.assertIn("/root/cppt/t1/output", remote_yaml)
        self.assertIn("/root/cppt/t1/dataset/train.jsonl", self.ssh.uploaded)  # 数据集按文件上传
        self.assertEqual(self.manager.cleaned, [])
        marker = next(item for item in sink.artifacts if item.get("type") == "youyun-instance")
        self.assertEqual(marker["state"], "awaiting_comparison")
        adapter = next(item for item in sink.artifacts if item.get("type") == "adapter")
        self.assertEqual(adapter["baseModelMetadata"]["maxContextLength"], 32768)
        completed = [u for u in sink.updates if "等待效果对比" in u.get("current_step", "")]
        self.assertTrue(completed)

    def test_yaml_paths_rewritten_to_remote(self) -> None:
        sink = FakeSink()
        asyncio.run(self.make_runner(sink).run("t1", {"taskId": "t1", "parameters": {}, "youyunResource": {"gpuType": "4090"}}))
        yaml_text = self.ssh.uploaded["/root/cppt/t1/train.yaml"]
        self.assertNotIn("/local/dataset", yaml_text)
        self.assertNotIn("/local/output", yaml_text)
        self.assertIn("bf16: true", yaml_text)

    def test_old_gpu_uses_fp16(self) -> None:
        sink = FakeSink()
        asyncio.run(self.make_runner(sink).run("t1", {"taskId": "t1", "parameters": {}, "youyunResource": {"gpuType": "2080"}}))
        yaml_text = self.ssh.uploaded["/root/cppt/t1/train.yaml"]
        self.assertIn("fp16: true", yaml_text)
        self.assertIn("bf16: false", yaml_text)


class RetainedCallbackTest(RunnerTestBase):
    def test_on_retained_fires_after_completion(self) -> None:
        events: list[tuple[str, dict]] = []

        async def on_retained(task_id: str, task: dict) -> None:
            events.append((task_id, dict(task)))

        sink = FakeSink()

        async def scenario() -> str:
            runner = self.make_runner(sink, on_retained=on_retained)
            outcome = await runner.run("t1", {"taskId": "t1", "parameters": {"loggingSteps": 5}})
            await asyncio.gather(*runner._retained_callbacks)
            return outcome

        outcome = asyncio.run(scenario())
        self.assertEqual(outcome, "completed")
        self.assertEqual([task_id for task_id, _ in events], ["t1"])

    def test_on_retained_failure_does_not_fail_task(self) -> None:
        async def on_retained(task_id: str, task: dict) -> None:
            raise RuntimeError("预热失败")

        sink = FakeSink()

        async def scenario() -> str:
            runner = self.make_runner(sink, on_retained=on_retained)
            outcome = await runner.run("t1", {"taskId": "t1", "parameters": {}})
            await asyncio.gather(*runner._retained_callbacks)
            return outcome

        outcome = asyncio.run(scenario())
        self.assertEqual(outcome, "completed")
        self.assertTrue(any("预热失败" in log for log in sink.logs))


class PrecisionSelectionTest(unittest.TestCase):
    def test_known_and_unknown_gpu_families(self) -> None:
        self.assertTrue(_gpu_supports_bf16("RTX 4090"))
        self.assertTrue(_gpu_supports_bf16("A100"))
        self.assertFalse(_gpu_supports_bf16("V100S"))
        self.assertFalse(_gpu_supports_bf16("future-gpu"))


class FailurePathTest(RunnerTestBase):
    def test_create_instance_failure_is_reported_at_creation_stage(self) -> None:
        from backend.providers.youyun.models import YouyunApiError

        class CreateFailureManager(FakeManager):
            def create_one_shot(self, *args, **kwargs):
                raise YouyunApiError("CreateCompShareInstance", 226645, "Service unavailable")

        self.manager = CreateFailureManager()
        sink = FakeSink()
        outcome = asyncio.run(self.make_runner(sink).run("create-failure", {"taskId": "create-failure"}))

        self.assertEqual(outcome, "failed")
        self.assertTrue(any("优云实例创建失败" in log for log in sink.logs))
        self.assertTrue(any(update.get("current_step") == "优云实例创建失败" for update in sink.updates))
        self.assertTrue(any("RetCode=226645" in update.get("error", "") for update in sink.updates))

    def test_dead_install_process_is_restarted(self) -> None:
        class DeadInstallSSH(FakeSSH):
            def __init__(self) -> None:
                super().__init__()
                self.install_launch_count = 0

            def run(self, command, timeout=120, check=True):
                if "nohup" in command and "env-install" in command:
                    self.commands.append(command)
                    self.install_launch_count += 1
                    return 0, ""
                if "env-install.done" in command and "cat" in command:
                    self.commands.append(command)
                    if self.install_launch_count == 1:
                        return 0, "---\n\n__XUNTUI_INSTALL_STATUS__=dead\n"
                    return 0, "ok\n---\nlast install line\n__XUNTUI_INSTALL_STATUS__=missing\n"
                return super().run(command, timeout, check)

        self.ssh = DeadInstallSSH()
        sink = FakeSink()
        runner = self.make_runner(sink)
        runner.install_timeout = 10

        outcome = asyncio.run(runner.run("install-retry", {"taskId": "install-retry"}))

        self.assertEqual(outcome, "completed")
        self.assertEqual(self.ssh.install_launch_count, 2)
        self.assertTrue(any("安装进程已退出" in log for log in sink.logs))

    def test_ssh_banner_failure_retries_and_then_succeeds(self) -> None:
        class FlakySSH(FakeSSH):
            attempts = 0

            def connect(self, host, port, user, password) -> None:
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("Error reading SSH protocol banner")

        self.ssh = FlakySSH()
        sink = FakeSink()
        runner = self.make_runner(sink)
        connected = runner._connect_ssh(self.manager.create_one_shot("ssh-banner"), "ssh-banner")
        self.assertIs(connected, self.ssh)
        self.assertEqual(self.ssh.attempts, 2)
        self.assertTrue(any("协议握手未完成" in log for log in sink.logs))
        self.assertTrue(any("SSH 连接已建立" in log for log in sink.logs))

    def test_ssh_auth_failure_does_not_retry(self) -> None:
        class AuthSSH(FakeSSH):
            attempts = 0

            def connect(self, host, port, user, password) -> None:
                self.attempts += 1
                raise RuntimeError("Authentication failed")

        self.ssh = AuthSSH()
        sink = FakeSink()
        runner = self.make_runner(sink)
        with self.assertRaisesRegex(RuntimeError, "认证失败"):
            runner._connect_ssh(self.manager.create_one_shot("ssh-auth"), "ssh-auth")
        self.assertEqual(self.ssh.attempts, 1)

    def test_ssh_retries_with_refreshed_port(self) -> None:
        class RefreshManager(FakeManager):
            def __init__(self) -> None:
                super().__init__()
                self.refreshes = 0

            def get(self, uhost_id):
                self.refreshes += 1
                instance = self.create_one_shot(uhost_id)
                if self.refreshes >= 2:
                    instance.ssh_port = 23
                return instance

        class PortSSH(FakeSSH):
            ports: list[int] = []

            def connect(self, host, port, user, password) -> None:
                self.ports.append(port)
                if port != 23:
                    raise TimeoutError("timed out")

        self.manager = RefreshManager()
        self.ssh = PortSSH()
        sink = FakeSink()
        runner = self.make_runner(sink)
        runner._connect_ssh(self.manager.create_one_shot("ssh-refresh"), "ssh-refresh")
        self.assertEqual(self.ssh.ports, [60022, 23])
        self.assertGreaterEqual(self.manager.refreshes, 2)

    def test_install_failure_cleans_instance(self) -> None:
        self.ssh = FakeSSH(install_polls=0, install_fail=True)
        sink = FakeSink()
        outcome = asyncio.run(self.make_runner(sink).run("t2", {"taskId": "t2"}))
        self.assertEqual(outcome, "failed")
        self.assertTrue(any(u.get("state") == "failed" for u in sink.updates))
        self.assertEqual(self.manager.cleaned, ["uhost-9"])

    def test_install_command_removes_deepspeed_when_nvcc_is_missing(self) -> None:
        self.assertIn("command -v nvcc", _INSTALL_COMMAND)
        self.assertIn("pip uninstall -y deepspeed", _INSTALL_COMMAND)
        self.assertIn("deepspeed detected without nvcc", _INSTALL_COMMAND)

    def test_nonzero_train_exit_returns_failed(self) -> None:
        class FailSSH(FakeSSH):
            def run(self, command, timeout=120, check=True):
                if command.startswith("echo code:"):
                    self.train_poll_count += 1
                    if self.train_poll_count > len(LOG_LINES):
                        return 0, "code:1\nRuntimeError: OOM\n"
                    return 0, f"code:none\n{LOG_LINES[self.train_poll_count - 1]}\n"
                return super().run(command, timeout, check)

        self.ssh = FailSSH()
        sink = FakeSink()
        outcome = asyncio.run(self.make_runner(sink).run("t3", {"taskId": "t3"}))
        self.assertEqual(outcome, "failed")
        self.assertTrue(any(u.get("state") == "failed" for u in sink.updates))
        self.assertEqual(self.manager.cleaned, ["uhost-9"])

    def test_cleanup_failure_recorded_not_raised(self) -> None:
        self.manager.cleanup_error = "账户欠费"
        self.ssh = FakeSSH(install_polls=0, install_fail=True)
        sink = FakeSink()
        outcome = asyncio.run(self.make_runner(sink).run("t4", {"taskId": "t4"}))
        self.assertEqual(outcome, "failed")
        self.assertIn("清理未完成：账户欠费", "\n".join(sink.logs))
        marker = next(item for item in sink.artifacts if item.get("type") == "youyun-instance")
        self.assertEqual(marker["state"], "cleanup_failed")
        self.assertEqual(marker["cleanupError"], "账户欠费")


class StopPathTest(RunnerTestBase):
    def test_manual_stop_retains_instance_when_adapter_was_saved(self) -> None:
        sink = FakeSink(cancel_after_logs=6)
        outcome = asyncio.run(self.make_runner(sink).run("t5", {"taskId": "t5", "parameters": {}}))
        self.assertEqual(outcome, "stopped")
        self.assertTrue(any(u.get("state") == "stopped" for u in sink.updates))
        self.assertEqual(self.manager.cleaned, [])
        marker = next(item for item in sink.artifacts if item.get("type") == "youyun-instance")
        self.assertEqual(marker["state"], "awaiting_comparison")
        self.assertTrue(marker["comparisonReady"])

    def test_manual_stop_during_install_does_not_start_training(self) -> None:
        self.ssh = FakeSSH(install_polls=20)
        sink = FakeSink(cancel_after_logs=3)
        outcome = asyncio.run(self.make_runner(sink).run("t6", {"taskId": "t6", "parameters": {}}))
        self.assertEqual(outcome, "stopped")
        self.assertTrue(any(u.get("current_step") == "正在停止远程环境安装" for u in sink.updates))
        self.assertTrue(any("/tmp/env-install.pid" in command for command in self.ssh.commands))
        self.assertEqual(self.ssh.uploaded, {})
        self.assertEqual(self.manager.cleaned, ["uhost-9"])


if __name__ == "__main__":
    unittest.main()
