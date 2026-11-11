"""Regression for a saved remote checkpoint whose download fails on stop."""
import asyncio
import json
import os
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.test_youyun_remote_training import FakeSink, RunnerTestBase
from backend.providers.youyun.remote_training import _RemoteAdapterMissing


class ManualStopRetentionTest(RunnerTestBase):
    def stopped_runner(self, sink):
        runner = self.make_runner(sink)

        async def stopped(*args):
            runner._stopped = True
            return 130

        def failed_download(*args):
            raise FileNotFoundError(2, "No such file")

        runner._train_and_follow = stopped
        self.ssh.get_tree = failed_download
        return runner

    def test_download_failure_must_not_delete_saved_remote_adapter(self):
        sink = FakeSink()
        runner = self.stopped_runner(sink)
        runner._inspect_remote_artifacts = AsyncMock(return_value=[{
            "type": "adapter", "name": "final", "variant": "final",
            "path": str(self.mirror / "t-stop" / "output" / "checkpoint-700"),
            "remotePath": "/root/cppt/t-stop/output/checkpoint-700",
            "selected": True, "syncState": "remote_ready",
        }])
        outcome = asyncio.run(runner.run("t-stop", {"taskId": "t-stop"}))

        self.assertEqual(outcome, "stopped")
        self.assertEqual(self.manager.cleaned, [], "A download failure must not destroy remote weights")
        marker = next(a for a in sink.artifacts if a["type"] == "youyun-instance")
        self.assertEqual(marker["state"], "awaiting_comparison")
        self.assertTrue(any(a.get("remotePath", "").endswith("checkpoint-700") for a in sink.artifacts))

    def test_unknown_remote_state_keeps_instance_without_claiming_comparison_ready(self):
        sink = FakeSink()
        runner = self.stopped_runner(sink)
        runner._inspect_remote_artifacts = AsyncMock(side_effect=TimeoutError("SSH unavailable"))
        self.assertEqual(asyncio.run(runner.run("t-stop", {"taskId": "t-stop"})), "stopped")
        self.assertEqual(self.manager.cleaned, [])
        marker = next(a for a in sink.artifacts if a["type"] == "youyun-instance")
        self.assertFalse(marker["comparisonReady"])
        self.assertTrue(marker["retainedAt"])

    def test_confirmed_missing_remote_adapter_releases_instance(self):
        sink = FakeSink()
        runner = self.stopped_runner(sink)
        runner._inspect_remote_artifacts = AsyncMock(side_effect=_RemoteAdapterMissing("missing"))
        self.assertEqual(asyncio.run(runner.run("t-stop", {"taskId": "t-stop"})), "stopped")
        self.assertEqual(self.manager.cleaned, ["uhost-9"])


@unittest.skipUnless(shutil.which("wsl" if os.name == "nt" else "bash"), "POSIX shell required")
class RemoteCheckpointProbeTest(RunnerTestBase):
    def setUp(self):
        super().setUp()
        self.output = self.mirror / "output"
        self.output.mkdir()
        self.remote_root = str(self.mirror)
        if os.name == "nt":
            self.remote_root = subprocess.check_output(
                ["wsl", "--exec", "wslpath", "-u", self.mirror.as_posix()], text=True,
            ).strip()

    def checkpoint(self, step, *, weights=True):
        directory = self.output / f"checkpoint-{step}"
        directory.mkdir()
        (directory / "adapter_config.json").write_text("{}")
        if weights:
            (directory / "adapter_model.safetensors").write_bytes(b"weights")
        return directory

    def probe(self):
        class ShellSSH:
            def run(self, command, *args):
                argv = ["wsl", "--exec", "bash", "-s"] if os.name == "nt" else ["bash", "-s"]
                result = subprocess.run(argv, input=command, capture_output=True, text=True, timeout=10)
                return result.returncode, result.stdout

        return asyncio.run(self.make_runner(FakeSink())._inspect_remote_artifacts(
            "probe", {}, ShellSSH(), SimpleNamespace(workdir=str(self.mirror), remote_workdir=self.remote_root),
        ))

    def test_checkpoint_only_output_uses_latest_complete_weights(self):
        self.checkpoint(25)
        latest = self.checkpoint(700)
        self.checkpoint(725, weights=False)
        (latest / "trainer_state.json").write_text(json.dumps({"best_model_checkpoint": "checkpoint-25"}))
        artifacts = self.probe()
        self.assertTrue(next(a for a in artifacts if a["variant"] == "best")["remotePath"].endswith("checkpoint-25"))
        final = next(a for a in artifacts if a["variant"] == "final")
        self.assertTrue(final["remotePath"].endswith("checkpoint-700"))
        self.assertEqual(final["path"], str(latest))

    def test_optimizer_files_are_not_adapter_weights(self):
        checkpoint = self.checkpoint(25, weights=False)
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "training_args.bin").write_bytes(b"arguments")
        with self.assertRaises(_RemoteAdapterMissing):
            self.probe()

    def test_root_config_does_not_borrow_checkpoint_weights(self):
        (self.output / "adapter_config.json").write_text("{}")
        checkpoint = self.checkpoint(700)
        self.assertEqual(self.probe()[0]["path"], str(checkpoint))
