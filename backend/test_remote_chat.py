from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TEST_RUNTIME = tempfile.TemporaryDirectory()
TEST_ROOT = Path(TEST_RUNTIME.name)
os.environ["XUNTUI_STORAGE_BACKEND"] = "sqlite"
os.environ["XUNTUI_DATABASE_PATH"] = str(TEST_ROOT / "training.db")
os.environ["XUNTUI_RUNTIME_ROOT"] = str(TEST_ROOT / "runtime")
os.environ["XUNTUI_DATASET_ROOT"] = str(TEST_ROOT / "datasets")
os.environ["XUNTUI_ARTIFACT_ROOT"] = str(TEST_ROOT / "artifacts")

from backend import app as app_module
from backend.providers.youyun.remote_chat import (
    EVENT_PREFIX,
    RemoteComparisonError,
    RemoteComparisonManager,
    remote_adapter_path,
)


class RemoteAdapterPathTest(unittest.TestCase):
    def test_maps_mirror_output_to_remote_root(self) -> None:
        self.assertEqual(
            remote_adapter_path(
                "/workspace/runtime/remote-jobs/ft-1/output", "ft-1", "/root/cppt"
            ),
            "/root/cppt/ft-1/output",
        )

    def test_maps_checkpoint_subdir(self) -> None:
        self.assertEqual(
            remote_adapter_path(
                "/workspace/runtime/remote-jobs/ft-1/output/checkpoint-6",
                "ft-1",
                "/root/cppt",
            ),
            "/root/cppt/ft-1/output/checkpoint-6",
        )

    def test_windows_style_local_path_is_normalized(self) -> None:
        self.assertEqual(
            remote_adapter_path(
                "C:\\workspace\\runtime\\remote-jobs\\ft-1\\output", "ft-1", "/root/cppt"
            ),
            "/root/cppt/ft-1/output",
        )

    def test_mismatched_task_is_rejected(self) -> None:
        with self.assertRaises(RemoteComparisonError):
            remote_adapter_path("/workspace/other-task/output", "ft-1", "/root/cppt")


class FakeChannel:
    def __init__(self, chunks: list[bytes], *, exit_when_drained: bool = True) -> None:
        self._chunks = list(chunks)
        self.closed = False
        self.exit_when_drained = exit_when_drained

    def recv_ready(self) -> bool:
        return bool(self._chunks)

    def recv(self, size: int) -> bytes:
        return self._chunks.pop(0)

    def exit_status_ready(self) -> bool:
        return self.exit_when_drained and not self._chunks


class FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.flushed = False

    def write(self, data: bytes) -> None:
        self.data += data

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        pass


class YouyunSSHSessionTest(unittest.TestCase):
    def make_session(self, chunks: list[bytes], *, exit_when_drained: bool = True):
        from backend.providers.youyun.ssh import YouyunSSHSession

        channel = FakeChannel(chunks, exit_when_drained=exit_when_drained)
        stdin = FakeStdin()
        session = YouyunSSHSession(stdin, SimpleNamespace(channel=channel))
        return session, stdin, channel

    def test_readline_buffers_partial_chunks(self) -> None:
        session, _, _ = self.make_session([b"he", b"llo\nwor", b"ld\n"])
        self.assertEqual(session.readline(timeout=1.0), "hello")
        self.assertEqual(session.readline(timeout=1.0), "world")

    def test_readline_returns_trailing_partial_then_reports_exit(self) -> None:
        session, _, _ = self.make_session([b"tail-bytes"])
        self.assertEqual(session.readline(timeout=1.0), "tail-bytes")
        with self.assertRaises(RuntimeError):
            session.readline(timeout=1.0)

    def test_readline_times_out_when_remote_silent(self) -> None:
        session, _, _ = self.make_session([], exit_when_drained=False)
        with self.assertRaises(TimeoutError):
            session.readline(timeout=0.2)

    def test_write_line_sends_framed_request(self) -> None:
        session, stdin, _ = self.make_session([])
        session.write_line('{"a": 1}')
        self.assertEqual(stdin.data, b'{"a": 1}\n')
        self.assertTrue(stdin.flushed)

    def test_write_line_on_closed_session_raises(self) -> None:
        session, _, channel = self.make_session([])
        channel.closed = True
        with self.assertRaises(RuntimeError):
            session.write_line("{}")


class FakeSession:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = list(lines or [])
        self.sent: list[str] = []
        self.closed = False
        self.fail_writes = 0

    @property
    def alive(self) -> bool:
        return not self.closed

    def write_line(self, text: str) -> None:
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise RuntimeError("SSH 会话写入失败。")
        self.sent.append(text)

    def readline(self, timeout: float) -> str:
        if self.lines:
            return self.lines.pop(0)
        if self.sent:
            request = json.loads(self.sent[-1])
            return (
                EVENT_PREFIX
                + json.dumps(
                    {
                        "type": "response",
                        "requestId": request["requestId"],
                        "text": "gpu answer",
                        "elapsedSeconds": 0.4,
                        "outputTokens": 10,
                    }
                )
            )
        raise AssertionError("no scripted lines and nothing sent")

    def close(self) -> None:
        self.closed = True


class FakeSSH:
    def __init__(self) -> None:
        self.connected: tuple | None = None
        self.uploads: list[tuple[str, str]] = []
        self.launched: list[tuple[str, FakeSession]] = []
        self.closed = False
        self.session_lines: list[str] = [EVENT_PREFIX + json.dumps({"type": "ready"})]

    def connect(self, host, port, user, password) -> None:
        self.connected = (host, port, user, password)

    def close(self) -> None:
        self.closed = True

    def put_file(self, local_path, remote_path) -> None:
        self.uploads.append((str(local_path), remote_path))

    def run(self, command: str, timeout: float = 120.0, check: bool = True):
        if "preflight-done" in command:
            return 0, "preflight-done"
        if "echo \"CACHE=" in command or "CACHE=$CACHE" in command:
            return 0, "CACHE=/root/.cache/modelscope"
        return 0, ""

    def open_session(self, command: str, timeout: float = 600.0) -> FakeSession:
        session = FakeSession(list(self.session_lines))
        self.launched.append((command, session))
        return session


class FakeYouyunManager:
    def __init__(self) -> None:
        self.client = SimpleNamespace(
            describe_instance=lambda instance_id: SimpleNamespace(uhost_id=instance_id)
        )

    def ssh_credentials(self, instance) -> dict:
        return {"host": "1.2.3.4", "port": 60022, "user": "root", "password": "pw"}


def build_manager(ssh: FakeSSH, **overrides) -> RemoteComparisonManager:
    script = TEST_ROOT / "chat_inference.py"
    script.write_text("# chat script\n", encoding="utf-8")
    options = dict(
        manager=FakeYouyunManager(),
        ssh_factory=lambda connect_timeout=20.0: ssh,
        remote_root="/root/cppt",
        ready_timeout=5.0,
        request_timeout=5.0,
        chat_script=script,
    )
    options.update(overrides)
    return RemoteComparisonManager(**options)


def build_task(marker_state: str = "awaiting_comparison", public_model_path: str = "") -> dict:
    marker = {
        "type": "youyun-instance",
        "provider": "youyun",
        "instanceId": "uhost-1",
        "state": marker_state,
    }
    task = {
        "id": "ft-remote",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "artifacts": [marker],
    }
    if public_model_path:
        task["modelSource"] = "public"
        task["youyunResource"] = {"publicModelPath": public_model_path}
    return task


class RemoteComparisonManagerTest(unittest.IsolatedAsyncioTestCase):
    def adapter_path(self) -> str:
        return "/workspace/runtime/remote-jobs/ft-remote/output"

    async def test_compare_launches_remote_process_and_returns_response(self) -> None:
        ssh = FakeSSH()
        manager = build_manager(ssh)
        response = await manager.compare(
            build_task(),
            self.adapter_path(),
            messages=[{"role": "user", "content": "影子晚上去哪儿了？"}],
            parameters={"maxNewTokens": 64, "temperature": 0, "topP": 0.9, "seed": 42},
            adapter_enabled=True,
            load_in_4bit=False,
        )
        self.assertEqual(response["text"], "gpu answer")
        self.assertEqual(len(ssh.launched), 1)
        command = ssh.launched[0][0]
        self.assertIn("/root/cppt/ft-remote/output", command)
        self.assertIn("MODELSCOPE_CACHE=/root/.cache/modelscope", command)
        self.assertIn("USE_MODELSCOPE_HUB=1", command)
        request = json.loads(ssh.launched[0][1].sent[0])
        self.assertTrue(request["adapterEnabled"])
        self.assertEqual(request["messages"][0]["content"], "影子晚上去哪儿了？")
        # 会话保留供后续请求复用
        self.assertIn("ft-remote", manager._handles)

    async def test_compare_prefers_saved_public_model_path(self) -> None:
        class PublicModelSSH(FakeSSH):
            def run(self, command: str, timeout: float = 120.0, check: bool = True):
                if "preflight-done" in command:
                    return 0, "preflight-done"
                if "PUBLIC_MODEL_PATH=" in command:
                    return 0, "PUBLIC_MODEL=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct"
                return super().run(command, timeout, check)

        ssh = PublicModelSSH()
        manager = build_manager(ssh)
        public_path = "/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct"
        await manager.compare(
            build_task(public_model_path=public_path),
            self.adapter_path(),
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            adapter_enabled=False,
            load_in_4bit=False,
        )
        command = ssh.launched[0][0]
        self.assertIn(f"--model {public_path}", command)
        self.assertIn("--source local", command)
        self.assertNotIn("--source modelscope", command)

    async def test_missing_original_public_path_does_not_replace_base_model(self):
        manager = build_manager(FakeSSH())
        with self.assertRaisesRegex(RemoteComparisonError, "不能替换"):
            manager._resolve_model_cache(FakeSSH(), "org/original", "/models/original")

    async def test_remote_repo_does_not_search_public_models_by_name(self):
        ssh = FakeSSH()
        manager = build_manager(ssh)
        commands = []
        original = ssh.run
        def record(command, *args, **kwargs):
            commands.append(command)
            return original(command, *args, **kwargs)
        ssh.run = record
        manager._resolve_model_cache(ssh, "org/original")
        self.assertFalse(any("for root in" in command for command in commands))

    async def test_released_marker_is_rejected(self) -> None:
        manager = build_manager(FakeSSH())
        with self.assertRaisesRegex(RemoteComparisonError, "已释放"):
            await manager.compare(
                build_task(marker_state="released"),
                self.adapter_path(),
                messages=[{"role": "user", "content": "hi"}],
                parameters={},
                adapter_enabled=True,
                load_in_4bit=False,
            )

    async def test_missing_instance_marker_is_rejected(self) -> None:
        manager = build_manager(FakeSSH())
        task = build_task()
        task["artifacts"] = []
        with self.assertRaisesRegex(RemoteComparisonError, "实例记录"):
            await manager.compare(
                task,
                self.adapter_path(),
                messages=[{"role": "user", "content": "hi"}],
                parameters={},
                adapter_enabled=True,
                load_in_4bit=False,
            )

    async def test_dead_session_is_rebuilt_once_and_request_succeeds(self) -> None:
        ssh = FakeSSH()
        manager = build_manager(ssh)
        first = await manager.compare(
            build_task(),
            self.adapter_path(),
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            adapter_enabled=True,
            load_in_4bit=False,
        )
        self.assertEqual(first["text"], "gpu answer")
        # 模拟会话中断：下次写入失败一次，应自动重建并成功。
        handle = manager._handles["ft-remote"]
        handle.session.fail_writes = 1
        second = await manager.compare(
            build_task(),
            self.adapter_path(),
            messages=[{"role": "user", "content": "hi again"}],
            parameters={},
            adapter_enabled=True,
            load_in_4bit=False,
        )
        self.assertEqual(second["text"], "gpu answer")
        self.assertEqual(len(ssh.launched), 2)

    async def test_adapter_variant_change_rebuilds_session(self) -> None:
        ssh = FakeSSH()
        manager = build_manager(ssh)
        await manager.compare(
            build_task(),
            self.adapter_path(),
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            adapter_enabled=True,
            load_in_4bit=False,
        )
        await manager.compare(
            build_task(),
            self.adapter_path() + "/checkpoint-6",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            adapter_enabled=True,
            load_in_4bit=False,
        )
        self.assertEqual(len(ssh.launched), 2)
        self.assertIn("/root/cppt/ft-remote/output/checkpoint-6", ssh.launched[1][0])

    async def test_warmup_builds_session_and_reuses_it(self) -> None:
        ssh = FakeSSH()
        manager = build_manager(ssh)
        ready = await manager.warmup(build_task(), self.adapter_path(), load_in_4bit=False)
        self.assertTrue(ready)
        self.assertEqual(len(ssh.launched), 1)
        ready_again = await manager.warmup(build_task(), self.adapter_path(), load_in_4bit=False)
        self.assertTrue(ready_again)
        self.assertEqual(len(ssh.launched), 1)  # 会话复用，未重建

    async def test_warmup_failure_raises_for_caller(self) -> None:
        class BrokenSSH(FakeSSH):
            def connect(self, host, port, user, password):
                raise OSError("network unreachable")

        manager = build_manager(BrokenSSH(), ssh_ready_timeout=0.1)
        with self.assertRaisesRegex(RemoteComparisonError, "无法连接"):
            await manager.warmup(build_task(), self.adapter_path(), load_in_4bit=False)

    async def test_warmup_ignores_non_retained_tasks(self) -> None:
        ssh = FakeSSH()
        manager = build_manager(ssh)
        ready = await manager.warmup(
            build_task(marker_state="released"), self.adapter_path(), load_in_4bit=False
        )
        self.assertFalse(ready)
        self.assertEqual(len(ssh.launched), 0)

    async def test_compare_while_warming_returns_busy_immediately(self) -> None:
        manager = build_manager(FakeSSH())
        lock = manager._task_lock("ft-remote")
        await lock.acquire()
        try:
            with self.assertRaisesRegex(RemoteComparisonError, "正在预热"):
                await manager.compare(
                    build_task(),
                    self.adapter_path(),
                    messages=[{"role": "user", "content": "hi"}],
                    parameters={},
                    adapter_enabled=True,
                    load_in_4bit=False,
                )
        finally:
            lock.release()

    async def test_read_failure_drops_handle(self) -> None:
        ssh = FakeSSH()
        manager = build_manager(ssh)
        await manager.compare(
            build_task(),
            self.adapter_path(),
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            adapter_enabled=True,
            load_in_4bit=False,
        )
        handle = manager._handles["ft-remote"]
        handle.session.lines.append(EVENT_PREFIX + json.dumps({"type": "error", "message": "boom"}))
        handle.session.sent.clear()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await manager.compare(
                build_task(),
                self.adapter_path(),
                messages=[{"role": "user", "content": "hi"}],
                parameters={},
                adapter_enabled=True,
                load_in_4bit=False,
            )
        self.assertNotIn("ft-remote", manager._handles)


def create_youyun_task(marker_state: str = "awaiting_comparison") -> dict:
    task = app_module.store.create(
        {
            "name": "youyun compare",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "modelSource": "modelscope",
            "tuningMode": "lora",
            "datasetId": "qwenruozhi_v1",
            "datasetName": "dataset",
            "parameters": {"epochs": 1},
            "provider": "youyun",
            "executionMode": "remote",
        }
    )
    adapter_path = app_module.youyun_mirror_executor.work_root / task["id"] / "output"
    adapter_path.mkdir(parents=True)
    app_module.store.update(
        task["id"],
        state="completed",
        artifacts=[
            {
                "type": "adapter",
                "name": "final",
                "path": str(adapter_path),
                "variant": "final",
                "selected": True,
                "checkpointStep": 6,
            },
            {
                "type": "youyun-instance",
                "provider": "youyun",
                "instanceId": "uhost-test",
                "state": marker_state,
            },
        ],
    )
    return app_module.store.get(task["id"])


class YouyunChatRoutingTest(unittest.IsolatedAsyncioTestCase):
    def remote_manager_stub(self, calls: dict):
        class StubRemoteManager:
            async def compare(self, task, adapter_path, **kwargs):
                calls["task_id"] = task["id"]
                calls["adapter_path"] = adapter_path
                calls.update(kwargs)
                return {"text": "gpu answer", "elapsedSeconds": 0.4, "outputTokens": 10}

        return StubRemoteManager()

    async def test_adapter_mode_routes_to_remote_gpu(self) -> None:
        task = create_youyun_task()
        calls: dict = {}
        with patch.object(
            app_module, "get_remote_chat_manager", lambda: self.remote_manager_stub(calls)
        ):
            response = await app_module.chat(
                app_module.ChatRequest(
                    mode="adapter",
                    adapterVariant="final",
                    taskId=task["id"],
                    model=task["model"],
                    messages=[{"role": "user", "content": "hello"}],
                )
            )
        self.assertEqual(calls["task_id"], task["id"])
        self.assertTrue(calls["adapter_enabled"])
        self.assertFalse(calls["load_in_4bit"])
        self.assertEqual(response["adapterVariant"], "final")
        self.assertEqual(response["adapterStep"], 6)
        self.assertEqual(response["text"], "gpu answer")
        updated = app_module.store.get(task["id"])
        marker = next(
            item for item in updated["artifacts"] if item["type"] == "youyun-instance"
        )
        self.assertIn("retainedAt", marker)

    async def test_base_mode_disables_adapter_remotely(self) -> None:
        task = create_youyun_task()
        calls: dict = {}
        with patch.object(
            app_module, "get_remote_chat_manager", lambda: self.remote_manager_stub(calls)
        ):
            response = await app_module.chat(
                app_module.ChatRequest(
                    mode="base",
                    taskId=task["id"],
                    model=task["model"],
                    messages=[{"role": "user", "content": "hello"}],
                )
            )
        self.assertFalse(calls["adapter_enabled"])
        self.assertNotIn("adapterVariant", response)

    async def test_slow_compare_returns_503_with_retry_hint(self) -> None:
        task = create_youyun_task()

        class SlowStub:
            async def compare(self, *args, **kwargs):
                await asyncio.sleep(0.4)
                return {"text": "late", "elapsedSeconds": 0.4, "outputTokens": 1}

        with patch.object(app_module, "get_remote_chat_manager", lambda: SlowStub()), patch.object(
            app_module, "REMOTE_CHAT_WAIT_SECONDS", 0.05
        ):
            with self.assertRaises(app_module.HTTPException) as raised:
                await app_module.chat(
                    app_module.ChatRequest(
                        mode="adapter",
                        adapterVariant="final",
                        taskId=task["id"],
                        model=task["model"],
                        messages=[{"role": "user", "content": "hello"}],
                    )
                )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("预热", str(raised.exception.detail))

    async def test_released_instance_is_rejected_with_409(self) -> None:
        task = create_youyun_task(marker_state="released")
        with patch.object(
            app_module, "get_remote_chat_manager", lambda: self.remote_manager_stub({})
        ):
            with self.assertRaises(app_module.HTTPException) as raised:
                await app_module.chat(
                    app_module.ChatRequest(
                        mode="adapter",
                        adapterVariant="final",
                        taskId=task["id"],
                        model=task["model"],
                        messages=[{"role": "user", "content": "hello"}],
                    )
                )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("已释放", str(raised.exception.detail))


class _WarmupStore:
    def __init__(self, tasks: list[dict]) -> None:
        self.tasks = tasks
        self.messages: list[str] = []
        self.list_include_details: list[bool] = []

    def get(self, task_id: str) -> dict | None:
        return next((task for task in self.tasks if task["id"] == task_id), None)

    def list(self, *, include_details: bool = True) -> list[dict]:
        self.list_include_details.append(include_details)
        return list(self.tasks)

    def append_log(self, task_id: str, message: str, **kwargs: object) -> None:
        self.messages.append(message)


def warmup_task() -> dict:
    return {
        "id": "ft-warm",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "tuningMode": "lora",
        "state": "completed",
        "artifacts": [
            {
                "type": "adapter",
                "name": "final",
                "variant": "final",
                "selected": True,
                "path": "/workspace/runtime/remote-jobs/ft-warm/output",
                "checkpointStep": 6,
            },
            {
                "type": "youyun-instance",
                "provider": "youyun",
                "instanceId": "uhost-warm",
                "state": "awaiting_comparison",
            },
        ],
    }


class RetainedWarmupAppTest(unittest.IsolatedAsyncioTestCase):
    def stub_manager(self, calls: dict, *, fail: bool = False):
        class Stub:
            async def warmup(self, task, adapter_path, *, load_in_4bit):
                calls["task_id"] = task["id"]
                calls["adapter_path"] = adapter_path
                calls["load_in_4bit"] = load_in_4bit
                if fail:
                    raise RuntimeError("实例不可达")
                return True

        return Stub()

    async def test_warmup_success_logs_ready(self) -> None:
        task = warmup_task()
        store = _WarmupStore([task])
        calls: dict = {}
        with patch.object(app_module, "store", store), patch.object(
            app_module, "get_remote_chat_manager", lambda: self.stub_manager(calls)
        ):
            await app_module._warmup_retained_comparison(task["id"], task)
        self.assertEqual(calls["task_id"], "ft-warm")
        self.assertFalse(calls["load_in_4bit"])
        self.assertTrue(any("已就绪" in message for message in store.messages))

    async def test_warmup_failure_logs_reason(self) -> None:
        task = warmup_task()
        store = _WarmupStore([task])
        calls: dict = {}
        with patch.object(app_module, "store", store), patch.object(
            app_module,
            "get_remote_chat_manager",
            lambda: self.stub_manager(calls, fail=True),
        ):
            await app_module._warmup_retained_comparison(task["id"], task)
        self.assertTrue(any("预热失败" in message and "实例不可达" in message for message in store.messages))

    async def test_warmup_refetches_fresh_task_artifacts(self) -> None:
        stale = {**warmup_task(), "artifacts": []}
        store = _WarmupStore([warmup_task()])  # store 中已有写入 Adapter 的新任务
        calls: dict = {}
        with patch.object(app_module, "store", store), patch.object(
            app_module, "get_remote_chat_manager", lambda: self.stub_manager(calls)
        ):
            await app_module._warmup_retained_comparison(stale["id"], stale)
        self.assertEqual(calls["task_id"], "ft-warm")
        self.assertTrue(any("已就绪" in message for message in store.messages))

    async def test_warmup_logs_skip_without_adapter(self) -> None:
        stale = {**warmup_task(), "artifacts": []}
        store = _WarmupStore([stale])  # store 里同样没有 Adapter
        with patch.object(app_module, "store", store), patch.object(
            app_module, "get_remote_chat_manager", lambda: self.stub_manager({})
        ):
            await app_module._warmup_retained_comparison(stale["id"], stale)
        self.assertTrue(any("预热跳过" in message for message in store.messages))

    async def test_startup_scan_warms_retained_tasks(self) -> None:
        store = _WarmupStore([warmup_task(), {**warmup_task(), "id": "ft-warm2", "state": "stopped"}])
        calls: list[str] = []

        class RecordingStub:
            async def warmup(self, task, adapter_path, *, load_in_4bit):
                calls.append(task["id"])
                return True

        with patch.object(app_module, "store", store), patch.object(
            app_module, "get_remote_chat_manager", lambda: RecordingStub()
        ):
            await app_module.warmup_retained_comparisons()
            for _ in range(5):
                await asyncio.sleep(0)
        self.assertEqual(calls, ["ft-warm", "ft-warm2"])  # 保留实例的已停止任务也可用于效果对比
        self.assertEqual(store.list_include_details, [False])


if __name__ == "__main__":
    unittest.main()
