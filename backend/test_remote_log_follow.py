"""CPU-only contract tests: real remote helper over a local subprocess transport."""
from __future__ import annotations
import asyncio
import shlex
import subprocess
import tempfile
import sys
import threading
import time
import unittest
from pathlib import Path

from backend.providers.youyun.log_follow import RemoteLogCollector

class LocalLogSSH:
    def __init__(self):
        self.process = None
        self.closed = False
        self.lock = threading.Lock()

    def follow_log(self, command, stop):
        self.process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while not stop.is_set():
                line = self.process.stdout.readline()
                if not line:
                    code = self.process.wait(timeout=5)
                    if code:
                        raise RuntimeError(self.process.stderr.read().decode())
                    return
                yield line
        finally:
            self.close()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                self.process.wait(timeout=5)
                for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                    pipe.close()

class MemoryLogSSH:
    """Framed remote transport double; source is append-only bytes."""
    def __init__(self, source=lambda: b''):
        self.source = source
        self.closed = False

    def connect(self, **kwargs):
        pass

    def close(self):
        self.closed = True

    def follow_log(self, command, stop):
        import base64
        import json
        args = shlex.split(command)
        offset, follow = int(args[-3]), args[-1] == 'follow'
        yield b'{"type":"ready"}\n'
        while not stop.is_set() and not self.closed:
            data = self.source()[offset:offset + 65536]
            if data:
                yield json.dumps(dict(type='data', fileId='fake:1', start=offset,
                    end=offset + len(data), data=base64.b64encode(data).decode())).encode()
                offset += len(data)
            elif not follow:
                yield b'{"type":"eof"}\n'
                return
            else:
                stop.wait(.01)


async def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        await asyncio.sleep(.01)

class CollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_lines_arrive_before_control_poll_and_final_partial_line_is_drained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.log'
            received, connections = [], []
            def connect():
                ssh = LocalLogSSH()
                connections.append(ssh)
                return ssh
            collector = RemoteLogCollector(connect, str(path), on_line=received.append, python=sys.executable)
            collector.start()
            try:
                path.write_bytes('first\n中文'.encode())
                await eventually(lambda: received == ['first'], timeout=1)
                with path.open('ab') as file:
                    file.write('日志\rprogress\r\nlast'.encode())
                await eventually(lambda: received == ['first', '中文日志', 'progress'], timeout=1)
                await collector.close(drain=True)
                self.assertEqual(received, ['first', '中文日志', 'progress', 'last'])
                self.assertEqual(collector.cursor.offset, path.stat().st_size)
                self.assertTrue(all(ssh.closed for ssh in connections))
            finally:
                await collector.close()
    async def test_backlog_over_256kb_is_not_skipped_in_polling_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vllm.log'
            expected = ['%06d:%s' % (n, 'x' * 120) for n in range(3000)]
            path.write_text('\n'.join(expected) + '\n', encoding='utf-8')
            received = []
            collector = RemoteLogCollector(LocalLogSSH, str(path), on_line=received.append,
                                           python=sys.executable, enabled=False)
            collector.start()
            try:
                await eventually(lambda: len(received) == 3000, timeout=5)
                self.assertEqual(received, expected)
            finally:
                await collector.close()

    async def test_restart_replays_partial_utf8_but_not_committed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vllm.log'
            path.write_bytes(b'first\n\xe4\xb8')
            received = []
            collector = RemoteLogCollector(LocalLogSSH, str(path), on_line=received.append, python=sys.executable)
            collector.start()
            await eventually(lambda: received == ['first'])
            await collector.close()
            cursor = collector.cursor
            self.assertEqual(cursor.offset, 6)
            with path.open('ab') as file:
                file.write(b'\xad\n')
            resumed = RemoteLogCollector(LocalLogSSH, str(path), on_line=received.append,
                                         cursor=cursor, python=sys.executable)
            resumed.start()
            try:
                await eventually(lambda: len(received) == 2)
                self.assertEqual(received, ['first', '中'])
            finally:
                await resumed.close()

    async def test_truncation_restarts_cursor_and_reports_possible_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vllm.log'
            path.write_bytes(b'long original line\n')
            received, diagnostics = [], []
            collector = RemoteLogCollector(LocalLogSSH, str(path), on_line=received.append,
                                           on_status=diagnostics.append, python=sys.executable)
            collector.start()
            try:
                await eventually(lambda: len(received) == 1)
                path.write_bytes(b'new\n')
                await eventually(lambda: len(received) == 2)
                self.assertEqual(received, ['long original line', 'new'])
                self.assertTrue(any('重置' in line for line in diagnostics))
            finally:
                await collector.close()

    async def test_repeated_disconnect_degrades_and_recovers_without_duplicates(self):
        calls, diagnostics, received = [], [], []
        class FlakySSH(LocalLogSSH):
            def follow_log(self, command, stop):
                calls.append(shlex.split(command)[-1])
                if len(calls) <= 3:
                    raise OSError('disconnected')
                yield from super().follow_log(command, stop)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vllm.log'
            path.write_bytes(b'one\n')
            collector = RemoteLogCollector(FlakySSH, str(path), on_line=received.append,
                on_status=diagnostics.append, python=sys.executable,
                retry_interval=.01, poll_interval=.02, recovery_interval=.1)
            collector.start()
            try:
                await eventually(lambda: any('恢复 SSH' in line for line in diagnostics), timeout=3)
                with path.open('ab') as file:
                    file.write(b'two\n')
                await eventually(lambda: received == ['one', 'two'])
                self.assertIn('snapshot', calls)
                self.assertTrue(any('降级' in line for line in diagnostics))
            finally:
                await collector.close()

    async def test_failed_consumer_does_not_advance_past_failed_line(self):
        received = []
        fail_once = True
        def consume(line):
            nonlocal fail_once
            if line == 'two' and fail_once:
                fail_once = False
                raise RuntimeError('temporary store failure')
            received.append(line)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vllm.log'
            path.write_bytes(b'one\ntwo\nthree\n')
            collector = RemoteLogCollector(LocalLogSSH, str(path), on_line=consume,
                python=sys.executable, retry_interval=.01)
            collector.start()
            try:
                await eventually(lambda: len(received) == 3)
                self.assertEqual(received, ['one', 'two', 'three'])
            finally:
                await collector.close()
    async def test_missing_file_waits_and_can_be_cancelled_while_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'not-yet-created.log'
            received, connections = [], []
            def connect():
                ssh = LocalLogSSH()
                connections.append(ssh)
                return ssh
            collector = RemoteLogCollector(connect, str(path), python=sys.executable, on_line=received.append)
            collector.start()
            await asyncio.sleep(.25)
            self.assertEqual(received, [])
            self.assertFalse(collector.task.done())
            path.write_bytes(b'created later\n')
            await eventually(lambda: received == ['created later'], timeout=1)
            await collector.close()
            self.assertTrue(all(ssh.closed and ssh.process.poll() is not None for ssh in connections))

    async def test_replaced_file_after_disconnect_resets_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vllm.log'
            path.write_bytes(b'old\n')
            received, diagnostics = [], []
            first = RemoteLogCollector(LocalLogSSH, str(path), python=sys.executable, on_line=received.append)
            first.start()
            await eventually(lambda: received == ['old'])
            await first.close()
            path.rename(path.with_suffix('.old'))
            path.write_bytes(b'new\n')
            second = RemoteLogCollector(LocalLogSSH, str(path), python=sys.executable,
                cursor=first.cursor, on_line=received.append, on_status=diagnostics.append)
            second.start()
            try:
                await eventually(lambda: received == ['old', 'new'])
                self.assertNotEqual(second.cursor.file_id, first.cursor.file_id)
                self.assertTrue(any('重置' in line for line in diagnostics))
            finally:
                await second.close()

    async def test_slow_consumer_applies_bounded_backpressure_without_losing_frames(self):
        data = (b'x' * 65535 + b'\n') * 64
        produced, consumed = 0, 0
        class CountingSSH(MemoryLogSSH):
            def follow_log(self, command, stop):
                nonlocal produced
                for frame in super().follow_log(command, stop):
                    if b'"type": "data"' in frame:
                        produced += 1
                    yield frame
        def consume(line):
            nonlocal consumed
            self.assertEqual(len(line), 65535)
            if consumed == 0:
                time.sleep(.05)  # the SSH reader keeps running on its own thread
                self.assertLess(produced, 64)
            consumed += 1
        collector = RemoteLogCollector(lambda: CountingSSH(lambda: data), '/fake', on_line=consume)
        collector.start()
        try:
            await eventually(lambda: consumed == 64, timeout=5)
        finally:
            await collector.close()

    async def test_remote_helper_exits_when_parent_closes_stdin_even_without_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = RemoteLogCollector(LocalLogSSH, str(Path(directory) / 'missing'),
                python=sys.executable, on_line=lambda _: None)
            process = subprocess.Popen(shlex.split(collector._command(True)), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertIn(b'ready', await asyncio.to_thread(process.stdout.readline))
                process.stdin.close()
                self.assertEqual(await asyncio.to_thread(process.wait, 2), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
                process.stderr.close()

class SSHLogTransportTests(unittest.TestCase):
    def test_log_transport_is_independent_and_idle_follower_is_cancellable(self):
        from types import SimpleNamespace
        from backend.providers.youyun.ssh import YouyunSSHClient

        class Channel:
            closed = False
            def recv_ready(self):
                return False
            def recv_stderr_ready(self):
                return False
            def exit_status_ready(self):
                return False
            def close(self):
                self.closed = True

        class Pipe:
            def __init__(self, channel):
                self.channel = channel
                self.closed = False
            def close(self):
                self.closed = True

        class Session:
            def __init__(self):
                self.channel = Channel()
                self.pipes = [Pipe(self.channel) for _ in range(3)]
                self.closed = False
            def get_transport(self):
                return self
            def is_active(self):
                return not self.closed
            def exec_command(self, command, timeout):
                opened.set()
                return self.pipes
            def close(self):
                self.closed = True
                self.channel.close()

        class Factory:
            def __init__(self):
                self.sessions = []
            def connect(self, **kwargs):
                session = Session()
                self.sessions.append(session)
                return session

        factory, opened, stop = Factory(), threading.Event(), threading.Event()
        control = YouyunSSHClient(transport=factory)
        control.connect('example.invalid', 22, 'test', 'not-a-real-password')
        logs = control.log_connection()
        thread = threading.Thread(target=lambda: list(logs.follow_log('fake helper', stop)))
        try:
            thread.start()
            self.assertTrue(opened.wait(1))
            self.assertFalse(factory.sessions[1].pipes[0].closed, 'stdin must remain open')
            stop.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            logs.close()
            self.assertTrue(factory.sessions[1].closed)
            self.assertTrue(all(pipe.closed for pipe in factory.sessions[1].pipes))
            self.assertFalse(factory.sessions[0].closed, 'log close must not kill the control transport')
        finally:
            stop.set()
            logs.close()
            thread.join(1)
            control.close()
