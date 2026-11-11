"""One lifecycle-owned remote log reader, independent of UI subscribers.

The reader owns its SSH transport; the asyncio consumer owns all store writes.
Only complete consumed lines advance the durable byte cursor. Reconnecting
replays an unfinished line. Store/cursor writes are not an atomic transaction.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import queue
import re
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_LOG = logging.getLogger(__name__)
_REMOTE_SCRIPT = Path(__file__).with_name('log_follow_remote.py').read_text(encoding='utf-8-sig')


@dataclass(frozen=True)
class LogCursor:
    offset: int = 0
    file_id: str = ''


class RemoteLogCollector:
    def __init__(self, connect: Callable[[], Any], path: str, *,
                 on_line: Callable[[str], None], python: str = 'python3',
                 cursor: LogCursor | None = None,
                 on_cursor: Callable[[LogCursor], None] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 enabled: bool | None = None, poll_interval: float = 5,
                 recovery_interval: float = 30, retry_interval: float = 1) -> None:
        self.connect, self.path, self.python = connect, path, python
        self.on_line, self.on_cursor, self.on_status = on_line, on_cursor, on_status
        self.cursor = cursor or LogCursor()
        self.enabled = (os.environ.get('XUNTUI_REMOTE_LOG_FOLLOW_ENABLED', 'true').lower()
                        not in {'0', 'false', 'no'}) if enabled is None else enabled
        self.poll_interval, self.recovery_interval = poll_interval, recovery_interval
        self.retry_interval = retry_interval
        self.task: asyncio.Task[None] | None = None
        self._pending = b''
        self._closing = False
        self._close_lock = asyncio.Lock()
        self._fallback = not self.enabled
        self._last_error = 0.0

    def start(self) -> None:
        if self.task is None and not self._closing:
            self.task = asyncio.create_task(self._run(), name='remote-log-follow')

    def _status(self, text: str) -> None:
        try:
            if self.on_status:
                self.on_status(text)
        except Exception:
            _LOG.warning('Could not persist remote log diagnostic', exc_info=True)

    def _error(self, exc: Exception) -> None:
        if time.monotonic() - self._last_error >= 30:
            self._last_error = time.monotonic()
            # Do not include SSH commands/credentials in user-visible diagnostics.
            self._status('远端日志采集暂时失败，保留游标重试（%s）。' % type(exc).__name__)

    def _commit(self, offset: int, file_id: str) -> None:
        value = LogCursor(offset, file_id)
        if self.on_cursor:
            self.on_cursor(value)
        self.cursor = value

    def _flush(self) -> None:
        if self._pending:
            self.on_line(self._pending.decode('utf-8', errors='replace'))
            self._commit(self.cursor.offset + len(self._pending), self.cursor.file_id)
            self._pending = b''

    def _consume(self, frame: dict[str, Any]) -> None:
        kind = frame['type']
        if kind == 'reset':
            self._flush()
            self._commit(int(frame['offset']), str(frame['fileId']))
            self._status('远端日志文件已截断或替换，重置读取位置；已删除文件中未采集的内容可能无法恢复。')
        elif kind == 'data':
            data = base64.b64decode(frame['data'], validate=True)
            start, end, identity = int(frame['start']), int(frame['end']), str(frame['fileId'])
            if len(data) > 65536 or end - start != len(data):
                raise ValueError('invalid remote log frame')
            if self.cursor.file_id and identity != self.cursor.file_id:
                raise ValueError('file changed without reset')
            if start != self.cursor.offset + len(self._pending):
                raise ValueError('noncontiguous remote log frame')
            self.cursor = LogCursor(self.cursor.offset, identity)
            self._pending += data
            # Split on bytes, not decoded chunks: a UTF-8 code point can span frames.
            while match := re.search(br'[\r\n]', self._pending):
                count = match.end()
                raw = self._pending[:match.start()]
                if raw:
                    self.on_line(raw.decode('utf-8', errors='replace'))
                self._commit(self.cursor.offset + count, identity)
                self._pending = self._pending[count:]
            # Bound pathological single-line buffering without silently losing bytes.
            if len(self._pending) >= 1024 * 1024:
                self._status('远端日志单行超过 1 MiB，按片段保存。')
                # Avoid splitting an incomplete UTF-8 suffix at this explicit boundary.
                import codecs
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                text = decoder.decode(self._pending)
                suffix, _ = decoder.getstate()
                consumed = len(self._pending) - len(suffix)
                self.on_line(text)
                self._commit(self.cursor.offset + consumed, identity)
                self._pending = suffix

    def _command(self, follow: bool) -> str:
        return ' '.join(shlex.quote(str(value)) for value in (
            self.python, '-u', '-c', _REMOTE_SCRIPT, self.path, self.cursor.offset,
            self.cursor.file_id, 'follow' if follow else 'snapshot'))

    async def _session(self, follow: bool) -> None:
        # Discard only uncommitted partial bytes. The remote command replays them.
        self._pending = b''
        command = self._command(follow)
        messages: queue.Queue[Any] = queue.Queue(maxsize=32)
        stop = threading.Event()
        transport: list[Any] = []
        completed = object()

        def put(value: Any) -> None:
            while not stop.is_set():
                try:
                    messages.put(value, timeout=.1)
                    return
                except queue.Full:
                    pass

        def worker() -> None:
            ssh = None
            try:
                ssh = self.connect()
                transport.append(ssh)
                if stop.is_set():
                    return
                for line in ssh.follow_log(command, stop):
                    if stop.is_set():
                        break
                    if len(line) > 100_000:
                        raise ValueError('remote log protocol line too large')
                    put(json.loads(line))
                put(completed)
            except Exception as exc:
                put(exc)
            finally:
                if ssh is not None:
                    ssh.close()

        thread = threading.Thread(target=worker, name='remote-log-reader', daemon=True)
        thread.start()
        saw_eof = False
        last_received = time.monotonic()
        try:
            while True:
                try:
                    frame = messages.get_nowait()
                except queue.Empty:
                    if time.monotonic() - last_received > 45:
                        raise TimeoutError('SSH log follower stopped responding')
                    await asyncio.sleep(.01)
                    continue
                last_received = time.monotonic()
                if frame is completed:
                    if follow or not saw_eof:
                        raise EOFError('remote log follower exited')
                    return
                if isinstance(frame, Exception):
                    raise frame
                if frame.get('type') == 'ready' and follow and self._fallback:
                    self._fallback = False
                    self._status('远端日志已恢复 SSH 持续采集。')
                if frame.get('type') == 'eof':
                    saw_eof = True
                self._consume(frame)
                # Yield even while a backlog keeps the bounded queue nonempty.
                await asyncio.sleep(0)
        finally:
            stop.set()
            if transport:
                transport[0].close()
            # Connections and command opening have their own finite timeouts.
            # Never start a replacement while its predecessor is still alive.
            await asyncio.to_thread(thread.join)

    async def _run(self) -> None:
        failures = 0
        next_recovery = time.monotonic() + self.recovery_interval
        if self._fallback:
            self._status('SSH 持续日志采集已关闭，使用单一采集器增量轮询。')
        while not self._closing:
            follow = self.enabled and (not self._fallback or time.monotonic() >= next_recovery)
            started = time.monotonic()
            try:
                await self._session(follow)
                if not follow:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._error(exc)
                if follow:
                    failures = 1 if time.monotonic() - started >= 30 else failures + 1
                    if failures >= 3:
                        if not self._fallback:
                            self._status('SSH 日志流不可用，已降级为 5 秒增量轮询，将自动尝试恢复。')
                        self._fallback = True
                        next_recovery = time.monotonic() + self.recovery_interval
                await asyncio.sleep(self.poll_interval if self._fallback else self.retry_interval)

    async def close(self, *, drain: bool = False) -> None:
        async with self._close_lock:
            if self._closing:
                return
            self._closing = True
            if self.task:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            if drain:
                try:
                    # A finite snapshot catches up after the producer has exited.
                    await asyncio.wait_for(self._session(False), timeout=30)
                    self._flush()
                except Exception as exc:
                    self._error(exc)
                    self._status('远端日志尾部补读未完成，保留已入库日志及游标。')
            else:
                # Pending text remains replayable from the last committed cursor.
                self._pending = b''
