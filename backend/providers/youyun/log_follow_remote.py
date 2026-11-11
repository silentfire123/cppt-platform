"""Executed over SSH using the existing remote Python (stdlib only).

stdout is a bounded JSON-lines protocol; stdin EOF terminates an idle follower.
No credentials, daemon, dependencies, or writes to the remote filesystem.
"""
import base64
import json
import os
import sys
import threading
import time


def main():
    path, offset, identity, mode = sys.argv[1:5]
    offset = int(offset)
    follow = mode == 'follow'
    stream = None
    last_heartbeat = 0.0
    snapshot_end = None

    def parent_closed():
        # Keep stdin open at the SSH client. EOF also stops a silent follower.
        while os.read(sys.stdin.fileno(), 1):
            pass
        os._exit(0)

    threading.Thread(target=parent_closed, daemon=True).start()

    def emit(kind, **values):
        print(json.dumps(dict(type=kind, **values), separators=(',', ':')), flush=True)

    def file_id(stat):
        return '%s:%s' % (stat.st_dev, stat.st_ino)

    def same_file(stat):
        return ':'.join(identity.split(':')[:2]) == file_id(stat)

    emit('ready')
    try:
        while True:
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                stat = None
            if stream is not None:
                # Drain the old descriptor before following a replacement path.
                old_stat = os.fstat(stream.fileno())
                changed = stat is not None and not same_file(stat)
                if old_stat.st_size < offset:
                    identity = file_id(old_stat) + ':' + str(time.time_ns())
                    emit('reset', fileId=identity, offset=0, reason='truncated')
                    offset = 0
                    stream.seek(0)
                remaining = 65536 if follow else max(0, min(65536, snapshot_end - offset))
                data = stream.read(remaining)
                if data:
                    emit('data', fileId=identity, start=offset, end=offset + len(data),
                         data=base64.b64encode(data).decode('ascii'))
                    offset += len(data)
                    continue
                if changed and follow:
                    stream.close()
                    stream = None
                    offset = 0
            if stream is None and stat is not None:
                candidate = open(path, 'rb')
                actual = os.fstat(candidate.fileno())
                new_identity = identity if identity and same_file(actual) else file_id(actual) + ':' + str(time.time_ns())
                if (identity and identity != new_identity) or actual.st_size < offset:
                    new_identity = file_id(actual) + ':' + str(time.time_ns())
                    emit('reset', fileId=new_identity, offset=0, reason='replaced_or_truncated')
                    offset = 0
                identity = new_identity
                snapshot_end = actual.st_size
                stream = candidate
                stream.seek(offset)
                continue
            if not follow:
                emit('eof', fileId=identity, offset=offset, exists=stat is not None)
                return
            now = time.monotonic()
            if now - last_heartbeat >= 1:
                emit('heartbeat')
                last_heartbeat = now
            time.sleep(.1)
    finally:
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    main()
