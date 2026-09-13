"""Ship the service's own log output to the store, under ``EILAF_ENV=dev``.

The container's logs are only reachable over SSH with ``docker compose logs``,
and they go with the container on the next ``--force-recreate``. In dev mode we
want them to outlive it, next to the conversations they explain.

Records are buffered and uploaded in chunks — one object per flush, never
appended to, since S3 has no append and a read-modify-write would lose records
whenever two flushes overlapped. Chunks sort chronologically within a run:
``logs/YYYY/MM/DD/<run id>-00001.log``.

A flush happens when the buffer fills, when enough time has passed *and another
record arrives*, or at interpreter shutdown (``logging.shutdown`` closes every
handler, and ``close`` flushes). The middle one is the caveat worth knowing: a
process that goes completely quiet holds its last few lines until it exits. For
a dev log that is a fair trade against running a timer thread.

These logs are **not** redacted. ``server`` logs the sender's ``wa_id`` — the
user's phone number — when a turn fails, so this is dev-only for the same reason
the chat capture is: the welcome message says it is happening.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from app.whatsapp.reporting.base import ReportStore

TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"


class StoreLogHandler(logging.Handler):
    """Buffer log records and upload them to a :class:`ReportStore` in chunks."""

    def __init__(
        self,
        store: ReportStore,
        prefix: str = "logs",
        max_records: int = 100,
        max_seconds: float = 30.0,
    ):
        super().__init__()
        self.store = store
        self.prefix = prefix
        self.max_records = max_records
        self.max_seconds = max_seconds
        # Identifies one run of the process, so chunks from a restart cannot be
        # confused with (or overwrite) the previous boot's.
        self.run_id = uuid.uuid4().hex[:8]
        self._buffer: list[str] = []
        self._sequence = 0
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        # Set while uploading. boto3 logs as it works, so without this a flush
        # would re-enter emit, append to the buffer it is draining, and (at
        # DEBUG level) never finish.
        self._flushing = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._flushing:
            return
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 — a bad format string must not kill the app
            self.handleError(record)
            return
        with self._lock:
            self._buffer.append(line)
            due = len(self._buffer) >= self.max_records or (
                time.monotonic() - self._last_flush >= self.max_seconds
            )
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if self._flushing or not self._buffer:
                return
            chunk, self._buffer = self._buffer, []
            self._sequence += 1
            sequence = self._sequence
            self._last_flush = time.monotonic()
            self._flushing = True
        try:
            self.store.put(
                self._path(sequence),
                ("\n".join(chunk) + "\n").encode(),
                TEXT_CONTENT_TYPE,
            )
        except Exception:  # noqa: BLE001 — never let logging break the service
            # Deliberately not logged: that would recurse straight back here.
            # stderr still reaches `docker compose logs`, which is where someone
            # debugging a broken log shipper would actually be looking.
            import sys
            import traceback

            print(f"log upload failed ({len(chunk)} records dropped):", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            self._flushing = False

    def close(self) -> None:
        try:
            self.flush()
        finally:
            super().close()

    def _path(self, sequence: int) -> str:
        stamp = time.strftime("%Y/%m/%d", time.gmtime())
        return f"{self.prefix}/{stamp}/{self.run_id}-{sequence:05d}.log"
