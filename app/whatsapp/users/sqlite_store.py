"""User records in a SQLite file — one table, one row per user.

The file lives on a bind-mounted host directory (``data/users``), so it survives
the ``--force-recreate`` every deploy does. Not S3: a lookup happens at the
start of every conversation, and the instance role can only *write* to the
bucket, by design.

A connection per call rather than one shared: handlers run on the server's
threadpool, and SQLite connections are not shareable across threads by default.
The table is tiny and the calls rare (two per conversation at most), so the
connect cost does not matter.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from app.whatsapp.users.base import UserRecord, UserStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    key        TEXT PRIMARY KEY,
    language   TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen  TEXT NOT NULL
)
"""


class SQLiteUserStore(UserStore):
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def get(self, key: str) -> UserRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT language, first_seen, last_seen FROM users WHERE key = ?",
                (key,),
            ).fetchone()
        return UserRecord(*row) if row else None

    def touch(self, key: str, language: str, today: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO users (key, language, first_seen, last_seen)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET"
                " language = excluded.language, last_seen = excluded.last_seen",
                (key, language, today, today),
            )
