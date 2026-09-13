"""Store backed by the local filesystem — dev machines and tests.

Not a production path: on the EC2 host the container is recreated on every
deploy (``docker compose up --force-recreate``), so anything written inside it
is thrown away. ``config`` therefore *requires* a bucket under ``EILAF_ENV=prod``
rather than letting a misconfigured host fall back to here and silently lose
reports.

It is a real destination for ``EILAF_ENV=dev`` on a laptop, though: dev-mode
chats and logs land under ``data/reports/`` and can be read straight off disk
without a bucket or a single AWS credential.
"""

from __future__ import annotations

from pathlib import Path

from app.whatsapp.reporting.base import ReportStore


class LocalReportStore(ReportStore):
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def put(self, path: str, data: bytes, content_type: str) -> str:
        target = self.directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)
