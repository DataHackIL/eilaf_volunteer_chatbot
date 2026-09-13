"""Report store backed by the local filesystem — dev machines and tests.

Not a production path: on the EC2 host the container is recreated on every
deploy (``docker compose up --force-recreate``), so anything written inside it
is thrown away. ``config`` therefore *requires* a bucket under ``EILAF_ENV=prod``
rather than letting a misconfigured host fall back to here and silently lose
reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.whatsapp.reporting.base import ReportStore, report_path


class LocalReportStore(ReportStore):
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def save(self, report: dict[str, Any]) -> str:
        path = self.directory / report_path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)
