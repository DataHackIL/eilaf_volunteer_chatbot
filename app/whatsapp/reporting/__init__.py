"""Long-term storage for conversations the user asked us to keep.

``get_report_store()`` picks the implementation from configuration: the bucket
when one is set (the production path), the local directory otherwise. Under
``EILAF_ENV=prod`` a bucket is mandatory — ``config`` refuses to import without
one — so the local branch cannot be reached by accident on a deployed host.
"""

from __future__ import annotations

from functools import lru_cache

from app.whatsapp import config
from app.whatsapp.reporting.base import ReportStore, build_report, report_path
from app.whatsapp.reporting.local_store import LocalReportStore
from app.whatsapp.reporting.s3_store import S3ReportStore

__all__ = [
    "LocalReportStore",
    "ReportStore",
    "S3ReportStore",
    "build_report",
    "get_report_store",
    "report_path",
]


@lru_cache(maxsize=1)
def get_report_store() -> ReportStore:
    """The process-wide report store (cached, like ``get_pipeline``)."""
    if config.REPORT_BUCKET:
        return S3ReportStore(
            config.REPORT_BUCKET, config.REPORT_PREFIX, config.REPORT_REGION
        )
    return LocalReportStore(config.REPORT_DIR)
