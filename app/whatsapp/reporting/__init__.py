"""Storage for conversations, and for the service's own logs in dev mode.

``get_report_store()`` picks the implementation from configuration: the bucket
when one is set (the production path), the local directory otherwise. Under
``EILAF_ENV=prod`` a bucket is mandatory — the server refuses to boot without
one (``config.wiring_problems``) — so the local branch cannot be reached by
accident on a deployed host.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.whatsapp import config
from app.whatsapp.reporting.base import ReportStore, build_report, report_path
from app.whatsapp.reporting.local_store import LocalReportStore
from app.whatsapp.reporting.log_handler import StoreLogHandler
from app.whatsapp.reporting.s3_store import S3ReportStore

__all__ = [
    "LocalReportStore",
    "ReportStore",
    "S3ReportStore",
    "StoreLogHandler",
    "build_report",
    "get_report_store",
    "install_log_handler",
    "report_path",
]


@lru_cache(maxsize=1)
def get_report_store() -> ReportStore:
    """The process-wide store (cached, like ``get_pipeline``)."""
    if config.REPORT_BUCKET:
        return S3ReportStore(
            config.REPORT_BUCKET, config.REPORT_PREFIX, config.REPORT_REGION
        )
    return LocalReportStore(config.REPORT_DIR)


def install_log_handler(
    formatter: logging.Formatter | None = None,
) -> StoreLogHandler | None:
    """Attach a :class:`StoreLogHandler` to the root logger, in dev mode only.

    Returns the handler (or ``None`` when not in dev mode) so a caller can flush
    it deliberately. Attached to the *root* logger because that is where the
    app's own records converge — ``server`` calls ``basicConfig`` for exactly
    that reason.
    """
    if not config.IS_DEV:
        return None
    handler = StoreLogHandler(get_report_store(), prefix=config.LOG_PREFIX)
    handler.setFormatter(
        formatter
        or logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler
