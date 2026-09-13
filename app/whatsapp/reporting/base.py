"""What a report store is, and what one report looks like.

A *report* is a conversation the user explicitly asked us to keep: they typed
the report phrase, read what approving means, and tapped Approve. Everything
here therefore runs only on the far side of that tap — nothing in this package
is called for a conversation nobody reported.

The document deliberately carries **no ``wa_id``**. On WhatsApp that id *is* the
user's phone number, so storing it would quietly undo the promise the consent
message makes ("your phone number is not saved"). Reports are addressed by a
random ``report_id`` instead, and the only identifying string in the file is the
``contact`` the user chose to type — ``None`` when they tapped Skip.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any


class ReportStore(ABC):
    """Somewhere a reported conversation can be written."""

    @abstractmethod
    def save(self, report: dict[str, Any]) -> str:
        """Persist ``report``; return where it landed, for the log.

        Implementations raise on failure rather than swallowing it — the caller
        tells the user their report was *not* saved, which is only honest if a
        failure is actually visible here.
        """


def build_report(
    *,
    language: str,
    turns: Sequence[dict[str, Any]],
    reason: str | None = None,
    contact: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON document for one reported conversation.

    ``created_at`` is wall-clock UTC: the session's own ``last_seen`` is
    monotonic (see ``conversation.Session``) and is a duration, not a date.
    """
    return {
        "report_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "language": language,
        "reason": reason,
        "contact": contact,
        "turns": list(turns),
    }


def report_path(report: dict[str, Any]) -> str:
    """Relative path for ``report``: ``YYYY/MM/DD/<epoch>-<report_id>.json``.

    Date-partitioned so a bucket listing stays navigable once there are many,
    and prefixed with the epoch inside the day so a plain lexical listing is
    also chronological. Derived from the report's own ``created_at`` rather than
    from the clock, so the path can never disagree with the document — including
    for a report assembled a second before midnight and written just after.
    """
    created = datetime.fromisoformat(report["created_at"])
    return f"{created:%Y/%m/%d}/{int(created.timestamp())}-{report['report_id']}.json"
