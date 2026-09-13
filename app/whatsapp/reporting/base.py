"""What a store is, and what one stored conversation looks like.

Three kinds of thing end up here, under one prefix:

``reports/``
    A conversation the user explicitly asked us to keep: they typed the report
    phrase, read what approving means, and tapped Approve.
``chats/``
    Every conversation, saved automatically — but **only** under
    ``EILAF_ENV=dev``, where the welcome message says so in the user's language
    before they have typed anything. Off in production.
``logs/``
    The service's own log output, also dev-only (see :mod:`.log_handler`).

The conversation documents deliberately carry **no ``wa_id``**. On WhatsApp that
id *is* the user's phone number, so storing it would quietly undo the promise
the consent message makes ("your phone number is not saved"). They are addressed
by a random id instead, and the only identifying string in the file is the
``contact`` the user chose to type — ``None`` when they tapped Skip.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

JSON_CONTENT_TYPE = "application/json; charset=utf-8"


class ReportStore(ABC):
    """Somewhere the bot can write a file and say where it went."""

    @abstractmethod
    def put(self, path: str, data: bytes, content_type: str) -> str:
        """Write ``data`` at ``path`` (relative to the store's root).

        The one primitive: conversation documents and log chunks both go
        through it. Implementations raise on failure rather than swallowing
        it — a caller that tells the user their report was *not* saved can
        only be honest if a failure is actually visible here.
        """

    def save(self, report: dict[str, Any]) -> str:
        """Write one conversation document, at the path its contents imply."""
        # ensure_ascii=False keeps Hebrew and Arabic readable in the console and
        # in anything that reads the object back, rather than storing every
        # letter as a \uXXXX escape.
        body = json.dumps(report, ensure_ascii=False, indent=2).encode()
        return self.put(report_path(report), body, JSON_CONTENT_TYPE)


def build_report(
    *,
    language: str,
    turns: Sequence[dict[str, Any]],
    reason: str | None = None,
    contact: str | None = None,
    kind: str = "reports",
    report_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON document for one conversation.

    ``created_at`` is wall-clock UTC: the session's own ``last_seen`` is
    monotonic (see ``conversation.Session``) and is a duration, not a date.

    ``report_id`` and ``created_at`` are passed in for a dev-mode chat, which is
    re-saved after every turn: holding both fixed for the life of the session
    keeps the path stable, so each save overwrites the last rather than leaving
    a trail of partial copies of the same conversation.
    """
    return {
        "kind": kind,
        "report_id": report_id or str(uuid.uuid4()),
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "language": language,
        "reason": reason,
        "contact": contact,
        "turns": list(turns),
    }


def report_path(report: dict[str, Any]) -> str:
    """``<kind>/YYYY/MM/DD/<epoch>-<report_id>.json`` for ``report``.

    Date-partitioned so a listing stays navigable once there are many, and
    prefixed with the epoch inside the day so a plain lexical listing is also
    chronological. Derived from the report's own ``created_at`` rather than from
    the clock, so the path can never disagree with the document — including for
    a report assembled a second before midnight and written just after, and for
    a dev chat whose every save must land on the path the first one chose.
    """
    created = datetime.fromisoformat(report["created_at"])
    kind = report.get("kind", "reports")
    return f"{kind}/{created:%Y/%m/%d}/{int(created.timestamp())}-{report['report_id']}.json"
