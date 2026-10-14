"""What the bot remembers about a user between conversations, and under what key.

A conversation (``conversation.Session``) is forgotten after the idle TTL, a
restart keyword, or a process restart. This is the one thing that outlives it:
enough to tell a returning user from a new one, and the language they last used,
so the welcome can be shorter and in the right language from the first word.

Nothing here is keyed by the phone number. On WhatsApp the ``wa_id`` *is* the
number, and the consent message promises it is not saved, so records are keyed
by ``HMAC-SHA256(salt, wa_id)``. Without the salt the key cannot be recomputed
from a number, and a leaked database is a list of opaque ids and languages. With
the salt, anyone holding a number can still test whether it has written to us,
so the salt is a secret and lives in ``.env`` like the others.
"""

from __future__ import annotations

import hashlib
import hmac
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class UserRecord:
    language: str  # ``Language`` value: "he" / "ar" / "en"
    first_seen: str  # UTC date, YYYY-MM-DD; a day is all anything here needs
    last_seen: str


class UserStore(ABC):
    @abstractmethod
    def get(self, key: str) -> UserRecord | None:
        """The record for ``key``, or None for someone we have not seen."""

    @abstractmethod
    def touch(self, key: str, language: str, today: str) -> None:
        """Record ``key`` as seen ``today`` in ``language``, creating it if new.

        ``first_seen`` is set once and kept; the rest is overwritten.
        """


def user_key(wa_id: str, salt: str) -> str:
    """The pseudonymous key a user's record is stored under."""
    return hmac.new(salt.encode(), wa_id.encode(), hashlib.sha256).hexdigest()
