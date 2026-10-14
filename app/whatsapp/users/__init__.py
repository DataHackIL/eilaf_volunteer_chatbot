"""Recognising a returning user — see :mod:`.base` for what is kept and why.

Off unless ``EILAF_USER_SALT`` is set: without a salt there is no safe key, and
an unsalted hash of a phone number is reversible by enumeration in minutes.
"""

from __future__ import annotations

from functools import lru_cache

from app.whatsapp import config
from app.whatsapp.users.base import UserRecord, UserStore, user_key
from app.whatsapp.users.sqlite_store import SQLiteUserStore

__all__ = ["SQLiteUserStore", "UserRecord", "UserStore", "get_user_store", "user_key"]


@lru_cache(maxsize=1)
def get_user_store() -> UserStore | None:
    """The process-wide store, or None when recognition is off."""
    if not config.USER_SALT:
        return None
    return SQLiteUserStore(config.USER_DB)
