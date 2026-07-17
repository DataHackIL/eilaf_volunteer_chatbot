"""Cheap, deterministic prefilter: obvious junk that needs no LLM to reject.

Keeps obvious non-content out of both the corpus and the (billed) Claude pass.
Deliberately conservative — it should only fire on segments that are clearly not
a rights/benefits statement, leaving every borderline case to the enricher.
"""

from __future__ import annotations

import re

# Below this many characters a segment is too short to carry a useful
# rights/benefits statement on its own (Hebrew is compact, so keep this low).
_MIN_USEFUL_CHARS = 12

# Section-scaffolding phrases legal / rights pages emit as their own nodes.
# Matched after stripping trailing separators (``:`` / ``-`` / ``–``).
_STUB_PHRASES = {
    "בסעיף זה",
    "בחוק זה",
    "בתקנות אלה",
    "בקצרה",
    "כללי",
    "הגדרות",
    "ראו גם",
    "ראה גם",
    "לקריאה נוספת",
    "קישורים חיצוניים",
    "הערות שוליים",
    "חקיקה ופסיקה",
    "פסקי דין",
    "תוכן עניינים",
}

# Trailing separators to peel off before phrase matching.
_TRAILING = " \t‏‎:—–-"

# A segment with no letters at all (pure punctuation / numbering / bullets).
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def is_obvious_junk(text: str) -> bool:
    """True when a segment is clearly not worth embedding or asking Claude about.

    Fires on: empty/whitespace, letter-free fragments (bare numbering, bullets),
    sub-threshold snippets, and a small set of known section-scaffolding stubs.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if not _HAS_LETTER.search(stripped):
        return True
    normalized = stripped.strip(_TRAILING)
    if normalized in _STUB_PHRASES:
        return True
    if len(stripped) < _MIN_USEFUL_CHARS:
        return True
    return False
