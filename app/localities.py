"""Loads the CBS locality list (``app/localities.csv``) for the locality filter.

The file is Israel Central Bureau of Statistics data in ``windows-1255``. Only
two columns are used: the Hebrew name and its Latin transliteration; no Arabic
name is available. The **Hebrew name is canonical** — it's what gets stored in
``facts["locality"]`` and what the corpus tags will eventually match — while the
transliteration is display-only, shown when the UI language is English.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.visualizer.i18n import Language

_CSV_PATH = Path(__file__).resolve().parent / "localities.csv"
_ENCODING = "windows-1255"
_HEBREW_COLUMN = "שם_ישוב"
_LATIN_COLUMN = "שם_ישוב_לועזי"


@dataclass(frozen=True)
class Locality:
    hebrew: str  # canonical value stored in facts["locality"]
    latin: str  # Latin transliteration; "" when the CBS row has none

    def label(self, lang: Language) -> str:
        """Display name for the UI language.

        English shows the transliteration, falling back to Hebrew when the CBS
        row has none. Arabic names are unavailable, so Hebrew and Arabic both
        show the Hebrew name.
        """
        if lang is Language.ENGLISH and self.latin:
            return self.latin
        return self.hebrew


@lru_cache(maxsize=1)
def load_localities() -> list[Locality]:
    """Return the deduplicated locality list, sorted by Hebrew name."""
    localities: list[Locality] = []
    seen: set[str] = set()
    with _CSV_PATH.open(encoding=_ENCODING, newline="") as handle:
        for row in csv.DictReader(handle):
            hebrew = (row.get(_HEBREW_COLUMN) or "").strip()
            latin = (row.get(_LATIN_COLUMN) or "").strip()
            if not hebrew or hebrew in seen:
                continue
            seen.add(hebrew)
            localities.append(Locality(hebrew=hebrew, latin=latin))
    localities.sort(key=lambda locality: locality.hebrew)
    return localities
