"""Detects he/ar/en by counting which script the letters belong to.

For the three languages we support, the writing system *is* the language: Hebrew
and Arabic each own an exclusive Unicode block, and English is the only one of
the three written in Latin letters. So a character count needs no model, no
network and no dependency, and it is right on inputs a statistical detector
struggles with — a two-word greeting ("שלום", "مرحبا", "hi").

The one blind spot is transliteration: Levantine Arabic typed in Latin letters
("marhaba", "kifak") counts as English. Nothing in the text distinguishes it
cheaply, so the front-end covers that case by offering a one-tap language
switch rather than by guessing harder here.

Only letters vote. Digits, punctuation and emoji are skipped, so Arabic-Indic
digits or a trailing "?" never decide the outcome, and a message made only of
those yields ``None`` (no opinion) instead of a coin flip.
"""

from __future__ import annotations

from collections import Counter

from app.language.base import LanguageDetector
from app.visualizer.i18n import Language

# Inclusive code-point ranges per language. ``Language.ENGLISH`` stands in for
# "Latin script" — of our three languages it is the only one written that way.
_SCRIPT_RANGES: dict[Language, tuple[tuple[int, int], ...]] = {
    Language.HEBREW: (
        (0x0590, 0x05FF),  # Hebrew
        (0xFB1D, 0xFB4F),  # Hebrew presentation forms
    ),
    Language.ARABIC: (
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic presentation forms-A
        (0xFE70, 0xFEFF),  # Arabic presentation forms-B
    ),
    Language.ENGLISH: (
        (0x0041, 0x007A),  # basic Latin letters (non-letters are filtered out)
        (0x00C0, 0x024F),  # Latin-1 Supplement .. Latin Extended-B
    ),
}


def _script_of(codepoint: int) -> Language | None:
    for language, ranges in _SCRIPT_RANGES.items():
        if any(low <= codepoint <= high for low, high in ranges):
            return language
    return None


class ScriptDetector(LanguageDetector):
    """Majority vote over the letters' Unicode scripts."""

    def detect(self, text: str | None) -> Language | None:
        if not text:
            return None

        votes: Counter[Language] = Counter()
        for char in text:
            if not char.isalpha():
                continue
            language = _script_of(ord(char))
            if language is not None:
                votes[language] += 1

        if not votes:
            return None
        ranked = votes.most_common(2)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None  # a genuine tie is no signal; let the caller default
        return ranked[0][0]


_DEFAULT_DETECTOR = ScriptDetector()


def detect_language(text: str | None) -> Language | None:
    """Detect ``text``'s language with the default (script-based) detector."""
    return _DEFAULT_DETECTOR.detect(text)
