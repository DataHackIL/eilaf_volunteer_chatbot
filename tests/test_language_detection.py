"""Tests for the script-based language detector (``app/language``).

``None`` is a first-class result here — it means "no signal", which the callers
turn into their own default language — so the cases that must *not* guess are
tested as carefully as the ones that must.
"""

from __future__ import annotations

import pytest

from app.language import ScriptDetector, detect_language
from app.visualizer.i18n import Language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Greetings — the realistic opening message, and the input length that
        # statistical detectors handle worst.
        ("שלום", Language.HEBREW),
        ("היי", Language.HEBREW),
        ("مرحبا", Language.ARABIC),
        ("السلام عليكم", Language.ARABIC),
        ("hi", Language.ENGLISH),
        ("hello there", Language.ENGLISH),
        # Real questions.
        ("איך אני מגיש תביעה לביטוח לאומי?", Language.HEBREW),
        ("كيف بقدر أقدّم طلب؟", Language.ARABIC),
        ("what am I entitled to?", Language.ENGLISH),
        # Latin letters outside ASCII still count as Latin script.
        ("dónde puedo", Language.ENGLISH),
    ],
)
def test_detects_the_script(text, expected):
    assert detect_language(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,  # a non-text WhatsApp message (voice note, image)
        "",
        "   ",
        "2",
        "0501234567",
        "👋",
        "?!...",
        "١٢٣",  # Arabic-Indic digits are digits, not letters: they do not vote
    ],
)
def test_no_signal_yields_none(text):
    assert detect_language(text) is None


def test_majority_script_wins_in_mixed_text():
    # A Hebrew question carrying one Latin word is still Hebrew.
    assert detect_language("מה הזכויות שלי לפי ה-PTSD?") is Language.HEBREW
    # An English question naming a Hebrew term is still English.
    assert detect_language("am I entitled to דמי פגיעה") is Language.ENGLISH


def test_an_even_split_is_no_signal():
    assert detect_language("שלום abcd") is None  # four letters each way


def test_unsupported_scripts_do_not_vote():
    assert detect_language("привет") is None
    assert detect_language("привет hello") is Language.ENGLISH


def test_detector_is_reusable_and_stateless():
    detector = ScriptDetector()
    assert detector.detect("שלום") is Language.HEBREW
    assert detector.detect("hello") is Language.ENGLISH
    assert detector.detect("שלום") is Language.HEBREW
