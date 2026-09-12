"""Language-detection interface: guesses a UI language from a piece of text.

The front-ends need this because asking "which language?" costs a whole turn
before the user can say anything useful — their opening message already carries
the answer. Detection is deliberately allowed to *fail*: ``detect`` returns
``None`` when the text carries no signal (digits, emoji, an empty string), so
the caller — not the detector — owns the fallback language. Swap
implementations freely; a script-range count is cheap and offline, an LLM-backed
detector could later read transliterated Arabic that the script test cannot see.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.visualizer.i18n import Language


class LanguageDetector(ABC):
    @abstractmethod
    def detect(self, text: str | None) -> Language | None:
        """Return the language ``text`` appears to be written in.

        ``None`` means "no opinion" — an empty, missing, or scriptless text, or
        a tie between candidates — and is a normal result, not an error. The
        caller should substitute its own default language.
        """
        raise NotImplementedError
