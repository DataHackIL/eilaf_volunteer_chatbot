"""Translator interface: puts a user's question into the corpus language.

The corpus is Hebrew, and two parts of retrieval only work well on Hebrew text:
the BM25 reranker is lexical, so an Arabic query shares almost no tokens with
the passages, and e5 matches Arabic against Hebrew more weakly than Hebrew
against Hebrew. Translating the query first gives both a Hebrew query to work
on. Only this direction is needed: the generator already writes its answer in
the user's language.

Languages are named the way the generator names them ("Hebrew", "Arabic",
"English"), so the pipeline can hand the same string to both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

HEBREW = "Hebrew"


def is_hebrew(language: str | None) -> bool:
    """True for the corpus language (or no language given), case-insensitively."""
    return not language or language.strip().lower() == HEBREW.lower()


class Translator(ABC):
    @abstractmethod
    def to_hebrew(self, text: str, source: str) -> str:
        """Translate ``text``, written in ``source`` ("Arabic" / "English"), to Hebrew.

        May raise on a backend failure; the pipeline catches that and retrieves
        on the original text alone.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """A short label for logs and the benchmark table."""
        return type(self).__name__


class IdentityTranslator(Translator):
    """Returns the text unchanged. For tests, and to exercise the wiring."""

    def to_hebrew(self, text: str, source: str) -> str:
        return text
