"""Generator interface: turns a query + retrieved contexts into an answer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from chatbot.rag.documents import Document


class Generator(ABC):
    @abstractmethod
    def generate(
        self, query: str, contexts: Sequence[Document], language: str = "Hebrew"
    ) -> str:
        """Answer ``query`` from ``contexts``, replying in ``language``.

        ``language`` is the natural-language name of the answer language
        ("Hebrew" / "Arabic" / "English"); non-LLM heads may ignore it.
        """
        raise NotImplementedError
