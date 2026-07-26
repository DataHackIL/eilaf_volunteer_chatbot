"""Placeholder generator that returns random passage(s) from the contexts.

Ignores the query — it only proves the wiring. Swap for a real (Hebrew-output)
LLM generator once a provider is chosen.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from chatbot.rag.documents import Document
from chatbot.rag.generator.base import Generator


class RandomPassagesGenerator(Generator):
    def __init__(self, n: int = 1, seed: int | None = None):
        self.n = n
        self._rng = random.Random(seed)

    def generate(
        self, query: str, contexts: Sequence[Document], language: str = "Hebrew"
    ) -> str:
        if not contexts:
            return ""
        k = min(self.n, len(contexts))
        chosen = self._rng.sample(list(contexts), k)
        return "\n\n---\n\n".join(doc.text for doc in chosen)
