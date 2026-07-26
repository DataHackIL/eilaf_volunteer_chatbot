"""Generator that surfaces the retrieved nearest-neighbour passages.

The semantic nearest-neighbour search happens upstream (a real embedder +
``RAGPipeline.retrieve``); this generator simply returns those passages in
ranked order instead of sampling random ones. Stand-in until a real
(Hebrew-output) LLM generator is added.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot.rag.documents import Document
from chatbot.rag.generator.base import Generator


class ContextEchoGenerator(Generator):
    def generate(
        self, query: str, contexts: Sequence[Document], language: str = "Hebrew"
    ) -> str:
        # Verbatim passage echo — no phrasing, so ``language`` is ignored.
        return "\n\n---\n\n".join(doc.text for doc in contexts)
