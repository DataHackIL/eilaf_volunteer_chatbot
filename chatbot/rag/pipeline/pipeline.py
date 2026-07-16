"""The RAG pipeline: documents -> embedder -> retrieve -> generator -> answer.

All components are swappable. The first slice wires placeholder components
(random embedder + random-passage generator) so the whole thing runs today.

Terminal demo::

    echo "שאלה לדוגמה" | python -m chatbot.rag.pipeline
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chatbot.rag.documents import Document, load_documents
from chatbot.rag.embedder import Embedder, SentenceTransformerEmbedder
from chatbot.rag.filters import MetadataFilter, SoftMetadataFilter
from chatbot.rag.generator import ContextEchoGenerator, Generator

# repo_root/data/static  (this file is chatbot/rag/pipeline/pipeline.py)
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[3] / "data" / "static"


class RAGPipeline:
    def __init__(
        self,
        documents: Sequence[Document],
        embedder: Embedder,
        generator: Generator,
        metadata_filter: MetadataFilter | None = None,
        top_k: int = 5,
    ):
        self.documents = list(documents)
        self.embedder = embedder
        self.generator = generator
        self.metadata_filter = metadata_filter or SoftMetadataFilter()
        self.top_k = top_k
        self._embeddings = (
            self.embedder.encode([d.text for d in self.documents])
            if self.documents
            else np.zeros((0, 1), dtype=np.float32)
        )

    @classmethod
    def from_static_dir(
        cls, static_dir: str | Path = DEFAULT_STATIC_DIR, **kwargs
    ) -> "RAGPipeline":
        """Build a pipeline from files on disk using the pretrained embedder.

        Retrieves the nearest-neighbour passages to the query and echoes them.
        Swap in ``RandomEmbedder`` / ``RandomPassagesGenerator`` for a no-download
        smoke test, or a real LLM generator once a provider is chosen.
        """
        return cls(
            documents=load_documents(static_dir),
            embedder=SentenceTransformerEmbedder(),
            generator=ContextEchoGenerator(),
            **kwargs,
        )

    def retrieve(self, query: str, facts: dict | None = None) -> list[Document]:
        """Retrieve the top-k chunks, optionally re-scored by closed-form facts.

        ``facts`` maps answered fields (``age``/``gender``/``locality``) to
        their values; omitted fields don't constrain anything. See
        :class:`~chatbot.rag.filters.MetadataFilter`.
        """
        if not self.documents:
            return []
        query_vec = self.embedder.encode([query])[0]
        sims = self._cosine(self._embeddings, query_vec)
        if facts:
            sims = sims + self.metadata_filter.adjust(self.documents, facts)
        top_idx = np.argsort(-sims)[: self.top_k]
        return [self.documents[i] for i in top_idx]

    def answer(self, query: str, facts: dict | None = None) -> str:
        return self.generator.generate(query, self.retrieve(query, facts))

    @staticmethod
    def _cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
        vector_norm = np.linalg.norm(vector) + 1e-8
        return (matrix @ vector) / (matrix_norm * vector_norm)


def _main() -> None:
    pipeline = RAGPipeline.from_static_dir()
    if not pipeline.documents:
        print(f"No documents found in {DEFAULT_STATIC_DIR}", file=sys.stderr)
    line = input("Please enter a query (or Ctrl+C to exit):\n")
    query = line.strip()
    print(pipeline.answer(query))


if __name__ == "__main__":
    _main()
