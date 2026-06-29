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
from chatbot.rag.embedder import Embedder, RandomEmbedder
from chatbot.rag.generator import Generator, RandomPassagesGenerator

# repo_root/data/static  (this file is chatbot/rag/pipeline/pipeline.py)
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[3] / "data" / "static"


class RAGPipeline:
    def __init__(
        self,
        documents: Sequence[Document],
        embedder: Embedder,
        generator: Generator,
        top_k: int = 5,
    ):
        self.documents = list(documents)
        self.embedder = embedder
        self.generator = generator
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
        """Build a pipeline with placeholder components from files on disk."""
        return cls(
            documents=load_documents(static_dir),
            embedder=RandomEmbedder(),
            generator=RandomPassagesGenerator(n=5),
            **kwargs,
        )

    def retrieve(self, query: str) -> list[Document]:
        if not self.documents:
            return []
        query_vec = self.embedder.encode([query])[0]
        sims = self._cosine(self._embeddings, query_vec)
        top_idx = np.argsort(-sims)[: self.top_k]
        return [self.documents[i] for i in top_idx]

    def answer(self, query: str) -> str:
        return self.generator.generate(query, self.retrieve(query))

    @staticmethod
    def _cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
        vector_norm = np.linalg.norm(vector) + 1e-8
        return (matrix @ vector) / (matrix_norm * vector_norm)


def _main() -> None:
    pipeline = RAGPipeline.from_static_dir()
    if not pipeline.documents:
        print(f"No documents found in {DEFAULT_STATIC_DIR}", file=sys.stderr)
    for line in sys.stdin:
        query = line.strip()
        if not query:
            continue
        print(pipeline.answer(query))


if __name__ == "__main__":
    _main()
