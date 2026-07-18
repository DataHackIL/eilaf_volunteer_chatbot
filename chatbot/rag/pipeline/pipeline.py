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
from chatbot.rag.pipeline.embedding_cache import load_or_encode

# repo_root/data/static  (this file is chatbot/rag/pipeline/pipeline.py)
_STATIC_DIR = Path(__file__).resolve().parents[3] / "data" / "static"
# The static store is split in two: scrapers write raw section-trees to ``raw/``;
# the enrich stage (``data/enrich``) reads ``raw/`` and writes annotated copies to
# ``enriched/`` — which is what the RAG pipeline actually reads.
RAW_STATIC_DIR = _STATIC_DIR / "raw"
DEFAULT_STATIC_DIR = _STATIC_DIR / "enriched"


class RAGPipeline:
    def __init__(
        self,
        documents: Sequence[Document],
        embedder: Embedder,
        generator: Generator,
        metadata_filter: MetadataFilter | None = None,
        top_k: int = 5,
        cache_dir: str | Path | None = None,
    ):
        self.documents = list(documents)
        self.embedder = embedder
        self.generator = generator
        self.metadata_filter = metadata_filter or SoftMetadataFilter()
        self.top_k = top_k
        # Index by id so retrieval can reattach a matched node's parent. Ids are
        # unique per corpus; a heading-only parent simply isn't in the map.
        self._by_id = {d.id: d for d in self.documents if d.id}
        self._embeddings = self._embed(cache_dir)

    def _embed(self, cache_dir: str | Path | None) -> np.ndarray:
        """Embed the corpus, via the on-disk cache when ``cache_dir`` is set.

        The cache turns the (minutes-long, CPU-bound) corpus embedding into a
        one-time cost; see :mod:`chatbot.rag.pipeline.embedding_cache`.
        """
        texts = [d.text for d in self.documents]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if cache_dir is not None:
            return load_or_encode(self.embedder, texts, cache_dir)
        return self.embedder.encode(texts)

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
            # Persist embeddings in a subdir (not matched by the loader's
            # top-level *.json/*.docx/... globs), so startup pays the encode
            # cost once instead of on every launch.
            cache_dir=Path(static_dir) / ".embeddings",
            **kwargs,
        )

    def retrieve(self, query: str, facts: dict | None = None) -> list[Document]:
        """Retrieve the top-k chunks, each followed by its immediate parent.

        Scoring and top-k selection are unchanged; after selecting the matches,
        each one's immediate parent section (from the source tree) is appended as
        extra context so a matched clause carries its parent's framing prose.
        Parents are looked up by ``parent_id`` and deduped by id, so siblings
        sharing a parent add it once and a match that is another match's parent
        is never doubled. A heading-only parent that never became a Document
        simply doesn't resolve. Result size is at most ~2×``top_k``.

        ``facts`` maps answered fields (``age``/``gender``/``locality``) to
        their values; omitted fields don't constrain anything. See
        :class:`~chatbot.rag.filters.MetadataFilter`. Note that appended parents
        are not themselves re-scored or metadata-filtered.
        """
        if not self.documents:
            return []
        query_vec = self.embedder.encode([query])[0]
        sims = self._cosine(self._embeddings, query_vec)
        if facts:
            sims = sims + self.metadata_filter.adjust(self.documents, facts)
        top_idx = np.argsort(-sims)[: self.top_k]

        results: list[Document] = []
        seen: set[str] = set()

        def add(doc: Document) -> None:
            # Docs without an id (defensive) can't be deduped by id, so always keep.
            if doc.id and doc.id in seen:
                return
            if doc.id:
                seen.add(doc.id)
            results.append(doc)

        for i in top_idx:
            doc = self.documents[i]
            add(doc)
            parent = self._by_id.get(doc.parent_id) if doc.parent_id else None
            if parent is not None:
                add(parent)
        return results

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
