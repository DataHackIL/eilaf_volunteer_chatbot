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

# A short domain "anchor" prepended to the query in embedding space. The corpus
# is single-domain (violence-victim material), so this doesn't discriminate
# between docs — it re-centres the *query* into that region, which empirically
# surfaces better matches. Fed as a separate query line, blended by weight (see
# ``retrieve``), so a long user question doesn't dilute the short anchor.
DEFAULT_CONTEXT_ANCHOR = "נפגע אירוע אלימות"
# How much the anchor pulls the query vector, in [0, 1]: 0 ignores it, 1 discards
# the user's question. Decoupled from text length, unlike plain concatenation.
DEFAULT_CONTEXT_WEIGHT = 0.4

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
        sibling_k: int = 3,
        context_weight: float = DEFAULT_CONTEXT_WEIGHT,
        cache_dir: str | Path | None = None,
    ):
        self.documents = list(documents)
        self.embedder = embedder
        self.generator = generator
        self.metadata_filter = metadata_filter or SoftMetadataFilter()
        self.top_k = top_k
        self.sibling_k = sibling_k
        self.context_weight = context_weight
        # Index by id so retrieval can reattach a matched node's parent. Ids are
        # unique per corpus; a heading-only parent simply isn't in the map.
        self._by_id = {d.id: d for d in self.documents if d.id}
        # Group document *indices* by structural parent, so retrieval can pull a
        # match's nearest siblings. Only real parents group: top-level nodes (and
        # non-tree docx/pdf/txt chunks) carry ``parent_id=None`` and must not be
        # lumped into one giant corpus-wide sibling set — so they're left out.
        self._siblings: dict[str, list[int]] = {}
        for idx, d in enumerate(self.documents):
            if d.parent_id:
                self._siblings.setdefault(d.parent_id, []).append(idx)
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

    def retrieve(
        self,
        query: str,
        facts: dict | None = None,
        context: str | None = DEFAULT_CONTEXT_ANCHOR,
        context_weight: float | None = None,
        sibling_k: int | None = None,
    ) -> list[Document]:
        """Retrieve the top-k chunks, each with its nearest siblings and parent.

        ``context`` is an optional second query line — a brief description of the
        violent event — blended into the query embedding by ``context_weight``
        (see :func:`_encode_query`) to steer retrieval toward the right domain.
        It defaults to :data:`DEFAULT_CONTEXT_ANCHOR`; pass ``None``/``""`` to
        query on ``query`` alone. ``context_weight`` overrides the instance
        default for this call only (it affects the query vector, not the cached
        corpus embeddings), so a UI can tune it live without rebuilding.

        Scoring and top-k selection are unchanged. After selecting the matches,
        each one is expanded with two kinds of structural context from the source
        tree: its ``sibling_k`` nearest siblings (peer clauses under the same
        parent, ranked by the same query similarity) for lateral context, then
        its immediate parent section for its framing prose. ``sibling_k``
        overrides the instance default for this call only, so a UI can tune it
        live. Siblings share the match's already-computed similarity ranking, so
        they respect any metadata adjustment; the parent is looked up by id and
        is not re-scored. Everything is deduped by id via one ``seen`` set, so a
        shared parent or sibling is added once and a match that is another
        match's parent/sibling is never doubled. Heading-only parents (and the
        top-level nodes that have no structural parent) simply don't resolve.
        Result size is at most ~(2 + ``sibling_k``)×``top_k``.

        ``facts`` maps answered fields (``age``/``gender``/``locality``) to
        their values; omitted fields don't constrain anything. See
        :class:`~chatbot.rag.filters.MetadataFilter`. Note that appended parents
        are not themselves re-scored or metadata-filtered.
        """
        if not self.documents:
            return []
        n_sib = self.sibling_k if sibling_k is None else sibling_k
        query_vec = self._encode_query(query, context, context_weight)
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
            for sib_idx in self._nearest_siblings(i, doc.parent_id, sims, n_sib):
                add(self.documents[sib_idx])
            parent = self._by_id.get(doc.parent_id) if doc.parent_id else None
            if parent is not None:
                add(parent)
        return results

    def _nearest_siblings(
        self, idx: int, parent_id: str | None, sims: np.ndarray, k: int
    ) -> list[int]:
        """The ``k`` doc indices sharing ``parent_id``, excluding ``idx``, by sim.

        Reuses the already-computed ``sims`` (so no extra encoding), returning the
        highest-scoring peers under the same structural parent. Empty when the
        node has no real parent (top-level / non-tree chunks) or when ``k <= 0``.
        """
        if not parent_id or k <= 0:
            return []
        peers = [j for j in self._siblings.get(parent_id, ()) if j != idx]
        peers.sort(key=lambda j: sims[j], reverse=True)
        return peers[:k]

    def answer(
        self,
        query: str,
        facts: dict | None = None,
        context: str | None = DEFAULT_CONTEXT_ANCHOR,
        context_weight: float | None = None,
        sibling_k: int | None = None,
    ) -> str:
        return self.generator.generate(
            query, self.retrieve(query, facts, context, context_weight, sibling_k)
        )

    def _encode_query(
        self, query: str, context: str | None, weight: float | None = None
    ) -> np.ndarray:
        """Embed the query, blended with the optional domain ``context`` line.

        With no context, this is a plain single-text encode. Otherwise both
        lines are encoded and each L2-normalised (so the blend is a true
        interpolation on the unit sphere regardless of the embedder), then mixed
        ``(1 - w)·query + w·context``. The weight w — ``weight`` when given, else
        the instance ``context_weight`` — not the two lines' relative token
        lengths, sets the anchor's pull, which is the whole point of feeding it
        separately rather than concatenating. ``_cosine`` re-norms at scoring
        time, so the blended vector needn't be unit-length here.
        """
        if not context or not context.strip():
            return self.embedder.encode([query])[0]
        w = self.context_weight if weight is None else weight
        vecs = self.embedder.encode([query, context]).astype(np.float64)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        query_vec, context_vec = vecs
        return (1 - w) * query_vec + w * context_vec

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
