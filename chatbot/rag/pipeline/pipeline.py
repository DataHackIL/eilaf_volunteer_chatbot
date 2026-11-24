"""The RAG pipeline: documents -> embedder -> retrieve -> generator -> answer.

All components are swappable. The first slice wires placeholder components
(random embedder + random-passage generator) so the whole thing runs today.

Terminal demo::

    echo "שאלה לדוגמה" | python -m chatbot.rag.pipeline
"""

from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chatbot.rag.documents import Document, load_documents
from chatbot.rag.embedder import E5Embedder, Embedder
from chatbot.rag.filters import MetadataFilter, SoftMetadataFilter
from chatbot.rag.generator import ContextEchoGenerator, Generator
from chatbot.rag.pipeline.embedding_cache import load_or_encode
from chatbot.rag.reranker import BM25Reranker, Reranker
from chatbot.rag.translator import Translator, is_hebrew

log = logging.getLogger(__name__)

# A short domain "anchor" prepended to the query in embedding space. The corpus
# is single-domain (violence-victim material), so this doesn't discriminate
# between docs — it re-centres the *query* into that region, which empirically
# surfaces better matches. Fed as a separate query line, blended by weight (see
# ``retrieve``), so a long user question doesn't dilute the short anchor.
DEFAULT_CONTEXT_ANCHOR = "נפגע אירוע אלימות"
# How much the anchor pulls the query vector, in [0, 1]: 0 ignores it, 1 discards
# the user's question. Decoupled from text length, unlike plain concatenation.
DEFAULT_CONTEXT_WEIGHT = 0.4
# Reciprocal-rank-fusion constant for merging the translated and original-language
# rankings (see ``_fuse``). 60 is the value from the original RRF paper and the
# usual default: it damps the gap between rank 1 and rank 2 so that a chunk both
# passes agree on beats one that only a single pass ranks first.
RRF_K = 60
# Weight of the original-language ranking in that fusion (the translated one
# weighs 1.0). 0 means translation only, which scored best in
# scripts/translation_bench.py (2026-11-24, 30 draft pairs, NLLB-600M): top-5
# overlap with the Hebrew question's matches fell from 0.30 at w=0 to 0.25 at
# 0.25 and 0.20 at 1.0. The bench's reference is Hebrew retrieval, so it can't
# reward the original pass for finding a passage the corpus holds in Arabic;
# raise this if that starts to matter.
DEFAULT_ORIGINAL_WEIGHT = 0.0

# The static store, defaulting to repo_root/data/static (this file is
# chatbot/rag/pipeline/pipeline.py). ``EILAF_STATIC_DIR`` moves the whole store
# elsewhere for a deployment — an EC2 volume, or an S3 bucket exposed as a
# directory (mountpoint-s3 / s3fs / a sync into local disk at boot). Everything
# downstream hangs off this one root, so scrapers, enrichment and the embedding
# cache all follow it together. It is read once, at import: set it as a real
# environment variable (docker compose's ``env_file`` does), not later in code.
_STATIC_DIR = Path(
    os.environ.get("EILAF_STATIC_DIR")
    or Path(__file__).resolve().parents[3] / "data" / "static"
)
# The static store is split in two: scrapers write raw section-trees to ``raw/``;
# the enrich stage (``data/enrich``) reads ``raw/`` and writes annotated copies to
# ``enriched/`` — which is what the RAG pipeline actually reads.
RAW_STATIC_DIR = _STATIC_DIR / "raw"
DEFAULT_STATIC_DIR = _STATIC_DIR / "enriched"
# The Arabic dialect glossary the query translator applies (see
# chatbot/rag/translator/glossary.py). Data, not code: it lives with the corpus,
# out of git, and travels the same way (S3 sync / bind mount). Read once, when
# the translator is built, so an edit takes effect on the next restart.
GLOSSARY_PATH = _STATIC_DIR / "glossary_ar.csv"


class RAGPipeline:
    def __init__(
        self,
        documents: Sequence[Document],
        embedder: Embedder,
        generator: Generator,
        metadata_filter: MetadataFilter | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
        candidate_k: int = 20,
        sibling_k: int = 3,
        context_weight: float = DEFAULT_CONTEXT_WEIGHT,
        cache_dir: str | Path | None = None,
        translator: Translator | None = None,
        translation_cache_size: int = 256,
        original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
    ):
        self.documents = list(documents)
        self.embedder = embedder
        self.generator = generator
        self.metadata_filter = metadata_filter or SoftMetadataFilter()
        # Optional second-stage lexical reranker. When set, the embedder's
        # ``candidate_k`` nearest chunks are re-ordered by the reranker and the
        # top-k of *that* become the matches; when None, retrieval is pure
        # embedder ranking (behaviour unchanged). ``candidate_k`` is the pool the
        # reranker sees — it only matters when a reranker is set, and clamps to
        # ``top_k`` if set smaller.
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.sibling_k = sibling_k
        self.context_weight = context_weight
        # Optional query translator. When set, a question in another language is
        # also retrieved in Hebrew (the corpus language) and the two rankings
        # are fused; see ``_match_indices``. None keeps retrieval as it was.
        # Recent translations, so the same question asked twice (``matches`` then
        # ``retrieve``, or "add details" re-answering it) is translated once.
        self._translations: OrderedDict[tuple[str, str], str | None] = OrderedDict()
        self._translation_cache_size = translation_cache_size
        # How much the original-language pass counts in the fusion, relative to
        # the translated pass (1.0). 0 skips that pass and ranks on the
        # translation alone.
        self.original_weight = original_weight
        self.translator = translator
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

    @property
    def translator(self) -> Translator | None:
        return self._translator

    @translator.setter
    def translator(self, translator: Translator | None) -> None:
        # The cached translations came from the previous translator.
        self._translator = translator
        self._translations.clear()

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
        return self.embedder.encode_documents(texts)

    @classmethod
    def from_static_dir(
        cls, static_dir: str | Path = DEFAULT_STATIC_DIR, **kwargs
    ) -> RAGPipeline:
        """Build a pipeline from files on disk using the pretrained embedder.

        Retrieves the nearest-neighbour passages to the query and echoes them.
        Swap in ``RandomEmbedder`` / ``RandomPassagesGenerator`` for a no-download
        smoke test, or a real LLM generator once a provider is chosen.

        A BM25 lexical reranker is fitted on the loaded corpus and wired in by
        default (the caller may override with ``reranker=None`` for pure embedder
        ranking, or pass their own). It re-orders the embedder's candidate pool
        so exact-term matches surface; fitting is cheap (in-memory token counts).
        """
        documents = load_documents(static_dir)
        kwargs.setdefault("reranker", BM25Reranker(documents))
        return cls(
            documents=documents,
            embedder=E5Embedder(),
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
        language: str | None = None,
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

        ``language`` names the language ``query`` is written in ("Hebrew",
        "Arabic", "English"; None means Hebrew). It only matters when a
        ``translator`` is set; see :meth:`_match_indices`.
        """
        if not self.documents:
            return []
        n_sib = self.sibling_k if sibling_k is None else sibling_k
        top_idx, sims = self._match_indices(
            query, facts, context, context_weight, language
        )

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

    def _match_indices(
        self,
        query: str,
        facts: dict | None,
        context: str | None,
        context_weight: float | None,
        language: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The top-k match indices and the full embedder similarity vector.

        Embedder similarity (metadata-adjusted) ranks the corpus; the returned
        ``sims`` drives the later sibling ranking either way. Without a reranker
        the matches are the top-k of that ranking. With one, the embedder's
        ``candidate_k`` nearest are handed to the reranker and its top-k become
        the matches — a lexical second stage over a semantic shortlist. Siblings
        stay ranked by ``sims`` (embedder similarity), not the reranker.

        A non-Hebrew ``language`` with a ``translator`` set runs two passes: one
        on the Hebrew translation (with the reranker, whose lexical signal only
        works on Hebrew text) and one on the original text (without it). Their
        ``candidate_k`` heads are merged by reciprocal-rank fusion, so a chunk
        both passes find ranks highest, and the original pass still covers for
        a bad translation or a passage the corpus holds in the user's language.
        Siblings then rank by the higher of the two similarities. If the
        translation fails, retrieval falls back to the original text alone.
        """
        if self.translator is None or is_hebrew(language):
            ranked, sims = self._rank(query, facts, context, context_weight, True)
            return ranked[: self.top_k], sims
        hebrew_query = self._translate(query, language)
        if hebrew_query is None:
            ranked, sims = self._rank(query, facts, context, context_weight, True)
            return ranked[: self.top_k], sims
        # The default anchor is already Hebrew; only a user-written event
        # description needs translating. If that fails, keep it as written.
        hebrew_context = context
        if context and context.strip() and context != DEFAULT_CONTEXT_ANCHOR:
            hebrew_context = self._translate(context, language) or context
        he_ranked, he_sims = self._rank(
            hebrew_query, facts, hebrew_context, context_weight, True
        )
        if self.original_weight <= 0:
            return he_ranked[: self.top_k], he_sims
        orig_ranked, orig_sims = self._rank(
            query, facts, context, context_weight, False
        )
        fused = self._fuse(
            [he_ranked[: self.candidate_k], orig_ranked[: self.candidate_k]],
            [1.0, self.original_weight],
        )
        return fused[: self.top_k], np.maximum(he_sims, orig_sims)

    def _rank(
        self,
        query: str,
        facts: dict | None,
        context: str | None,
        context_weight: float | None,
        rerank: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The whole corpus in match order, and the embedder similarity vector.

        With ``rerank`` and a reranker set, the first ``candidate_k`` are in the
        reranker's order; the rest stay in embedder order behind them.
        """
        query_vec = self._encode_query(query, context, context_weight)
        sims = self._cosine(self._embeddings, query_vec)
        # Always consult the filter, even with no answered facts: the ``track``
        # axis penalises chunks scoped to a separate compensation track unless a
        # fact affirms it, so an unanswered form has to reach the filter to take
        # effect. Every other axis is a no-op on empty facts.
        sims = sims + self.metadata_filter.adjust(self.documents, facts or {})
        ranked = np.argsort(-sims)
        if not rerank or self.reranker is None:
            return ranked, sims
        candidates = ranked[: self.candidate_k]
        rr = self.reranker.score(query, [self.documents[i] for i in candidates])
        head = candidates[np.argsort(-rr, kind="stable")]
        return np.concatenate([head, ranked[self.candidate_k :]]), sims

    @staticmethod
    def _fuse(
        rankings: Sequence[np.ndarray], weights: Sequence[float] | None = None
    ) -> np.ndarray:
        """Weighted reciprocal-rank fusion: each list adds ``w / (RRF_K + rank)``.

        Ties keep first-seen order, and the first list is read first, so the
        translated pass wins a tie against the original-language one.
        """
        weights = weights or [1.0] * len(rankings)
        scores: dict[int, float] = {}
        for ranking, weight in zip(rankings, weights, strict=True):
            for rank, idx in enumerate(ranking, start=1):
                scores[int(idx)] = scores.get(int(idx), 0.0) + weight / (RRF_K + rank)
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        return np.array(ordered, dtype=np.int64)

    def hebrew_query(self, query: str, language: str | None) -> str | None:
        """The Hebrew text retrieval runs on for ``query``, or None if it runs on
        ``query`` as written (Hebrew input, no translator, or a failed translation).

        Shares the translation cache, so calling it around ``retrieve`` costs no
        extra translation. For front-ends that show what was searched.
        """
        if self.translator is None or is_hebrew(language):
            return None
        return self._translate(query, language)

    def _translate(self, text: str, language: str) -> str | None:
        """``text`` in Hebrew, or None when the translator fails or returns nothing.

        Cached per (text, language), failures included: a backend that just
        failed on this question would most likely fail again.
        """
        key = (text, language)
        if key in self._translations:
            self._translations.move_to_end(key)
            return self._translations[key]
        try:
            result = self.translator.to_hebrew(text, language).strip() or None
        except Exception:  # noqa: BLE001 — any backend error means "retrieve untranslated"
            log.warning(
                "translation to Hebrew failed (%s); retrieving on the original text",
                getattr(self.translator, "name", type(self.translator).__name__),
                exc_info=True,
            )
            result = None
        self._translations[key] = result
        if len(self._translations) > self._translation_cache_size:
            self._translations.popitem(last=False)
        return result

    def matches(
        self,
        query: str,
        facts: dict | None = None,
        context: str | None = DEFAULT_CONTEXT_ANCHOR,
        context_weight: float | None = None,
        language: str | None = None,
    ) -> list[Document]:
        """The ranked top-k match Documents, without sibling/parent expansion.

        This is the pipeline's core retrieval output — what a precision gauge
        should score — before :meth:`retrieve` pads it with structural context.
        """
        if not self.documents:
            return []
        top_idx, _ = self._match_indices(
            query, facts, context, context_weight, language
        )
        return [self.documents[i] for i in top_idx]

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
        language: str = "Hebrew",
    ) -> str:
        return self.generator.generate(
            query,
            self.retrieve(query, facts, context, context_weight, sibling_k, language),
            language,
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
            return self.embedder.encode_queries([query])[0]
        w = self.context_weight if weight is None else weight
        vecs = self.embedder.encode_queries([query, context]).astype(np.float64)
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
