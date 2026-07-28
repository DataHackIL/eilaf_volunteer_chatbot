"""BM25 reranker: lexical scoring, its effect on pipeline matches, and the
threshold precision gauge built on it."""

from __future__ import annotations

import numpy as np
import pytest

from chatbot.rag.documents.documents import Document
from chatbot.rag.embedder.random_embedder import RandomEmbedder
from chatbot.rag.eval.precision import bm25_precision_at_k
from chatbot.rag.generator.context_echo import ContextEchoGenerator
from chatbot.rag.pipeline.pipeline import RAGPipeline
from chatbot.rag.reranker import BM25Reranker

_CORPUS = [
    Document(text="חוק זכויות נפגעי עבירה", id="d0"),
    Document(text="ביטוח לאומי קצבת נכות", id="d1"),
    Document(text="נפגעי תאונת דרכים פיצוי", id="d2"),
    Document(text="עורך דין ייעוץ משפטי", id="d3"),
]


def test_score_ranks_lexical_overlap_first():
    bm25 = BM25Reranker(_CORPUS)
    scores = bm25.score("נפגעי עבירה זכויות", _CORPUS)
    assert int(np.argmax(scores)) == 0  # d0 shares the most query terms


def test_out_of_corpus_query_scores_zero():
    bm25 = BM25Reranker(_CORPUS)
    scores = bm25.score("xyzzy unseen tokens", _CORPUS)
    assert np.all(scores == 0.0)


def test_score_corpus_matches_score_on_same_docs():
    bm25 = BM25Reranker(_CORPUS)
    query = "נפגעי עבירה"
    np.testing.assert_allclose(bm25.score_corpus(query), bm25.score(query, _CORPUS))


def test_empty_documents_returns_empty():
    bm25 = BM25Reranker(_CORPUS)
    assert bm25.score("q", []).shape == (0,)


def _pipeline_with_reranker(reranker) -> RAGPipeline:
    return RAGPipeline(
        documents=_CORPUS,
        embedder=RandomEmbedder(dim=8),
        generator=ContextEchoGenerator(),
        reranker=reranker,
        top_k=1,
        candidate_k=4,
        sibling_k=0,
    )


def test_reranker_reorders_embedder_candidates(monkeypatch):
    # Embedder ranks d3 first; BM25 for a d0-heavy query should override it so d0
    # becomes the single top match.
    pipe = _pipeline_with_reranker(BM25Reranker(_CORPUS))
    # Force the embedder shortlist to include all docs with d3 on top.
    monkeypatch.setattr(
        pipe, "_cosine", lambda m, v: np.array([0.1, 0.2, 0.3, 0.9])
    )
    matches = pipe.matches("נפגעי עבירה זכויות")
    assert [d.id for d in matches] == ["d0"]


def test_no_reranker_keeps_embedder_order(monkeypatch):
    pipe = _pipeline_with_reranker(reranker=None)
    monkeypatch.setattr(pipe, "_cosine", lambda m, v: np.array([0.1, 0.2, 0.3, 0.9]))
    assert [d.id for d in pipe.matches("נפגעי עבירה")] == ["d3"]  # top cosine wins


def test_candidate_k_clamps_up_to_top_k():
    pipe = RAGPipeline(
        documents=_CORPUS,
        embedder=RandomEmbedder(dim=8),
        generator=ContextEchoGenerator(),
        top_k=5,
        candidate_k=2,  # smaller than top_k -> clamped
    )
    assert pipe.candidate_k == 5


def test_precision_all_relevant_when_tau_low(monkeypatch):
    pipe = _pipeline_with_reranker(BM25Reranker(_CORPUS))
    # Make the top match be d0, the exact best lexical match -> normalised 1.0.
    monkeypatch.setattr(pipe, "_cosine", lambda m, v: np.array([0.9, 0.1, 0.1, 0.1]))
    report = bm25_precision_at_k(pipe, ["נפגעי עבירה זכויות"], tau=0.5, context=None)
    assert report.mean == 1.0


def test_precision_zero_for_out_of_corpus_query():
    pipe = _pipeline_with_reranker(BM25Reranker(_CORPUS))
    report = bm25_precision_at_k(pipe, ["completely unseen"], tau=0.5, context=None)
    assert report.per_query == [("completely unseen", 0.0)]


def test_precision_requires_bm25_reranker():
    pipe = _pipeline_with_reranker(reranker=None)
    with pytest.raises(ValueError, match="BM25Reranker"):
        bm25_precision_at_k(pipe, ["q"], context=None)
