"""Tests for retrieval's immediate-parent expansion and dedup.

Scoring is stubbed so the tests exercise only the expansion logic: given a fixed
ranking of matches, each match's immediate parent is appended once, on top of the
top-k, deduped by id.
"""

from __future__ import annotations

import numpy as np
import pytest

from chatbot.rag.documents.documents import Document
from chatbot.rag.embedder.base import Embedder
from chatbot.rag.embedder.random_embedder import RandomEmbedder
from chatbot.rag.generator.context_echo import ContextEchoGenerator
from chatbot.rag.pipeline.pipeline import RAGPipeline

# id -> parent_id; mirrors the tree in tests.test_documents (A section + two
# clauses; B1 is a clause whose heading-only parent "src.json#1" is not a Document).
_DOCS = [
    Document(text="parent A body", id="src.json#0", parent_id=None),
    Document(text="clause A1", id="src.json#0/0", parent_id="src.json#0"),
    Document(text="clause A2", id="src.json#0/1", parent_id="src.json#0"),
    Document(text="clause B1", id="src.json#1/0", parent_id="src.json#1"),
]


def _pipeline(monkeypatch, ranked_ids: list[str], top_k: int = 2) -> RAGPipeline:
    """Build a pipeline whose retrieval scores ``ranked_ids`` highest, in order.

    ``_cosine`` is stubbed on the instance to return similarities derived from the
    desired ranking, so ``retrieve`` selects exactly those ids as its top-k and we
    observe only the parent-expansion behaviour.
    """
    pipe = RAGPipeline(
        documents=_DOCS,
        embedder=RandomEmbedder(dim=8),
        generator=ContextEchoGenerator(),
        top_k=top_k,
    )
    order = {doc_id: rank for rank, doc_id in enumerate(ranked_ids)}

    def fake_cosine(matrix, vector):
        # Higher score = earlier in ranked_ids; unranked docs score lowest.
        return np.array(
            [len(ranked_ids) - order.get(d.id, len(ranked_ids)) for d in pipe.documents],
            dtype=np.float64,
        )

    monkeypatch.setattr(pipe, "_cosine", fake_cosine)
    return pipe


def test_parent_appended_after_match(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0"], top_k=1)
    ids = [d.id for d in pipe.retrieve("q")]
    assert ids == ["src.json#0/0", "src.json#0"]  # clause, then its section


def test_shared_parent_added_once(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0/1"], top_k=2)
    ids = [d.id for d in pipe.retrieve("q")]
    # A1, its parent A, then A2 — the shared parent A is not repeated.
    assert ids == ["src.json#0/0", "src.json#0", "src.json#0/1"]


def test_match_that_is_a_parent_is_not_doubled(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0"], top_k=2)
    ids = [d.id for d in pipe.retrieve("q")]
    assert ids == ["src.json#0/0", "src.json#0"]  # A added by expansion, not again


def test_heading_only_parent_is_not_appended(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#1/0"], top_k=1)
    ids = [d.id for d in pipe.retrieve("q")]
    assert ids == ["src.json#1/0"]  # B1's parent isn't a Document -> nothing added


def test_result_size_bounded_by_twice_top_k(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0/1"], top_k=2)
    assert len(pipe.retrieve("q")) <= 2 * pipe.top_k


class _FixedEmbedder(Embedder):
    """Returns preset vectors for known texts (orthogonal axes) for blend tests."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 2):
        self._vectors = {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}
        self._dim = dim

    def encode(self, texts):  # noqa: D102
        # Corpus docs (embedded at construction) aren't under test here, so any
        # unknown text maps to a zero vector of the right width.
        return np.stack([self._vectors.get(t, np.zeros(self._dim, np.float32)) for t in texts])


def _blend_pipeline(context_weight: float) -> RAGPipeline:
    # Query and context map to distinct orthogonal axes, so the blended vector's
    # components read off the mix weights directly.
    embedder = _FixedEmbedder({"q": [1.0, 0.0], "ctx": [0.0, 1.0]})
    return RAGPipeline(
        documents=_DOCS,
        embedder=embedder,
        generator=ContextEchoGenerator(),
        context_weight=context_weight,
    )


def test_context_blend_mixes_by_weight():
    pipe = _blend_pipeline(context_weight=0.4)
    vec = pipe._encode_query("q", "ctx")
    assert vec == pytest.approx([0.6, 0.4])  # (1-w)·query + w·context


def test_empty_context_uses_query_only():
    pipe = _blend_pipeline(context_weight=0.4)
    for empty in (None, "", "   "):
        assert pipe._encode_query("q", empty) == pytest.approx([1.0, 0.0])


def test_per_call_weight_overrides_instance_default():
    pipe = _blend_pipeline(context_weight=0.4)
    # Override wins; None falls back to the instance default.
    assert pipe._encode_query("q", "ctx", 0.25) == pytest.approx([0.75, 0.25])
    assert pipe._encode_query("q", "ctx", None) == pytest.approx([0.6, 0.4])


def test_blend_normalises_each_line_first():
    # Give the context a non-unit norm; it must be normalised before mixing, so
    # the weight — not the raw magnitude — governs its pull.
    embedder = _FixedEmbedder({"q": [1.0, 0.0], "ctx": [0.0, 10.0]})
    pipe = RAGPipeline(
        documents=_DOCS,
        embedder=embedder,
        generator=ContextEchoGenerator(),
        context_weight=0.4,
    )
    assert pipe._encode_query("q", "ctx") == pytest.approx([0.6, 0.4])


def test_empty_corpus_returns_empty():
    pipe = RAGPipeline(
        documents=[],
        embedder=RandomEmbedder(dim=8),
        generator=ContextEchoGenerator(),
    )
    assert pipe.retrieve("q") == []
