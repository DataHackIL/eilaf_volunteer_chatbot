"""Tests for retrieval's sibling + immediate-parent expansion and dedup.

Scoring is stubbed so the tests exercise only the expansion logic: given a fixed
ranking of matches, each match is expanded with its nearest siblings and its
immediate parent, on top of the top-k, all deduped by id. Parent-only behaviour
is isolated with ``sibling_k=0``; sibling behaviour has its own cases below.
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
    ids = [d.id for d in pipe.retrieve("q", sibling_k=0)]
    assert ids == ["src.json#0/0", "src.json#0"]  # clause, then its section


def test_shared_parent_added_once(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0/1"], top_k=2)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=0)]
    # A1, its parent A, then A2 — the shared parent A is not repeated.
    assert ids == ["src.json#0/0", "src.json#0", "src.json#0/1"]


def test_match_that_is_a_parent_is_not_doubled(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0"], top_k=2)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=0)]
    assert ids == ["src.json#0/0", "src.json#0"]  # A added by expansion, not again


def test_heading_only_parent_is_not_appended(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#1/0"], top_k=1)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=0)]
    assert ids == ["src.json#1/0"]  # B1's parent isn't a Document -> nothing added


def test_nearest_sibling_appended_before_parent(monkeypatch):
    # A1 is the sole match; its peer A2 (same parent) is pulled in, then parent A.
    pipe = _pipeline(monkeypatch, ["src.json#0/0"], top_k=1)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=1)]
    assert ids == ["src.json#0/0", "src.json#0/1", "src.json#0"]


def test_sibling_k_zero_disables_siblings(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0"], top_k=1)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=0)]
    assert ids == ["src.json#0/0", "src.json#0"]  # peer A2 not pulled in


def test_top_level_node_has_no_siblings(monkeypatch):
    # Parent A (src.json#0) is top-level (parent_id=None) -> no sibling set, so a
    # generous sibling_k pulls in nothing lateral.
    pipe = _pipeline(monkeypatch, ["src.json#0"], top_k=1)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=3)]
    assert ids == ["src.json#0"]


def test_sibling_that_is_also_a_match_not_doubled(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0/1"], top_k=2)
    ids = [d.id for d in pipe.retrieve("q", sibling_k=3)]
    # A1, its sibling A2, shared parent A; A2 as a match adds nothing new.
    assert ids == ["src.json#0/0", "src.json#0/1", "src.json#0"]


def test_siblings_ranked_nearest_first_and_capped(monkeypatch):
    # A section with four clauses; match C0, so its three peers C1/C2/C3 rank as
    # siblings. Cap at 2 -> only the two nearest by similarity, in that order.
    docs = [
        Document(text="parent C", id="c.json#0", parent_id=None),
        Document(text="clause C0", id="c.json#0/0", parent_id="c.json#0"),
        Document(text="clause C1", id="c.json#0/1", parent_id="c.json#0"),
        Document(text="clause C2", id="c.json#0/2", parent_id="c.json#0"),
        Document(text="clause C3", id="c.json#0/3", parent_id="c.json#0"),
    ]
    pipe = RAGPipeline(
        documents=docs,
        embedder=RandomEmbedder(dim=8),
        generator=ContextEchoGenerator(),
        top_k=1,
    )
    # C0 is the top match; among its peers, C3 is nearest, then C1, then C2.
    ranking = ["c.json#0/0", "c.json#0/3", "c.json#0/1", "c.json#0/2"]
    order = {doc_id: rank for rank, doc_id in enumerate(ranking)}
    monkeypatch.setattr(
        pipe, "_cosine",
        lambda m, v: np.array(
            [len(ranking) - order.get(d.id, len(ranking)) for d in pipe.documents],
            dtype=np.float64,
        ),
    )
    ids = [d.id for d in pipe.retrieve("q", sibling_k=2)]
    # C0, then its two nearest peers (C3 before C1), then parent C.
    assert ids == ["c.json#0/0", "c.json#0/3", "c.json#0/1", "c.json#0"]


def test_result_size_bounded(monkeypatch):
    pipe = _pipeline(monkeypatch, ["src.json#0/0", "src.json#0/1"], top_k=2)
    sibling_k = 3
    bound = (2 + sibling_k) * pipe.top_k
    assert len(pipe.retrieve("q", sibling_k=sibling_k)) <= bound


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
