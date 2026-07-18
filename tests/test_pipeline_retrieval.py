"""Tests for retrieval's immediate-parent expansion and dedup.

Scoring is stubbed so the tests exercise only the expansion logic: given a fixed
ranking of matches, each match's immediate parent is appended once, on top of the
top-k, deduped by id.
"""

from __future__ import annotations

import numpy as np

from chatbot.rag.documents.documents import Document
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


def test_empty_corpus_returns_empty():
    pipe = RAGPipeline(
        documents=[],
        embedder=RandomEmbedder(dim=8),
        generator=ContextEchoGenerator(),
    )
    assert pipe.retrieve("q") == []
