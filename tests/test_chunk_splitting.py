"""Tests for the ``max_chars`` split: no chunk may exceed the embedder's window.

The embedder truncates silently at its context window, so an over-long chunk
loses its tail with no error and that text becomes unretrievable. These pin the
split that prevents it, and the id/parent bookkeeping it has to preserve — the
substrate retrieval's parent and sibling expansion relies on (see
:mod:`tests.test_pipeline_retrieval`).
"""

from __future__ import annotations

import pytest

from chatbot.rag.documents.documents import _flatten_json_tree, _split_text

SENTENCE = "כל אדם שנפגע מעבירה זכאי לקבל מידע על ההליך הפלילי המתנהל בעניינו. "


def tree(text: str) -> dict:
    return {"title": "T", "children": [{"heading": "A", "text": text}]}


def docs_of(text: str, max_chars: int = 100):
    return _flatten_json_tree(
        tree(text), source="src.json", merge_chars=0, max_chars=max_chars
    )


def test_long_node_splits_into_several_documents():
    docs = docs_of(SENTENCE * 10)
    assert len(docs) > 1
    # The breadcrumb rides on every piece, so the body alone must fit the budget.
    for d in docs:
        body = d.text.split("\n", 1)[1]
        assert len(body) <= 100


def test_short_node_is_untouched():
    assert len(docs_of("קצר מדי כדי להתפצל")) == 1


def test_disabled_when_max_chars_is_zero():
    assert len(docs_of(SENTENCE * 10, max_chars=0)) == 1


def test_first_piece_keeps_the_node_id():
    docs = docs_of(SENTENCE * 10)
    # Children reference their parent by the node's own id, so the first piece
    # must keep it or those links stop resolving.
    assert docs[0].id == "src.json#0"
    assert [d.id for d in docs[1:]] == [f"src.json#0~{i}" for i in range(1, len(docs))]


def test_pieces_are_unique_and_share_a_parent():
    docs = docs_of(SENTENCE * 10)
    assert len({d.id for d in docs}) == len(docs)
    # Same parent => they are each other's siblings, so sibling expansion can
    # pull a split section back together.
    assert {d.parent_id for d in docs} == {None}


def test_every_piece_keeps_breadcrumb_and_metadata():
    data = tree(SENTENCE * 10)
    data["track"] = "hostile_acts"
    docs = _flatten_json_tree(data, source="src.json", merge_chars=0, max_chars=100)
    assert len(docs) > 1
    for d in docs:
        assert d.text.startswith("T › A\n")
        assert d.meta["track"] == "hostile_acts"


def test_splits_on_lines_before_sentences():
    text = "\n".join(["שורה ראשונה.", "שורה שנייה.", "שורה שלישית."])
    assert _split_text(text, 20) == ["שורה ראשונה.", "שורה שנייה.", "שורה שלישית."]


def test_falls_through_to_sentences_then_words():
    two = "משפט ראשון כאן. משפט שני כאן."
    assert _split_text(two, 20) == ["משפט ראשון כאן.", "משפט שני כאן."]
    # No sentence break available -> word boundaries.
    pieces = _split_text("אחת שתיים שלוש ארבע חמש שש", 12)
    assert len(pieces) > 1
    assert all(len(p) <= 12 for p in pieces)


def test_unbreakable_run_is_returned_whole():
    # Better an over-long chunk than a word cut in half — and it must terminate
    # rather than recurse forever looking for a finer boundary.
    word = "א" * 50
    assert _split_text(word, 10) == [word]


def test_packing_fills_the_budget():
    # Greedy packing: adjacent short lines share a chunk instead of each
    # becoming its own, which would re-introduce over-granularity.
    text = "\n".join(["אבג"] * 10)
    assert _split_text(text, 20) == ["אבג\nאבג\nאבג\nאבג\nאבג", "אבג\nאבג\nאבג\nאבג\nאבג"]


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_yields_nothing(text):
    assert _split_text(text, 100) == []
