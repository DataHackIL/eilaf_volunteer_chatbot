"""Tests for section-tree flattening: node ids and parent links.

These pin the id/adjacency substrate that retrieval's parent-expansion relies on
(see :mod:`tests.test_pipeline_retrieval`).
"""

from __future__ import annotations

import pytest

from chatbot.rag.documents.documents import _flatten_json_tree


@pytest.fixture
def tree() -> dict:
    """A 3-level tree: A has body text + two clause children; B is heading-only.

    B (a marker with no ``text``) is never emitted as a Document, so its child
    B1 points at a ``parent_id`` that won't resolve — the intended behaviour for
    heading-only parents.
    """
    return {
        "title": "T",
        "children": [
            {
                "marker": "A",
                "text": "parent A body",
                "children": [
                    {"marker": "A1", "text": "clause A1"},
                    {"marker": "A2", "text": "clause A2"},
                ],
            },
            {
                "marker": "B",  # heading only, no text -> not emitted
                "children": [{"marker": "B1", "text": "clause B1"}],
            },
        ],
    }


def test_ids_encode_index_path(tree):
    docs = _flatten_json_tree(tree, source="src.json", merge_chars=0)
    by_id = {d.id: d for d in docs}

    # child 0 (A), its children 0/0 (A1) and 0/1 (A2), and 1/0 (B1).
    assert set(by_id) == {"src.json#0", "src.json#0/0", "src.json#0/1", "src.json#1/0"}
    assert by_id["src.json#0"].text.endswith("parent A body")
    assert by_id["src.json#0/0"].text.endswith("clause A1")


def test_parent_id_links_to_immediate_parent(tree):
    by_id = {d.id: d for d in _flatten_json_tree(tree, source="src.json", merge_chars=0)}

    # Clauses point at their section; the section is top-level so has no parent.
    assert by_id["src.json#0/0"].parent_id == "src.json#0"
    assert by_id["src.json#0/1"].parent_id == "src.json#0"
    assert by_id["src.json#0"].parent_id is None


def test_heading_only_parent_is_absent_from_map(tree):
    docs = _flatten_json_tree(tree, source="src.json", merge_chars=0)
    by_id = {d.id: d for d in docs}

    b1 = by_id["src.json#1/0"]
    # B1 records its structural parent id, but B was never emitted, so the id
    # doesn't resolve — parent-expansion will (correctly) skip it.
    assert b1.parent_id == "src.json#1"
    assert "src.json#1" not in by_id


def test_lead_is_top_level_with_no_parent():
    docs = _flatten_json_tree(
        {"title": "T", "lead": "intro text", "children": []}, source="src.json"
    )
    lead = next(d for d in docs if d.id == "src.json#lead")
    assert lead.parent_id is None
    assert lead.text.endswith("intro text")


def test_small_subtree_collapses_to_one_document(tree):
    # With a generous merge budget, section A's tiny clause list rolls up into a
    # single Document instead of three — sub-markers kept so structure survives.
    docs = _flatten_json_tree(tree, source="src.json", merge_chars=500)
    by_id = {d.id: d for d in docs}

    # A collapses to just A's id; its clause ids no longer exist on their own.
    assert "src.json#0" in by_id
    assert "src.json#0/0" not in by_id
    assert "src.json#0/1" not in by_id
    a = by_id["src.json#0"]
    assert "parent A body" in a.text
    assert "A1 clause A1" in a.text  # descendant marker preserved inline
    assert "A2 clause A2" in a.text
    # B (heading-only) collapses around its single clause under B's id.
    assert "src.json#1" in by_id
    assert "B1 clause B1" in by_id["src.json#1"].text


def test_large_subtree_still_recurses(tree):
    # A merge budget below the subtree total keeps the section split into nodes,
    # so the id/parent substrate is intact for genuinely large sections.
    docs = _flatten_json_tree(tree, source="src.json", merge_chars=5)
    by_id = {d.id: d for d in docs}
    assert {"src.json#0", "src.json#0/0", "src.json#0/1", "src.json#1/0"} <= set(by_id)
