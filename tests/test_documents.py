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
    docs = _flatten_json_tree(tree, source="src.json")
    by_id = {d.id: d for d in docs}

    # child 0 (A), its children 0/0 (A1) and 0/1 (A2), and 1/0 (B1).
    assert set(by_id) == {"src.json#0", "src.json#0/0", "src.json#0/1", "src.json#1/0"}
    assert by_id["src.json#0"].text.endswith("parent A body")
    assert by_id["src.json#0/0"].text.endswith("clause A1")


def test_parent_id_links_to_immediate_parent(tree):
    by_id = {d.id: d for d in _flatten_json_tree(tree, source="src.json")}

    # Clauses point at their section; the section is top-level so has no parent.
    assert by_id["src.json#0/0"].parent_id == "src.json#0"
    assert by_id["src.json#0/1"].parent_id == "src.json#0"
    assert by_id["src.json#0"].parent_id is None


def test_heading_only_parent_is_absent_from_map(tree):
    docs = _flatten_json_tree(tree, source="src.json")
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
