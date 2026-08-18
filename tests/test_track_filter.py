"""Tests for the ``track`` eligibility axis: inheritance + off-track penalties.

``track`` marks content belonging to a separate compensation track (hostile
acts, military bereavement, road/work accidents). It is the one filter axis that
penalises on a *missing* fact, so these pin the asymmetry — an unanswered form
must still push track-scoped chunks down, while never touching untagged ones.
"""

from __future__ import annotations

import pytest

from chatbot.rag.documents.documents import _flatten_json_tree
from chatbot.rag.filters import SoftMetadataFilter

PENALTY = 0.15


@pytest.fixture
def tagged_tree() -> dict:
    """A source stamped hostile-acts at the root, with one segment overriding."""
    return {
        "title": "T",
        "track": "hostile_acts",
        "children": [
            {"heading": "A", "text": "hostile-acts body"},
            {
                "heading": "B",
                "text": "road-accident body",
                "meta": {"track": "road_accident"},
            },
        ],
    }


def docs_of(tree: dict):
    return _flatten_json_tree(tree, source="src.json", merge_chars=0)


def test_root_track_inherits_to_every_segment(tagged_tree):
    tracks = [d.meta.get("track") for d in docs_of(tagged_tree)]
    # Segment B carries its own tag, so it overrides the root's rather than
    # inheriting it — the enricher gets the last word on a mixed page.
    assert tracks == ["hostile_acts", "road_accident"]


def test_untracked_source_tags_nothing():
    docs = docs_of({"title": "T", "children": [{"heading": "A", "text": "body"}]})
    assert "track" not in docs[0].meta


def test_missing_fact_penalises_tracked_chunks(tagged_tree):
    # The corpus serves victims of ordinary violence, so a chunk that declares a
    # track is off-track until a fact says otherwise.
    adjustment = SoftMetadataFilter(PENALTY).adjust(docs_of(tagged_tree), {})
    assert list(adjustment) == pytest.approx([-PENALTY, -PENALTY])


def test_matching_fact_clears_only_its_own_track(tagged_tree):
    adjustment = SoftMetadataFilter(PENALTY).adjust(
        docs_of(tagged_tree), {"track": "hostile_acts"}
    )
    assert list(adjustment) == pytest.approx([0.0, -PENALTY])


def test_untagged_chunks_are_never_penalised():
    docs = docs_of({"title": "T", "children": [{"heading": "A", "text": "body"}]})
    for facts in ({}, {"track": "hostile_acts"}, {"age": 30, "gender": "f"}):
        assert list(SoftMetadataFilter(PENALTY).adjust(docs, facts)) == [0.0]


def test_track_penalty_stacks_with_age(tagged_tree):
    tree = dict(tagged_tree)
    tree["children"] = [{"heading": "A", "text": "body", "meta": {"min_age": 67}}]
    # Off-track (inherited) *and* age-ineligible: soft penalties accumulate
    # rather than one masking the other.
    adjustment = SoftMetadataFilter(PENALTY).adjust(docs_of(tree), {"age": 30})
    assert list(adjustment) == pytest.approx([-2 * PENALTY])
