"""Tests for what happens when the LLM back-end fails on a segment.

A failed call must leave the node un-annotated so a later run retries it.
Recording the enricher's defaults instead is indistinguishable from a genuine
"useful, no constraints" verdict, and the reuse cache would treat it as done
forever — which is how a retired Gemini model name silently burned a whole run.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest

from data.enrich.base import SegmentAnnotation, SegmentEnricher
from data.enrich.enrich_store import annotate_tree


class StubEnricher(SegmentEnricher):
    """Returns the queued annotations in order, recording what it was sent."""

    def __init__(self, *annotations: SegmentAnnotation):
        self.queued = list(annotations)
        self.seen: list[str] = []

    def enrich(self, segments: Sequence[str]) -> list[SegmentAnnotation]:
        self.seen.extend(segments)
        return [self.queued.pop(0) for _ in segments]


def tree() -> dict:
    return {
        "title": "T",
        "children": [
            {"heading": "A", "text": "a segment long enough to pass the junk rule"},
            {"heading": "B", "text": "another segment long enough to pass the rule"},
        ],
    }


def test_failed_segment_is_left_unannotated():
    enricher = StubEnricher(SegmentAnnotation(), SegmentAnnotation(ok=False))
    out, pending, sent, annotated = annotate_tree(tree(), enricher)

    assert (pending, sent, annotated) == (2, 2, 1)
    assert "useful" in out["children"][0]
    # The failed one carries no verdict at all, so it still looks pending.
    assert "useful" not in out["children"][1]


def test_failed_segment_is_retried_on_the_next_run():
    first = StubEnricher(SegmentAnnotation(), SegmentAnnotation(ok=False))
    enriched, *_ = annotate_tree(tree(), first)

    second = StubEnricher(SegmentAnnotation(track="hostile_acts"))
    out, pending, sent, annotated = annotate_tree(
        tree(), second, existing_tree=enriched
    )

    # Only the previously-failed segment is re-sent; the successful one is reused.
    assert second.seen == ["another segment long enough to pass the rule"]
    assert (pending, sent, annotated) == (1, 1, 1)
    assert out["children"][1]["meta"] == {"track": "hostile_acts"}


def test_ok_is_transport_state_and_never_reaches_meta():
    assert SegmentAnnotation(ok=False).tags() == {}
    assert SegmentAnnotation(ok=True, track="military").tags() == {"track": "military"}


def test_budget_counts_attempts_not_successes():
    # Both fail; the caller still learns 2 were sent, so --limit cannot be
    # outspent by a back-end that answers nothing.
    enricher = StubEnricher(SegmentAnnotation(ok=False), SegmentAnnotation(ok=False))
    _, pending, sent, annotated = annotate_tree(tree(), enricher)
    assert (pending, sent, annotated) == (2, 2, 0)


def test_ok_is_not_part_of_any_backend_schema():
    """``ok`` is ours, not the model's — it must never be asked for or accepted.

    If it leaked into a request schema the model would start reporting its own
    success, and a hallucinated ``ok=false`` would defer a segment forever.
    """
    from data.enrich.claude_enricher import _ANNOTATION_SCHEMA
    from data.enrich.openai_enricher import _SYSTEM_JSON

    assert "ok" not in _ANNOTATION_SCHEMA["properties"]
    assert "ok" not in _ANNOTATION_SCHEMA["required"]
    assert '"ok"' not in _SYSTEM_JSON
    # A reply without the key still parses, defaulting to a usable answer.
    parsed = SegmentAnnotation.model_validate_json(
        '{"useful":true,"min_age":null,"max_age":null,"gender":null,"track":null}'
    )
    assert parsed.ok is True


# --- checkpointing ---------------------------------------------------------


def big_tree(n: int) -> dict:
    return {
        "title": "T",
        "children": [
            {
                "heading": f"H{i}",
                "text": f"segment number {i} long enough to pass the rule",
            }
            for i in range(n)
        ],
    }


def test_checkpoint_sends_in_chunks_and_persists_each():
    enricher = StubEnricher(*[SegmentAnnotation() for _ in range(5)])
    snapshots: list[int] = []

    def on_chunk(tree):
        snapshots.append(sum("useful" in node for node in tree["children"]))

    _, pending, sent, annotated = annotate_tree(
        big_tree(5), enricher, chunk_size=2, on_chunk=on_chunk
    )

    assert (pending, sent, annotated) == (5, 5, 5)
    assert snapshots == [2, 4, 5]


def test_checkpoint_keeps_earlier_chunks_when_a_later_one_crashes():
    class CrashOnSecondChunk(StubEnricher):
        def enrich(self, segments):
            if self.seen:
                raise RuntimeError("AccessDenied")
            return super().enrich(segments)

    enricher = CrashOnSecondChunk(*[SegmentAnnotation() for _ in range(4)])
    saved: list[dict] = []
    with pytest.raises(RuntimeError):
        annotate_tree(
            big_tree(4),
            enricher,
            chunk_size=2,
            on_chunk=lambda t: saved.append(copy.deepcopy(t)),
        )

    (last,) = saved
    assert [("useful" in n) for n in last["children"]] == [True, True, False, False]
    # And the saved copy is a valid "existing" tree: a re-run sends only the rest.
    retry = StubEnricher(SegmentAnnotation(), SegmentAnnotation())
    _, pending, sent, _ = annotate_tree(big_tree(4), retry, existing_tree=last)
    assert (pending, sent) == (2, 2)


def test_checkpoint_respects_the_budget_and_sends_each_segment_once():
    # Failures stay pending but are not re-sent within the same call.
    enricher = StubEnricher(*[SegmentAnnotation(ok=False) for _ in range(3)])
    calls: list[int] = []
    _, pending, sent, annotated = annotate_tree(
        big_tree(5),
        enricher,
        max_new=3,
        chunk_size=2,
        on_chunk=lambda t: calls.append(1),
    )
    assert (pending, sent, annotated) == (5, 3, 0)
    assert len(enricher.seen) == len(set(enricher.seen)) == 3
    assert len(calls) == 2
