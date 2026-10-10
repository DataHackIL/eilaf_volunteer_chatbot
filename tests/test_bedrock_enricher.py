"""Tests for the Bedrock enrichment back-end: the request, ordering, and failures.

The classification spec is shared with the direct-API Claude enricher, so the
thing to pin here is that it is *sent* the same way, plus the two behaviours
that are new: concurrent calls must come back in input order (``annotate_tree``
zips results to nodes by position), and a failure that would repeat for every
segment must stop the run instead of deferring the whole backlog one call at a
time.

No network and no AWS: a hand-rolled fake stands in for the client, as in
``test_bedrock_generator.py``.
"""

from __future__ import annotations

import json
import threading
import time

import anthropic
import httpx
import pytest

from chatbot.rag.generator.bedrock import MODEL_ENV, REGION_ENV
from data.enrich.bedrock_enricher import BedrockSegmentEnricher
from data.enrich.claude_enricher import _ANNOTATION_SCHEMA, _SYSTEM, MODEL

_REQUEST = httpx.Request("POST", "https://bedrock-mantle.example/v1/messages")


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    monkeypatch.delenv(REGION_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)


def _reply(segment: str) -> str:
    """A deterministic per-segment answer, so ordering is checkable."""
    return json.dumps(
        {
            "useful": True,
            "min_age": len(segment),
            "max_age": None,
            "gender": None,
            "track": None,
        }
    )


class _FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 3})()


class _FakeClient:
    """Answers each segment via ``respond(segment)``; thread-safe call log."""

    def __init__(self, respond=lambda seg: _FakeResponse(_reply(seg))):
        self.calls = []
        self._lock = threading.Lock()
        self._respond = respond
        self.messages = type(
            "Messages", (), {"create": lambda _self, **kw: self._create(**kw)}
        )()

    def _create(self, **kw):
        with self._lock:
            self.calls.append(kw)
        return self._respond(kw["messages"][0]["content"])


def _enricher(client, **kw) -> BedrockSegmentEnricher:
    enricher = BedrockSegmentEnricher(region="eu-north-1", **kw)
    enricher._client = client
    return enricher


# --- configuration --------------------------------------------------------

def test_construction_is_inert_and_enrich_demands_a_region():
    enricher = BedrockSegmentEnricher()
    assert enricher._client is None
    assert enricher.enrich([]) == []  # nothing to send, nothing resolved
    with pytest.raises(RuntimeError, match=REGION_ENV):
        enricher.enrich(["a segment"])


def test_model_env_overrides_the_derived_id(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "eu.anthropic.claude-haiku-4-5")
    assert BedrockSegmentEnricher(region="eu-north-1").model == (
        "eu.anthropic.claude-haiku-4-5"
    )


def test_pinned_model_needs_no_region(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "eu.anthropic.pinned")
    assert BedrockSegmentEnricher().model == "eu.anthropic.pinned"


# --- the request ----------------------------------------------------------

def test_request_matches_the_claude_enricher_spec():
    enricher = _enricher(fake := _FakeClient())
    (annotation,) = enricher.enrich(["סעיף על זכויות"])

    (call,) = fake.calls
    assert call["model"] == f"eu.anthropic.{MODEL}"
    assert call["system"] == _SYSTEM
    assert call["thinking"] == {"type": "disabled"}
    assert call["output_config"] == {
        "format": {"type": "json_schema", "schema": _ANNOTATION_SCHEMA}
    }
    assert call["messages"] == [{"role": "user", "content": "סעיף על זכויות"}]
    assert annotation.ok and annotation.min_age == len("סעיף על זכויות")
    assert enricher.last_usage == {"calls": 1, "input_tokens": 10, "output_tokens": 3}


def test_concurrent_results_keep_input_order():
    segments = [f"segment {'x' * i}" for i in range(20)]

    def slow_for_early(seg):
        # Earlier segments finish *later*, so completion order is reversed.
        time.sleep(0.002 * (40 - len(seg)))
        return _FakeResponse(_reply(seg))

    enricher = _enricher(_FakeClient(slow_for_early), workers=8)
    annotations = enricher.enrich(segments)

    assert [a.min_age for a in annotations] == [len(s) for s in segments]


# --- failures -------------------------------------------------------------

@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse("not json"),
        _FakeResponse(_reply("x"), stop_reason="max_tokens"),
        _FakeResponse("", stop_reason="refusal"),
    ],
)
def test_unusable_reply_defers_the_segment(response):
    enricher = _enricher(_FakeClient(lambda seg: response))
    (annotation,) = enricher.enrich(["a segment"])
    assert annotation.ok is False


def test_transient_api_error_defers_only_that_segment():
    def flaky(seg):
        if seg == "bad":
            raise anthropic.APIConnectionError(request=_REQUEST)
        return _FakeResponse(_reply(seg))

    enricher = _enricher(_FakeClient(flaky))
    good, bad = enricher.enrich(["good", "bad"])
    assert good.ok and not bad.ok


def test_access_error_aborts_instead_of_deferring_everything():
    def denied(seg):
        response = httpx.Response(403, request=_REQUEST)
        raise anthropic.PermissionDeniedError("denied", response=response, body=None)

    enricher = _enricher(_FakeClient(denied), workers=2)
    with pytest.raises(anthropic.PermissionDeniedError):
        enricher.enrich([f"segment {i}" for i in range(50)])
