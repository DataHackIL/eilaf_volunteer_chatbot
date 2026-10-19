"""Tests for the Bedrock answer head: configuration, not transport.

The head shares every answer-shaping line with :class:`ClaudeGenerator` — the
prompt, the citation rules, the usage read-out — so there is nothing new to test
about what it says. What *is* new is where the call goes, and the two ways that
can go wrong quietly: a region resolved by botocore's fallback instead of by us
(answers generated outside the EU under an EU-only assumption), and a model id
guessed for a geography nobody has checked (an AccessDenied naming regions that
appear nowhere in the config).

No network and no SDK: a hand-rolled fake stands in for the client, matching the
house pattern in ``test_whatsapp_conversation.py``.
"""

from __future__ import annotations

import pytest

from chatbot.rag.documents.documents import Document
from chatbot.rag.generator._prompt import NO_CONTEXT
from chatbot.rag.generator.bedrock import (
    MODEL_ENV,
    REGION_ENV,
    BedrockGenerator,
)
from chatbot.rag.generator.claude_api import MODEL

DOCS = [Document(text="מי שנפגע מאלימות זכאי לסיוע נפשי.", source="s")]


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    """Neither variable may leak in from the developer's shell or .env."""
    monkeypatch.delenv(REGION_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)


class _FakeResponse:
    def __init__(self, text):
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.usage = type("Usage", (), {"input_tokens": 11, "output_tokens": 7})()


class _FakeClient:
    """Records the kwargs ``generate`` would have sent to Bedrock."""

    def __init__(self, text="תשובה"):
        self.calls = []
        self.messages = type(
            "Messages", (), {"create": lambda _self, **kw: self._create(**kw)}
        )()
        self._text = text

    def _create(self, **kw):
        self.calls.append(kw)
        return _FakeResponse(self._text)


# --- construction is inert -------------------------------------------------


def test_construction_touches_nothing():
    """The Streamlit app builds every head up front; this one must stay cheap."""
    gen = BedrockGenerator()
    assert gen.last_usage is None
    assert gen._client is None


def test_empty_contexts_short_circuits_without_any_config():
    """No passages means no call, so an unconfigured head still answers here."""
    gen = BedrockGenerator()
    assert gen.generate("שאלה", []) == NO_CONTEXT
    assert gen._client is None


# --- region resolution -----------------------------------------------------


def test_missing_region_raises_and_does_not_default():
    gen = BedrockGenerator()
    with pytest.raises(RuntimeError) as excinfo:
        gen.region
    message = str(excinfo.value)
    assert REGION_ENV in message
    # The whole point: no silent us-east-1.
    assert "us-east-1" in message, "the error should name the fallback it refuses"


def test_region_comes_from_env(monkeypatch):
    monkeypatch.setenv(REGION_ENV, "eu-north-1")
    assert BedrockGenerator().region == "eu-north-1"


def test_explicit_region_beats_env(monkeypatch):
    monkeypatch.setenv(REGION_ENV, "eu-north-1")
    assert BedrockGenerator(region="us-east-1").region == "us-east-1"


# --- model id derivation ---------------------------------------------------


@pytest.mark.parametrize(
    ("region", "geo"),
    [("eu-north-1", "eu"), ("eu-central-1", "eu"), ("us-east-1", "us")],
)
def test_model_id_derives_geo_prefix(region, geo):
    assert BedrockGenerator(region=region).model == f"{geo}.anthropic.{MODEL}"


def test_unmapped_geography_refuses_to_guess():
    """ap-* is not mapped: a guessed profile id fails confusingly at invoke time."""
    gen = BedrockGenerator(region="ap-south-1")
    with pytest.raises(RuntimeError) as excinfo:
        gen.model
    assert MODEL_ENV in str(excinfo.value)


def test_model_env_overrides_derivation(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "anthropic.claude-sonnet-5")
    assert BedrockGenerator(region="ap-south-1").model == "anthropic.claude-sonnet-5"


def test_explicit_model_beats_env(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "eu.anthropic.from-env")
    gen = BedrockGenerator(region="eu-north-1", model="eu.anthropic.explicit")
    assert gen.model == "eu.anthropic.explicit"


# --- generation reuses the Claude head's behaviour -------------------------


def test_generate_sends_the_resolved_model_and_records_usage():
    gen = BedrockGenerator(region="eu-north-1")
    gen._client = fake = _FakeClient("תשובה מבוססת מקורות [1]")

    answer = gen.generate("מה הזכויות שלי?", DOCS)

    assert answer == "תשובה מבוססת מקורות [1]"
    assert gen.last_usage == {"input_tokens": 11, "output_tokens": 7}
    (call,) = fake.calls
    assert call["model"] == f"eu.anthropic.{MODEL}"
    assert call["max_tokens"] == gen.max_tokens
    # Same grounding contract as the direct-API head.
    assert "Hebrew" in call["system"]
    assert DOCS[0].text in call["messages"][0]["content"]


def test_language_threads_through_to_the_system_prompt():
    gen = BedrockGenerator(region="eu-north-1")
    gen._client = fake = _FakeClient()
    gen.generate("question", DOCS, language="Arabic")
    assert "Arabic" in fake.calls[0]["system"]
