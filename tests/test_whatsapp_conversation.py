"""Offline tests for the WhatsApp front-end.

The conversation state machine is driven end-to-end with a **fake pipeline**
(no e5 model, no LLM call, no network), so we exercise the flow, the Skip
branches, age validation, and the ``facts`` assembly deterministically. A second
group drives the FastAPI webhook: verification handshake + inbound parsing.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib

import httpx

import pytest
from fastapi.testclient import TestClient

from app.whatsapp import client, config, conversation, server
from chatbot.rag.documents.documents import Document


class _FakeGenerator:
    def __init__(self):
        self.last = None

    def generate(self, query, contexts, language="Hebrew"):
        self.last = (query, list(contexts), language)
        return f"answer[{language}]"


class _FakePipeline:
    """Records the facts/context passed to ``retrieve`` for assertions."""

    def __init__(self):
        self.generator = _FakeGenerator()
        self.retrieve_calls = []

    def retrieve(self, query, facts=None, context=None, **kwargs):
        self.retrieve_calls.append({"query": query, "facts": facts, "context": context})
        return [Document(text="clause", source="benefits.json", id="benefits.json#0")]


@pytest.fixture
def fake_pipeline(monkeypatch):
    pipe = _FakePipeline()
    monkeypatch.setattr(conversation, "get_pipeline", lambda: pipe)
    return pipe


def _button_ids(reply):
    return [bid for bid, _ in (reply.buttons or [])]


# --------------------------------------------------------------------------- #
# Conversation flow                                                            #
# --------------------------------------------------------------------------- #


def test_happy_path_builds_facts_and_answers(fake_pipeline):
    wa = "happy"
    conversation.reset(wa)

    # First contact → language menu with the three language buttons.
    replies = conversation.handle_message(wa, text="hi")
    assert _button_ids(replies[0]) == ["lang:he", "lang:ar", "lang:en"]

    # Pick English → question prompt (English string).
    replies = conversation.handle_message(wa, button_id="lang:en")
    assert replies[0].body == "What would you like to ask?"

    # Ask a question → event step with a Skip button.
    replies = conversation.handle_message(wa, text="What am I entitled to?")
    assert _button_ids(replies[0]) == ["skip"]

    # Type an event description → gender step (Female/Male/Skip).
    replies = conversation.handle_message(wa, text="assault at home")
    assert _button_ids(replies[0]) == ["gender:f", "gender:m", "skip"]

    # Female → age step.
    replies = conversation.handle_message(wa, button_id="gender:f")
    assert _button_ids(replies[0]) == ["skip"]

    # Age 30 → runs the pipeline; answer + "ask another".
    replies = conversation.handle_message(wa, text="30")
    assert "answer[English]" in replies[0].body
    assert "benefits.json" in replies[0].body  # sources listed
    assert len(replies) == 2  # answer + ask-another

    call = fake_pipeline.retrieve_calls[-1]
    assert call["facts"] == {"gender": "f", "age": 30}
    assert call["context"] == "assault at home"
    # Language threaded into the generator as its English name.
    assert fake_pipeline.generator.last[2] == "English"

    # State reset to accept a new question (language kept).
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY


def test_all_skips_default_context_and_empty_facts(fake_pipeline):
    wa = "skips"
    conversation.reset(wa)
    conversation.handle_message(wa, button_id="lang:he")
    conversation.handle_message(wa, text="שאלה")
    conversation.handle_message(wa, button_id="skip")  # event skipped
    conversation.handle_message(wa, button_id="skip")  # gender skipped
    conversation.handle_message(wa, button_id="skip")  # age skipped

    call = fake_pipeline.retrieve_calls[-1]
    assert call["facts"] is None  # empty facts → None (unconstrained retrieval)
    assert call["context"] == "נפגע אירוע אלימות"  # DEFAULT_CONTEXT_ANCHOR


def test_invalid_age_reprompts_then_accepts(fake_pipeline):
    wa = "age"
    conversation.reset(wa)
    conversation.handle_message(wa, button_id="lang:en")
    conversation.handle_message(wa, text="q")
    conversation.handle_message(wa, button_id="skip")  # event
    conversation.handle_message(wa, button_id="skip")  # gender

    replies = conversation.handle_message(wa, text="not-a-number")
    assert "valid age" in replies[0].body.lower()
    assert conversation._SESSIONS[wa].step is conversation.Step.AGE  # still on age

    replies = conversation.handle_message(wa, text="42")
    assert fake_pipeline.retrieve_calls[-1]["facts"] == {"age": 42}


def test_empty_question_reprompts(fake_pipeline):
    wa = "empty"
    conversation.reset(wa)
    conversation.handle_message(wa, button_id="lang:en")
    replies = conversation.handle_message(wa, text="   ")
    assert replies[0].body == "What would you like to ask?"
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY


# --------------------------------------------------------------------------- #
# Webhook server                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_token_probe(monkeypatch):
    """Keep the boot-time token check off the network in every test.

    ``server.lifespan`` probes the Graph API, so without this any test that
    starts the app would make a real call — slow, flaky, and dependent on a
    live credential. ``usable=None`` is the "couldn't tell" verdict, which the
    check treats as non-fatal.
    """
    monkeypatch.setattr(
        server.client,
        "check_token",
        lambda: client.TokenStatus(None, None, "stubbed in tests"),
    )


@pytest.fixture
def webhook_client(monkeypatch):
    # Avoid loading the real e5 pipeline at lifespan startup.
    monkeypatch.setattr(server, "get_pipeline", lambda: None)
    monkeypatch.setattr(conversation, "get_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(server.config, "VERIFY_TOKEN", "verify-me")
    monkeypatch.setattr(server.config, "APP_SECRET", "")  # skip signature in tests
    sent: list = []
    monkeypatch.setattr(server, "_send", lambda wa_id, reply: sent.append((wa_id, reply)))
    with TestClient(server.app) as client:
        yield client, sent


def test_webhook_verification_echoes_challenge(webhook_client):
    client, _ = webhook_client
    resp = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-me",
            "hub.challenge": "1234567",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "1234567"


def test_webhook_verification_rejects_bad_token(webhook_client):
    client, _ = webhook_client
    resp = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "x"},
    )
    assert resp.status_code == 403


def test_webhook_dispatches_inbound_text(webhook_client):
    client, sent = webhook_client
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messages": [
                                {"from": "972500000000", "type": "text", "text": {"body": "hi"}}
                            ]
                        },
                    }
                ]
            }
        ],
    }
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    # Background task ran: first contact → language menu sent to the sender.
    assert sent and sent[0][0] == "972500000000"
    assert _button_ids(sent[0][1]) == ["lang:he", "lang:ar", "lang:en"]


def test_webhook_ignores_status_callbacks(webhook_client):
    client, sent = webhook_client
    payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]}
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    assert sent == []


def test_signature_rejected_when_secret_set(monkeypatch):
    monkeypatch.setattr(server, "get_pipeline", lambda: None)
    monkeypatch.setattr(server.config, "APP_SECRET", "s3cret")
    with TestClient(server.app) as client:
        resp = client.post(
            "/webhook",
            content=b"{}",
            headers={"X-Hub-Signature-256": "sha256=deadbeef"},
        )
        assert resp.status_code == 403

        good = "sha256=" + hmac.new(b"s3cret", b"{}", hashlib.sha256).hexdigest()
        resp = client.post(
            "/webhook", content=b"{}", headers={"X-Hub-Signature-256": good}
        )
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Production guards (config imported with EILAF_ENV set)
# --------------------------------------------------------------------------- #


@pytest.fixture
def reload_config(monkeypatch):
    """Re-import ``config`` under a patched environment, then restore it.

    The guards run at import time, so they can only be exercised by reloading
    the module. Setting a key to "" still counts as *present* for ``load_dotenv``
    (which never overrides os.environ), so a real repo-root .env cannot leak in
    and make these pass by accident.
    """

    def _reload(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)


def test_prod_refuses_to_import_without_app_secret(reload_config):
    with pytest.raises(RuntimeError, match="WHATSAPP_APP_SECRET"):
        reload_config(EILAF_ENV="prod", WHATSAPP_APP_SECRET="")


def test_prod_imports_with_app_secret(reload_config):
    reloaded = reload_config(EILAF_ENV="prod", WHATSAPP_APP_SECRET="s3cret")
    assert reloaded.APP_SECRET == "s3cret"


def test_dev_still_tolerates_a_missing_app_secret(reload_config):
    reloaded = reload_config(EILAF_ENV="", WHATSAPP_APP_SECRET="")
    assert reloaded.APP_SECRET == ""


def test_send_failure_carries_the_graph_api_explanation(monkeypatch):
    """A failed send must surface Meta's reason, not just the status line.

    ``raise_for_status()`` reports "400 Bad Request" and discards the body, which
    is where Meta actually says *why* — an expired token and a recipient missing
    from the test allow-list look identical in the log without it.
    """
    body = '{"error":{"message":"(#131030) Recipient phone number not in allowed list"}}'
    request = httpx.Request("POST", "https://graph.facebook.com/v21.0/1/messages")
    monkeypatch.setattr(
        client.httpx, "post", lambda *a, **k: httpx.Response(400, text=body, request=request)
    )
    monkeypatch.setattr(client.config, "ACCESS_TOKEN", "t0ken")
    monkeypatch.setattr(client.config, "PHONE_NUMBER_ID", "1")

    with pytest.raises(httpx.HTTPStatusError, match="131030"):
        client.send_text("972500000000", "hi")


def test_prod_refuses_to_serve_on_a_rejected_token(monkeypatch):
    """A token Meta rejects must stop the boot, not degrade to a mute bot."""
    monkeypatch.setattr(server, "get_pipeline", lambda: None)
    monkeypatch.setattr(server.config, "ENV", "prod")
    monkeypatch.setattr(
        server.client,
        "check_token",
        lambda: client.TokenStatus(False, None, "190: Session has expired"),
    )
    with pytest.raises(RuntimeError, match="rejected by Meta"):
        with TestClient(server.app):
            pass


def test_dev_serves_on_a_rejected_token(monkeypatch):
    """Outside prod the same token only warns — offline work stays possible."""
    monkeypatch.setattr(server, "get_pipeline", lambda: None)
    monkeypatch.setattr(server.config, "ENV", "")
    monkeypatch.setattr(
        server.client,
        "check_token",
        lambda: client.TokenStatus(False, None, "190: Session has expired"),
    )
    with TestClient(server.app) as c:
        assert c.get("/webhook").status_code == 403  # up and serving


def test_seconds_left_is_none_when_the_token_never_expires():
    assert client.TokenStatus(True, 0, "ok").seconds_left is None
    assert client.TokenStatus(True, None, "ok").seconds_left is None
