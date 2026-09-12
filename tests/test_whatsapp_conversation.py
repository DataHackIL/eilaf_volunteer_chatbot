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

from app.visualizer.i18n import TEXTS, ArabicText, EnglishText, HebrewText, Language
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
# Conversation flow — the default, one-turn path                               #
# --------------------------------------------------------------------------- #


def test_a_question_is_answered_on_the_turn_it_arrives(fake_pipeline):
    wa = "happy"
    conversation.reset(wa)

    # First contact → language detected from the greeting (no menu turn), and
    # the welcome offers the other two languages in case the guess was wrong.
    replies = conversation.handle_message(wa, text="hi")
    assert _button_ids(replies[0]) == ["lang:he", "lang:ar"]
    assert "What would you like to ask?" in replies[0].body

    # The question is answered immediately — no event/gender/age turns.
    replies = conversation.handle_message(wa, text="What am I entitled to?")
    assert "answer[English]" in replies[0].body
    assert "benefits.json" in replies[0].body  # sources listed
    assert len(replies) == 2  # answer + follow-up
    # The follow-up offers the one lever that still moves retrieval.
    assert _button_ids(replies[1]) == ["refine"]

    call = fake_pipeline.retrieve_calls[-1]
    # Unconstrained, and on the default anchor: exactly what skipping every
    # question used to produce.
    assert call["facts"] is None
    assert call["context"] == "נפגע אירוע אלימות"  # DEFAULT_CONTEXT_ANCHOR
    # The opening message is kept, prepended to the question.
    assert call["query"] == "hi What am I entitled to?"
    # Language threaded into the generator as its English name.
    assert fake_pipeline.generator.last[2] == "English"

    # Back to accepting a new question (language kept).
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY


def test_add_details_re_answers_the_same_question(fake_pipeline):
    wa = "refine"
    conversation.reset(wa)
    conversation.handle_message(wa, text="שלום")
    conversation.handle_message(wa, text="מה הזכויות שלי?")
    assert len(fake_pipeline.retrieve_calls) == 1

    replies = conversation.handle_message(wa, button_id="refine")
    assert replies[0].body == HebrewText.EXPAND_PROMPT.value
    assert conversation._SESSIONS[wa].step is conversation.Step.REFINE
    assert len(fake_pipeline.retrieve_calls) == 1  # nothing re-run yet

    replies = conversation.handle_message(wa, text="תקיפה בבית")
    first, second = fake_pipeline.retrieve_calls
    assert second["query"] == first["query"]  # same question…
    assert second["context"] == "תקיפה בבית"  # …new anchor
    assert first["context"] == "נפגע אירוע אלימות"
    # And the answer is re-issued, itself refinable again.
    assert _button_ids(replies[1]) == ["refine"]
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY


def test_add_details_with_nothing_to_refine_asks_for_a_question(fake_pipeline):
    """A tap on an answer the process no longer remembers (it was restarted).

    Either way the user is asked for a question rather than answered nothing,
    and the pipeline is never run on an empty query.
    """
    # Nothing known at all → the welcome, which carries the question prompt.
    wa = "refine-cold"
    conversation.reset(wa)
    replies = conversation.handle_message(wa, button_id="refine")
    assert HebrewText.QUERY_PROMPT.value in replies[0].body

    # Past the welcome but with no question yet → the bare question prompt.
    wa = "refine-no-query"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hello")
    replies = conversation.handle_message(wa, button_id="refine")
    assert replies[0].body == EnglishText.QUERY_PROMPT.value

    assert fake_pipeline.retrieve_calls == []


# Pictographs, dingbats, misc symbols, and the variation selector that turns
# some of them into emoji. Written as code points so this file does not itself
# contain what it forbids.
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0xFE0F, 0xFE0F),
)


def test_no_ui_string_carries_an_emoji():
    """Emoji are out of place in a service for victims of violence.

    Asserted over every string, not just the follow-up that used to carry one,
    so a cheerful addition anywhere fails here instead of reaching a user.
    """
    offenders = [
        f"{texts.__name__}.{member.name}"
        for texts in TEXTS.values()
        for member in texts
        if isinstance(member.value, str)
        for char in member.value
        if any(low <= ord(char) <= high for low, high in _EMOJI_RANGES)
    ]
    assert offenders == []


def test_empty_question_reprompts(fake_pipeline):
    wa = "empty"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hello")
    replies = conversation.handle_message(wa, text="   ")
    assert replies[0].body == "What would you like to ask?"
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY
    # A bare greeting must never become the query on its own, so it is held.
    assert conversation._SESSIONS[wa].pending_query == "hello"


# --------------------------------------------------------------------------- #
# Language: detection, correction, and where it lands                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("greeting", "language"),
    [
        ("שלום", Language.HEBREW),
        ("مرحبا", Language.ARABIC),
        ("hi there", Language.ENGLISH),
        ("👋", Language.HEBREW),  # no script signal → default
        ("2", Language.HEBREW),
    ],
)
def test_first_message_sets_the_language(fake_pipeline, greeting, language):
    wa = f"detect-{language.value}-{greeting}"
    conversation.reset(wa)
    replies = conversation.handle_message(wa, text=greeting)

    assert conversation._SESSIONS[wa].language is language
    assert TEXTS[language].QUERY_PROMPT.value in replies[0].body
    # The switch buttons offer the two languages we did *not* pick.
    assert _button_ids(replies[0]) == [
        f"lang:{lang.value}" for lang in Language if lang is not language
    ]


def test_a_second_question_is_not_polluted_by_the_greeting(fake_pipeline):
    wa = "second"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    conversation.handle_message(wa, text="what am I entitled to?")
    assert fake_pipeline.retrieve_calls[-1]["query"] == "hi what am I entitled to?"

    # The opening message was consumed, not kept for every later question.
    conversation.handle_message(wa, text="and who is eligible?")
    assert fake_pipeline.retrieve_calls[-1]["query"] == "and who is eligible?"


def test_language_button_switches_without_losing_the_step(fake_pipeline):
    wa = "switch"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hello")
    conversation.handle_message(wa, text="a question")  # answered
    conversation.handle_message(wa, button_id="refine")  # now on the refine step

    replies = conversation.handle_message(wa, button_id="lang:he")
    assert conversation._SESSIONS[wa].language is Language.HEBREW
    # Same step, re-asked in Hebrew — progress is kept, not restarted.
    assert conversation._SESSIONS[wa].step is conversation.Step.REFINE
    assert replies[0].body == HebrewText.EXPAND_PROMPT.value

    conversation.handle_message(wa, text="תקיפה בבית")
    # The switch reached the generator: Hebrew, though the question was English.
    assert fake_pipeline.generator.last[2] == "Hebrew"


def test_language_keyword_reopens_the_full_menu_mid_flow(fake_pipeline):
    wa = "keyword"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hello")
    conversation.handle_message(wa, text="a question")
    conversation.handle_message(wa, button_id="refine")  # refine step

    replies = conversation.handle_message(wa, text="language")
    assert replies[0].body == conversation.LANGUAGE_PROMPT
    assert _button_ids(replies[0]) == ["lang:he", "lang:ar", "lang:en"]
    # The menu itself does not advance or reset the flow.
    assert conversation._SESSIONS[wa].step is conversation.Step.REFINE


def test_a_question_mentioning_language_is_not_taken_as_the_keyword(fake_pipeline):
    wa = "not-keyword"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    replies = conversation.handle_message(wa, text="in which language do I apply?")
    assert "answer[English]" in replies[0].body  # answered, not treated as a command
    assert "language" in fake_pipeline.retrieve_calls[-1]["query"]


def test_keyword_as_the_very_first_message_still_reaches_the_question(fake_pipeline):
    wa = "keyword-first"
    conversation.reset(wa)
    replies = conversation.handle_message(wa, text="שפה")
    assert _button_ids(replies[0]) == ["lang:he", "lang:ar", "lang:en"]

    replies = conversation.handle_message(wa, button_id="lang:ar")
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY
    assert replies[0].body == ArabicText.QUERY_PROMPT.value
    # The keyword is not mistaken for the start of a question.
    assert conversation._SESSIONS[wa].pending_query == ""


# --------------------------------------------------------------------------- #
# The fact-collecting flow, behind WHATSAPP_COLLECT_FACTS                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def collect_facts(monkeypatch):
    """Turn the retired pre-answer questions back on, as the env var does.

    Patched on the module object rather than via the environment because
    ``config`` reads its variables at import time.
    """
    monkeypatch.setattr(conversation.config, "COLLECT_FACTS", True)


def test_flagged_flow_builds_facts_and_answers(fake_pipeline, collect_facts):
    wa = "flagged"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")

    # Ask a question → event step with a Skip button, not an answer.
    replies = conversation.handle_message(wa, text="What am I entitled to?")
    assert _button_ids(replies[0]) == ["skip"]
    assert fake_pipeline.retrieve_calls == []

    # Type an event description → gender step (Female/Male/Skip).
    replies = conversation.handle_message(wa, text="assault at home")
    assert _button_ids(replies[0]) == ["gender:f", "gender:m", "skip"]

    # Female → age step.
    replies = conversation.handle_message(wa, button_id="gender:f")
    assert _button_ids(replies[0]) == ["skip"]

    # Age 30 → runs the pipeline.
    replies = conversation.handle_message(wa, text="30")
    assert "answer[English]" in replies[0].body

    call = fake_pipeline.retrieve_calls[-1]
    assert call["facts"] == {"gender": "f", "age": 30}
    assert call["context"] == "assault at home"
    assert call["query"] == "hi What am I entitled to?"


def test_flagged_flow_all_skips_default_context_and_empty_facts(
    fake_pipeline, collect_facts
):
    wa = "skips"
    conversation.reset(wa)
    conversation.handle_message(wa, text="שלום")
    conversation.handle_message(wa, text="שאלה")
    conversation.handle_message(wa, button_id="skip")  # event skipped
    conversation.handle_message(wa, button_id="skip")  # gender skipped
    conversation.handle_message(wa, button_id="skip")  # age skipped

    call = fake_pipeline.retrieve_calls[-1]
    assert call["facts"] is None  # empty facts → None (unconstrained retrieval)
    assert call["context"] == "נפגע אירוע אלימות"  # DEFAULT_CONTEXT_ANCHOR


def test_flagged_flow_invalid_age_reprompts_then_accepts(fake_pipeline, collect_facts):
    wa = "age"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hello")
    conversation.handle_message(wa, text="q")
    conversation.handle_message(wa, button_id="skip")  # event
    conversation.handle_message(wa, button_id="skip")  # gender

    replies = conversation.handle_message(wa, text="not-a-number")
    assert "valid age" in replies[0].body.lower()
    assert conversation._SESSIONS[wa].step is conversation.Step.AGE  # still on age

    replies = conversation.handle_message(wa, text="42")
    assert fake_pipeline.retrieve_calls[-1]["facts"] == {"age": 42}


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
    conversation.reset("972500000000")
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
    # Background task ran: first contact → welcome sent to the sender, in the
    # language detected from "hi", with the two switch buttons.
    assert sent and sent[0][0] == "972500000000"
    assert _button_ids(sent[0][1]) == ["lang:he", "lang:ar"]


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
