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
import json
import logging
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.i18n import TEXTS, ArabicText, EnglishText, HebrewText, Language
from app.whatsapp import client, config, conversation, server
from app.whatsapp.reporting import (
    LocalReportStore,
    S3ReportStore,
    StoreLogHandler,
    build_report,
    report_path,
)
from app.whatsapp.users import SQLiteUserStore, user_key
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
        return [
            Document(
                text="clause",
                source="benefits.json",
                id="benefits.json#0",
                citations=[{"text": "Benefits guide", "url": "https://example.org/benefits"}],
            )
        ]


@pytest.fixture(autouse=True)
def _no_user_store(monkeypatch):
    """Returning-user recognition off unless a test turns it on.

    The repo ``.env`` may well set ``EILAF_USER_SALT``, and without this every
    test would read and write the real ``data/users`` database.
    """
    monkeypatch.setattr(conversation, "get_user_store", lambda: None)


@pytest.fixture
def user_store(monkeypatch, tmp_path):
    store = SQLiteUserStore(tmp_path / "users.sqlite")
    monkeypatch.setattr(conversation, "get_user_store", lambda: store)
    monkeypatch.setattr(conversation.config, "USER_SALT", "pepper")
    return store


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
    # Sources are held back behind a button rather than appended to the answer.
    assert "Benefits guide" not in replies[0].body
    assert len(replies) == 2  # answer + follow-up
    # The follow-up offers the one lever that still moves retrieval, and the
    # sources.
    assert _button_ids(replies[1]) == ["refine", "sources:1"]

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
    assert _button_ids(replies[1]) == ["refine", "sources:2"]
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


def test_show_sources_on_tap(fake_pipeline):
    wa = "sources"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    conversation.handle_message(wa, text="What am I entitled to?")

    replies = conversation.handle_message(wa, button_id="sources:1")
    assert len(replies) == 1
    # Sources are the passages' citations, never the corpus file name.
    assert replies[0].body == (
        f"{EnglishText.SOURCES.value}:\n• Benefits guide (https://example.org/benefits)"
    )
    assert "benefits.json" not in replies[0].body
    assert conversation._SESSIONS[wa].step is conversation.Step.QUERY
    assert len(fake_pipeline.retrieve_calls) == 1  # nothing re-run


def test_show_sources_on_an_older_answer_shows_its_own(fake_pipeline, monkeypatch):
    wa = "sources-old"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    conversation.handle_message(wa, text="first?")
    monkeypatch.setattr(
        conversation, "_format_sources", lambda contexts: "• Second source"
    )
    conversation.handle_message(wa, text="second?")

    assert "Benefits guide" in conversation.handle_message(wa, button_id="sources:1")[0].body
    assert "Second source" in conversation.handle_message(wa, button_id="sources:2")[0].body


def test_show_sources_after_they_are_gone(fake_pipeline):
    """A tap from before a restart, or on an id we never issued."""
    wa = "sources-gone"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    conversation.handle_message(wa, text="What am I entitled to?")
    conversation.handle_message(wa, text="restart")
    for bid in ("sources:1", "sources:x"):
        replies = conversation.handle_message(wa, button_id=bid)
        assert replies[0].body == EnglishText.SOURCES_UNAVAILABLE.value


def test_sources_kept_for_a_bounded_number_of_answers(fake_pipeline, monkeypatch):
    monkeypatch.setattr(conversation, "_MAX_SOURCES", 2)
    wa = "sources-bound"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    for q in ("a?", "b?", "c?"):
        conversation.handle_message(wa, text=q)
    assert sorted(conversation._SESSIONS[wa].sources) == [2, 3]


def test_no_sources_button_when_there_are_none(fake_pipeline, monkeypatch):
    monkeypatch.setattr(conversation, "_format_sources", lambda contexts: "")
    wa = "sources-none"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    replies = conversation.handle_message(wa, text="What am I entitled to?")
    assert _button_ids(replies[1]) == ["refine"]


def test_a_new_user_gets_the_full_welcome_and_is_remembered(fake_pipeline, user_store):
    wa = "972500000100"
    conversation.reset(wa)
    replies = conversation.handle_message(wa, text="hi")
    assert EnglishText.WELCOME.value in replies[0].body
    record = user_store.get(user_key(wa, "pepper"))
    assert record is not None and record.language == "en"


def test_a_returning_user_gets_welcome_back_in_their_language(fake_pipeline, user_store):
    wa = "972500000101"
    user_store.touch(user_key(wa, "pepper"), "ar", "2026-10-01")
    conversation.reset(wa)
    # Digits carry no script signal, so the remembered language decides.
    replies = conversation.handle_message(wa, text="123")
    assert replies[0].body == (
        f"{ArabicText.WELCOME_BACK.value}\n\n{ArabicText.QUERY_PROMPT.value}"
    )
    assert _button_ids(replies[0]) == ["lang:he", "lang:en"]


def test_the_message_script_beats_the_remembered_language(fake_pipeline, user_store):
    wa = "972500000102"
    key = user_key(wa, "pepper")
    user_store.touch(key, "ar", "2026-10-01")
    conversation.reset(wa)
    replies = conversation.handle_message(wa, text="hello")
    assert replies[0].body.startswith(EnglishText.WELCOME_BACK.value)
    record = user_store.get(key)
    assert record.language == "en"
    assert record.first_seen == "2026-10-01"  # kept across updates


def test_a_language_switch_is_remembered(fake_pipeline, user_store):
    wa = "972500000103"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    conversation.handle_message(wa, button_id="lang:he")
    assert user_store.get(user_key(wa, "pepper")).language == "he"


def test_the_record_holds_no_phone_number(fake_pipeline, user_store):
    wa = "972500000104"
    conversation.reset(wa)
    conversation.handle_message(wa, text="hi")
    assert wa.encode() not in user_store.path.read_bytes()


def test_a_broken_user_store_does_not_break_the_conversation(fake_pipeline, monkeypatch):
    class Broken:
        def get(self, key):
            raise OSError("disk gone")

        def touch(self, *args):
            raise OSError("disk gone")

    monkeypatch.setattr(conversation, "get_user_store", lambda: Broken())
    monkeypatch.setattr(conversation.config, "USER_SALT", "pepper")
    wa = "972500000105"
    conversation.reset(wa)
    replies = conversation.handle_message(wa, text="hi")
    assert EnglishText.WELCOME.value in replies[0].body
    replies = conversation.handle_message(wa, text="What am I entitled to?")
    assert "answer[English]" in replies[0].body


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


@pytest.fixture(autouse=True)
def _wired(monkeypatch):
    """Pass the boot-time wiring check; tests/test_whatsapp_wiring.py covers it.

    It needs real credentials and a corpus on disk, neither of which a test
    starting the app should depend on.
    """
    monkeypatch.setattr(server.config, "check_wiring", lambda: None)


@pytest.fixture
def webhook_client(monkeypatch):
    # Avoid loading the real e5 pipeline at lifespan startup.
    monkeypatch.setattr(server, "get_pipeline", lambda: None)
    monkeypatch.setattr(conversation, "get_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(server.config, "VERIFY_TOKEN", "verify-me")
    monkeypatch.setattr(server.config, "APP_SECRET", "")  # skip signature in tests
    sent: list = []
    monkeypatch.setattr(server, "_send", lambda wa_id, reply: sent.append((wa_id, reply)))
    # The typing indicator lands in the same list, so tests can check ordering.
    monkeypatch.setattr(
        server.client, "show_typing", lambda message_id: sent.append(("typing", message_id))
    )
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


def _inbound(message: dict) -> dict:
    """A webhook payload carrying one inbound message object."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {"changes": [{"field": "messages", "value": {"messages": [message]}}]}
        ],
    }


@pytest.mark.parametrize("mtype", ["request_welcome", "system", "reaction"])
def test_webhook_stays_silent_on_events_the_user_did_not_send(webhook_client, mtype):
    """The welcome is a reply, never an opening move.

    ``request_welcome`` is the one that was reaching real phones: WhatsApp
    emits it when the user merely opens the chat thread with no conversation
    live, so once a session had expired it arrived — and was answered — every
    few hours, unprompted. It carries no text and no button, which is exactly
    what a silent first contact looks like to the state machine.
    """
    wa = "972500000001"
    client, sent = webhook_client
    conversation.reset(wa)
    resp = client.post("/webhook", json=_inbound({"from": wa, "type": mtype}))
    assert resp.status_code == 200
    assert sent == []
    # And nothing was started on our side either, so the next real message is
    # still a first contact.
    assert wa not in conversation._SESSIONS


def test_saying_hi_after_one_is_ignored_still_gets_the_welcome(webhook_client):
    wa = "972500000002"
    client, sent = webhook_client
    conversation.reset(wa)
    client.post("/webhook", json=_inbound({"from": wa, "type": "request_welcome"}))
    assert sent == []
    client.post(
        "/webhook",
        json=_inbound({"from": wa, "type": "text", "text": {"body": "hi"}}),
    )
    assert len(sent) == 1
    assert EnglishText.WELCOME.value in sent[0][1].body


def test_an_unreadable_message_the_user_sent_is_still_answered(webhook_client):
    """A voice note is not a question we can read, but it *is* the user talking.

    It stays on the answered path (the welcome, in the default language) — only
    events the user never sent are dropped.
    """
    wa = "972500000003"
    client, sent = webhook_client
    conversation.reset(wa)
    resp = client.post("/webhook", json=_inbound({"from": wa, "type": "audio"}))
    assert resp.status_code == 200
    assert len(sent) == 1
    assert HebrewText.WELCOME.value in sent[0][1].body


def test_typing_indicator_shown_before_the_reply(webhook_client):
    wa = "972500000004"
    client, sent = webhook_client
    conversation.reset(wa)
    client.post(
        "/webhook",
        json=_inbound({"from": wa, "id": "wamid.X", "type": "text", "text": {"body": "hi"}}),
    )
    assert sent[0] == ("typing", "wamid.X")
    assert sent[1][0] == wa


def test_typing_indicator_failure_does_not_block_the_reply(webhook_client, monkeypatch):
    def boom(message_id):
        raise httpx.HTTPError("graph down")

    monkeypatch.setattr(server.client, "show_typing", boom)
    wa = "972500000005"
    client, sent = webhook_client
    conversation.reset(wa)
    client.post(
        "/webhook",
        json=_inbound({"from": wa, "id": "wamid.Y", "type": "text", "text": {"body": "hi"}}),
    )
    assert len(sent) == 1 and sent[0][0] == wa


def test_show_typing_payload(monkeypatch):
    posted: list = []
    monkeypatch.setattr(client, "_post", posted.append)
    client.show_typing("wamid.Z")
    assert posted == [
        {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": "wamid.Z",
            "typing_indicator": {"type": "text"},
        }
    ]


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


def test_prod_refuses_to_boot_without_app_secret(reload_config):
    reloaded = reload_config(EILAF_ENV="prod", WHATSAPP_APP_SECRET="")
    problems = reloaded.wiring_problems()
    assert any(p.startswith("WHATSAPP_APP_SECRET") for p in problems)


def test_prod_imports_with_app_secret(reload_config):
    reloaded = reload_config(
        EILAF_ENV="prod",
        WHATSAPP_APP_SECRET="s3cret",
        EILAF_REPORT_BUCKET="eilaf-reports",
    )
    assert reloaded.APP_SECRET == "s3cret"


def test_prod_refuses_to_boot_without_a_report_bucket(reload_config):
    """Without one the report store falls back to a directory inside the
    container, which --force-recreate deletes — losing reports the user was
    told we had kept, with the save itself reporting success."""
    reloaded = reload_config(
        EILAF_ENV="prod", WHATSAPP_APP_SECRET="s3cret", EILAF_REPORT_BUCKET=""
    )
    problems = reloaded.wiring_problems()
    assert any(p.startswith("EILAF_REPORT_BUCKET") for p in problems)


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
    monkeypatch.setattr(server.config, "IS_PROD", True)
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
    monkeypatch.setattr(server.config, "IS_PROD", False)
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


def test_boot_check_actually_logs_the_token_lifetime(monkeypatch, caplog):
    """The healthy path must *reach a handler*, not just avoid raising.

    This regressed once invisibly: the check ran, decided the token was fine,
    and logged at INFO to a logger whose records propagated to a handler-less
    root — so `docker compose logs` showed nothing and there was no way to tell
    the check had run. Asserting on the exception paths alone missed it.
    """
    expires = int(time.time()) + 30 * 86400 + 3600
    monkeypatch.setattr(
        server.client, "check_token", lambda: client.TokenStatus(True, expires, "ok")
    )
    with caplog.at_level(logging.INFO, logger="whatsapp"):
        server._check_access_token()
    assert "expires in 30 days" in caplog.text


def test_boot_check_warns_when_expiry_is_close(monkeypatch, caplog):
    expires = int(time.time()) + 3 * 86400
    monkeypatch.setattr(
        server.client, "check_token", lambda: client.TokenStatus(True, expires, "ok")
    )
    with caplog.at_level(logging.INFO, logger="whatsapp"):
        server._check_access_token()
    assert "rotate it" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# --------------------------------------------------------------------------- #
# Session expiry and explicit restart                                          #
# --------------------------------------------------------------------------- #


def _drive(wa_id, *messages):
    """Send each message in turn, returning the final batch of replies."""
    replies = []
    for m in messages:
        replies = conversation.handle_message(wa_id, text=m)
    return replies


def test_idle_session_expires_and_next_message_starts_fresh(fake_pipeline, monkeypatch):
    """Coming back after the TTL is a new conversation, not a continuation."""
    conversation.reset("u-ttl")
    _drive("u-ttl", "שלום", "מה מגיע לי?")
    assert conversation._SESSIONS["u-ttl"].query  # mid-conversation

    # Age the session past the TTL rather than sleeping.
    session = conversation._SESSIONS["u-ttl"]
    session.last_seen -= config.SESSION_TTL_SECONDS + 1

    conversation.handle_message("u-ttl", text="שאלה חדשה")
    fresh = conversation._SESSIONS["u-ttl"]
    assert fresh.query == "", "stale query survived expiry"
    assert fresh.context == conversation.DEFAULT_CONTEXT_ANCHOR


def test_sweep_drops_idle_sessions_so_the_dict_does_not_grow(fake_pipeline):
    """The only bound on _SESSIONS: without this it grows for every wa_id ever."""
    conversation._SESSIONS.clear()
    conversation._last_sweep = 0.0
    for i in range(5):
        conversation.handle_message(f"u-sweep-{i}", text="hi")
    for session in conversation._SESSIONS.values():
        session.last_seen -= config.SESSION_TTL_SECONDS + 1

    conversation._last_sweep = 0.0  # allow the throttled sweep to run
    conversation.handle_message("u-live", text="hi")
    assert set(conversation._SESSIONS) == {"u-live"}


def test_sweep_is_throttled(fake_pipeline, monkeypatch):
    """A sweep on every turn would be O(sessions) per message for no gain."""
    conversation._SESSIONS.clear()
    conversation.handle_message("u-a", text="hi")
    conversation._SESSIONS["u-a"].last_seen -= config.SESSION_TTL_SECONDS + 1
    # _last_sweep was just set by the call above, so this one must not sweep;
    # u-a still goes, but via the per-user check, not the sweep.
    conversation.handle_message("u-b", text="hi")
    assert "u-b" in conversation._SESSIONS


def test_restart_keyword_restarts_immediately(fake_pipeline):
    """Exact keyword, no confirmation: the conversation is cleared on the spot."""
    conversation.reset("u-exit")
    conversation.handle_message("u-exit", text="what am I entitled to?")
    conversation.handle_message("u-exit", text="what am I entitled to?")
    assert conversation._SESSIONS["u-exit"].query

    replies = conversation.handle_message("u-exit", text="restart")
    session = conversation._SESSIONS["u-exit"]
    assert session.query == ""
    assert session.context == conversation.DEFAULT_CONTEXT_ANCHOR
    assert session.step is conversation.Step.QUERY
    assert replies[0].buttons is None, "no confirmation buttons any more"


def test_restart_keeps_the_language(fake_pipeline):
    """The language was detected, not part of what the user asked to clear —
    and a confirmation nobody can read is a poor way to start over."""
    conversation.reset("u-yes")
    conversation.handle_message("u-yes", text="what am I entitled to?")
    conversation._SESSIONS["u-yes"].language = Language.ENGLISH

    replies = conversation.handle_message("u-yes", text="restart")
    assert conversation._SESSIONS["u-yes"].language is Language.ENGLISH
    assert TEXTS[Language.ENGLISH].RESTART_DONE.value in replies[0].body


def test_restart_works_from_any_step(fake_pipeline):
    """Including mid-refine, where a stale context would otherwise persist."""
    conversation.reset("u-mid")
    conversation.handle_message("u-mid", text="what am I entitled to?")
    conversation.handle_message("u-mid", text="what am I entitled to?")
    conversation._SESSIONS["u-mid"].step = conversation.Step.REFINE
    conversation.handle_message("u-mid", text="התחל מחדש")
    assert conversation._SESSIONS["u-mid"].step is conversation.Step.QUERY


def test_a_question_containing_an_exit_word_is_not_eaten(fake_pipeline):
    """Exact-match only — "stop" inside a real sentence must stay a question."""
    conversation.reset("u-word")
    conversation.handle_message("u-word", text="hello")
    replies = conversation.handle_message(
        "u-word", text="can my employer stop paying me while I recover?"
    )
    assert any("answer[" in r.body for r in replies)


def test_only_the_three_documented_phrases_restart(fake_pipeline):
    """Synonyms must NOT restart: the phrase is advertised, and destructive."""
    for phrase in ("reset", "exit", "quit", "stop", "/restart", "איפוס", "خروج"):
        conversation.reset("u-syn")
        conversation.handle_message("u-syn", text="what am I entitled to?")
        conversation.handle_message("u-syn", text="what am I entitled to?")
        conversation.handle_message("u-syn", text=phrase)
        assert conversation._SESSIONS["u-syn"].query, f"{phrase!r} wrongly restarted"


def test_each_language_phrase_restarts(fake_pipeline):
    for phrase in ("restart", "התחל מחדש", "من البداية"):
        conversation.reset("u-lang")
        conversation.handle_message("u-lang", text="what am I entitled to?")
        conversation.handle_message("u-lang", text="what am I entitled to?")
        conversation.handle_message("u-lang", text=phrase)
        assert conversation._SESSIONS["u-lang"].query == "", f"{phrase!r} did not restart"


def test_welcome_tells_the_user_how_to_restart(fake_pipeline):
    """The phrase is only discoverable if we say it — so assert we do."""
    conversation.reset("u-hint")
    replies = conversation.handle_message("u-hint", text="hello")
    assert TEXTS[Language.ENGLISH].RESTART_HINT.value in replies[0].body


# --------------------------------------------------------------------------- #
# Reporting a conversation                                                     #
# --------------------------------------------------------------------------- #


class _FakeStore:
    """Records what would have been written, and can be made to fail."""

    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail

    def save(self, report):
        if self.fail:
            raise RuntimeError("bucket on fire")
        self.saved.append(report)
        return f"fake://{report['report_id']}"


@pytest.fixture
def store(monkeypatch):
    fake = _FakeStore()
    monkeypatch.setattr(conversation, "get_report_store", lambda: fake)
    return fake


def _report(wa_id, reason=None, contact=None):
    """Walk the whole report exchange; None means "tap Skip". Returns the last replies."""
    conversation.handle_message(wa_id, text="report")
    conversation.handle_message(wa_id, button_id="report:approve")
    conversation.handle_message(
        wa_id, **({"button_id": "skip"} if reason is None else {"text": reason})
    )
    return conversation.handle_message(
        wa_id, **({"button_id": "skip"} if contact is None else {"text": contact})
    )


def test_the_transcript_records_both_directions(fake_pipeline):
    """Nothing was kept before this; without it a report has nothing to carry."""
    conversation.reset("u-tr")
    conversation.handle_message("u-tr", text="hello")
    conversation.handle_message("u-tr", text="what am I entitled to?")

    transcript = conversation._SESSIONS["u-tr"].transcript
    assert [t.role for t in transcript] == ["user", "bot", "user", "bot", "bot"]
    assert transcript[0].text == "hello"
    assert "answer[" in transcript[3].text


def test_the_transcript_is_capped(fake_pipeline, monkeypatch):
    """A session lives up to 3h; nothing else bounds this list."""
    monkeypatch.setattr(conversation, "_MAX_TURNS", 6)
    conversation.reset("u-cap")
    for _ in range(20):
        conversation.handle_message("u-cap", text="what am I entitled to?")

    transcript = conversation._SESSIONS["u-cap"].transcript
    assert len(transcript) == 6
    assert transcript[-1].role == "bot", "the newest turns are the ones kept"


def test_the_report_phrase_offers_consent_with_buttons(fake_pipeline, store):
    conversation.reset("u-rep")
    conversation.handle_message("u-rep", text="hello")
    replies = conversation.handle_message("u-rep", text="report")

    assert TEXTS[Language.ENGLISH].REPORT_CONSENT.value in replies[0].body
    assert _button_ids(replies[0]) == ["report:approve", "report:cancel"]
    assert conversation._SESSIONS["u-rep"].step is conversation.Step.REPORT_CONSENT
    assert store.saved == [], "nothing is stored before the tap"


def test_typed_text_at_the_consent_step_is_not_consent(fake_pipeline, store):
    """Reading a typed "yes" as approval means guessing at consent for permanent
    storage — and guessing wrong for exactly the user who missed the message."""
    for answer in ("yes", "ok", "כן", "نعم", "approve"):
        conversation.reset("u-typed")
        conversation.handle_message("u-typed", text="hello")
        conversation.handle_message("u-typed", text="report")
        replies = conversation.handle_message("u-typed", text=answer)

        assert store.saved == [], f"{answer!r} wrongly approved a report"
        assert _button_ids(replies[0]) == ["report:approve", "report:cancel"]
        assert conversation._SESSIONS["u-typed"].step is conversation.Step.REPORT_CONSENT


def test_cancel_saves_nothing_and_returns_to_the_question_step(fake_pipeline, store):
    conversation.reset("u-cancel")
    conversation.handle_message("u-cancel", text="hello")
    conversation.handle_message("u-cancel", text="report")
    replies = conversation.handle_message("u-cancel", button_id="report:cancel")

    assert store.saved == []
    assert TEXTS[Language.ENGLISH].REPORT_CANCELLED.value in replies[0].body
    assert conversation._SESSIONS["u-cancel"].step is conversation.Step.QUERY
    assert conversation._SESSIONS["u-cancel"].report is None


def test_approving_and_skipping_both_questions_saves_once(fake_pipeline, store):
    conversation.reset("u-skip")
    conversation.handle_message("u-skip", text="hello")
    conversation.handle_message("u-skip", text="what am I entitled to?")
    replies = _report("u-skip")

    assert len(store.saved) == 1
    report = store.saved[0]
    assert report["reason"] is None and report["contact"] is None
    assert report["language"] == "en"
    assert report["turns"], "the conversation itself was saved"
    assert TEXTS[Language.ENGLISH].REPORT_DONE.value in replies[0].body
    assert conversation._SESSIONS["u-skip"].step is conversation.Step.QUERY


def test_a_typed_reason_and_contact_are_stored(fake_pipeline, store):
    conversation.reset("u-detail")
    conversation.handle_message("u-detail", text="hello")
    _report("u-detail", reason="the answer was wrong", contact="me@example.com")

    report = store.saved[0]
    assert report["reason"] == "the answer was wrong"
    assert report["contact"] == "me@example.com"


def test_the_saved_report_carries_no_phone_number(fake_pipeline, store):
    """The consent message promises this, and the wa_id *is* the phone number.

    Asserted over the serialized document rather than a key, so a number
    reaching the file by any route — a new field, a nested value — fails here.
    """
    wa_id = "972501234567"
    conversation.reset(wa_id)
    conversation.handle_message(wa_id, text="hello")
    conversation.handle_message(wa_id, text="what am I entitled to?")
    _report(wa_id)

    report = store.saved[0]
    assert "wa_id" not in report
    assert wa_id not in json.dumps(report, ensure_ascii=False)
    assert report["contact"] is None


def test_the_snapshot_excludes_the_report_exchange(fake_pipeline, store):
    """What is stored is the conversation the user was looking at when they
    decided to report it — not the consent turns that decision produced."""
    conversation.reset("u-snap")
    conversation.handle_message("u-snap", text="hello")
    conversation.handle_message("u-snap", text="what am I entitled to?")
    before = len(conversation._SESSIONS["u-snap"].transcript)
    _report("u-snap", reason="a reason", contact="me@example.com")

    turns = store.saved[0]["turns"]
    assert len(turns) == before
    blob = json.dumps(turns, ensure_ascii=False)
    assert TEXTS[Language.ENGLISH].REPORT_CONSENT.value not in blob
    assert "a reason" not in blob and "me@example.com" not in blob
    # The trigger itself is a command, not something the user said to us. (The
    # welcome does contain the word, since it advertises the phrase — so this
    # checks for the turn, not the substring.)
    assert not [t for t in turns if t["text"].lower() == "report"]


def test_a_second_report_does_not_carry_the_first_ones_contact_details(
    fake_pipeline, store
):
    """The consent turns are machinery, not conversation, so they are never
    recorded. Otherwise the contact details typed into one report would reappear
    inside the next as if the user had said them to us."""
    conversation.reset("u-twice")
    conversation.handle_message("u-twice", text="hello")
    conversation.handle_message("u-twice", text="what am I entitled to?")
    _report("u-twice", reason="first reason", contact="me@example.com")
    _report("u-twice")

    first, second = store.saved
    assert second["contact"] is None and second["reason"] is None
    blob = json.dumps(second["turns"], ensure_ascii=False)
    assert "me@example.com" not in blob
    assert "first reason" not in blob
    assert second["turns"] == first["turns"], "the conversation itself is unchanged"


def test_a_failing_store_tells_the_user_and_does_not_crash(
    fake_pipeline, monkeypatch, caplog
):
    """Claiming a report was kept when it was not is the worst outcome here."""
    monkeypatch.setattr(conversation, "get_report_store", lambda: _FakeStore(fail=True))
    conversation.reset("u-fail")
    conversation.handle_message("u-fail", text="hello")
    with caplog.at_level(logging.ERROR):
        replies = _report("u-fail")

    assert TEXTS[Language.ENGLISH].REPORT_FAILED.value in replies[0].body
    assert "bucket on fire" in caplog.text
    assert conversation._SESSIONS["u-fail"].step is conversation.Step.QUERY
    assert conversation._SESSIONS["u-fail"].report is None


def test_restart_mid_report_abandons_it(fake_pipeline, store):
    """Someone who wants out of the report gets out, and nothing is stored."""
    conversation.reset("u-abort")
    conversation.handle_message("u-abort", text="hello")
    conversation.handle_message("u-abort", text="report")
    conversation.handle_message("u-abort", button_id="report:approve")
    conversation.handle_message("u-abort", text="restart")

    assert store.saved == []
    assert conversation._SESSIONS["u-abort"].step is conversation.Step.QUERY
    assert conversation._SESSIONS["u-abort"].report is None


def test_the_report_phrase_works_from_any_step(fake_pipeline, store):
    conversation.reset("u-anystep")
    conversation.handle_message("u-anystep", text="hello")
    conversation.handle_message("u-anystep", text="what am I entitled to?")
    conversation._SESSIONS["u-anystep"].step = conversation.Step.REFINE
    replies = conversation.handle_message("u-anystep", text="report")

    assert _button_ids(replies[0]) == ["report:approve", "report:cancel"]


def test_a_question_containing_the_word_report_is_not_eaten(fake_pipeline, store):
    """Exact-match only — a real question about police reports stays a question."""
    conversation.reset("u-rword")
    conversation.handle_message("u-rword", text="hello")
    replies = conversation.handle_message(
        "u-rword", text="do I have to file a police report first?"
    )
    assert any("answer[" in r.body for r in replies)
    assert store.saved == []


def test_each_language_phrase_opens_a_report(fake_pipeline, store):
    for phrase in ("report", "דיווח", "تبليغ"):
        conversation.reset("u-rlang")
        conversation.handle_message("u-rlang", text="hello")
        conversation.handle_message("u-rlang", text=phrase)
        step = conversation._SESSIONS["u-rlang"].step
        assert step is conversation.Step.REPORT_CONSENT, f"{phrase!r} did not report"


def test_welcome_tells_the_user_how_to_report(fake_pipeline):
    """Like the restart phrase, it is only discoverable if we name it."""
    conversation.reset("u-rhint")
    replies = conversation.handle_message("u-rhint", text="hello")
    assert TEXTS[Language.ENGLISH].REPORT_HINT.value in replies[0].body


@pytest.mark.parametrize(
    "phrase,language",
    [("report", Language.ENGLISH), ("דיווח", Language.HEBREW), ("تبليغ", Language.ARABIC)],
)
def test_reporting_as_the_very_first_message_reads_in_that_language(
    fake_pipeline, store, phrase, language
):
    """The trigger phrase is language-specific, so it doubles as the sample —
    the consent message is the one that most needs to be understood."""
    conversation.reset("u-first")
    replies = conversation.handle_message("u-first", text=phrase)
    assert TEXTS[language].REPORT_CONSENT.value in replies[0].body


def test_the_report_hint_names_the_phrase_that_actually_works():
    """The hint is quoted verbatim to the user; a drift here advertises a
    phrase that does nothing. Mirrors the restart-hint check above."""
    for texts in TEXTS.values():
        quoted = texts.REPORT_HINT.value.split('"')[1]
        assert quoted.strip().lower() in conversation._REPORT_KEYWORDS


# --------------------------------------------------------------------------- #
# Report stores                                                                #
# --------------------------------------------------------------------------- #


def _report_doc(**kw):
    """A report with a pinned date, so key assertions do not rot next January."""
    report = build_report(language="he", turns=[{"role": "user", "text": "שלום"}], **kw)
    report["created_at"] = "2026-09-13T12:00:00+00:00"
    return report


def test_report_path_is_date_partitioned_and_derived_from_the_document():
    """From created_at, not the clock: a report built a second before midnight
    and written just after must not land on a day its own contents deny."""
    report = _report_doc()
    report["created_at"] = "2026-09-13T23:59:59+00:00"
    path = report_path(report)
    assert path.startswith("reports/2026/09/13/")
    assert path.endswith(f"-{report['report_id']}.json")


def test_local_store_writes_readable_utf8_json(tmp_path):
    """Hebrew and Arabic must survive as letters, not \\uXXXX escapes."""
    store = LocalReportStore(tmp_path)
    location = store.save(_report_doc(reason="סיבה"))

    written = Path(location)
    assert written.parent.parent.parent.parent.name == "reports"  # <kind>/YYYY/MM/DD/
    assert "שלום" in written.read_text(encoding="utf-8")
    assert json.loads(written.read_text(encoding="utf-8"))["reason"] == "סיבה"


class _StubS3:
    def __init__(self):
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)


def test_s3_store_key_layout_and_body():
    client = _StubS3()
    store = S3ReportStore("eilaf-reports", "conversations/", client=client)
    report = _report_doc()
    location = store.save(report)

    (call,) = client.calls
    assert call["Bucket"] == "eilaf-reports"
    assert call["Key"] == f"conversations/{report_path(report)}"
    assert call["ContentType"] == "application/json; charset=utf-8"
    assert json.loads(call["Body"].decode()) == report
    assert location == f"s3://eilaf-reports/{call['Key']}"


@pytest.mark.parametrize("prefix", ["conversations", "conversations/", "/conversations/"])
def test_s3_prefix_is_normalised(prefix):
    """Joining a prefix blindly gives "conversations2026/…" or a double slash,
    each of which reads as a different folder in the console."""
    client = _StubS3()
    S3ReportStore("b", prefix, client=client).save(_report_doc())
    assert client.calls[0]["Key"].startswith("conversations/reports/2026/09/13/")


def test_s3_store_with_no_prefix_writes_at_the_root():
    client = _StubS3()
    S3ReportStore("b", "", client=client).save(_report_doc())
    assert client.calls[0]["Key"].startswith("reports/2026/")


def test_s3_put_object_params_are_valid_for_the_real_api():
    """The stub above accepts anything, including a misspelled parameter.

    botocore's Stubber validates the call against the actual S3 service model,
    so this is the one check that would fail before a deploy rather than on the
    first real report — and there is no bucket to try it against yet.
    """
    import json as _json

    import boto3
    from botocore.stub import Stubber

    client = boto3.client(
        "s3",
        region_name="eu-north-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    report = _report_doc()
    stubber = Stubber(client)
    stubber.add_response(
        "put_object",
        {"ETag": '"abc"'},
        {
            "Bucket": "eilaf-reports",
            "Key": f"conversations/{report_path(report)}",
            "Body": _json.dumps(report, ensure_ascii=False, indent=2).encode(),
            "ContentType": "application/json; charset=utf-8",
        },
    )
    with stubber:
        location = S3ReportStore("eilaf-reports", "conversations/", client=client).save(
            report
        )
        stubber.assert_no_pending_responses()
    assert location.startswith("s3://eilaf-reports/conversations/")


# --------------------------------------------------------------------------- #
# EILAF_ENV=dev: every conversation kept, and the user told so up front        #
# --------------------------------------------------------------------------- #


@pytest.fixture
def dev_mode(monkeypatch, store, fake_pipeline):
    """Turn on dev mode for the conversation layer, with a recording store.

    ``fake_pipeline`` is not optional here even for tests that never look at an
    answer: without it these drive the real pipeline, loading e5 and reading the
    corpus off disk.
    """
    monkeypatch.setattr(conversation.config, "IS_DEV", True)
    return store


def test_dev_mode_notice_comes_first_in_the_welcome(dev_mode):
    """Saving starts on this turn, so telling the user later is telling them
    too late — the notice leads, before the greeting."""
    conversation.reset("u-dev")
    replies = conversation.handle_message("u-dev", text="hello")
    assert replies[0].body.startswith(EnglishText.DEV_NOTICE.value)
    assert EnglishText.WELCOME.value in replies[0].body


def test_no_dev_notice_when_dev_mode_is_off(fake_pipeline, store):
    conversation.reset("u-nodev")
    replies = conversation.handle_message("u-nodev", text="hello")
    assert EnglishText.DEV_NOTICE.value not in replies[0].body
    assert store.saved == [], "nothing is kept outside dev mode"


@pytest.mark.parametrize("language", [Language.HEBREW, Language.ARABIC])
def test_correcting_the_language_repeats_the_notice(dev_mode, language):
    """The realistic reason to tap a language button is that you could not read
    the welcome — which is where the notice was."""
    conversation.reset("u-devlang")
    conversation.handle_message("u-devlang", text="hello")
    replies = conversation.handle_message("u-devlang", button_id=f"lang:{language.value}")
    assert TEXTS[language].DEV_NOTICE.value in replies[0].body


def test_the_notice_is_not_repeated_once_a_question_has_been_asked(dev_mode):
    conversation.reset("u-devlate")
    conversation.handle_message("u-devlate", text="hello")
    conversation.handle_message("u-devlate", text="what am I entitled to?")
    replies = conversation.handle_message("u-devlate", button_id="lang:he")
    assert HebrewText.DEV_NOTICE.value not in replies[0].body


def test_dev_mode_saves_every_conversation_without_being_asked(dev_mode):
    conversation.reset("u-devsave")
    conversation.handle_message("u-devsave", text="hello")
    conversation.handle_message("u-devsave", text="what am I entitled to?")

    assert len(dev_mode.saved) == 2, "saved after each turn"
    latest = dev_mode.saved[-1]
    assert latest["kind"] == "chats"
    assert any("what am I entitled to?" == t["text"] for t in latest["turns"])


def test_dev_saves_overwrite_one_path_per_conversation(dev_mode):
    """Fixed id and timestamp for the session, so the store keeps one current
    copy instead of a trail of partial ones."""
    conversation.reset("u-devpath")
    for _ in range(4):
        conversation.handle_message("u-devpath", text="what am I entitled to?")

    paths = {report_path(r) for r in dev_mode.saved}
    assert len(paths) == 1
    assert paths.pop().startswith("chats/")


def test_a_restart_begins_a_new_dev_object(dev_mode):
    """A restart is a new conversation, so it must not overwrite the old one."""
    conversation.reset("u-devrestart")
    conversation.handle_message("u-devrestart", text="hello")
    first = report_path(dev_mode.saved[-1])
    conversation.handle_message("u-devrestart", text="restart")
    conversation.handle_message("u-devrestart", text="what am I entitled to?")

    assert report_path(dev_mode.saved[-1]) != first


def test_the_dev_autosave_never_breaks_a_conversation(
    fake_pipeline, monkeypatch, caplog
):
    """Capture is for us, not the user: a broken bucket must not cost them an
    answer, or even show them an error."""
    monkeypatch.setattr(conversation.config, "IS_DEV", True)
    monkeypatch.setattr(conversation, "get_report_store", lambda: _FakeStore(fail=True))
    conversation.reset("u-devfail")
    conversation.handle_message("u-devfail", text="hello")
    with caplog.at_level(logging.WARNING):
        replies = conversation.handle_message("u-devfail", text="what am I entitled to?")

    assert "answer[" in replies[0].body
    assert "bucket on fire" in caplog.text


def test_dev_mode_does_not_store_the_phone_number_either(dev_mode):
    """The autosaved document is the same shape as a report, and the wa_id is
    just as absent from it."""
    wa_id = "972501234567"
    conversation.reset(wa_id)
    conversation.handle_message(wa_id, text="hello")
    assert wa_id not in json.dumps(dev_mode.saved[-1], ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Dev-mode log shipping                                                        #
# --------------------------------------------------------------------------- #


class _MemoryStore:
    """Records put() calls; stands in for S3 or the filesystem."""

    def __init__(self, fail=False):
        self.puts = []
        self.fail = fail

    def put(self, path, data, content_type):
        if self.fail:
            raise RuntimeError("bucket on fire")
        self.puts.append((path, data, content_type))
        return f"mem://{path}"


def _handler(store, **kw):
    handler = StoreLogHandler(store, **kw)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    return handler


def _record(message, level=logging.INFO):
    return logging.LogRecord("t", level, __file__, 1, message, None, None)


def test_log_handler_uploads_a_chunk_once_the_buffer_fills():
    store = _MemoryStore()
    handler = _handler(store, max_records=3, max_seconds=1e9)
    for i in range(3):
        handler.emit(_record(f"line {i}"))

    (path, data, content_type) = store.puts[0]
    assert data.decode() == "INFO line 0\nINFO line 1\nINFO line 2\n"
    assert content_type == "text/plain; charset=utf-8"
    assert path.startswith("logs/")
    assert path.endswith("-00001.log")


def test_log_chunks_are_sequenced_within_a_run():
    """S3 has no append, so each flush is its own object; the names have to
    sort in the order the lines were written."""
    store = _MemoryStore()
    handler = _handler(store, max_records=1, max_seconds=1e9)
    for i in range(3):
        handler.emit(_record(f"line {i}"))

    names = [p.rsplit("/", 1)[1] for p, _, _ in store.puts]
    assert names == sorted(names)
    assert [n.split("-")[1] for n in names] == ["00001.log", "00002.log", "00003.log"]
    assert len({n.split("-")[0] for n in names}) == 1, "one run id"


def test_closing_the_handler_flushes_the_tail():
    """logging.shutdown closes handlers at exit, which is what stops the last
    few lines of a run being lost."""
    store = _MemoryStore()
    handler = _handler(store, max_records=1000, max_seconds=1e9)
    handler.emit(_record("nearly lost"))
    assert store.puts == []

    handler.close()
    assert "nearly lost" in store.puts[0][1].decode()


def test_a_failing_log_upload_does_not_raise(capsys):
    """A broken log shipper must not take the service down with it."""
    handler = _handler(_MemoryStore(fail=True), max_records=1, max_seconds=1e9)
    handler.emit(_record("hello"))  # must not raise
    assert "log upload failed" in capsys.readouterr().err


def test_the_handler_does_not_re_enter_itself_while_uploading():
    """boto3 logs while it works. Without the guard those records land in the
    buffer being drained, and at DEBUG level the flush never ends."""
    store = _MemoryStore()
    handler = _handler(store, max_records=1, max_seconds=1e9)

    logged_during_upload = []
    real_put = store.put

    def chatty_put(path, data, content_type):
        handler.emit(_record("botocore says hello"))
        logged_during_upload.append(path)
        return real_put(path, data, content_type)

    store.put = chatty_put
    handler.emit(_record("real line"))

    assert len(logged_during_upload) == 1, "the nested emit did not recurse"
    assert store.puts[0][1].decode() == "INFO real line\n"


def test_install_log_handler_is_dev_only(monkeypatch):
    import app.whatsapp.reporting as reporting

    monkeypatch.setattr(reporting.config, "IS_DEV", False)
    assert reporting.install_log_handler() is None

    monkeypatch.setattr(reporting.config, "IS_DEV", True)
    monkeypatch.setattr(reporting, "get_report_store", lambda: _MemoryStore())
    handler = reporting.install_log_handler()
    try:
        assert handler in logging.getLogger().handlers
    finally:
        logging.getLogger().removeHandler(handler)
