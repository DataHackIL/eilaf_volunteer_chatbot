"""Per-user conversation state machine for the WhatsApp front-end.

WhatsApp has no equivalent of Streamlit's ``st.session_state``, so we keep our
own per-user state (keyed by the sender's WhatsApp id, ``wa_id``) and serialize
the Streamlit app's one-page inputs into a multi-turn dialog:

    language → question → violent-event description → gender → age → answer

Each optional step offers a **Skip** button; the collected answers become the
same ``facts`` dict the Streamlit app builds (``streamlit_app.py`` ~L146), then
we call the shared pipeline and reply in the user's chosen language.

State is in-memory (``_SESSIONS``); it is per-process and resets on restart,
which is fine for the single-process interim deployment. Swapping it for SQLite
later is a localized change. ``handle_message`` is synchronous and may call the
(blocking) pipeline — the server runs it in a threadpool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

from app.visualizer.i18n import TEXTS, Language
from app.whatsapp.pipeline_singleton import get_pipeline
from chatbot.rag.pipeline.pipeline import DEFAULT_CONTEXT_ANCHOR

# Button ids (echoed back by WhatsApp as interactive ``button_reply.id``).
_SKIP = "skip"
_LANG_PREFIX = "lang:"
_GENDER_PREFIX = "gender:"

# Shown before a language is known, so it is deliberately trilingual.
LANGUAGE_PROMPT = "בחרו שפה · اختاروا اللغة · Choose a language"


class Step(Enum):
    LANGUAGE = auto()
    QUERY = auto()
    EVENT = auto()
    GENDER = auto()
    AGE = auto()


@dataclass
class Session:
    step: Step = Step.LANGUAGE
    language: Language | None = None
    query: str = ""
    context: str = DEFAULT_CONTEXT_ANCHOR
    facts: dict = field(default_factory=dict)


@dataclass
class Reply:
    """One outbound message: plain text, or text + interactive reply buttons."""

    body: str
    buttons: list[tuple[str, str]] | None = None  # (id, title); None → plain text


# In-memory per-user state. Process-global, resets on restart.
_SESSIONS: dict[str, Session] = {}


def reset(wa_id: str) -> None:
    """Drop a user's state (e.g. for tests or an explicit restart)."""
    _SESSIONS.pop(wa_id, None)


def handle_message(
    wa_id: str, text: str | None = None, button_id: str | None = None
) -> list[Reply]:
    """Advance ``wa_id``'s conversation by one message; return the reply/-ies.

    Exactly one of ``text`` (a typed message) or ``button_id`` (a tapped
    interactive reply) is normally set. Unrecognised input at a step simply
    re-prompts that step.
    """
    session = _SESSIONS.setdefault(wa_id, Session())

    if session.step is Step.LANGUAGE:
        return _handle_language(session, button_id)
    if session.step is Step.QUERY:
        return _handle_query(session, text)
    if session.step is Step.EVENT:
        return _handle_event(session, text, button_id)
    if session.step is Step.GENDER:
        return _handle_gender(session, button_id)
    if session.step is Step.AGE:
        return _handle_age(session, text, button_id)
    raise AssertionError(f"unhandled step {session.step}")  # pragma: no cover


def _handle_language(session: Session, button_id: str | None) -> list[Reply]:
    if button_id and button_id.startswith(_LANG_PREFIX):
        session.language = Language(button_id[len(_LANG_PREFIX) :])
        session.step = Step.QUERY
        return [Reply(TEXTS[session.language].QUERY_PROMPT.value)]
    # First contact or anything that isn't a language pick → (re)offer the menu.
    return [
        Reply(
            LANGUAGE_PROMPT,
            buttons=[(f"{_LANG_PREFIX}{lang.value}", lang.native_name) for lang in Language],
        )
    ]


def _handle_query(session: Session, text: str | None) -> list[Reply]:
    t = TEXTS[session.language]
    if not text or not text.strip():
        return [Reply(t.QUERY_PROMPT.value)]
    session.query = text.strip()
    session.step = Step.EVENT
    return [Reply(t.CONTEXT_PROMPT.value, buttons=[(_SKIP, t.SKIP.value)])]


def _handle_event(
    session: Session, text: str | None, button_id: str | None
) -> list[Reply]:
    t = TEXTS[session.language]
    # Skip keeps the default domain anchor; typed text overrides it.
    if button_id != _SKIP and text and text.strip():
        session.context = text.strip()
    session.step = Step.GENDER
    return [_gender_prompt(t)]


def _handle_gender(session: Session, button_id: str | None) -> list[Reply]:
    t = TEXTS[session.language]
    if button_id and button_id.startswith(_GENDER_PREFIX):
        session.facts["gender"] = button_id[len(_GENDER_PREFIX) :]  # "f" / "m"
    elif button_id != _SKIP:
        # Typed text instead of tapping a button → re-offer the choices.
        return [_gender_prompt(t)]
    session.step = Step.AGE
    return [Reply(t.AGE.value, buttons=[(_SKIP, t.SKIP.value)])]


def _handle_age(
    session: Session, text: str | None, button_id: str | None
) -> list[Reply]:
    t = TEXTS[session.language]
    if button_id != _SKIP:
        raw = (text or "").strip()
        if not raw.isdigit() or not (0 <= int(raw) <= 120):
            return [Reply(t.INVALID_AGE.value, buttons=[(_SKIP, t.SKIP.value)])]
        session.facts["age"] = int(raw)
    return _run_pipeline(session)


def _gender_prompt(t) -> Reply:
    return Reply(
        t.GENDER.value,
        buttons=[
            (f"{_GENDER_PREFIX}f", t.GENDER_FEMALE.value),
            (f"{_GENDER_PREFIX}m", t.GENDER_MALE.value),
            (_SKIP, t.SKIP.value),
        ],
    )


def _run_pipeline(session: Session) -> list[Reply]:
    """Retrieve + generate, then reset to accept another question."""
    t = TEXTS[session.language]
    pipeline = get_pipeline()
    contexts = pipeline.retrieve(
        session.query, session.facts or None, context=session.context
    )
    # Retrieval already succeeded; a generator key/quota error still lets us show
    # the sources, mirroring the Streamlit app's try/except.
    try:
        answer = pipeline.generator.generate(
            session.query, contexts, session.language.prompt_name
        )
    except Exception as exc:  # noqa: BLE001 — surface any SDK/quota error to the user
        answer = t.GENERATOR_ERROR.value.format(error=exc)

    body = f"{t.ANSWER.value}:\n{answer or t.NO_ANSWER.value}"
    sources = _format_sources(contexts)
    if sources:
        body += f"\n\n{t.SOURCES.value}:\n{sources}"

    # Keep the language, clear the rest, and go back to accepting a question.
    session.step = Step.QUERY
    session.query = ""
    session.context = DEFAULT_CONTEXT_ANCHOR
    session.facts = {}
    return [Reply(body), Reply(t.ASK_ANOTHER.value)]


def _format_sources(contexts: Sequence) -> str:
    """Compact, de-duplicated bullet list of the retrieved sources."""
    seen: list[str] = []
    for doc in contexts:
        if doc.source and doc.source not in seen:
            seen.append(doc.source)
    return "\n".join(f"• {src}" for src in seen)
