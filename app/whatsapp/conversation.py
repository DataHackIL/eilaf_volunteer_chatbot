"""Per-user conversation state machine for the WhatsApp front-end.

WhatsApp has no equivalent of Streamlit's ``st.session_state``, so we keep our
own per-user state (keyed by the sender's WhatsApp id, ``wa_id``) and serialize
the Streamlit app's one-page inputs into a multi-turn dialog:

    question → answer (language auto-detected; event details optional, on request)

A question is answered on the turn it arrives. The old forced follow-ups —
violent-event description, gender, age — are gone from the default path: the
corpus carries almost no age/gender/locality tags, so ``SoftMetadataFilter`` did
nothing with two of the three answers, and they only delayed the answer. They
remain behind ``WHATSAPP_COLLECT_FACTS=1`` for the day an enrichment pass
populates those tags. The event description *does* move retrieval (it replaces
``DEFAULT_CONTEXT_ANCHOR`` in the anchor blend), so it survives as an opt-in:
every answer offers an **Add details** button which re-runs the same question
against whatever the user then describes.

Language is **not** a step. Asking "which language?" costs a whole turn before
the user can say anything useful, and their opening message already answers it —
its script identifies he/ar/en (see :mod:`app.language`). So the first message is
used twice: as the language sample, and as the start of the question, which it is
prepended to so nothing the user typed is thrown away. When detection has no
signal we fall back to :data:`DEFAULT_LANGUAGE`; a wrong guess (Arabic typed in
Latin letters is the realistic case) is corrected with the reply buttons on the
opening message, or by typing "language" at any point.

Each optional step offers a **Skip** button; the collected answers become the
same ``facts`` dict the Streamlit app builds (``streamlit_app.py`` ~L146), then
we call the shared pipeline and reply in the user's language.

State is in-memory (``_SESSIONS``); it is per-process and resets on restart,
which is fine for the single-process interim deployment. Swapping it for SQLite
later is a localized change. ``handle_message`` is synchronous and may call the
(blocking) pipeline — the server runs it in a threadpool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

from app.language import detect_language
from app.visualizer.i18n import TEXTS, Language
from app.whatsapp import config
from app.whatsapp.pipeline_singleton import get_pipeline
from chatbot.rag.pipeline.pipeline import DEFAULT_CONTEXT_ANCHOR

# Button ids (echoed back by WhatsApp as interactive ``button_reply.id``).
_SKIP = "skip"
_REFINE = "refine"
_LANG_PREFIX = "lang:"
_GENDER_PREFIX = "gender:"

# Used when the opening message carries no script signal (digits, emoji, or a
# non-text message such as a voice note).
DEFAULT_LANGUAGE = Language.HEBREW

# Shown when the user asks for the menu explicitly, so it stays trilingual —
# whatever language we guessed may be the one they cannot read.
LANGUAGE_PROMPT = "בחרו שפה · اختاروا اللغة · Choose a language"

# Typing one of these re-opens the language menu, at any step. Matched exactly
# (after strip/lower) so a real question mentioning "language" is never eaten.
_LANGUAGE_KEYWORDS = frozenset(
    {
        "שפה",
        "שינוי שפה",
        "החלפת שפה",
        "לשנות שפה",
        "لغة",
        "اللغة",
        "تغيير اللغة",
        "language",
        "change language",
        "lang",
        "/lang",
    }
)


class Step(Enum):
    WELCOME = auto()  # first contact: detect the language
    QUERY = auto()  # the steady state — a message is a question, answered at once
    REFINE = auto()  # opt-in: the user tapped "Add details" after an answer
    # Reached only under ``config.COLLECT_FACTS``.
    EVENT = auto()
    GENDER = auto()
    AGE = auto()


@dataclass
class Session:
    step: Step = Step.WELCOME
    language: Language = DEFAULT_LANGUAGE
    # The opening message, held until the question arrives and prepended to it.
    pending_query: str = ""
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

    # Language changes are handled ahead of the step dispatch: a WhatsApp button
    # stays tappable in the chat history, so ``lang:`` can arrive many turns
    # after the message that offered it, and the keyword deliberately works
    # anywhere. Neither loses the user's place in the flow.
    if button_id and button_id.startswith(_LANG_PREFIX):
        return _switch_language(session, button_id[len(_LANG_PREFIX) :])
    if text and text.strip().lower() in _LANGUAGE_KEYWORDS:
        return [Reply(LANGUAGE_PROMPT, buttons=_language_buttons())]

    # "Add details" travels the same way, and needs a question to refine: a tap
    # on an answer this process no longer remembers (a restart wiped the
    # session) falls through to asking for one rather than answering nothing.
    if button_id == _REFINE and session.query:
        session.step = Step.REFINE
        return _reprompt(session)

    if session.step is Step.WELCOME:
        return _handle_welcome(session, text)
    if session.step is Step.QUERY:
        return _handle_query(session, text)
    if session.step is Step.REFINE:
        return _handle_refine(session, text)
    if session.step is Step.EVENT:
        return _handle_event(session, text, button_id)
    if session.step is Step.GENDER:
        return _handle_gender(session, button_id)
    if session.step is Step.AGE:
        return _handle_age(session, text, button_id)
    raise AssertionError(f"unhandled step {session.step}")  # pragma: no cover


def _handle_welcome(session: Session, text: str | None) -> list[Reply]:
    """First contact: guess the language, keep the message, ask the question."""
    session.language = detect_language(text) or DEFAULT_LANGUAGE
    session.pending_query = (text or "").strip()
    session.step = Step.QUERY

    t = TEXTS[session.language]
    body = "\n\n".join(
        (t.WELCOME.value, t.QUERY_PROMPT.value, t.CHANGE_LANGUAGE.value)
    )
    return [Reply(body, buttons=_switch_buttons(session.language))]


def _switch_language(session: Session, code: str) -> list[Reply]:
    """Adopt the tapped language and re-ask the current step in it."""
    try:
        session.language = Language(code)
    except ValueError:  # pragma: no cover — ids are ours, but never trust input
        return _reprompt(session)
    if session.step is Step.WELCOME:
        # Reached only when the very first message was the keyword itself, so
        # there is no question to collect yet — just move on to asking for one.
        session.step = Step.QUERY
    return _reprompt(session)


def _language_buttons() -> list[tuple[str, str]]:
    """All three languages, for the explicitly-requested menu."""
    return [(f"{_LANG_PREFIX}{lang.value}", lang.native_name) for lang in Language]


def _switch_buttons(current: Language) -> list[tuple[str, str]]:
    """The two languages the user is *not* in, to correct a wrong guess."""
    return [
        (f"{_LANG_PREFIX}{lang.value}", lang.native_name)
        for lang in Language
        if lang is not current
    ]


def _handle_query(session: Session, text: str | None) -> list[Reply]:
    new_text = (text or "").strip()
    if not new_text:
        # Keep ``pending_query``: on its own, an opening "hi" is not a question.
        return _reprompt(session)
    session.query = " ".join(p for p in (session.pending_query, new_text) if p)
    session.pending_query = ""  # consumed; later questions stand alone
    if not config.COLLECT_FACTS:
        return _run_pipeline(session)
    session.step = Step.EVENT
    return _reprompt(session)


def _handle_refine(session: Session, text: str | None) -> list[Reply]:
    """Re-answer the question we already have, against a described event."""
    if not text or not text.strip():
        return _reprompt(session)
    session.context = text.strip()
    return _run_pipeline(session)


def _handle_event(
    session: Session, text: str | None, button_id: str | None
) -> list[Reply]:
    # Skip keeps the default domain anchor; typed text overrides it.
    if button_id != _SKIP and text and text.strip():
        session.context = text.strip()
    session.step = Step.GENDER
    return _reprompt(session)


def _handle_gender(session: Session, button_id: str | None) -> list[Reply]:
    if button_id and button_id.startswith(_GENDER_PREFIX):
        session.facts["gender"] = button_id[len(_GENDER_PREFIX) :]  # "f" / "m"
    elif button_id != _SKIP:
        # Typed text instead of tapping a button → re-offer the choices.
        return _reprompt(session)
    session.step = Step.AGE
    return _reprompt(session)


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


def _reprompt(session: Session) -> list[Reply]:
    """The prompt for the step we are now on, in the session's language.

    One place per step, so advancing a step and re-asking it after a language
    switch cannot drift apart.
    """
    t = TEXTS[session.language]
    if session.step is Step.REFINE:
        return [Reply(t.EXPAND_PROMPT.value)]
    if session.step is Step.EVENT:
        return [Reply(t.CONTEXT_PROMPT.value, buttons=[(_SKIP, t.SKIP.value)])]
    if session.step is Step.GENDER:
        return [
            Reply(
                t.GENDER.value,
                buttons=[
                    (f"{_GENDER_PREFIX}f", t.GENDER_FEMALE.value),
                    (f"{_GENDER_PREFIX}m", t.GENDER_MALE.value),
                    (_SKIP, t.SKIP.value),
                ],
            )
        ]
    if session.step is Step.AGE:
        return [Reply(t.AGE.value, buttons=[(_SKIP, t.SKIP.value)])]
    return [Reply(t.QUERY_PROMPT.value)]  # Step.QUERY


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
    # ``query`` deliberately survives: "Add details" re-answers *this* question.
    # A stale one is harmless — ``_handle_query`` overwrites it.
    session.step = Step.QUERY
    session.context = DEFAULT_CONTEXT_ANCHOR
    session.facts = {}
    return [
        Reply(body),
        Reply(t.ASK_ANOTHER.value, buttons=[(_REFINE, t.ADD_DETAILS.value)]),
    ]


def _format_sources(contexts: Sequence) -> str:
    """Compact, de-duplicated bullet list of the retrieved sources."""
    seen: list[str] = []
    for doc in contexts:
        if doc.source and doc.source not in seen:
            seen.append(doc.source)
    return "\n".join(f"• {src}" for src in seen)
