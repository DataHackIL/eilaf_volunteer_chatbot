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

A conversation ends in one of three ways: the user types a restart keyword
(one phrase per language — ``restart`` / ``התחל מחדש`` / ``من البداية`` — named
in the welcome message, matched exactly, acted on at once, no confirmation), it goes idle for ``config.SESSION_TTL_SECONDS``
(default 3h), or the process restarts. The TTL is also the only bound on
``_SESSIONS``: nothing else ever removes an entry, so without it the dict grows
once per wa_id for the life of the process.

Typing a report phrase (``report`` / ``דיווח`` / ``تبليغ``, also named in the
welcome) offers to keep the conversation: the user is told that approving means
we store a copy long-term, and only an explicit button tap — never typed text —
sends it. A reason and contact details are then each offered with a Skip button.
What is stored is the conversation as it stood when the phrase arrived, and it
carries no ``wa_id``: that is the user's phone number, and the consent message
says it is not saved. See :mod:`app.whatsapp.reporting`.

Under ``EILAF_ENV=dev`` that consent step is skipped, because the conversation
is kept from the first turn regardless: the welcome leads with a notice saying
so, in the user's language, and every turn re-saves the conversation to the
store. This is a test-deployment mode — outside it nothing is kept without the
Approve tap.

State is in-memory (``_SESSIONS``); it is per-process and resets on restart,
which is fine for the single-process interim deployment. Swapping it for SQLite
later is a localized change. ``handle_message`` is synchronous and may call the
(blocking) pipeline — the server runs it in a threadpool.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto

from app.language import detect_language
from app.visualizer.i18n import TEXTS, Language
from app.whatsapp import config
from app.whatsapp.pipeline_singleton import get_pipeline
from app.whatsapp.reporting import build_report, get_report_store
from chatbot.rag.pipeline.pipeline import DEFAULT_CONTEXT_ANCHOR

logger = logging.getLogger("whatsapp.conversation")

# Button ids (echoed back by WhatsApp as interactive ``button_reply.id``).
_SKIP = "skip"
_REFINE = "refine"
_LANG_PREFIX = "lang:"
_GENDER_PREFIX = "gender:"
_REPORT_APPROVE = "report:approve"
_REPORT_CANCEL = "report:cancel"

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


# Exactly one phrase per language, and the welcome message names it. Matched
# exactly (after strip/lower), so a real question containing "restart" is still
# a question. A restart is destructive with no undo, so the list is deliberately
# short and explicit rather than a set of synonyms someone could hit by accident.
_RESTART_KEYWORDS = frozenset(
    {
        "restart",
        "התחל מחדש",
        "من البداية",
    }
)


# Same shape as the restart phrases, and named the same way in the welcome
# message: one exact phrase per language, matched after strip/lower, so a
# question *about* reporting is still a question. Unlike a restart this is not
# destructive — it opens a consent message and nothing is stored until the user
# taps Approve — so a stray match costs one tappable message, not a lost
# conversation.
_REPORT_KEYWORDS = frozenset(
    {
        "report",
        "דיווח",
        "تبليغ",
    }
)


class Step(Enum):
    WELCOME = auto()  # first contact: detect the language
    QUERY = auto()  # the steady state — a message is a question, answered at once
    REFINE = auto()  # opt-in: the user tapped "Add details" after an answer
    # The report exchange: consent, then two optional questions.
    REPORT_CONSENT = auto()
    REPORT_REASON = auto()
    REPORT_CONTACT = auto()
    # Reached only under ``config.COLLECT_FACTS``.
    EVENT = auto()
    GENDER = auto()
    AGE = auto()


@dataclass(frozen=True)
class Turn:
    """One message in either direction, as it would appear in a report."""

    role: str  # "user" or "bot"
    text: str
    button_id: str | None = None
    # Wall-clock UTC, as it will be read. ``Session.last_seen`` is monotonic and
    # measures elapsed time; it is not a date and cannot be rendered as one.
    # Nothing does arithmetic on this, so it is stored in its final form.
    at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


@dataclass
class ReportDraft:
    """A report being assembled, from the trigger until it is saved or dropped.

    ``turns`` is a snapshot taken when the report phrase arrived, so the consent
    exchange that follows — and anything the user says afterwards — is not in
    what gets stored. That keeps the file equal to the conversation the user was
    looking at when they decided to report it, which is what they approved.
    """

    turns: list[Turn]
    reason: str | None = None
    contact: str | None = None


@dataclass
class Session:
    # Monotonic, not wall-clock: only elapsed time matters here, and monotonic
    # cannot jump backwards on an NTP correction and strand a live session.
    last_seen: float = field(default_factory=time.monotonic)
    step: Step = Step.WELCOME
    language: Language = DEFAULT_LANGUAGE
    # Identity for the dev-mode autosave, fixed for the life of the session so
    # every save of this conversation lands on the same path and overwrites the
    # last. Unused (and unwritten) outside dev mode.
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    # The opening message, held until the question arrives and prepended to it.
    pending_query: str = ""
    query: str = ""
    context: str = DEFAULT_CONTEXT_ANCHOR
    facts: dict = field(default_factory=dict)
    # Every turn so far, kept only in memory and only for the life of the
    # session: without it there is nothing for a report to contain. It leaves
    # the process only if the user asks for that explicitly and approves.
    transcript: list[Turn] = field(default_factory=list)
    report: ReportDraft | None = None


@dataclass
class Reply:
    """One outbound message: plain text, or text + interactive reply buttons."""

    body: str
    buttons: list[tuple[str, str]] | None = None  # (id, title); None → plain text


# In-memory per-user state. Process-global, resets on restart.
_SESSIONS: dict[str, Session] = {}

# Sweeping every message would be O(len(_SESSIONS)) per turn for no benefit —
# a user's own stale session is caught by the per-user check in handle_message
# regardless. This sweep exists only to stop the dict growing without bound, so
# running it occasionally is enough.
_SWEEP_INTERVAL_SECONDS = 60.0
_last_sweep: float = 0.0

# Oldest turns are dropped past this. A session lives up to SESSION_TTL_SECONDS
# (3h by default) and nothing else bounds the transcript, so a long or abusive
# conversation would otherwise grow a process-global list without limit. A
# report of a conversation this long is already far more than anyone will read.
_MAX_TURNS = 200


def _is_expired(session: Session, now: float) -> bool:
    return now - session.last_seen >= config.SESSION_TTL_SECONDS


def _sweep(now: float) -> None:
    """Drop every session idle past the TTL. Throttled; safe to call often."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_INTERVAL_SECONDS:
        return
    _last_sweep = now
    for wa_id in [k for k, s in _SESSIONS.items() if _is_expired(s, now)]:
        del _SESSIONS[wa_id]


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

    This is a thin recording wrapper around :func:`_dispatch`. Keeping the
    transcript here — in one place, rather than in each step handler — is what
    stops it drifting as steps are added, and it captures typed text, button
    taps and every reply alike.

    Two rules keep the report exchange out of the transcript, and both matter:

    * Turns are recorded *after* the dispatch, so a handler running inside it
      sees the conversation as it stood before this message. That is what
      ``_start_report`` snapshots, and it is why the trigger phrase is not in
      the stored file.
    * The exchange itself is not recorded at all. The transcript is meant to be
      the *conversation*; the consent turns are machinery. Were they recorded,
      a second report in the same session would carry the first one's reason
      and contact details as if the user had said them to us.
    """
    existing = _SESSIONS.get(wa_id)
    before = existing.step if existing is not None else None

    replies = _dispatch(wa_id, text, button_id)

    # A restart replaces the session and an expiry can delete one, so re-read it
    # rather than holding a reference across the dispatch. The identity check
    # keeps a *replaced* session from inheriting the old one's step: coming back
    # after the TTL expired mid-report is a new conversation, and its first turn
    # is ordinary.
    session = _SESSIONS.get(wa_id)
    if session is None:
        return replies
    was_reporting = session is existing and _is_report_step(before)
    if not (was_reporting or _is_report_step(session.step)):
        _record(session, Turn("user", (text or "").strip(), button_id))
        for reply in replies:
            _record(session, Turn("bot", reply.body))
        _autosave(session)
    return replies


def _autosave(session: Session) -> None:
    """In dev mode only, keep the store's copy of this conversation current.

    Re-saved after every turn rather than flushed when the conversation ends:
    ``report_id`` and ``created_at`` are fixed for the session, so each save
    overwrites the same object. That costs one PutObject per turn — nothing
    beside the retrieval and generation the same turn just did — and in exchange
    a conversation is captured even when the process is killed mid-session,
    which an end-of-conversation flush cannot promise.

    A failure here is never the user's problem: dev capture is for us, so it is
    logged and swallowed rather than surfaced or retried.
    """
    if not config.IS_DEV or not session.transcript:
        return
    try:
        get_report_store().save(
            build_report(
                kind="chats",
                language=session.language.value,
                turns=[asdict(turn) for turn in session.transcript],
                report_id=session.session_id,
                created_at=session.started_at,
            )
        )
    except Exception:  # noqa: BLE001 — dev capture must never break a conversation
        logger.warning(
            "dev autosave failed for session %s", session.session_id, exc_info=True
        )


def _is_report_step(step: Step | None) -> bool:
    return step in (Step.REPORT_CONSENT, Step.REPORT_REASON, Step.REPORT_CONTACT)


def _record(session: Session, turn: Turn) -> None:
    session.transcript.append(turn)
    if len(session.transcript) > _MAX_TURNS:
        del session.transcript[: len(session.transcript) - _MAX_TURNS]


def _dispatch(
    wa_id: str, text: str | None = None, button_id: str | None = None
) -> list[Reply]:
    now = time.monotonic()
    _sweep(now)

    # A conversation that has been idle past the TTL is not resumed: the next
    # message starts a new one. Someone coming back hours later is asking a
    # fresh question, and silently answering it against stale context (a prior
    # query, a described event) is worse than starting over.
    session = _SESSIONS.get(wa_id)
    if session is not None and _is_expired(session, now):
        del _SESSIONS[wa_id]
        session = None
    if session is None:
        session = _SESSIONS[wa_id] = Session()
    session.last_seen = now

    # Restart is handled ahead of the step dispatch, like language below, so it
    # works from any step. An exact keyword match restarts immediately: there is
    # no confirmation and no attempt to infer the intent from free text, because
    # a wrong guess either wipes a conversation nobody asked to end or adds a
    # round-trip to every user who typed the word deliberately.
    if text and text.strip().lower() in _RESTART_KEYWORDS:
        return _restart(wa_id, session)

    # Reporting is handled here for the same reason, and deliberately keeps the
    # restart check ahead of it: someone who wants out mid-report gets out.
    if text and text.strip().lower() in _REPORT_KEYWORDS:
        return _start_report(session, text)

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
    # A report step with no draft behind it has nothing to collect into, and
    # would fail deep inside a background task (i.e. as silence for the user).
    # Nothing reaches this today; it costs one comparison to keep it that way.
    if _is_report_step(session.step) and session.report is None:
        session.step = Step.QUERY
        return _reprompt(session)
    if session.step is Step.REPORT_CONSENT:
        return _handle_report_consent(session, button_id)
    if session.step is Step.REPORT_REASON:
        return _handle_report_reason(session, text, button_id)
    if session.step is Step.REPORT_CONTACT:
        return _handle_report_contact(session, text, button_id)
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
    # The dev notice goes first, before anything else has been said: under
    # EILAF_ENV=dev the conversation is saved from this turn onward, so telling
    # the user afterwards would be telling them too late.
    parts = [t.DEV_NOTICE.value] if config.IS_DEV else []
    parts += [
        t.WELCOME.value,
        t.QUERY_PROMPT.value,
        t.CHANGE_LANGUAGE.value,
        t.RESTART_HINT.value,
        t.REPORT_HINT.value,
    ]
    return [Reply("\n\n".join(parts), buttons=_switch_buttons(session.language))]


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
    replies = _reprompt(session)
    # Correcting the language is the realistic case for someone who could not
    # read the welcome — which in dev mode is where the notice was. Repeat it in
    # the language they just chose, while there is still nothing recorded but
    # hellos. Once they have asked something they have seen it and moved on.
    if config.IS_DEV and not session.query and replies:
        t = TEXTS[session.language]
        replies[0] = Reply(
            f"{t.DEV_NOTICE.value}\n\n{replies[0].body}", buttons=replies[0].buttons
        )
    return replies


def _restart(wa_id: str, session: Session) -> list[Reply]:
    """Drop the conversation and start a new one, keeping the language.

    The language is carried across deliberately: it was detected or chosen, not
    part of what the user asked to clear, and a confirmation the user cannot
    read is a poor way to begin.
    """
    language = session.language
    reset(wa_id)
    fresh = _SESSIONS[wa_id] = Session()
    fresh.language = language
    fresh.step = Step.QUERY  # language is known; nothing to detect
    t = TEXTS[language]
    return [Reply(f"{t.RESTART_DONE.value}\n\n{t.QUERY_PROMPT.value}")]


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


def _start_report(session: Session, text: str | None) -> list[Reply]:
    """Snapshot the conversation and explain what approving would mean."""
    if session.step is Step.WELCOME:
        # The trigger phrase is itself language-specific, so it doubles as the
        # language sample the opening message would otherwise have been. Without
        # this, someone whose first message is "report" reads the consent text —
        # the one message that most needs to be understood — in the default
        # language rather than theirs.
        session.language = detect_language(text) or DEFAULT_LANGUAGE
    session.report = ReportDraft(turns=list(session.transcript))
    session.step = Step.REPORT_CONSENT
    return _reprompt(session)


def _handle_report_consent(session: Session, button_id: str | None) -> list[Reply]:
    """Only the button approves. Typed text re-asks; it is not consent.

    Reading a typed "yes" (or anything else) as approval would mean guessing at
    consent for permanent storage, and the guess is wrong precisely for the user
    who did not follow the message. The button is unambiguous, so it is the only
    way through.
    """
    if button_id == _REPORT_CANCEL:
        return _cancel_report(session)
    if button_id != _REPORT_APPROVE:
        return _reprompt(session)
    session.step = Step.REPORT_REASON
    return _reprompt(session)


def _handle_report_reason(
    session: Session, text: str | None, button_id: str | None
) -> list[Reply]:
    if button_id != _SKIP and text and text.strip():
        session.report.reason = text.strip()
    session.step = Step.REPORT_CONTACT
    return _reprompt(session)


def _handle_report_contact(
    session: Session, text: str | None, button_id: str | None
) -> list[Reply]:
    if button_id != _SKIP and text and text.strip():
        session.report.contact = text.strip()
    return _save_report(session)


def _cancel_report(session: Session) -> list[Reply]:
    session.report = None
    session.step = Step.QUERY
    return [Reply(TEXTS[session.language].REPORT_CANCELLED.value)]


def _save_report(session: Session) -> list[Reply]:
    """Write the approved snapshot, and say plainly whether it was written.

    The draft is cleared and the step reset *before* the write, so a failure
    leaves the user back in the normal flow rather than stuck in a report they
    cannot finish. Note what the document does not contain: the ``wa_id``. On
    WhatsApp that is the user's phone number, and the consent message promises
    it is not saved — the only contact here is the one they typed themselves.
    """
    t = TEXTS[session.language]
    draft = session.report
    session.report = None
    session.step = Step.QUERY

    report = build_report(
        language=session.language.value,
        turns=[asdict(turn) for turn in draft.turns],
        reason=draft.reason,
        contact=draft.contact,
    )
    try:
        location = get_report_store().save(report)
    except Exception:  # noqa: BLE001 — any storage failure is the user's to hear
        logger.exception("failed saving report %s", report["report_id"])
        return [Reply(t.REPORT_FAILED.value)]
    logger.info("saved report to %s", location)
    return [Reply(t.REPORT_DONE.value)]


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
    if session.step is Step.REPORT_CONSENT:
        return [
            Reply(
                t.REPORT_CONSENT.value,
                buttons=[
                    (_REPORT_APPROVE, t.REPORT_APPROVE.value),
                    (_REPORT_CANCEL, t.REPORT_CANCEL.value),
                ],
            )
        ]
    if session.step is Step.REPORT_REASON:
        return [Reply(t.REPORT_REASON_PROMPT.value, buttons=[(_SKIP, t.SKIP.value)])]
    if session.step is Step.REPORT_CONTACT:
        return [Reply(t.REPORT_CONTACT_PROMPT.value, buttons=[(_SKIP, t.SKIP.value)])]
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
