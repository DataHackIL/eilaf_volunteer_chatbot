"""FastAPI webhook server for the WhatsApp Cloud API front-end.

Two endpoints on ``/webhook``:

* ``GET``  — Meta's one-time verification handshake (echo ``hub.challenge``).
* ``POST`` — inbound messages. We validate the ``X-Hub-Signature-256`` HMAC,
  pull the text / tapped-button out of the payload, and hand it to the
  conversation state machine. The (blocking) retrieval + generation runs in a
  background task so we return ``200`` to Meta immediately — Meta retries on a
  slow/failed response, which would otherwise double-process a turn.

  Only what the user actually sent gets that far: WhatsApp also delivers events
  nobody typed (see ``_UNPROMPTED_TYPES``), and answering one means greeting a
  user who never said hello.

Run with ``python -m app.whatsapp.server`` (uvicorn, single worker so the warm
pipeline is loaded once and shared).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from app.whatsapp import client, config
from app.whatsapp.conversation import Reply, handle_message
from app.whatsapp.pipeline_singleton import get_pipeline
from app.whatsapp.reporting import install_log_handler

# uvicorn configures its own loggers but leaves the ROOT logger bare, so records
# from this one propagate to a handler-less root: anything below WARNING is
# dropped and never reaches `docker compose logs`. That silently hid the boot
# check's "token ok, expires in N days" line — the half of the check that tells
# you it ran at all. basicConfig is a no-op when handlers already exist, so this
# defers to any real logging config rather than fighting it.
logging.basicConfig(
    level=os.environ.get("EILAF_LOG_LEVEL", "INFO"),
    format="%(levelname)s:     %(name)s: %(message)s",
)
logger = logging.getLogger("whatsapp")

# Under EILAF_ENV=dev only, a second handler copies the log to the store, so it
# outlives the container that `--force-recreate` replaces. Installed here rather
# than in the lifespan so the boot check's own lines are captured too. Returns
# None in every other mode, and `logging.shutdown` flushes it on the way out.
_log_handler = install_log_handler()


# Warn this far ahead of the access token's expiry. The dashboard's tokens last
# 24h and an exchanged one 60 days, so a week is enough notice to rotate without
# crying wolf on every boot.
_TOKEN_WARN_SECONDS = 7 * 24 * 3600


def _check_access_token() -> None:
    """Report the access token's health at boot; refuse to serve on a dead one.

    An expired token fails *asymmetrically*: Meta keeps delivering webhooks and
    still shows a healthy callback URL, so the bot receives everything and
    answers nothing. That is invisible without watching the logs, which nobody
    does on an unattended host — hence a check at the one moment someone is
    likely to be looking, and a hard stop under EILAF_ENV=prod.
    """
    status = client.check_token()
    if status.usable is None:
        # Unknown, not broken — a Graph API blip must not keep the bot down.
        logger.warning("access token not verified: %s", status.detail)
        return
    if not status.usable:
        message = f"WHATSAPP_ACCESS_TOKEN was rejected by Meta — {status.detail}"
        if config.IS_PROD:
            raise RuntimeError(message + " (refusing to start: replies would fail)")
        logger.error("%s — replies WILL fail", message)
        return

    left = status.seconds_left
    if left is None:
        logger.info("access token ok (expiry unknown; set WHATSAPP_APP_ID to see it)")
    elif left <= 0:
        logger.error("access token expired %d days ago", -left // 86400)
    elif left < _TOKEN_WARN_SECONDS:
        logger.warning("access token expires in %d days — rotate it", left // 86400)
    else:
        logger.info("access token ok, expires in %d days", left // 86400)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if config.IS_DEV:
        # Say it plainly at boot: this mode keeps every conversation, and the
        # only thing telling users is a notice in the welcome message.
        logger.warning(
            "EILAF_ENV=dev — every conversation and this log are being saved to %s",
            config.REPORT_BUCKET or config.REPORT_DIR,
        )
        if not config.REPORT_BUCKET:
            logger.warning(
                "no EILAF_REPORT_BUCKET set: saving to %s. Under compose that "
                "path is bind-mounted from the host, so it survives a recreate "
                "— but it is one box's disk, not S3, and nothing backs it up",
                config.REPORT_DIR,
            )
    # Before the slow model load, so a dead credential is reported in seconds
    # rather than after a minute of warming a pipeline that cannot reply.
    await run_in_threadpool(_check_access_token)
    # Load the e5 model + embedding cache once at boot (in a thread so the event
    # loop isn't blocked), not lazily on the first user's message.
    await run_in_threadpool(get_pipeline)
    yield
    # Uploads the tail of the log that no later record would have pushed out.
    if _log_handler is not None:
        await run_in_threadpool(_log_handler.flush)


app = FastAPI(title="Eilaf WhatsApp chatbot", lifespan=lifespan)


@app.get("/webhook")
async def verify(request: Request) -> Response:
    """Meta webhook verification: echo the challenge when the token matches."""
    params = request.query_params
    expected = config.require("WHATSAPP_VERIFY_TOKEN", config.VERIFY_TOKEN)
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == expected:
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification failed")


@app.post("/webhook")
async def receive(request: Request, background: BackgroundTasks) -> Response:
    raw = await request.body()
    _verify_signature(request, raw)
    data = json.loads(raw or b"{}")
    for message in _iter_messages(data):
        wa_id = message.get("from")
        if not wa_id:
            continue
        text, button_id = _extract(message)
        background.add_task(_process, wa_id, text, button_id)
    return Response(status_code=200)


def _verify_signature(request: Request, raw: bytes) -> None:
    """Validate the App-Secret HMAC on the raw body (skipped if unconfigured).

    Leaving ``WHATSAPP_APP_SECRET`` unset disables the check — convenient for
    local/offline testing. Production cannot reach this state: ``EILAF_ENV=prod``
    makes the secret mandatory at import (see ``config``'s production guards),
    so the container refuses to boot rather than serve an open webhook.
    """
    secret = config.APP_SECRET
    if not secret:
        return
    sig = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=403, detail="bad signature")


# Inbound ``messages[].type`` values that are not the user saying anything to
# us. They arrive in the same array as a typed question, but carry no text and
# no tapped button — so handing one to the state machine is indistinguishable
# from a silent first contact, and the bot greets someone who never said hello.
#
# ``request_welcome`` is the one that bites. WhatsApp emits it when the user
# merely *opens* the chat thread while no conversation is live, which — once a
# session has expired — is every few hours, unprompted. (It is Meta's
# conversational-components welcome trigger, and can also be switched off in
# WhatsApp Manager; the bot should not depend on a dashboard toggle for this.)
# ``system`` is a number-change notice and ``reaction`` is a tap on an old
# message: neither is a question, and neither is worth a welcome either.
_UNPROMPTED_TYPES = frozenset({"request_welcome", "system", "reaction"})


def _iter_messages(data: dict) -> Iterator[dict]:
    """Yield each message the user actually sent.

    Status/delivery callbacks never reach here — they are not in ``messages``
    at all — and the unprompted event types above are dropped, so the welcome
    is a reply to the user and never an opening move of our own.
    """
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            for message in change.get("value", {}).get("messages", []):
                mtype = message.get("type", "")
                if mtype in _UNPROMPTED_TYPES:
                    # Logged rather than dropped in silence: this line is the
                    # only record that the event arrived at all, and it names
                    # the type to look for if something keeps emitting one.
                    logger.info("ignoring %r: the user sent nothing", mtype)
                    continue
                yield message


def _extract(message: dict) -> tuple[str | None, str | None]:
    """Pull ``(text, button_id)`` from a message; both None for unsupported types."""
    mtype = message.get("type")
    if mtype == "text":
        return message.get("text", {}).get("body"), None
    if mtype == "interactive":
        interactive = message.get("interactive", {})
        if interactive.get("type") == "button_reply":
            return None, interactive.get("button_reply", {}).get("id")
    # A voice note, a photo, a location: the user did send something, so it is
    # answered (with the welcome, or a re-prompt for the step they are on) —
    # we just have nothing to read. Logged because a type showing up here
    # regularly is a hint that it belongs in _UNPROMPTED_TYPES instead.
    logger.info("message type %r carries no text or button", mtype)
    return None, None


def _process(wa_id: str, text: str | None, button_id: str | None) -> None:
    """Advance the conversation and send the reply/-ies (runs off the request)."""
    try:
        for reply in handle_message(wa_id, text, button_id):
            _send(wa_id, reply)
    except Exception:  # noqa: BLE001 — background task: log, don't crash the worker
        logger.exception("failed handling message from %s", wa_id)


def _send(wa_id: str, reply: Reply) -> None:
    if reply.buttons:
        client.send_buttons(wa_id, reply.body, reply.buttons)
    else:
        client.send_text(wa_id, reply.body)


def main() -> None:
    import uvicorn

    uvicorn.run("app.whatsapp.server:app", host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
