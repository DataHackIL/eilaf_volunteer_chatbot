"""FastAPI webhook server for the WhatsApp Cloud API front-end.

Two endpoints on ``/webhook``:

* ``GET``  — Meta's one-time verification handshake (echo ``hub.challenge``).
* ``POST`` — inbound messages. We validate the ``X-Hub-Signature-256`` HMAC,
  pull the text / tapped-button out of the payload, and hand it to the
  conversation state machine. The (blocking) retrieval + generation runs in a
  background task so we return ``200`` to Meta immediately — Meta retries on a
  slow/failed response, which would otherwise double-process a turn.

Run with ``python -m app.whatsapp.server`` (uvicorn, single worker so the warm
pipeline is loaded once and shared).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from app.whatsapp import client, config
from app.whatsapp.conversation import Reply, handle_message
from app.whatsapp.pipeline_singleton import get_pipeline

logger = logging.getLogger("whatsapp")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the e5 model + embedding cache once at boot (in a thread so the event
    # loop isn't blocked), not lazily on the first user's message.
    await run_in_threadpool(get_pipeline)
    yield


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


def _iter_messages(data: dict) -> Iterator[dict]:
    """Yield each inbound message object; ignores status/delivery callbacks."""
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            yield from change.get("value", {}).get("messages", [])


def _extract(message: dict) -> tuple[str | None, str | None]:
    """Pull ``(text, button_id)`` from a message; both None for unsupported types."""
    mtype = message.get("type")
    if mtype == "text":
        return message.get("text", {}).get("body"), None
    if mtype == "interactive":
        interactive = message.get("interactive", {})
        if interactive.get("type") == "button_reply":
            return None, interactive.get("button_reply", {}).get("id")
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
