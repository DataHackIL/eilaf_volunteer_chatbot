"""Thin WhatsApp Cloud API sender: plain text and interactive reply buttons.

Outbound direction only (business → user): a normal HTTPS POST to the Graph API
``/messages`` endpoint, authenticated with the access token. The inbound
direction (user → business) arrives as webhooks handled in ``server.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.whatsapp import config

_TIMEOUT = 30.0
# WhatsApp caps interactive reply buttons at 3, and each title at 20 chars.
_MAX_BUTTONS = 3
_MAX_TITLE = 20


def _post(payload: dict) -> None:
    token = config.require("WHATSAPP_ACCESS_TOKEN", config.ACCESS_TOKEN)
    resp = httpx.post(
        config.graph_url(),
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=_TIMEOUT,
    )
    if resp.is_error:
        # Meta says *why* in the body — an expired token, a malformed payload,
        # or (#131030) a recipient missing from the test allow-list. Plain
        # ``raise_for_status()`` drops it and logs a bare "400 Bad Request",
        # which is indistinguishable from every other way a send can fail.
        raise httpx.HTTPStatusError(
            f"{resp.status_code} from {resp.request.url}: {resp.text}",
            request=resp.request,
            response=resp,
        )


def send_text(to: str, body: str) -> None:
    """Send a plain-text message to WhatsApp id ``to``."""
    _post(
        {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        }
    )


def send_buttons(to: str, body: str, buttons: Sequence[tuple[str, str]]) -> None:
    """Send an interactive reply-buttons message.

    ``buttons`` is a sequence of ``(id, title)`` pairs; the tapped button comes
    back as an interactive ``button_reply`` whose ``id`` matches. At most 3
    buttons are sent (WhatsApp's limit); titles are truncated to 20 chars.
    """
    _post(
        {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {"id": bid, "title": title[:_MAX_TITLE]},
                        }
                        for bid, title in list(buttons)[:_MAX_BUTTONS]
                    ]
                },
            },
        }
    )
