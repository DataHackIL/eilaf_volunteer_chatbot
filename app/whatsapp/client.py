"""Thin WhatsApp Cloud API sender: plain text and interactive reply buttons.

Outbound direction only (business → user): a normal HTTPS POST to the Graph API
``/messages`` endpoint, authenticated with the access token. The inbound
direction (user → business) arrives as webhooks handled in ``server.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import time

import httpx

from app.whatsapp import config

_TIMEOUT = 30.0
# WhatsApp caps interactive reply buttons at 3, and each title at 20 chars.
_MAX_BUTTONS = 3
_MAX_TITLE = 20


@dataclass(frozen=True)
class TokenStatus:
    """What we could learn about the access token, without assuming network."""

    usable: bool | None  # None = couldn't tell (network failed); don't act on it
    expires_at: int | None  # epoch seconds; 0 means never, None means unknown
    detail: str

    @property
    def seconds_left(self) -> int | None:
        if not self.expires_at:  # unknown, or never expires
            return None
        return self.expires_at - int(time.time())


def check_token() -> TokenStatus:
    """Probe the access token so a dead one fails loudly instead of silently.

    An expired or under-scoped token is invisible from the outside: inbound
    webhooks keep arriving and Meta still reports a healthy callback URL, while
    every reply fails. So ask Meta directly.

    A plain GET on the phone number proves the token still works. ``debug_token``
    also reports the expiry but needs an app access token, so it is attempted
    only when ``WHATSAPP_APP_ID`` is set and never downgrades the verdict.
    Network trouble yields ``usable=None`` — unknown, not broken; a bot that
    refuses to boot because Meta blipped is worse than one that logs and serves.
    """
    token = config.ACCESS_TOKEN
    if not token or not config.PHONE_NUMBER_ID:
        return TokenStatus(None, None, "WHATSAPP_ACCESS_TOKEN/PHONE_NUMBER_ID not set")

    base = f"https://graph.facebook.com/{config.GRAPH_API_VERSION}"
    try:
        resp = httpx.get(
            f"{base}/{config.PHONE_NUMBER_ID}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return TokenStatus(None, None, f"could not reach the Graph API: {exc}")

    if resp.is_error:
        return TokenStatus(False, None, f"{resp.status_code}: {resp.text[:300]}")

    expires_at = None
    if config.APP_ID and config.APP_SECRET:
        try:
            dbg = httpx.get(
                f"{base}/debug_token",
                params={
                    "input_token": token,
                    "access_token": f"{config.APP_ID}|{config.APP_SECRET}",
                },
                timeout=_TIMEOUT,
            )
            if not dbg.is_error:
                expires_at = dbg.json().get("data", {}).get("expires_at")
        except (httpx.HTTPError, ValueError):
            pass  # expiry is a nicety; the token already answered for itself

    return TokenStatus(True, expires_at, "accepted by the Graph API")


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
