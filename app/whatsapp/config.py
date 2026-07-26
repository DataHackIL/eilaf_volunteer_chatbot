"""Runtime configuration for the WhatsApp Cloud API front-end.

Every value comes from an environment variable, loaded from the repo-root
``.env`` (same explicit-path pattern as ``app/visualizer/streamlit_app.py``).
Secrets are **never** hard-coded here or baked into the Docker image — they are
injected at run time via ``--env-file .env``. See ``app/whatsapp/SETUP.md`` for
how to obtain each value from the Meta developer dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root is …/app/whatsapp/config.py → parents[2]. Explicit path so it
# resolves the same however the server is launched (uvicorn, docker, tests).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Graph API version pinned so a Meta-side default bump can't silently change
# behaviour; override via env if you need a newer one.
GRAPH_API_VERSION = os.environ.get("GRAPH_API_VERSION", "v21.0")

ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")


def require(name: str, value: str) -> str:
    """Return ``value`` or raise if it's empty — a clear miss-config message."""
    if not value:
        raise RuntimeError(
            f"{name} is not set — add it to .env (see app/whatsapp/SETUP.md)"
        )
    return value


def graph_url() -> str:
    """The Cloud API send-message endpoint for the configured phone number."""
    pid = require("WHATSAPP_PHONE_NUMBER_ID", PHONE_NUMBER_ID)
    return f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
