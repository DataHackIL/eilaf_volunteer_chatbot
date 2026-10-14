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
# Optional. Only used to build the app access token (``<id>|<secret>``) that
# ``debug_token`` needs to report *when* the access token expires. Without it
# the boot check can still tell whether the token works, just not for how long.
APP_ID = os.environ.get("WHATSAPP_APP_ID", "")

# Which deployment this is. Three values:
#
#   ""      (default) local/offline runs. Nothing is tightened, nothing is kept.
#   "prod"  reachable from the internet. Tightens the checks below.
#   "dev"   a test deployment. Every conversation and the service's own logs are
#           saved to the store, and the welcome message says so, in the user's
#           language, before they have typed anything. NOT for real traffic:
#           people reach this bot after a violent event, and a notice is only
#           informed consent if whoever set EILAF_ENV meant it to be read.
#
# Normalised, so EILAF_ENV=DEV and EILAF_ENV=Prod behave as written rather than
# silently landing back on the permissive default.
ENV = os.environ.get("EILAF_ENV", "").strip().lower()
IS_PROD = ENV == "prod"
IS_DEV = ENV == "dev"

# How long a conversation survives without a message before it is dropped and
# the next message starts fresh. Sessions are in-memory and never shrink on
# their own, so this is both a UX choice (a question hours later is a new
# conversation, not a continuation) and the only bound on _SESSIONS growth.
SESSION_TTL_SECONDS = int(os.environ.get("EILAF_SESSION_TTL", 3 * 60 * 60))

# Restores the old fact-collecting flow (event → gender → age before answering).
# Off by default: the corpus carries almost no age/gender/locality tags, so
# ``SoftMetadataFilter`` does nothing with the answers and the three turns only
# delay the answer. Turn it on once an enrichment pass populates those tags.
COLLECT_FACTS = os.environ.get("WHATSAPP_COLLECT_FACTS", "") == "1"

# --- reported conversations --------------------------------------------------
# Where a conversation goes once the user has read the consent message and
# tapped Approve. With a bucket set we write to S3; without one, to a local
# directory (dev machines and tests only — see the production guard below).
# No AWS key lives here or in .env: on EC2 boto3 takes the instance role.
REPORT_BUCKET = os.environ.get("EILAF_REPORT_BUCKET", "")
REPORT_PREFIX = os.environ.get("EILAF_REPORT_PREFIX", "conversations/")
# SET THIS. Measured on the box 2026-09-16: left empty, boto3 inside the
# container resolved us-east-1, not the instance's own region — botocore does
# not consult instance metadata for the region by default, and S3 falls back to
# us-east-1 when nothing else says otherwise. The credentials still come from
# the instance role; only the region is wrong, which fails against a bucket
# anywhere else as a confusing PermanentRedirect or AuthorizationHeaderMalformed.
REPORT_REGION = os.environ.get("EILAF_REPORT_REGION", "")
REPORT_DIR = os.environ.get("EILAF_REPORT_DIR", "") or str(
    Path(__file__).resolve().parents[2] / "data" / "reports"
)
# Where dev-mode log chunks go, under the same prefix as the conversations they
# explain — so one bucket prefix, and one IAM statement, covers everything.
LOG_PREFIX = os.environ.get("EILAF_LOG_PREFIX", "logs")


# --- returning users ---------------------------------------------------------
# Recognising someone who has written before (see ``app/whatsapp/users``). The
# salt keys each record by HMAC(salt, wa_id) rather than by the phone number;
# it is a secret, and empty turns the feature off. Rotating it forgets everyone.
USER_SALT = os.environ.get("EILAF_USER_SALT", "")
USER_DB = os.environ.get("EILAF_USER_DB", "") or str(
    Path(__file__).resolve().parents[2] / "data" / "users" / "users.sqlite"
)


def require(name: str, value: str) -> str:
    """Return ``value`` or raise if it's empty — a clear miss-config message."""
    if not value:
        raise RuntimeError(
            f"{name} is not set — add it to .env (see app/whatsapp/SETUP.md)"
        )
    return value


# --- production guards -------------------------------------------------------
# ``server._verify_signature`` skips the App-Secret HMAC when the secret is
# empty: convenient offline, an *open webhook* on a public host — anyone who
# finds the URL could drive the pipeline and send messages as us. Fail here, at
# import, so a missing secret stops the container before uvicorn binds instead
# of degrading silently. A botched secret refresh then crash-loops visibly
# (compose restarts it) rather than serving traffic unauthenticated.
if IS_PROD:
    require("WHATSAPP_APP_SECRET", APP_SECRET)
    # Without a bucket the report store falls back to a directory *inside the
    # container*, which every `docker compose up --force-recreate` deletes. That
    # loses reports the user was told we had kept — a promise broken invisibly,
    # since the save itself succeeds. Fail at import instead, like the secret.
    require("EILAF_REPORT_BUCKET", REPORT_BUCKET)


def graph_url() -> str:
    """The Cloud API send-message endpoint for the configured phone number."""
    pid = require("WHATSAPP_PHONE_NUMBER_ID", PHONE_NUMBER_ID)
    return f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
