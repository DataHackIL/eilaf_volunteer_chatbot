"""Runtime configuration for the WhatsApp Cloud API front-end.

Every value comes from an environment variable, loaded from the repo-root
``.env`` (explicit path, so it resolves however the server is launched).
Secrets are **never** hard-coded here or baked into the Docker image — they are
injected at run time via ``--env-file .env``. Every variable is listed, with
where to get it, in the repo-root ``.env.example``; :func:`check_wiring` runs at
boot and names whatever is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root is …/app/whatsapp/config.py → parents[2]. Explicit path so it
# resolves the same however the server is launched (uvicorn, docker, tests).
# Under docker compose there is no file here — env_file injects the variables.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE)

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

# --- answer engine -----------------------------------------------------------
# Which head phrases the answer from the retrieved passages, and the variables
# it cannot work without (any one of them, where there are several). The heads
# themselves are built in ``pipeline_singleton``; this is plain data so the boot
# check can name a missing key without importing the model stack.
ENGINE_KEYS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "claude": ("ANTHROPIC_API_KEY",),
    # Same model as "claude", on the project's AWS account: no key, but a region
    # (credentials come from the instance role / AWS_PROFILE).
    "bedrock": ("EILAF_BEDROCK_REGION",),
    "cerebras": ("CEREBRAS_API_KEY",),
    # Returns the retrieved passages verbatim. For trying the wiring before any
    # LLM key exists.
    "echo": (),
}
GENERATOR = os.environ.get("EILAF_GENERATOR", "").strip().lower() or "gemini"

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
            f"{name} is not set — add it to .env (.env.example says where to get it)"
        )
    return value


# Where each required setting comes from, for the boot check's message. The
# full story for every variable is in .env.example; this is enough to act on.
_REQUIRED_WHERE = {
    "WHATSAPP_ACCESS_TOKEN": (
        "Meta app dashboard > WhatsApp > API Setup > access token "
        "(a system-user token for anything that must outlive 24h)"
    ),
    "WHATSAPP_PHONE_NUMBER_ID": "Meta app dashboard > WhatsApp > API Setup > Phone number ID",
    "WHATSAPP_VERIFY_TOKEN": (
        "any string you invent; paste the same one into Meta's webhook config"
    ),
}


def corpus_files() -> list[Path]:
    """The documents the pipeline would load, without loading them."""
    # Imported here: the pipeline module pulls in the model stack, which a
    # config import must not.
    from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR

    return [
        path
        for pattern in ("*.json", "*.docx", "*.pdf", "*.txt")
        for path in Path(DEFAULT_STATIC_DIR).glob(pattern)
    ]


def wiring_problems() -> list[str]:
    """Everything that stops the bot from answering, each with its fix.

    All of it is checked, not just the first miss, so one boot names every gap
    instead of one per restart.
    """
    values = {
        "WHATSAPP_ACCESS_TOKEN": ACCESS_TOKEN,
        "WHATSAPP_PHONE_NUMBER_ID": PHONE_NUMBER_ID,
        "WHATSAPP_VERIFY_TOKEN": VERIFY_TOKEN,
    }
    problems = [
        f"{name} is not set — {_REQUIRED_WHERE[name]}"
        for name, value in values.items()
        if not value
    ]
    if GENERATOR not in ENGINE_KEYS:
        problems.append(
            f"EILAF_GENERATOR={GENERATOR!r} is not an engine — use one of "
            + ", ".join(ENGINE_KEYS)
        )
    else:
        keys = ENGINE_KEYS[GENERATOR]
        if keys and not any(os.environ.get(key, "") for key in keys):
            problems.append(
                f"{keys[0]} is not set, and the answer engine is {GENERATOR!r} "
                "(EILAF_GENERATOR) — set the key, or pick another engine"
            )
    if IS_PROD:
        # ``server._verify_signature`` skips the App-Secret HMAC when the secret
        # is empty: convenient offline, an *open webhook* on a public host —
        # anyone who finds the URL could drive the pipeline and send messages as
        # us. A botched secret refresh must crash-loop visibly (compose restarts
        # it) rather than serve traffic unauthenticated.
        if not APP_SECRET:
            problems.append(
                "WHATSAPP_APP_SECRET is not set, and EILAF_ENV=prod — Meta app "
                "dashboard > App settings > Basic > App secret"
            )
        # Without a bucket the report store falls back to a directory *inside
        # the container*, which every `docker compose up --force-recreate`
        # deletes — losing reports the user was told we had kept, while the
        # save itself reports success.
        if not REPORT_BUCKET:
            problems.append(
                "EILAF_REPORT_BUCKET is not set, and EILAF_ENV=prod — reports "
                "would land inside the container and die with it"
            )
    if not corpus_files():
        from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR

        problems.append(
            f"the corpus is empty ({DEFAULT_STATIC_DIR} has no documents), so "
            "every answer would be blank — run ./scripts/curate_data.sh, or set "
            "EILAF_CORPUS_S3_URI to pull it at boot"
        )
    # Inside a container there is never a file here (compose injects the values,
    # and itself refuses to start without the repo-root .env), so the hint
    # would only mislead there.
    if problems and not ENV_FILE.exists() and not Path("/.dockerenv").exists():
        problems.append(
            f"no .env at {ENV_FILE} — copy .env.example to .env and fill it in"
        )
    return problems


def check_wiring() -> None:
    """Refuse to boot on a gap :func:`wiring_problems` finds, listing them all."""
    problems = wiring_problems()
    if problems:
        raise RuntimeError("the bot is not wired up:\n  - " + "\n  - ".join(problems))


def graph_url() -> str:
    """The Cloud API send-message endpoint for the configured phone number."""
    pid = require("WHATSAPP_PHONE_NUMBER_ID", PHONE_NUMBER_ID)
    return f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
