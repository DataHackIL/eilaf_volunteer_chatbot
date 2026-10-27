""".env.example lists every setting the bot reads, and nothing it doesn't.

It is the one place a new deployer learns what to set, so a variable that the
code reads but the file omits is a setting nobody will know exists — and an
entry the code no longer reads is a setting someone will fill in for nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.whatsapp import config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"

# What the bot's own code reads: the server and the pipeline it runs, and the
# boot path that pulls and curates the corpus. scripts/ is left out on purpose —
# those are operator tools run from a laptop, not settings of the bot.
_PY_DIRS = ("app", "chatbot", "data")
_SHELL_FILES = ("app/whatsapp/docker-compose.yml", "app/whatsapp/entrypoint.sh")

_PY_READ = re.compile(
    r"""environ(?:\.get)?[(\[]\s*["']([A-Z][A-Z0-9_]+)["']"""  # os.environ.get("X") / ["X"]
    r"""|getenv\(\s*["']([A-Z][A-Z0-9_]+)["']"""  # os.getenv("X")
    r"""|^[A-Z_]*_ENV\s*=\s*["']([A-Z][A-Z0-9_]+)["']""",  # REGION_ENV = "X"
    re.M,
)
_SHELL_READ = re.compile(r"\$\{([A-Z][A-Z0-9_]+)")


def _read_by_code() -> set[str]:
    names: set[str] = set()
    for folder in _PY_DIRS:
        for path in (ROOT / folder).rglob("*.py"):
            for match in _PY_READ.finditer(path.read_text(encoding="utf-8")):
                names.add(next(group for group in match.groups() if group))
    for rel in _SHELL_FILES:
        names |= set(_SHELL_READ.findall((ROOT / rel).read_text(encoding="utf-8")))
    # The engines' keys are read by their SDKs, not by our code.
    for keys in config.ENGINE_KEYS.values():
        names |= set(keys)
    # GOOGLE_API_KEY is the Gemini SDK's alias for GEMINI_API_KEY; listing both
    # would only invite setting both.
    return names - {"GOOGLE_API_KEY"}


def _documented() -> tuple[set[str], set[str]]:
    """(``NAME=`` entries, names in the shell-only comment block)."""
    text = EXAMPLE.read_text(encoding="utf-8")
    entries = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, re.M))
    shell_block = text.split("NOT HERE", 1)[1]
    shell_only = set(re.findall(r"^#\s+([A-Z][A-Z0-9_]+)\s", shell_block, re.M))
    return entries, shell_only


def test_every_setting_the_code_reads_is_in_env_example():
    entries, shell_only = _documented()
    missing = _read_by_code() - entries - shell_only
    assert not missing, f"add to .env.example: {sorted(missing)}"


def test_env_example_lists_nothing_the_code_ignores():
    entries, shell_only = _documented()
    stale = (entries | shell_only) - _read_by_code()
    assert not stale, f"no longer read by the code: {sorted(stale)}"


def test_shell_only_settings_are_not_offered_as_env_entries():
    """compose ignores .env for these, so an entry would silently do nothing."""
    entries, shell_only = _documented()
    assert not entries & shell_only
