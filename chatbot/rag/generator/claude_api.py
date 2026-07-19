"""Claude-backed generator head (Messages API + grounded Hebrew answer).

One ``messages.create`` call per query: the retrieved passages go in as the user
message, the grounding rules as the system prompt, and Claude returns a Hebrew
answer cited back to the passages. Thinking is turned off — grounded QA over the
supplied context is light reasoning, and off is cheaper and lower-latency.

Claude is imported lazily so importing this module (and ``chatbot.rag``) never
requires the SDK or an API key — only actually calling :meth:`generate` does.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot.rag.documents import Document
from chatbot.rag.generator._prompt import NO_CONTEXT, SYSTEM, build_user_message
from chatbot.rag.generator.base import Generator

# Centralized so the model tier is a one-line change. Sonnet 5 matches the
# enrichment unit's tier: strong Hebrew, cheap, 1M context — plenty for a
# grounded chat answer. Bump to an Opus id here if answer quality needs it.
MODEL = "claude-sonnet-5"
_MAX_TOKENS = 1024


class ClaudeGenerator(Generator):
    """Answer a query from its retrieved contexts using Claude.

    ``client`` is created lazily on first use. ``last_usage`` exposes the most
    recent call's token counts (for the eval app / cost read-outs); it stays
    ``None`` until the first API-backed generation.
    """

    def __init__(self, model: str = MODEL, max_tokens: int = _MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self.last_usage: dict | None = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed to actually call the API

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, query: str, contexts: Sequence[Document]) -> str:
        # No passages retrieved → nothing to ground on; skip the API call.
        if not contexts:
            return NO_CONTEXT

        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "disabled"},
            system=SYSTEM,
            messages=[{"role": "user", "content": build_user_message(query, contexts)}],
        )

        usage = response.usage
        self.last_usage = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        return next((b.text for b in response.content if b.type == "text"), "")
