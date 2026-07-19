"""Gemini-backed generator head (free-tier Generate Content, grounded Hebrew answer).

A drop-in mirror of :class:`~chatbot.rag.generator.claude_api.ClaudeGenerator`
that targets Google AI Studio's *free* API: same job (a Hebrew answer grounded in
the retrieved passages), same shared prompt spec, only the transport differs.

The free tier is per-minute rate-limited (RPM), so on a 429 we back off and
retry the same query rather than dropping the turn. ``google-genai`` is imported
lazily so importing this module (and ``chatbot.rag``) never requires the SDK or a
key — only actually calling :meth:`generate` does.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from chatbot.rag.documents import Document
from chatbot.rag.generator._prompt import NO_CONTEXT, SYSTEM, build_user_message
from chatbot.rag.generator.base import Generator
from chatbot.rag.generator.fallback import FallbackGenerator

# Centralized so the model tier is a one-line change. flash-lite is the cheapest
# free-tier chat model, with a higher daily request cap than flash for interactive use.
MODEL = "gemini-3.1-flash-lite"
_MAX_OUTPUT_TOKENS = 1024

# On a 429 we wait and retry the same query rather than dropping the answer.
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 20

# Distinct free-tier models, each with its own per-day quota. Rotating across
# them via :func:`gemini_rotation` multiplies the free daily budget. Ordered
# cheapest / highest-cap first.
ROTATION_MODELS = ("gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash")


class GeminiGenerator(Generator):
    """Answer a query from its retrieved contexts using Gemini's free API.

    Interface is identical to :class:`ClaudeGenerator`, so it's a drop-in swap.
    ``client``/``config`` are built lazily on first use; ``last_usage`` exposes
    the most recent call's token counts and stays ``None`` until then.
    """

    def __init__(self, model: str = MODEL, max_retries: int = _MAX_RETRIES):
        self.model = model
        self.max_retries = max_retries
        self._client = None
        self._config = None
        self.last_usage: dict | None = None

    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy: only needed to actually call the API

            # Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
            self._client = genai.Client()
        return self._client

    def _get_config(self):
        if self._config is None:
            from google.genai import types

            self._config = types.GenerateContentConfig(
                system_instruction=SYSTEM,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.0,  # grounded QA — keep it deterministic
            )
        return self._config

    def generate(self, query: str, contexts: Sequence[Document]) -> str:
        # No passages retrieved → nothing to ground on; skip the API call.
        if not contexts:
            return NO_CONTEXT

        from google.genai import errors

        client = self._get_client()
        config = self._get_config()
        message = build_user_message(query, contexts)

        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=message, config=config
                )
            except errors.APIError as exc:
                # 429 == rate limited: wait and retry the same query.
                if getattr(exc, "code", None) == 429 and attempt < self.max_retries - 1:
                    wait = _BACKOFF_SECONDS * (attempt + 1)
                    print(f"  gemini: rate limited, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise

            usage = response.usage_metadata
            self.last_usage = {
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            }
            return response.text or ""

        return ""  # unreachable: last attempt either returns or re-raises


def gemini_rotation(
    models: Sequence[str] = ROTATION_MODELS, max_retries: int = 1
) -> FallbackGenerator:
    """A :class:`FallbackGenerator` over one Gemini head per model.

    On a 429 (typically the per-day free quota, which is *per model*) it fails
    over to the next model instead of waiting — so ``max_retries`` defaults to 1
    (fail fast; no per-model backoff) since the rotation, not the retry, is the
    recovery path. The chain still raises if *every* model is exhausted.
    """
    return FallbackGenerator(
        [GeminiGenerator(model=model, max_retries=max_retries) for model in models]
    )
