"""OpenAI-compatible generator head (Cerebras / Groq / OpenRouter / …).

Any provider that exposes an OpenAI-style ``/chat/completions`` endpoint plugs
in here: supply the ``base_url``, the ``model``, and the name of the env var
holding the key. Defaults target **Cerebras** — a generous free daily quota
(~14k requests/day) and very fast inference — so it's a good relief valve when
the Gemini free tier maxes out.

Same job as the other heads: a grounded Hebrew answer with ``[n]`` passage
citations, using the shared prompt spec. The ``openai`` SDK is imported lazily so
importing this module (and ``chatbot.rag``) never requires the SDK or a key.

Other free providers are a one-line construction change, e.g.::

    OpenAICompatibleGenerator(
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        model="llama-3.3-70b-versatile",
    )
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from chatbot.rag.documents import Document
from chatbot.rag.generator._prompt import NO_CONTEXT, build_system, build_user_message
from chatbot.rag.generator.base import Generator

# Cerebras defaults. ``gpt-oss-120b`` is Cerebras' production model; for stronger
# Hebrew try ``gemma-4-31b`` (Google lineage) — a one-line ``model=`` change. The
# live catalogue is at inference-docs.cerebras.ai/models/overview.
BASE_URL = "https://api.cerebras.ai/v1"
KEY_ENV = "CEREBRAS_API_KEY"
MODEL = "gpt-oss-120b"
# gpt-oss (and other reasoning models) spend completion budget on internal
# reasoning before the visible answer, so keep headroom to avoid a truncated
# reply — same failure mode as Gemini thinking models.
_MAX_TOKENS = 4096


class OpenAICompatibleGenerator(Generator):
    """Answer a query from its retrieved contexts via an OpenAI-style endpoint.

    ``client`` is built lazily on first use; ``last_usage`` exposes the most
    recent call's token counts and stays ``None`` until then. ``max_retries`` is
    handed to the SDK, which already retries 429/5xx with backoff — set it to 0
    for fast fail-over when this head sits inside a :class:`FallbackGenerator`.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        key_env: str = KEY_ENV,
        model: str = MODEL,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = 0.0,  # grounded QA — keep it deterministic
        max_retries: int = 2,
    ):
        self.base_url = base_url
        self.key_env = key_env
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
        self.last_usage: dict | None = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy: only needed to actually call the API

            api_key = os.environ.get(self.key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.key_env} is not set — add it to .env to use "
                    f"{self.base_url}"
                )
            self._client = OpenAI(
                base_url=self.base_url, api_key=api_key, max_retries=self.max_retries
            )
        return self._client

    def generate(
        self, query: str, contexts: Sequence[Document], language: str = "Hebrew"
    ) -> str:
        # No passages retrieved → nothing to ground on; skip the API call.
        if not contexts:
            return NO_CONTEXT

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": build_system(language)},
                {"role": "user", "content": build_user_message(query, contexts)},
            ],
        )

        usage = response.usage
        self.last_usage = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return response.choices[0].message.content or ""
