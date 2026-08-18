"""OpenAI-compatible segment enricher (Cerebras free tier by default).

A drop-in sibling of :class:`~data.enrich.claude_enricher.ClaudeSegmentEnricher`
and :class:`~data.enrich.gemini_enricher.GeminiSegmentEnricher` for any provider
that exposes an OpenAI-style ``/chat/completions`` endpoint. Defaults to
**Cerebras** running ``gpt-oss-120b``. Its free tier was once ~14k requests/day
(far above Gemini's), but as of 2026-08-18 this account's key answers every
request with ``402 payment_required``, so treat the free quota as gone until
billing is set up — ``--provider gemini`` is the working free path.

Like the Gemini free tier there's no Batches API, so segments are sent one
request at a time. The same annotation spec is reused (``_SYSTEM`` +
``SegmentAnnotation``); JSON is requested via ``response_format`` and validated
with pydantic, so a malformed reply defers the segment to a later run rather
than recording a guess. The ``openai`` SDK is imported lazily so the rest of
``data.enrich`` stays usable without the SDK or a key.

Other free providers are a one-line construction change, e.g.::

    OpenAICompatibleSegmentEnricher(
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        model="llama-3.3-70b-versatile",
    )
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import get_args

from data.enrich.base import SegmentAnnotation, SegmentEnricher, Track

# Reuse the exact classification spec from the Claude enricher so every back-end
# annotates against one prompt; only the transport differs.
from data.enrich.claude_enricher import _SYSTEM

# Cerebras defaults (fast; see the module docstring on its quota). Groq /
# OpenRouter are a one-line swap and are the route to try if Cerebras stays
# behind billing; the live catalogue is at inference-docs.cerebras.ai/models.
BASE_URL = "https://api.cerebras.ai/v1"
KEY_ENV = "CEREBRAS_API_KEY"
MODEL = "gpt-oss-120b"
# gpt-oss reasons before emitting the JSON, so leave headroom (max_tokens bounds
# reasoning + output combined) or the object gets truncated and fails to parse.
_MAX_TOKENS = 4096
_MAX_RETRIES = 5

# json_object mode is portable across OpenAI-compatible providers (unlike strict
# json_schema, whose support varies). The keys must be named for the model, so
# append the shape to the shared system prompt; pydantic still validates.
_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else, with exactly these "
    'keys: "useful" (boolean), "min_age" (integer or null), "max_age" (integer '
    'or null), "gender" ("m", "f", or null), "track" ('
    + ", ".join(f'"{value}"' for value in get_args(Track))
    + ", or null)."
)
_SYSTEM_JSON = _SYSTEM + _JSON_INSTRUCTION


class OpenAICompatibleSegmentEnricher(SegmentEnricher):
    """Annotate segments via an OpenAI-style endpoint, one request per segment.

    ``client`` is created lazily on first use so importing this module (and the
    rest of ``data.enrich``) never requires the SDK or credentials. Interface is
    identical to the other enrichers, so it's a drop-in swap. ``max_retries`` is
    handed to the SDK, which already retries 429/5xx with backoff.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        key_env: str = KEY_ENV,
        model: str = MODEL,
        max_retries: int = _MAX_RETRIES,
    ):
        self.base_url = base_url
        self.key_env = key_env
        self.model = model
        self.max_retries = max_retries
        self._client = None
        self.last_usage: dict | None = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy: only needed to actually call the API

            api_key = os.environ.get(self.key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.key_env} is not set — add it to .env to use {self.base_url}"
                )
            self._client = OpenAI(
                base_url=self.base_url, api_key=api_key, max_retries=self.max_retries
            )
        return self._client

    def _annotate_one(self, segment: str) -> tuple[SegmentAnnotation, int, int]:
        """Annotate a single segment; defer it on any API error.

        Returns ``(annotation, input_tokens, output_tokens)``. The SDK retries
        rate limits internally; a persistent failure marks the segment
        ``ok=False``, leaving it un-annotated for a later run to retry.
        """
        from openai import OpenAIError

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                temperature=0.0,  # deterministic bulk classification
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_JSON},
                    {"role": "user", "content": segment},
                ],
            )
        except OpenAIError as exc:
            print(f"  openai: request failed ({exc}); segment deferred")
            return SegmentAnnotation(ok=False), 0, 0

        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return self._parse_text(response.choices[0].message.content), in_tok, out_tok

    def enrich(self, segments: Sequence[str]) -> list[SegmentAnnotation]:
        segments = list(segments)
        if not segments:
            return []

        annotations: list[SegmentAnnotation] = []
        input_tokens = output_tokens = 0
        for segment in segments:
            annotation, in_tok, out_tok = self._annotate_one(segment)
            annotations.append(annotation)
            input_tokens += in_tok
            output_tokens += out_tok

        # Exposed for programmatic access + printed for a quick cost read-out.
        n = len(segments)
        self.last_usage = {
            "calls": n,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        print(
            f"openai_enricher: {n} calls, "
            f"{input_tokens} input + {output_tokens} output tokens "
            f"(avg {input_tokens / n:.0f} in / {output_tokens / n:.0f} out per call)"
        )
        return annotations

    @staticmethod
    def _parse_text(text: str | None) -> SegmentAnnotation:
        # A blocked/empty/malformed reply is a failed attempt, not a verdict:
        # defer it (``ok=False``) so a later run asks again.
        if not text:
            return SegmentAnnotation(ok=False)
        try:
            return SegmentAnnotation.model_validate_json(text)
        except (ValueError, json.JSONDecodeError):
            return SegmentAnnotation(ok=False)
