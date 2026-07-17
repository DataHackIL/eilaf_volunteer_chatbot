"""Gemini-backed segment enricher (free-tier Generate Content + structured outputs).

A drop-in mirror of :class:`~data.enrich.claude_enricher.ClaudeSegmentEnricher`
that targets Google AI Studio's *free* API. It does the same job — age/gender
group tags per segment — with the same strict JSON schema so every result
validates against :class:`SegmentAnnotation`.

The one structural difference: the free tier has no Batches API, so segments are
sent one request at a time, sequentially, with backoff on rate-limit (429)
errors. ``google-genai`` is imported lazily so the rest of ``data.enrich`` stays
usable without the SDK or a key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from data.enrich.base import SegmentAnnotation, SegmentEnricher

# Reuse the exact prompt from the Claude enricher so both back-ends classify
# against one spec; only the transport differs.
from data.enrich.claude_enricher import _SYSTEM

# Centralized so the model tier is a one-line change. flash-lite is the cheapest
# free-tier model and plenty for this bulk closed-form classification.
MODEL = "gemini-3.1-flash-lite"

# Free-tier rate limits are per-minute (RPM); on a 429 we back off and retry the
# same segment rather than dropping it.
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 20


def _annotation_schema():
    """Strict response schema mirroring :data:`claude_enricher._ANNOTATION_SCHEMA`.

    Built lazily inside the function so importing this module never imports the
    SDK. Nullable tags (``None`` == unconstrained) map to ``nullable=True``;
    ``useful`` is deliberately absent — usefulness comes from the rule pass, not
    the model.
    """
    from google.genai import types

    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "min_age": types.Schema(type=types.Type.INTEGER, nullable=True),
            "max_age": types.Schema(type=types.Type.INTEGER, nullable=True),
            "gender": types.Schema(
                type=types.Type.STRING, enum=["m", "f"], nullable=True
            ),
        },
        required=["min_age", "max_age", "gender"],
    )


class GeminiSegmentEnricher(SegmentEnricher):
    """Annotate segments with Gemini's free API, one request per segment.

    ``client`` is created lazily on first use so importing this module (and the
    rest of ``data.enrich``) never requires the SDK or credentials. Interface is
    identical to :class:`ClaudeSegmentEnricher`, so it's a drop-in swap.
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
                system_instruction=_SYSTEM,
                response_mime_type="application/json",
                response_schema=_annotation_schema(),
                temperature=0.0,  # deterministic bulk classification
            )
        return self._config

    def _annotate_one(self, segment: str) -> tuple[SegmentAnnotation, int, int]:
        """Annotate a single segment; retry with backoff on rate limits.

        Returns ``(annotation, input_tokens, output_tokens)``. On any
        non-retryable error, or after exhausting retries, keeps the segment
        (useful, untagged) so content is never silently dropped.
        """
        from google.genai import errors

        client = self._get_client()
        config = self._get_config()
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=segment, config=config
                )
            except errors.APIError as exc:
                # 429 == rate limited: wait and retry the same segment.
                if getattr(exc, "code", None) == 429 and attempt < self.max_retries - 1:
                    wait = _BACKOFF_SECONDS * (attempt + 1)
                    print(f"  gemini: rate limited, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                print(f"  gemini: request failed ({exc}); keeping segment untagged")
                return SegmentAnnotation(), 0, 0

            usage = response.usage_metadata
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0
            return self._parse_text(response.text), in_tok, out_tok

        return SegmentAnnotation(), 0, 0

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
            f"gemini_enricher: {n} calls, "
            f"{input_tokens} input + {output_tokens} output tokens "
            f"(avg {input_tokens / n:.0f} in / {output_tokens / n:.0f} out per call)"
        )
        return annotations

    @staticmethod
    def _parse_text(text: str | None) -> SegmentAnnotation:
        if not text:
            return SegmentAnnotation()  # keep on empty/blocked response
        try:
            return SegmentAnnotation.model_validate_json(text)
        except (ValueError, json.JSONDecodeError):
            return SegmentAnnotation()
