"""Gemini-backed segment enricher (free-tier Generate Content + structured outputs).

A drop-in mirror of :class:`~data.enrich.claude_enricher.ClaudeSegmentEnricher`
that targets Google AI Studio's *free* API. It does the same job — age/gender
group tags per segment — with the same strict JSON schema so every result
validates against :class:`SegmentAnnotation`.

The one structural difference: the free tier has no Batches API, so segments are
sent one request at a time, sequentially. Each free-tier model has its own
per-day quota, so a 429 rotates to the next model (and only backs off when every
model is rate-limited at once), keeping a bulk run going past a single model's
daily cap. ``google-genai`` is imported lazily so the rest of ``data.enrich``
stays usable without the SDK or a key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from data.enrich.base import SegmentAnnotation, SegmentEnricher

# Reuse the exact prompt from the Claude enricher so both back-ends classify
# against one spec; only the transport differs.
from data.enrich.claude_enricher import _SYSTEM

# Primary model + the fall-over rotation. Each free-tier model has its own
# per-day quota, so rotating on a 429 (see _annotate_one) multiplies the daily
# budget for a bulk run. Ordered primary-first.
MODEL = "gemini-3-flash-preview"
ROTATION_MODELS = (MODEL, "gemini-2.5-flash", "gemini-2.0-flash")

# Only when *every* model is rate-limited in one pass do we back off (the cap may
# be per-minute) and retry the whole rotation, rather than dropping the segment.
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
            "useful": types.Schema(type=types.Type.BOOLEAN),
        },
        required=["min_age", "max_age", "gender", "useful"],
    )


class GeminiSegmentEnricher(SegmentEnricher):
    """Annotate segments with Gemini's free API, one request per segment.

    ``models`` is a fall-over rotation: a 429 advances to the next model (each
    has its own per-day quota), so a bulk run isn't stalled by one model's daily
    cap. ``client`` is created lazily on first use so importing this module (and
    the rest of ``data.enrich``) never requires the SDK or credentials. Interface
    is identical to :class:`ClaudeSegmentEnricher`, so it's a drop-in swap.
    """

    def __init__(
        self, models: Sequence[str] = ROTATION_MODELS, max_retries: int = _MAX_RETRIES
    ):
        self.models = list(models)
        self._model_idx = 0  # sticky: stays on the last working model across segments
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
        """Annotate a single segment, rotating models on rate limits.

        Returns ``(annotation, input_tokens, output_tokens)``. A 429 rotates to
        the next model (each has its own per-day quota); only when every model is
        rate-limited in one pass do we back off and retry the rotation. On a
        non-rate-limit error, or after exhausting the backoff cycles, keeps the
        segment (useful, untagged) so content is never silently dropped.
        """
        from google.genai import errors

        client = self._get_client()
        config = self._get_config()
        n = len(self.models)
        for cycle in range(self.max_retries):
            for _ in range(n):  # try each model once before backing off
                model = self.models[self._model_idx]
                try:
                    response = client.models.generate_content(
                        model=model, contents=segment, config=config
                    )
                except errors.APIError as exc:
                    if getattr(exc, "code", None) != 429:
                        # Rotating or waiting won't fix a non-rate-limit error.
                        print(f"  gemini: request failed ({exc}); keeping segment untagged")
                        return SegmentAnnotation(), 0, 0
                    # Rate limited: advance to the next model (sticky).
                    self._model_idx = (self._model_idx + 1) % n
                    continue

                usage = response.usage_metadata
                in_tok = getattr(usage, "prompt_token_count", 0) or 0
                out_tok = getattr(usage, "candidates_token_count", 0) or 0
                return self._parse_text(response.text), in_tok, out_tok

            # Every model was rate-limited this pass; the cap may be per-minute,
            # so wait and retry the whole rotation (unless we're out of cycles).
            if cycle < self.max_retries - 1:
                wait = _BACKOFF_SECONDS * (cycle + 1)
                print(f"  gemini: all {n} models rate limited, waiting {wait}s")
                time.sleep(wait)

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
