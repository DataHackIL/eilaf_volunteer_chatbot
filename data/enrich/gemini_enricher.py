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
from typing import get_args

from data.enrich.base import SegmentAnnotation, SegmentEnricher, Track

# Reuse the exact prompt from the Claude enricher so both back-ends classify
# against one spec; only the transport differs.
from data.enrich.claude_enricher import _SYSTEM

# Primary model + the fall-over rotation. Each free-tier model has its own
# per-day quota, so rotating on a 429 (see _annotate_one) multiplies the daily
# budget for a bulk run. Ordered primary-first.
#
# Google retires these faster than this repo gets touched, and a retired name
# fails every call rather than rotating past it (see _annotate_one's 404 branch),
# so re-check the list against ``client.models.list()`` if a bulk run starts
# reporting model errors. ``gemini-3.7-flash`` is the newest flash tier but was
# answering 503 "high demand" when this rotation was last verified.
MODEL = "gemini-3.6-flash"
ROTATION_MODELS = (
    MODEL,
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
)

# Only when *every* model is rate-limited in one pass do we back off (the cap may
# be per-minute) and retry the whole rotation, rather than dropping the segment.
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 20

# Status codes worth trying another model for: the quota is per-model, and an
# overloaded tier usually has a healthy sibling.
_ROTATE_CODES = (429, 503)


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
            "track": types.Schema(
                type=types.Type.STRING, enum=list(get_args(Track)), nullable=True
            ),
            "useful": types.Schema(type=types.Type.BOOLEAN),
        },
        required=["min_age", "max_age", "gender", "track", "useful"],
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

    def _drop_model(self, model: str) -> None:
        """Retire a model name the API no longer serves, for this run.

        A 404 is permanent, and the model index is sticky — so leaving a dead
        name in the rotation makes every later segment fail on it. Dropping it
        keeps the run alive on the models that do work.
        """
        self.models.remove(model)
        self._model_idx = 0
        print(f"  gemini: {model} is no longer served; dropped from the rotation")
        if not self.models:
            raise RuntimeError(
                "gemini: no usable models left in the rotation. Update "
                "ROTATION_MODELS in data/enrich/gemini_enricher.py against "
                "client.models.list()."
            )

    def _annotate_one(self, segment: str) -> tuple[SegmentAnnotation, int, int]:
        """Annotate a single segment, rotating models on rate limits.

        Returns ``(annotation, input_tokens, output_tokens)``. A 429 or 503
        rotates to the next model (each has its own per-day quota); only when
        every model is exhausted in one pass do we back off and retry the
        rotation. A 404 means the name was retired, so it leaves the rotation
        entirely. Any other error marks the segment *not annotated*
        (``ok=False``), which defers it to a later run rather than recording a
        guess as the final answer.
        """
        from google.genai import errors

        client = self._get_client()
        config = self._get_config()
        for cycle in range(self.max_retries):
            for _ in range(len(self.models)):  # try each model before backing off
                model = self.models[self._model_idx]
                try:
                    response = client.models.generate_content(
                        model=model, contents=segment, config=config
                    )
                except errors.APIError as exc:
                    code = getattr(exc, "code", None)
                    if code == 404:
                        self._drop_model(model)
                        continue
                    if code not in _ROTATE_CODES:
                        # Rotating or waiting won't fix this one.
                        print(f"  gemini: request failed ({exc}); segment deferred")
                        return SegmentAnnotation(ok=False), 0, 0
                    # Rate limited / overloaded: advance to the next model (sticky).
                    self._model_idx = (self._model_idx + 1) % len(self.models)
                    continue

                usage = response.usage_metadata
                in_tok = getattr(usage, "prompt_token_count", 0) or 0
                out_tok = getattr(usage, "candidates_token_count", 0) or 0
                return self._parse_text(response.text), in_tok, out_tok

            # Every model was rate-limited this pass; the cap may be per-minute,
            # so wait and retry the whole rotation (unless we're out of cycles).
            if cycle < self.max_retries - 1:
                wait = _BACKOFF_SECONDS * (cycle + 1)
                print(f"  gemini: all {len(self.models)} models exhausted, waiting {wait}s")
                time.sleep(wait)

        # Out of cycles: defer rather than bank an un-annotated guess as final.
        return SegmentAnnotation(ok=False), 0, 0

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
        # A blocked/empty/malformed reply is a failed attempt, not a verdict:
        # defer it (``ok=False``) so a later run asks again.
        if not text:
            return SegmentAnnotation(ok=False)
        try:
            return SegmentAnnotation.model_validate_json(text)
        except (ValueError, json.JSONDecodeError):
            return SegmentAnnotation(ok=False)
