"""Claude-backed segment enricher (Message Batches API + structured outputs).

One Claude request per segment does both jobs — usefulness *and* group tags — via
the Batches API (50% cost, non-latency-sensitive) with a strict JSON schema so
every result validates against :class:`SegmentAnnotation`. Claude is imported
lazily, so the rest of ``data.enrich`` stays usable without the SDK or a key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import get_args

from data.enrich.base import SegmentAnnotation, SegmentEnricher, Track

# Centralized so the model tier is a one-line change. Sonnet 5 has adaptive
# thinking on by default; this is a cheap bulk classification, so we turn it off.
MODEL = "claude-sonnet-5"
_MAX_TOKENS = 512
_POLL_SECONDS = 30

_SYSTEM = (
    "You classify short segments from Israeli rights/benefits documents (mostly "
    "Hebrew) for a retrieval knowledge base. For each segment decide:\n"
    "- useful: false if the segment is a bare heading, a table-of-contents / "
    "navigation entry, or a meaningless fragment that carries no rights or "
    "benefits information on its own; true otherwise.\n"
    "- min_age / max_age: the age range the segment restricts eligibility to, if "
    "it states one; null if it applies to any age.\n"
    "- gender: \"m\" or \"f\" if eligibility is restricted to that gender; null "
    "if it applies to anyone.\n"
    "- track: the separate compensation track the segment is about, when it is "
    "about one — \"hostile_acts\" (נפגעי פעולות איבה / terror), \"military\" "
    "(נפגעי צה\"ל, משפחות שכולות), \"road_accident\" (תאונת דרכים, פלת\"ד), "
    "\"work_accident\" (תאונת עבודה, נפגעי עבודה). These are routed by separate "
    "laws and paying authorities, so their entitlements do not carry over to a "
    "victim of ordinary criminal violence. Use null when the segment is about "
    "crime victims generally rather than one of these tracks.\n"
    "Base every field only on what the segment itself states. When unsure, leave "
    "a tag null and keep useful=true."
)

# Strict schema for structured outputs: every key required, closed object,
# nullable tags via anyOf. Mirrors SegmentAnnotation.
_ANNOTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "useful": {"type": "boolean"},
        "min_age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "max_age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "gender": {"anyOf": [{"type": "string", "enum": ["m", "f"]}, {"type": "null"}]},
        "track": {
            "anyOf": [
                {"type": "string", "enum": list(get_args(Track))},
                {"type": "null"},
            ]
        },
    },
    "required": ["useful", "min_age", "max_age", "gender", "track"],
}


def _custom_id(index: int) -> str:
    return f"seg-{index}"


def _index_of(custom_id: str) -> int:
    return int(custom_id.removeprefix("seg-"))


class ClaudeSegmentEnricher(SegmentEnricher):
    """Annotate segments with Claude via one Batches API job.

    ``client`` is created lazily on first use so importing this module (and the
    rest of ``data.enrich``) never requires the SDK or credentials.
    """

    def __init__(self, model: str = MODEL, poll_seconds: int = _POLL_SECONDS):
        self.model = model
        self.poll_seconds = poll_seconds
        self._client = None
        self.last_usage: dict | None = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed to actually call the API

            self._client = anthropic.Anthropic()
        return self._client

    def _request(self, index: int, segment: str):
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
        )
        from anthropic.types.messages.batch_create_params import Request

        return Request(
            custom_id=_custom_id(index),
            params=MessageCreateParamsNonStreaming(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                thinking={"type": "disabled"},
                system=_SYSTEM,
                messages=[{"role": "user", "content": segment}],
                output_config={
                    "format": {"type": "json_schema", "schema": _ANNOTATION_SCHEMA}
                },
            ),
        )

    def enrich(self, segments: Sequence[str]) -> list[SegmentAnnotation]:
        segments = list(segments)
        if not segments:
            return []

        client = self._get_client()
        batch = client.messages.batches.create(
            requests=[self._request(i, seg) for i, seg in enumerate(segments)]
        )
        batch = self._await_batch(client, batch.id)

        # Default keeps a segment (useful, untagged) so an errored/missing result
        # never silently drops content.
        annotations = [SegmentAnnotation() for _ in segments]
        input_tokens = output_tokens = 0
        for result in client.messages.batches.results(batch.id):
            index = _index_of(result.custom_id)
            annotations[index] = self._parse_result(result)
            if result.result.type == "succeeded":
                usage = result.result.message.usage
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens

        # Exposed for programmatic access + printed for a quick cost read-out.
        n = len(segments)
        self.last_usage = {
            "calls": n,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        print(
            f"claude_enricher: {n} calls, "
            f"{input_tokens} input + {output_tokens} output tokens "
            f"(avg {input_tokens / n:.0f} in / {output_tokens / n:.0f} out per call)"
        )
        return annotations

    def _await_batch(self, client, batch_id: str):
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return batch
            counts = batch.request_counts
            print(
                f"  batch {batch_id}: {batch.processing_status} "
                f"(processing={counts.processing}, succeeded={counts.succeeded}, "
                f"errored={counts.errored})"
            )
            time.sleep(self.poll_seconds)

    @staticmethod
    def _parse_result(result) -> SegmentAnnotation:
        if result.result.type != "succeeded":
            return SegmentAnnotation()  # keep on error/expiry/cancel
        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            return SegmentAnnotation.model_validate_json(text)
        except (ValueError, json.JSONDecodeError):
            return SegmentAnnotation()
