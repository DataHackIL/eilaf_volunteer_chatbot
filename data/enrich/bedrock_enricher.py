"""Claude-on-Bedrock segment enricher (concurrent Messages calls + structured outputs).

The Bedrock sibling of :class:`~data.enrich.claude_enricher.ClaudeSegmentEnricher`:
same model tier, same system prompt, same strict JSON schema — so a segment
annotated here is judged against exactly the spec the direct-API back-end uses.
What differs is the transport:

* **No Batches API.** Bedrock serves the Messages endpoint but not Message
  Batches, so each segment is its own ``messages.create`` call. A small thread
  pool keeps a 4–7k-segment law file to minutes rather than hours; the SDK
  retries 429/5xx itself.
* **AWS-native credentials.** Like the Bedrock answer head, the client is built
  with a region only and botocore resolves the EC2 instance role. Region and
  model id resolve by the head's rules (``EILAF_BEDROCK_REGION`` /
  ``EILAF_BEDROCK_MODEL``, no region default) via the shared helpers in
  :mod:`chatbot.rag.generator.bedrock`.

A failed or malformed reply defers the segment (``ok=False``) for a later run.
A failure that would hit *every* segment — no access, a wrong model id, a
rejected request shape — aborts the run instead, so a misconfigured grant costs
one call rather than ten thousand deferrals.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from chatbot.rag.generator.bedrock import resolve_model_id, resolve_region
from data.enrich.base import SegmentAnnotation, SegmentEnricher

# Reuse the exact classification spec + model tier from the Claude enricher so
# both Claude back-ends annotate identically; only the transport differs.
from data.enrich.claude_enricher import _ANNOTATION_SCHEMA, _MAX_TOKENS, _SYSTEM, MODEL

_WORKERS = 8
_MAX_RETRIES = 5


class BedrockSegmentEnricher(SegmentEnricher):
    """Annotate segments with Claude on Bedrock, ``workers`` requests at a time.

    ``region`` / ``model`` override the ``EILAF_BEDROCK_*`` environment
    variables; neither is resolved until the first :meth:`enrich`, so
    constructing this back-end needs no AWS at all.
    """

    def __init__(
        self,
        region: str | None = None,
        model: str | None = None,
        workers: int = _WORKERS,
        max_retries: int = _MAX_RETRIES,
    ):
        self._region_override = region
        self._model_override = model
        self._region: str | None = None
        self._model: str | None = None
        self.workers = max(1, workers)
        self.max_retries = max_retries
        self._client = None
        self.last_usage: dict | None = None

    @property
    def region(self) -> str:
        if self._region is None:
            self._region = resolve_region(self._region_override)
        return self._region

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = resolve_model_id(
                self._model_override, self._region_override, base_model=MODEL
            )
        return self._model

    def _get_client(self):
        if self._client is None:
            # Lazy: importing this module must not require the AWS SDK extra.
            from anthropic import AnthropicBedrockMantle

            # Credentials are deliberately omitted so botocore resolves the
            # instance role (or the caller's AWS_PROFILE in local dev).
            self._client = AnthropicBedrockMantle(
                aws_region=self.region, max_retries=self.max_retries
            )
        return self._client

    def _annotate_one(self, segment: str) -> tuple[SegmentAnnotation, int, int]:
        """Annotate a single segment; defer it on a per-segment failure.

        Returns ``(annotation, input_tokens, output_tokens)``. Errors that no
        retry or later run could fix for *this* segment alone propagate.
        """
        import anthropic

        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                thinking={"type": "disabled"},
                system=_SYSTEM,
                messages=[{"role": "user", "content": segment}],
                output_config={
                    "format": {"type": "json_schema", "schema": _ANNOTATION_SCHEMA}
                },
            )
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ):
            raise  # the same for every segment: stop the run, don't defer 10k
        except anthropic.APIError as exc:
            print(f"  bedrock: request failed ({exc}); segment deferred")
            return SegmentAnnotation(ok=False), 0, 0

        usage = response.usage
        return self._parse(response), usage.input_tokens, usage.output_tokens

    def enrich(self, segments: Sequence[str]) -> list[SegmentAnnotation]:
        segments = list(segments)
        if not segments:
            return []

        # Resolve config + build the client once, before fanning out, so a
        # missing region fails here rather than once per worker.
        self._get_client()
        _ = self.model

        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            futures = [pool.submit(self._annotate_one, seg) for seg in segments]
            # Collected in submission order: annotate_tree zips the result
            # against the nodes it sent, strictly by position.
            results = [future.result() for future in futures]
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown()

        annotations = [annotation for annotation, _, _ in results]
        input_tokens = sum(in_tok for _, in_tok, _ in results)
        output_tokens = sum(out_tok for _, _, out_tok in results)

        # Exposed for programmatic access + printed for a quick cost read-out.
        n = len(segments)
        self.last_usage = {
            "calls": n,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        print(
            f"bedrock_enricher: {n} calls to {self.model}, "
            f"{input_tokens} input + {output_tokens} output tokens "
            f"(avg {input_tokens / n:.0f} in / {output_tokens / n:.0f} out per call)"
        )
        return annotations

    @staticmethod
    def _parse(response) -> SegmentAnnotation:
        # Anything but a clean finish (max_tokens, refusal) is a failed attempt,
        # not a verdict: defer it so a later run asks again.
        if response.stop_reason != "end_turn":
            return SegmentAnnotation(ok=False)
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return SegmentAnnotation.model_validate_json(text)
        except (ValueError, json.JSONDecodeError):
            return SegmentAnnotation(ok=False)
