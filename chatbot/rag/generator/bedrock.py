"""Claude via Amazon Bedrock — same prompt and transport, AWS-native credentials.

Identical to :class:`ClaudeGenerator` in every respect that shapes the answer: the
grounding system prompt, the numbered-passage user message and the citation rules
all come from ``_prompt``, so an answer from this head is comparable to one from
the direct-API head. Only two things differ, and both are about *where* the call
goes rather than what it says:

* **Credentials.** Nothing is read from ``.env``. ``AnthropicBedrockMantle`` is
  constructed with a region and nothing else, so botocore's own chain resolves
  the EC2 instance role from instance metadata — the same mechanism the report
  store already relies on (``app/whatsapp/reporting/s3_store.py``). No long-lived
  key reaches the box for answer generation.
* **Model id.** Bedrock namespaces Anthropic models under ``anthropic.`` and
  serves the current ones through *cross-region inference profiles*, whose ids
  additionally carry a geography prefix (``eu.``/``us.``). The derivation below is
  a convenience default; the authoritative id for a region comes from
  ``aws bedrock list-inference-profiles`` and can be pinned via
  ``EILAF_BEDROCK_MODEL``.

Both the SDK import and the region/model resolution are lazy, so constructing
this head costs nothing and needs no AWS at all — the Streamlit app builds every
generator up front and must not fail because Bedrock happens to be unconfigured.
"""

from __future__ import annotations

import os

# Same unit, shared deliberately: the model tier and output cap stay a one-line
# change in claude_api.py rather than drifting between the two Claude heads.
from chatbot.rag.generator.claude_api import MODEL, _MAX_TOKENS, ClaudeGenerator

REGION_ENV = "EILAF_BEDROCK_REGION"
MODEL_ENV = "EILAF_BEDROCK_MODEL"

# Geography prefixes for the cross-region inference profiles, keyed by the first
# segment of the AWS region name. Deliberately not a catch-all: deriving a
# plausible-looking profile id for a geography we have never checked produces an
# AccessDenied naming regions nobody configured, which is a miserable thing to
# debug. Unmapped geographies must name the id explicitly instead.
_GEO_BY_REGION_PREFIX = {"eu": "eu", "us": "us"}


class BedrockGenerator(ClaudeGenerator):
    """Answer a query from its retrieved contexts using Claude on Bedrock.

    ``region`` and ``model`` override the ``EILAF_BEDROCK_REGION`` /
    ``EILAF_BEDROCK_MODEL`` environment variables. Neither is resolved until the
    first generation, so an unconfigured instance is inert rather than broken.
    """

    def __init__(
        self,
        region: str | None = None,
        model: str | None = None,
        max_tokens: int = _MAX_TOKENS,
    ):
        # Note: ClaudeGenerator.__init__ is deliberately not called. It would
        # eagerly assign self.model, and here the id depends on a region we
        # refuse to resolve this early.
        self._region_override = region
        self._model_override = model
        self._region: str | None = None
        self._model: str | None = None
        self.max_tokens = max_tokens
        self._client = None
        self.last_usage: dict | None = None

    @property
    def region(self) -> str:
        """The Bedrock region, resolved on first access.

        Raises rather than letting botocore fall back to ``us-east-1``: the
        fallback is silent, and these are conversations with victims of
        violence, so a deployment that believes it is EU-only must fail loudly
        instead of quietly generating answers across the Atlantic. The report
        store learned this the hard way — see ``app/whatsapp/config.py``.
        """
        if self._region is None:
            region = self._region_override or os.environ.get(REGION_ENV, "").strip()
            if not region:
                raise RuntimeError(
                    f"{REGION_ENV} is not set, so the Bedrock region is unknown. "
                    "Set it explicitly (e.g. eu-north-1) — it is deliberately not "
                    "defaulted, because botocore would silently fall back to "
                    "us-east-1 and move conversations out of the EU."
                )
            self._region = region
        return self._region

    @property
    def model(self) -> str:
        """The Bedrock model id, resolved on first access."""
        if self._model is None:
            pinned = self._model_override or os.environ.get(MODEL_ENV, "").strip()
            self._model = pinned or self._derive_model_id(self.region)
        return self._model

    @staticmethod
    def _derive_model_id(region: str) -> str:
        """``eu-north-1`` -> ``eu.anthropic.claude-sonnet-5``.

        A default, not a guarantee: confirm against
        ``aws bedrock list-inference-profiles --region <region>`` and pin
        ``EILAF_BEDROCK_MODEL`` if the account is served a different id.
        """
        geo = _GEO_BY_REGION_PREFIX.get(region.split("-", 1)[0])
        if geo is None:
            raise RuntimeError(
                f"No inference-profile geography is mapped for region {region!r}. "
                f"Set {MODEL_ENV} to the id from "
                f"`aws bedrock list-inference-profiles --region {region}`."
            )
        return f"{geo}.anthropic.{MODEL}"

    def _get_client(self):
        if self._client is None:
            # Lazy: importing this module must not require the AWS SDK extra.
            from anthropic import AnthropicBedrockMantle

            # Credentials are deliberately omitted so botocore resolves the
            # instance role (or the caller's AWS_PROFILE in local dev).
            self._client = AnthropicBedrockMantle(aws_region=self.region)
        return self._client
