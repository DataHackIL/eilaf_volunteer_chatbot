"""Enrichment stage of the scraping pipeline: raw section-trees -> enriched.

Reads scraped section-trees from ``data/static/raw/`` and writes annotated
copies to ``data/static/enriched/`` — the store the RAG pipeline actually reads.
Each text-bearing segment is annotated with a usefulness judgment (drop headers,
stubs, meaningless lines) and closed-form group tags (age/gender/locality) that
the :class:`~chatbot.rag.filters.SoftMetadataFilter` re-scores retrieval with.

The pipeline is: **scrape (raw/) -> enrich (enriched/) -> load**. See
:mod:`data.enrich.enrich_store` for the driver.
"""

from data.enrich.base import SegmentAnnotation, SegmentEnricher
from data.enrich.claude_enricher import ClaudeSegmentEnricher
from data.enrich.rules import is_obvious_junk

__all__ = [
    "SegmentAnnotation",
    "SegmentEnricher",
    "ClaudeSegmentEnricher",
    "is_obvious_junk",
]
