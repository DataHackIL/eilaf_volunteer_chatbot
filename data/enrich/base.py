"""Segment enrichment interface + the annotation schema it produces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel


class SegmentAnnotation(BaseModel):
    """Enrichment result for one segment: usefulness + per-segment group tags.

    ``useful=False`` marks a segment the retriever should skip (a bare heading,
    a stub, a meaningless line). The age/gender tags mirror the keys
    :class:`~chatbot.rag.filters.SoftMetadataFilter` reads; ``None`` means
    *unconstrained* on that axis (the segment applies to everyone) and never
    causes a chunk to be filtered out.

    ``locality`` is deliberately absent here: it is a *dataset-level* attribute
    (national sources have none; a municipal dataset carries a single locality
    for all its segments), set on the tree root and propagated by the loader —
    not inferred per segment.
    """

    useful: bool = True
    min_age: int | None = None
    max_age: int | None = None
    gender: Literal["m", "f"] | None = None

    def tags(self) -> dict:
        """The group tags only (no ``useful``), dropping ``None`` values.

        This is exactly what gets stored under a node's ``meta`` and read back by
        the loader / metadata filter — absent keys mean unconstrained.
        """
        return {
            key: value
            for key, value in self.model_dump(exclude={"useful"}).items()
            if value is not None
        }


class SegmentEnricher(ABC):
    """Annotates scraped text segments with usefulness + group tags."""

    @abstractmethod
    def enrich(self, segments: Sequence[str]) -> list[SegmentAnnotation]:
        """Return one :class:`SegmentAnnotation` per input segment, in order."""
        raise NotImplementedError
