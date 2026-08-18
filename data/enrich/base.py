"""Segment enrichment interface + the annotation schema it produces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

# Compensation tracks that Israeli law routes separately from ordinary criminal
# violence — each with its own paying authority, eligibility test and forms.
# Their material *reads* like general victim support, which is exactly the
# hazard: a chunk about hostile-act bereavement is confidently wrong advice for
# someone hurt in a neighbourhood assault. ``None`` means the segment is not
# track-specific and applies to a victim of ordinary violence.
Track = Literal["hostile_acts", "military", "road_accident", "work_accident"]


class SegmentAnnotation(BaseModel):
    """Enrichment result for one segment: usefulness + per-segment group tags.

    ``useful=False`` marks a segment the retriever should skip (a bare heading,
    a stub, a meaningless line). The age/gender/track tags mirror the keys
    :class:`~chatbot.rag.filters.SoftMetadataFilter` reads; ``None`` means
    *unconstrained* on that axis (the segment applies to everyone) and never
    causes a chunk to be filtered out.

    ``track`` is annotated per segment *and* settable dataset-wide: a source
    that is entirely about one track (e.g. gov.il's hostile-acts guides) stamps
    it on the tree root, and the loader inherits it down to every segment — a
    per-segment value, when the enricher finds one, overrides it. That matters
    because a lone sentence like "the allowance is paid monthly" carries no
    signal on its own; only the page it came from knows which track it is on.

    ``locality`` is deliberately absent here: unlike ``track`` it is a *purely*
    dataset-level attribute (national sources have none; a municipal dataset
    carries a single locality for all its segments), so it is set on the tree
    root and propagated by the loader — never inferred per segment.
    """

    useful: bool = True
    min_age: int | None = None
    max_age: int | None = None
    gender: Literal["m", "f"] | None = None
    track: Track | None = None

    # False when the back-end never produced an answer for this segment (a dead
    # model, an exhausted quota, a malformed reply). Transport state, not an
    # annotation: the caller must leave such a node un-annotated so a later run
    # retries it. Recording the defaults instead would look identical to a
    # genuine "useful, no constraints" verdict and would never be revisited.
    ok: bool = True

    def tags(self) -> dict:
        """The group tags only (no ``useful``/``ok``), dropping ``None`` values.

        This is exactly what gets stored under a node's ``meta`` and read back by
        the loader / metadata filter — absent keys mean unconstrained.
        """
        return {
            key: value
            for key, value in self.model_dump(exclude={"useful", "ok"}).items()
            if value is not None
        }


class SegmentEnricher(ABC):
    """Annotates scraped text segments with usefulness + group tags."""

    @abstractmethod
    def enrich(self, segments: Sequence[str]) -> list[SegmentAnnotation]:
        """Return one :class:`SegmentAnnotation` per input segment, in order."""
        raise NotImplementedError
