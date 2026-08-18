"""Default metadata filter: soft penalties, with locality as a hard exclusion.

Rationale: eligibility carve-outs live in sibling clauses (e.g. a general
"ages 20-67" section right next to a "before age 20" exception), so an age or
gender mismatch only *down-weights* a chunk rather than dropping it — the
general context can still surface. Locality is different: a benefit scoped to
one municipality genuinely does not apply elsewhere, so a locality mismatch is
a hard exclusion.

``track`` (the separate compensation tracks — hostile acts, military
bereavement, road and work accidents) is the one axis that penalises on a
*missing* fact rather than only a conflicting one. The asymmetry is deliberate:
this corpus serves victims of ordinary violence, so a chunk that declares a
track is off-track by default, and only an explicit matching fact clears it.
Age and gender cannot work that way — most users answer neither, and treating
silence as a mismatch would penalise half the corpus. The penalty stays soft
because track-specific pages still carry general procedural material worth
surfacing when nothing better exists.

``track`` is the first axis carrying real values (stamped by the scrapers, then
per segment by the enricher). Age, gender and locality are still populated by a
later pass; until then every chunk is NaN on them (missing keys), so they
contribute nothing and only ``track`` moves scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chatbot.rag.documents import Document
from chatbot.rag.filters.base import MetadataFilter


class SoftMetadataFilter(MetadataFilter):
    def __init__(self, penalty: float = 0.15):
        # Cosine sims sit in ~[-1, 1]; a 0.15 nudge reorders near-ties without
        # burying a strong textual match. Tune once real tags exist.
        self.penalty = penalty

    def adjust(self, documents: Sequence[Document], facts: dict) -> np.ndarray:
        adjustment = np.zeros(len(documents), dtype=np.float32)
        age = facts.get("age")
        gender = facts.get("gender")
        locality = facts.get("locality")
        track = facts.get("track")
        for i, doc in enumerate(documents):
            meta = doc.meta
            if meta.get("track") and meta["track"] != track:
                adjustment[i] -= self.penalty
            if age is not None:
                low, high = meta.get("min_age"), meta.get("max_age")
                if (low is not None and age < low) or (high is not None and age > high):
                    adjustment[i] -= self.penalty
            if gender and meta.get("gender") and meta["gender"] != gender:
                adjustment[i] -= self.penalty
            if locality and meta.get("locality") and meta["locality"] != locality:
                adjustment[i] = -np.inf  # hard exclusion: wrong municipality
        return adjustment
