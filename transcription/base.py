"""Transcriber interface: maps a spoken waveform to text. Swap implementations freely."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 ``waveform`` (sampled at ``sample_rate`` Hz) to text."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Human-readable identity of this transcriber (for UI / logging)."""
        return type(self).__qualname__
