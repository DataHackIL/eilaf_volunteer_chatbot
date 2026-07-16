"""faster-whisper (CTranslate2) transcribers, incl. Hebrew/Arabic presets.

One shared backend runs every Whisper variant; the preset subclasses just pin a
model and language. faster-whisper expects mono 16 kHz float32 audio, which is
what the evaluation app feeds in.
"""

from __future__ import annotations

import numpy as np

from transcription.base import Transcriber


# ivrit.ai's Hebrew finetune of Whisper large-v3-turbo, converted to CTranslate2.
HEBREW_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
# Vanilla multilingual Whisper large-v3 (downloaded in CTranslate2 form).
ARABIC_MODEL = "large-v3"


class FasterWhisperTranscriber(Transcriber):
    def __init__(
        self,
        model_name: str,
        language: str,
        device: str | None = None,
        compute_type: str = "auto",
    ):
        # Imported lazily: keeps the heavy faster-whisper/ctranslate2 import off
        # the path of code that only needs the light `Transcriber` interface.
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.language = language
        self._model = WhisperModel(
            model_name,
            device=device or "auto",
            compute_type=compute_type,
        )

    @property
    def name(self) -> str:
        return f"{type(self).__qualname__}({self.model_name}, {self.language})"

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(
                f"faster-whisper expects 16 kHz audio, got {sample_rate} Hz; "
                "resample before calling transcribe()."
            )
        segments, _ = self._model.transcribe(
            waveform.astype(np.float32),
            language=self.language,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class IvritHebrewTranscriber(FasterWhisperTranscriber):
    """Hebrew transcription via ivrit.ai's Whisper large-v3-turbo CTranslate2 model."""

    def __init__(self, device: str | None = None):
        super().__init__(HEBREW_MODEL, language="he", device=device)


class WhisperArabicTranscriber(FasterWhisperTranscriber):
    """Arabic transcription via vanilla Whisper large-v3."""

    def __init__(self, device: str | None = None):
        super().__init__(ARABIC_MODEL, language="ar", device=device)
