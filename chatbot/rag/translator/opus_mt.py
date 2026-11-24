"""Helsinki-NLP opus-mt (MarianMT) via transformers: small, fast, permissive.

``Helsinki-NLP/opus-mt-ar-he`` is about 300 MB and Apache-licensed, but it was
trained on Modern Standard Arabic, so expect it to stumble on spoken Palestinian
phrasing. It's here mainly as a baseline for the benchmark. Marian's tokenizer
needs ``sentencepiece`` (the ``translate-bench`` extra).
"""

from __future__ import annotations

import threading

from chatbot.rag.translator.base import Translator

# One Marian model per language pair.
DEFAULT_MODELS = {
    "arabic": "Helsinki-NLP/opus-mt-ar-he",
    "english": "Helsinki-NLP/opus-mt-en-he",
}


class OpusMTTranslator(Translator):
    def __init__(self, models: dict[str, str] | None = None, num_beams: int = 4):
        self.models = {k.lower(): v for k, v in (models or DEFAULT_MODELS).items()}
        self.num_beams = num_beams
        # Pairs load on first use: a deployment that only sees Arabic never
        # downloads the English model.
        self._loaded: dict[str, tuple[object, object]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"opus-mt[{self.models.get('arabic', '?')}]"

    def _pair(self, source: str):
        key = source.strip().lower()
        if key not in self.models:
            raise ValueError(f"OpusMTTranslator has no model for {source!r}")
        with self._lock:
            if key not in self._loaded:
                from transformers import MarianMTModel, MarianTokenizer

                model_id = self.models[key]
                self._loaded[key] = (
                    MarianTokenizer.from_pretrained(model_id),
                    MarianMTModel.from_pretrained(model_id).eval(),
                )
        return self._loaded[key]

    def to_hebrew(self, text: str, source: str) -> str:
        if not text.strip():
            return text
        import torch

        tokenizer, model = self._pair(source)
        batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            # Length is capped by the model's own generation config (512 tokens).
            out = model.generate(**batch, num_beams=self.num_beams)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()
