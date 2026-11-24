"""NLLB-200 on CTranslate2: local, CPU, int8.

NLLB has a dedicated code for South Levantine Arabic (``ajp_Arab``, the
Palestinian/Jordanian dialect), which is the main way users write to the bot.
CTranslate2 runs the int8-quantised model on CPU at a fraction of the memory of
the original. Measured sizes on disk: about 0.6 GB for distilled-600M and
1.4 GB for distilled-1.3B. The model is CC-BY-NC-4.0.

The model directory is a CTranslate2 conversion with the tokenizer files copied
alongside. Make one with::

    ct2-transformers-converter --model facebook/nllb-200-distilled-600M \\
        --quantization int8 --output_dir models/nllb-600m-ct2 \\
        --copy_files tokenizer.json tokenizer_config.json special_tokens_map.json

(``ct2-transformers-converter`` ships with the ``ctranslate2`` package and needs
torch + transformers, which sentence-transformers already brings.)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from chatbot.rag.translator.base import Translator

# Our language names -> NLLB (FLORES-200) codes. Arabic defaults to the
# Palestinian dialect; pass ``arabic_code="arb_Arab"`` for Modern Standard Arabic.
DEFAULT_ARABIC_CODE = "ajp_Arab"
_CODES = {"english": "eng_Latn"}
_TARGET = "heb_Hebr"


class NLLBTranslator(Translator):
    def __init__(
        self,
        model_dir: str | Path,
        arabic_code: str = DEFAULT_ARABIC_CODE,
        beam_size: int = 4,
        threads: int | None = None,
    ):
        # Imported here so code that never translates doesn't pay for it.
        import ctranslate2
        from transformers import AutoTokenizer

        self.model_dir = str(model_dir)
        self.arabic_code = arabic_code
        self.beam_size = beam_size
        self._model = ctranslate2.Translator(
            self.model_dir,
            device="cpu",
            compute_type="int8",
            # The query is one short sentence: spend the cores on that one
            # sentence rather than on parallel batches that never come.
            inter_threads=1,
            intra_threads=threads or os.cpu_count() or 1,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # The tokenizer's source language is a setting on the shared object, so
        # two translations at once must not interleave between set and encode.
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"nllb[{Path(self.model_dir).name},{self.arabic_code}]"

    def _code(self, source: str) -> str:
        key = source.strip().lower()
        if key == "arabic":
            return self.arabic_code
        if key in _CODES:
            return _CODES[key]
        raise ValueError(f"NLLBTranslator has no code for {source!r}")

    def to_hebrew(self, text: str, source: str) -> str:
        if not text.strip():
            return text
        code = self._code(source)
        with self._lock:
            self._tokenizer.src_lang = code
            ids = self._tokenizer(text, truncation=True, max_length=256)["input_ids"]
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            result = self._model.translate_batch(
                [tokens],
                target_prefix=[[_TARGET]],
                beam_size=self.beam_size,
                max_decoding_length=256,
            )
            # The first output token is the forced target-language code.
            out = result[0].hypotheses[0][1:]
            return self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(out), skip_special_tokens=True
            ).strip()
