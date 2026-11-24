"""Translation by a chat LLM behind an OpenAI-compatible endpoint.

One class covers both a *local* model and a hosted one, because local runners
speak the same protocol:

* Ollama: ``base_url="http://localhost:11434/v1"``, ``model="gemma3:4b"``
* llama.cpp's ``llama-server``: ``base_url="http://localhost:8080/v1"``
* Cerebras / Groq / OpenRouter: their base URL plus an API key

General LLMs tend to cope with dialect and informal spelling better than a
dedicated translation model, at the cost of memory (a 4B model at 4-bit is about
3 GB) and seconds rather than milliseconds per query on CPU.
"""

from __future__ import annotations

from chatbot.rag.translator.base import Translator

_SYSTEM = (
    "You translate questions into Hebrew for a search engine over Israeli "
    "rights-and-benefits material for victims of violence. The input may be "
    "spoken Palestinian Arabic, other Arabic dialects, Modern Standard Arabic, "
    "or English, possibly with typos. Translate its meaning faithfully into "
    "plain Hebrew, using the standard Israeli terms for institutions, benefits "
    "and legal concepts. Output only the Hebrew translation: no quotes, no "
    "notes, no answer to the question."
)


class LLMTranslator(Translator):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return f"llm[{self.model}]"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            # Local runners ignore the key, but the SDK insists on one.
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
                timeout=self.timeout,
            )
        return self._client

    def to_hebrew(self, text: str, source: str) -> str:
        if not text.strip():
            return text
        response = self._get_client().chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"({source})\n{text}"},
            ],
        )
        return (response.choices[0].message.content or "").strip()
