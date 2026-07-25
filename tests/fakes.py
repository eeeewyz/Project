from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class FakeLLM:
    """A deterministic LLM fake that records prompts without using a network."""

    def __init__(self, responses: Iterable[str | Exception]):
        self.responses = iter(responses)
        self.calls: list[str] = []

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls.append(prompt)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class FakeEmbedder:
    """Deterministic, offline embeddings for retrieval tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("blue" in lowered),
                    float("t-shirt" in lowered),
                    float("dress" in lowered),
                    float("return" in lowered),
                ]
            )
        array = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)
