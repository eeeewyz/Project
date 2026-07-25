from __future__ import annotations

from collections.abc import Iterable


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
