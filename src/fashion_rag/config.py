from __future__ import annotations

import os
from dataclasses import dataclass


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    together_api_key: str | None
    together_model: str
    embedding_model: str
    top_k: int
    max_history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            together_api_key=os.getenv("TOGETHER_API_KEY") or None,
            together_model=os.getenv("TOGETHER_MODEL", "Qwen/Qwen3.5-9B"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            top_k=int(os.getenv("TOP_K", "5")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "4")),
        )

    def require_api_key(self) -> str:
        if self.together_api_key is None:
            raise MissingAPIKeyError(
                "TOGETHER_API_KEY is not configured. Add it as an environment "
                "variable or Hugging Face Space secret."
            )
        return self.together_api_key
