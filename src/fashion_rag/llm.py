"""Provider-neutral LLM access and validated structured generation."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from fashion_rag.config import Settings

if TYPE_CHECKING:
    from together import Together


T = TypeVar("T", bound=BaseModel)


class LLMServiceError(RuntimeError):
    """Raised when the language-model provider cannot fulfil a request."""


class StructuredOutputError(LLMServiceError):
    """Raised when an LLM fails to return schema-valid JSON twice."""


class LLMClient(Protocol):
    def complete(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str: ...


def _status_code(error: Exception) -> int | None:
    """Read common HTTP status locations without depending on one SDK version."""

    for owner in (error, getattr(error, "response", None)):
        code = getattr(owner, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_transient_error(error: Exception) -> bool:
    """Return true only for failures that are reasonable to retry once."""

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True

    status = _status_code(error)
    if status is not None:
        return status == 429 or status == 408 or 500 <= status <= 599

    name = type(error).__name__.lower()
    return any(
        marker in name
        for marker in (
            "timeout",
            "connection",
            "ratelimit",
            "rate_limit",
            "serviceunavailable",
            "service_unavailable",
            "transient",
        )
    )


class TogetherLLM:
    """Small adapter around Together's chat-completions client."""

    def __init__(
        self,
        settings: Settings,
        client: "Together | Any | None" = None,
        retry_delay: float = 0.25,
    ) -> None:
        self.settings = settings
        self.client = client if client is not None else self._create_client()
        self.retry_delay = retry_delay

    def _create_client(self) -> Any:
        try:
            from together import Together
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise LLMServiceError(
                "The Together SDK is not installed in this environment."
            ) from exc
        return Together(api_key=self.settings.require_api_key())

    def complete(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str:
        for attempt in range(2):
            should_retry = False
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.together_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    chat_template_kwargs={"enable_thinking": False},
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise LLMServiceError(
                        "The language-model service returned no text."
                    )
                return content
            except LLMServiceError:
                raise
            except Exception as exc:
                # Retain only the retry decision. Python clears ``exc`` after
                # this handler, keeping provider request/authentication text
                # out of the public exception's traceback locals.
                should_retry = _is_transient_error(exc)

            # This runs outside the exception handler so the public exception
            # does not retain provider text in __cause__ or __context__.
            if not should_retry or attempt == 1:
                raise LLMServiceError("The language-model service is unavailable.")
            time.sleep(self.retry_delay)

        raise AssertionError("unreachable")


def strip_json_fence(text: str) -> str:
    """Remove a single Markdown JSON fence while leaving ordinary JSON intact."""

    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return match.group(1) if match else stripped


class StructuredGenerator:
    """Generate a Pydantic model, correcting one invalid LLM response once."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        prompt: str,
        schema: type[T],
        *,
        temperature: float,
        max_tokens: int,
    ) -> T:
        current_prompt = prompt
        for attempt in range(2):
            raw = self.llm.complete(
                current_prompt, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return schema.model_validate_json(strip_json_fence(raw))
            except ValidationError as exc:
                if attempt == 1:
                    raise StructuredOutputError(
                        f"Model output did not match {schema.__name__}."
                    ) from exc
                current_prompt = (
                    f"{prompt}\n\n"
                    "The previous response did not match the required JSON schema. "
                    "Return corrected JSON only."
                )

        raise AssertionError("unreachable")
