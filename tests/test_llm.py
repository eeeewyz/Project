from __future__ import annotations

from types import SimpleNamespace

import pytest

from fashion_rag.config import Settings
from fashion_rag.llm import (
    LLMServiceError,
    StructuredGenerator,
    StructuredOutputError,
    TogetherLLM,
)
from fashion_rag.schemas import Route, RouteDecision
from tests.fakes import FakeLLM


def _contains_secret(value: object, secret: str, seen: set[int] | None = None) -> bool:
    """Inspect simple local-value graphs without invoking arbitrary repr methods."""

    seen = seen or set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, str):
        return secret in value
    if isinstance(value, BaseException):
        return _contains_secret(value.args, secret, seen)
    if isinstance(value, dict):
        return any(
            _contains_secret(item, secret, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_secret(item, secret, seen) for item in value)
    return False


def _traceback_locals_contain(error: BaseException, secret: str) -> bool:
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        # The caller naturally owns the test secret; inspect every adapter
        # frame carried by the public error instead of that external caller.
        if (
            frame.f_globals.get("__name__") == "fashion_rag.llm"
            and any(
                _contains_secret(value, secret)
                for value in frame.f_locals.values()
            )
        ):
            return True
        traceback = traceback.tb_next
    return False


def settings() -> Settings:
    return Settings(
        together_api_key="test-key",
        together_model="Qwen/Qwen3.5-9B",
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=5,
        max_history_turns=4,
    )


def test_structured_generator_accepts_fenced_json():
    llm = FakeLLM(['```json\n{"route":"faq"}\n```'])
    result = StructuredGenerator(llm).generate(
        "route this", RouteDecision, temperature=0, max_tokens=40
    )
    assert result.route is Route.FAQ


def test_structured_generator_retries_once_after_invalid_json():
    llm = FakeLLM(["not-json", '{"route":"product"}'])
    result = StructuredGenerator(llm).generate(
        "route this", RouteDecision, temperature=0, max_tokens=40
    )
    assert result.route is Route.PRODUCT
    assert len(llm.calls) == 2
    assert "Return corrected JSON only" in llm.calls[1]


def test_structured_generator_stops_after_second_invalid_response():
    llm = FakeLLM(["bad", "still bad"])
    with pytest.raises(StructuredOutputError):
        StructuredGenerator(llm).generate(
            "route this", RouteDecision, temperature=0, max_tokens=40
        )


def test_together_adapter_retries_one_transient_failure():
    class StubCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            message = SimpleNamespace(content="recovered")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = StubCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = TogetherLLM(settings(), client=client, retry_delay=0).complete(
        "hello", temperature=0, max_tokens=20
    )
    assert result == "recovered"
    assert completions.calls == 2


@pytest.mark.parametrize("error", [PermissionError("invalid API key"), ValueError("bad request")])
def test_together_adapter_does_not_retry_permanent_failures(error):
    class StubCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))
    with pytest.raises(LLMServiceError) as exc_info:
        TogetherLLM(settings(), client=client, retry_delay=0).complete(
            "hello", temperature=0, max_tokens=20
        )
    assert client.chat.completions.calls == 1
    assert "test-key" not in str(exc_info.value)


def test_together_adapter_does_not_chain_provider_secrets():
    secret = "fake-provider-secret"

    class AuthenticationError(RuntimeError):
        status_code = 401

    class StubCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise AuthenticationError(f"provider rejected {secret}")

    client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))
    with pytest.raises(LLMServiceError) as exc_info:
        TogetherLLM(settings(), client=client, retry_delay=0).complete(
            "hello", temperature=0, max_tokens=20
        )

    error = exc_info.value
    assert client.chat.completions.calls == 1
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not _traceback_locals_contain(error, secret)


def test_together_adapter_retries_one_rate_limit_failure():
    class RateLimitError(RuntimeError):
        status_code = 429

    class StubCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("slow down")
            message = SimpleNamespace(content="recovered")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = StubCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert TogetherLLM(settings(), client=client, retry_delay=0).complete(
        "hello", temperature=0, max_tokens=20
    ) == "recovered"
    assert completions.calls == 2
