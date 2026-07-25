import pytest

from fashion_rag.config import MissingAPIKeyError, Settings


def test_defaults_do_not_require_api_key_at_startup(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.together_model == "Qwen/Qwen3.5-9B"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.top_k == 5
    assert settings.max_history_turns == 4
    assert settings.together_api_key is None


def test_require_api_key_returns_secret_without_logging_it(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "secret-value")
    assert Settings.from_env().require_api_key() == "secret-value"


def test_require_api_key_raises_typed_error(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="TOGETHER_API_KEY"):
        Settings.from_env().require_api_key()
