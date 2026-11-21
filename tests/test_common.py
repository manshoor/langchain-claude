"""Offline tests for the shared model factory in common.py."""

import pytest

from acme_support import common

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Fake key so ChatAnthropic builds; no LC_* overrides leak in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LC_MODEL", raising=False)
    monkeypatch.delenv("LC_EFFORT", raising=False)


def _has_fallback(model) -> bool:
    """True when the request will carry the server-side fallback."""
    return common.FALLBACK_BETA in (model.betas or []) and "fallbacks" in model.model_kwargs


def test_defaults_unchanged() -> None:
    model = common.make_model()

    assert model.model == common.DEFAULT_MODEL
    assert model.reasoning_effort == common.DEFAULT_EFFORT
    assert model.max_tokens == common.MAX_TOKENS
    assert model.model_kwargs == {"fallbacks": "default"}
    assert _has_fallback(model)


def test_env_applies_without_args(monkeypatch) -> None:
    monkeypatch.setenv("LC_MODEL", SONNET)
    monkeypatch.setenv("LC_EFFORT", "medium")

    model = common.make_model()

    assert model.model == SONNET
    assert model.reasoning_effort == "medium"


def test_args_beat_env(monkeypatch) -> None:
    monkeypatch.setenv("LC_MODEL", SONNET)
    monkeypatch.setenv("LC_EFFORT", "medium")

    model = common.make_model(HAIKU, common.Effort.LOW)

    assert model.model == HAIKU
    assert model.reasoning_effort == "low"


def test_haiku_gets_no_fallback() -> None:
    # Haiku 5.5 has no server-side fallback: sending one is wrong.
    assert not _has_fallback(common.make_model(HAIKU))


def test_haiku_from_env_gets_no_fallback(monkeypatch) -> None:
    monkeypatch.setenv("LC_MODEL", HAIKU)

    assert not _has_fallback(common.make_model())


def test_sonnet_keeps_fallback() -> None:
    assert _has_fallback(common.make_model(SONNET, common.Effort.MEDIUM))


def test_prices_cover_role_models() -> None:
    for name in (common.DEFAULT_MODEL, SONNET, HAIKU):
        assert name in common.PRICES


def _result(model: str, input_tokens: int, output_tokens: int):
    """One finished model call, as LangChain reports it to callbacks."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    message = AIMessage("x", usage_metadata=usage, response_metadata={"model_name": model})
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_cost() -> None:
    # 1M input + 1M output tokens on Opus 5.5 at $4 / $20 per million.
    assert common.cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert common.cost("unknown-model", 1_000, 1_000) == 0.0


def test_usage_meter() -> None:
    meter = common.UsageMeter()
    meter.on_llm_end(_result(HAIKU, 1_000_000, 0))
    meter.on_llm_end(_result(HAIKU, 0, 1_000_000))
    meter.on_llm_end(_result(SONNET, 0, 0))

    assert meter.calls() == 3
    assert meter.cost() == pytest.approx(0.10 + 0.50)
    assert f"{HAIKU} ×2" in meter.summary()


def test_unpriced_model_is_flagged() -> None:
    # Review: a model missing from PRICES showed as $0 with no warning.
    meter = common.UsageMeter()
    meter.on_llm_end(_result("claude-new-model", 1_000, 1_000))

    assert "unpriced: claude-new-model" in meter.summary()
