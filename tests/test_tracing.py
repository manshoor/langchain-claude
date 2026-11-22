"""Offline tests for tracing.py: tracing must never break or slow a turn."""

import pytest

from acme_support import tracing

KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}


@pytest.fixture
def no_keys(monkeypatch) -> None:
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def keys(monkeypatch) -> None:
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)


def _boom(*args, **kwargs):
    raise RuntimeError("langfuse is down")


def test_off_without_keys(no_keys, monkeypatch) -> None:
    # Any attempt to reach Langfuse fails the test.
    monkeypatch.setattr(tracing, "_start", _boom)

    with tracing.turn_trace("support-turn", "thread", ["workflow"], "hi") as trace:
        assert trace.callbacks == []
        trace.score("verdict", "pass")
        trace.finish("bye")
        assert trace.url() is None

    tracing.shutdown()


def test_one_key_is_not_enough(no_keys, monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

    assert not tracing.enabled()


def test_setup_failure_keeps_turn_running(keys, monkeypatch) -> None:
    monkeypatch.setattr(tracing, "_start", _boom)
    ran = []

    with tracing.turn_trace("support-turn", "thread", ["workflow"], "hi") as trace:
        ran.append(True)
        assert trace.callbacks == []

    assert ran == [True]


def test_turn_errors_still_raise(no_keys) -> None:
    # Tracing swallows its own failures, never the turn's.
    with pytest.raises(ValueError), tracing.turn_trace("t", "s", [], "hi"):
        raise ValueError("model failed")


def _span(scope: str, attributes: dict):
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    return ReadableSpan(
        name="span", instrumentation_scope=InstrumentationScope(scope), attributes=attributes
    )


def test_drops_anthropic_sdk_spans() -> None:
    # Live run: the SDK's own span duplicated every LangChain generation and its cost.
    sdk = _span("com.anthropic.sdk.python", {"gen_ai.request.model": "claude-opus-5-5"})

    assert not tracing._keep_span(sdk)


def test_keeps_langfuse_spans() -> None:
    assert tracing._keep_span(_span("langfuse-sdk", {}))


ITEMS = [{"input": "a"}, {"input": "b"}, {"input": "c"}]


def test_experiment_runs_locally_in_order(no_keys) -> None:
    results = tracing.run_experiment("eval", "run", ITEMS, lambda item, cb: item["input"], 2)

    assert results == [("a", None), ("b", None), ("c", None)]


def test_experiment_failed_item_is_none(no_keys) -> None:
    def task(item: dict, callbacks: list) -> str:
        if item["input"] == "b":
            raise RuntimeError("API error")
        return item["input"]

    results = tracing.run_experiment("eval", "run", ITEMS, task, 2)

    assert [output for output, _ in results] == ["a", None, "c"]


def test_experiment_offline_has_no_callbacks(no_keys) -> None:
    results = tracing.run_experiment("eval", "run", ITEMS[:1], lambda item, cb: cb, 1)

    assert results == [([], None)]


def test_experiment_failure_reruns_only_missing(keys, monkeypatch) -> None:
    # Langfuse finishes item 0, then breaks: only items 1 and 2 may run again.
    def half_done(name, run_name, items, task, workers, results) -> None:
        results[0] = ("from langfuse", "trace-0")
        raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_langfuse_experiment", half_done)
    ran: list[str] = []

    def task(item: dict, callbacks: list) -> str:
        ran.append(item["input"])
        return item["input"]

    results = tracing.run_experiment("eval", "run", ITEMS, task, 2)

    assert sorted(ran) == ["b", "c"]
    assert results == [("from langfuse", "trace-0"), ("b", None), ("c", None)]
