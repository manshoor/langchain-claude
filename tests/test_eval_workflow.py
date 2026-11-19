"""Offline tests for the scoring in eval_workflow.py. No model calls."""

import importlib

import pytest

ev = importlib.import_module("eval_workflow")


def _scored(points: float, cost: float = 0.01, seconds: float = 5.0, **flags) -> dict:
    base = {
        "points": points,
        "cited_expected": 1,
        "cited_any": 1,
        "handoff": 0,
        "cost": cost,
        "seconds": seconds,
        "calls": 4,
        "failed": 0,
    }
    return base | flags


def test_cites_expected() -> None:
    reply = "Within 30 days [refund-policy.md]."

    assert ev._cites(reply, "refund-policy.md")
    assert not ev._cites(reply, "shipping.md")


def test_summary_maths() -> None:
    scored = [
        _scored(1.0, cost=0.01, seconds=2.0),
        _scored(0.5, cost=0.03, seconds=4.0, cited_expected=0),
        _scored(0.0, cost=0.02, seconds=9.0, cited_any=0, cited_expected=0, handoff=1),
    ]

    summary = ev._summarize(scored)

    assert summary["judge"] == pytest.approx(0.5)
    assert summary["expected_cited"] == pytest.approx(1 / 3)
    assert summary["no_citation"] == pytest.approx(1 / 3)
    assert summary["handoff"] == pytest.approx(1 / 3)
    assert summary["cost"] == pytest.approx(0.02)
    assert summary["median_s"] == pytest.approx(4.0)
    assert summary["failed"] == 0


def test_failed_items_score_zero_and_skip_cost() -> None:
    failed = _scored(0.0, cost=0.0, seconds=0.0, cited_expected=0, cited_any=0, calls=0, failed=1)

    summary = ev._summarize([_scored(1.0, cost=0.04, seconds=6.0), failed])

    assert summary["judge"] == pytest.approx(0.5)
    assert summary["cost"] == pytest.approx(0.04)
    assert summary["median_s"] == pytest.approx(6.0)
    assert summary["failed"] == 1


def test_estimate_counts_judge_per_answer() -> None:
    systems = [ev.System.BOT, ev.System.MIXED]
    expected = 10 * (ev.EST_COST[ev.System.BOT] + ev.EST_COST[ev.System.MIXED])
    expected += 2 * 10 * ev.EST_JUDGE_COST

    assert ev._estimate(systems, 10) == pytest.approx(expected)


ITEM = {"input": "Refund?", "expected_output": "refund-policy.md"}


def _answer(**values) -> dict:
    base = {"reply": "x", "handoff": 0, "cost": 0.0123, "calls": 3, "seconds": 2.0, "error": None}
    return base | values


def test_judge_failure_keeps_cost() -> None:
    # Review: an answer whose judge call failed was reported as $0.
    scored = ev._score(ITEM, _answer(), None)

    assert scored["failed"] == 1
    assert scored["cost"] == pytest.approx(0.0123)
    assert scored["calls"] == 3


def test_failed_turn_keeps_its_cost() -> None:
    # Review: a turn that raised after paid calls lost its meter.
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}
    paid = AIMessage("x", usage_metadata=usage, response_metadata={"model_name": "claude-opus-5-5"})

    class BrokenGraph:
        def invoke(self, payload, config):
            config["callbacks"][0].on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=paid)]])
            )
            raise RuntimeError("reviewer failed")

    answer = ev._ask_workflow(BrokenGraph(), "Refund?", [])

    assert answer["error"] == "reviewer failed"
    assert answer["cost"] == pytest.approx(4.0)
    assert ev._score(ITEM, answer, {"grade": "correct", "reason": ""})["failed"] == 1


def test_stops_when_every_item_failed() -> None:
    failed = _scored(0.0, failed=1)

    with pytest.raises(SystemExit):
        ev._check_failures(ev.System.MIXED, [failed, failed])

    ev._check_failures(ev.System.MIXED, [failed, _scored(1.0)])
