"""Compare the single-agent bot (05) with the multi-model workflow (06).

Every question in evals/retrieval_questions.json goes to each system, one
fresh conversation per question. Each reply is scored:

    judge           Opus 5.5 grades the reply against the full text of the doc
                    that answers it: correct 1, partial 0.5, wrong 0
    expected cited  the reply cites that doc
    no citation     the reply cites nothing (it declined or handed off)
    handoff         06 only: the workflow asked to open a ticket (resumed "no")
    cost, median time and model calls per question

Known limits: Opus grades replies Opus wrote (grading against the doc text
reduces this bias); 24 single-doc questions; replies vary between runs.

With LANGFUSE_* keys each system's run is a Langfuse experiment run, one
trace per question with the scores attached. Without keys it runs locally.

This calls the API and costs money. See the estimate first:

    uv run eval_workflow.py --dry-run
    uv run eval_workflow.py                                # 05, 06-mixed, 06-opus
    uv run eval_workflow.py --systems 05 06-mixed --limit 5
"""

import argparse
import importlib
import json
import re
import statistics
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import tracing
from common import Effort, UsageMeter, make_model
from retrieval import DOCS_DIR, load_chunks, vector_store

support_bot = importlib.import_module("05_support_bot")
workflow = importlib.import_module("06_support_workflow")

QUESTIONS_PATH = Path(__file__).parent / "evals" / "retrieval_questions.json"
EXPERIMENT_NAME = "support-eval"
JUDGE_MODEL = "claude-opus-5-5"
DECLINE = "no"  # answer to the workflow's ticket question, so no ticket opens
DEFAULT_WORKERS = 4
CITATION = re.compile(r"\[([\w-]+\.md)\]")


class System(StrEnum):
    """A support system under test."""

    BOT = "05"
    MIXED = "06-mixed"
    OPUS = "06-opus"
    HAIKU = "06-haiku"


DEFAULT_SYSTEMS = [System.BOT, System.MIXED, System.OPUS]
PRESETS = {
    System.MIXED: workflow.Preset.MIXED,
    System.OPUS: workflow.Preset.OPUS,
    System.HAIKU: workflow.Preset.HAIKU,
}

# Rough US dollars per question, from the Phase 5 live runs. Only for --dry-run.
EST_COST = {
    System.BOT: 0.02,
    System.MIXED: 0.015,
    System.OPUS: 0.06,
    System.HAIKU: 0.004,
}
EST_JUDGE_COST = 0.012


class GradeLevel(StrEnum):
    """The judge's verdict on one reply."""

    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"


POINTS = {GradeLevel.CORRECT: 1.0, GradeLevel.PARTIAL: 0.5, GradeLevel.WRONG: 0.0}

JUDGE_PROMPT = """You grade a customer support reply against the policy document
that answers the customer's question.

- correct: the reply answers the question, and every policy fact in it agrees
  with the document.
- partial: right, but incomplete, or it adds a minor detail the document
  doesn't support.
- wrong: it contradicts the document, misses the answer, or says the policies
  don't cover something the document does cover.

Judge only the facts. Ignore style, length and citation format."""


class Grade(TypedDict):
    """The judge's structured output."""

    grade: Annotated[GradeLevel, ..., "correct, partial or wrong"]
    reason: Annotated[str, ..., "One sentence"]


class Answer(TypedDict):
    """One system's reply to one question, with what it cost."""

    reply: str
    handoff: int  # 1 when the workflow asked to open a ticket
    cost: float
    calls: int
    seconds: float
    error: str | None  # set when the turn failed part way; cost still counts


class Scored(TypedDict):
    """One answer's scores. Flags are 0 or 1 so they average into shares."""

    points: float
    cited_expected: int
    cited_any: int
    handoff: int
    cost: float
    seconds: float
    calls: int
    failed: int


Ask = Callable[[str, list], Answer]


# ---- systems -----------------------------------------------------------------


def _system(system: System, store: InMemoryVectorStore) -> Ask:
    """Build one system, returning a function that asks it one question."""
    if system == System.BOT:
        return partial(_ask_bot, support_bot.build_bot(store, InMemorySaver()))

    team = workflow.build_team(PRESETS[system], store)
    return partial(_ask_workflow, workflow.build_graph(team, InMemorySaver()))


def _fresh_config(callbacks: list) -> dict:
    """A new conversation per question, so no answer sees another question."""
    return {"configurable": {"thread_id": uuid.uuid4().hex}, "callbacks": callbacks}


def _ask_bot(bot, question: str, callbacks: list) -> Answer:
    """Ask 05 one question."""
    meter = UsageMeter()
    started = time.monotonic()
    config = _fresh_config([meter, *callbacks])
    try:
        result = bot.invoke({"messages": [HumanMessage(question)]}, config)
    except Exception as error:  # noqa: BLE001 - keep what the failed turn already cost
        return _failed(meter, started, error)

    return Answer(
        reply=result["messages"][-1].text,
        handoff=0,
        cost=meter.cost(),
        calls=meter.calls(),
        seconds=time.monotonic() - started,
        error=None,
    )


def _ask_workflow(graph, question: str, callbacks: list) -> Answer:
    """Ask 06 one question. A ticket question is answered "no"."""
    meter = UsageMeter()
    started = time.monotonic()
    config = _fresh_config([meter, *callbacks])

    try:
        graph.invoke({"messages": [HumanMessage(question)]}, config)
        handoff = bool(graph.get_state(config).interrupts)
        if handoff:
            graph.invoke(Command(resume=DECLINE), config)
    except Exception as error:  # noqa: BLE001 - keep what the failed turn already cost
        return _failed(meter, started, error)

    return Answer(
        reply=graph.get_state(config).values["messages"][-1].text,
        handoff=int(handoff),
        cost=meter.cost(),
        calls=meter.calls(),
        seconds=time.monotonic() - started,
        error=None,
    )


def _failed(meter: UsageMeter, started: float, error: Exception) -> Answer:
    """A turn that raised part way. Its paid calls still count toward the total."""
    print(f"  item failed: {error}")
    return Answer(
        reply="",
        handoff=0,
        cost=meter.cost(),
        calls=meter.calls(),
        seconds=time.monotonic() - started,
        error=str(error),
    )


# ---- scoring -----------------------------------------------------------------


def _judge() -> Runnable:
    """Opus 5.5 at medium effort, returning a Grade."""
    model = make_model(JUDGE_MODEL, Effort.MEDIUM)
    return model.with_structured_output(Grade, method="json_schema")


def _grade(judge: Runnable, item: dict, reply: str, meter: UsageMeter) -> Grade:
    """Grade one reply against the full text of the doc that answers it."""
    source = item["expected_output"]
    prompt = (
        f"Question:\n{item['input']}\n\n"
        f"Policy document ({source}):\n{(DOCS_DIR / source).read_text()}\n\n"
        f"Reply:\n{reply}"
    )
    messages = [SystemMessage(JUDGE_PROMPT), HumanMessage(prompt)]
    return judge.invoke(messages, {"callbacks": [meter]})


def _cites(reply: str, source: str) -> bool:
    """True when the reply cites `source` as [source]."""
    return source in CITATION.findall(reply)


def _score(item: dict, answer: Answer | None, grade: Grade | None) -> Scored:
    """Turn one answer and its grade into numbers.

    A failed item scores 0 but keeps whatever it cost, so totals stay honest.
    """
    if answer is None:
        return Scored(
            points=0.0,
            cited_expected=0,
            cited_any=0,
            handoff=0,
            cost=0.0,
            seconds=0.0,
            calls=0,
            failed=1,
        )

    if answer["error"] is not None or grade is None:
        return Scored(
            points=0.0,
            cited_expected=0,
            cited_any=0,
            handoff=0,
            cost=answer["cost"],
            seconds=answer["seconds"],
            calls=answer["calls"],
            failed=1,
        )

    reply = answer["reply"]
    return Scored(
        points=POINTS[GradeLevel(grade["grade"])],
        cited_expected=int(_cites(reply, item["expected_output"])),
        cited_any=int(bool(CITATION.search(reply))),
        handoff=answer["handoff"],
        cost=answer["cost"],
        seconds=answer["seconds"],
        calls=answer["calls"],
        failed=0,
    )


def _summarize(scored: list[Scored]) -> dict[str, float]:
    """Quality over every item; cost, time and calls over items that ran."""
    ran = [s for s in scored if not s["failed"]] or scored
    return {
        "judge": statistics.mean(s["points"] for s in scored),
        "expected_cited": statistics.mean(s["cited_expected"] for s in scored),
        "no_citation": statistics.mean(1 - s["cited_any"] for s in scored),
        "handoff": statistics.mean(s["handoff"] for s in scored),
        "cost": statistics.mean(s["cost"] for s in ran),
        "median_s": statistics.median(s["seconds"] for s in ran),
        "calls": statistics.mean(s["calls"] for s in ran),
        "failed": sum(s["failed"] for s in scored),
    }


def _estimate(systems: list[System], questions: int) -> float:
    """Rough US dollars for a run: each system's answers plus one judge call each."""
    answers = sum(EST_COST[s] for s in systems) * questions
    return answers + len(systems) * questions * EST_JUDGE_COST


# ---- run ---------------------------------------------------------------------


def _run_system(
    system: System, items: list[dict], store, judge, judge_meter, workers: int, stamp: str
) -> list[Scored]:
    """Answer every question with one system, grade each answer, attach scores."""
    ask = _system(system, store)

    def task(item: dict, callbacks: list) -> Answer:
        return ask(item["input"], callbacks)

    results = tracing.run_experiment(EXPERIMENT_NAME, f"{system} {stamp}", items, task, workers)

    scored = []
    for item, (answer, trace_id) in zip(items, results, strict=True):
        grade = _safe_grade(judge, item, answer, judge_meter)
        scored.append(_score(item, answer, grade))
        _attach_scores(trace_id, scored[-1], grade)

    return scored


def _safe_grade(judge, item: dict, answer: Answer | None, meter: UsageMeter) -> Grade | None:
    """Grade one answer; None when there is no answer or the judge fails."""
    if answer is None or answer["error"] is not None:
        return None

    try:
        return _grade(judge, item, answer["reply"], meter)
    except Exception as error:  # noqa: BLE001 - one bad grade must not stop the eval
        print(f"  judge failed on {item['input']!r}: {error}")
        return None


def _check_failures(system: System, scored: list[Scored]) -> None:
    """Stop the eval when every item of a system failed: the next system would fail too."""
    if scored and all(s["failed"] for s in scored):
        raise SystemExit(
            f"Every {system} item failed. Stopping before the next system spends more."
        )


def _attach_scores(trace_id: str | None, scored: Scored, grade: Grade | None) -> None:
    """Send one item's scores to its Langfuse trace (nothing happens offline)."""
    trace = tracing.Trace(trace_id)
    trace.score("judge", scored["points"])
    trace.score("expected_cited", scored["cited_expected"])
    trace.score("handoff", scored["handoff"])
    if grade is not None:
        trace.score("grade", grade["grade"])


def _items(limit: int | None) -> list[dict]:
    """Questions in Langfuse's experiment item format."""
    questions = json.loads(QUESTIONS_PATH.read_text())[:limit]
    return [
        {"input": q["question"], "expected_output": q["source"], "metadata": {"style": q["style"]}}
        for q in questions
    ]


def _table(rows: dict[System, dict[str, float]]) -> str:
    """Markdown results table."""
    lines = [
        (
            "| System | Judge | Expected doc cited | No citation | Handoff "
            "| Cost / question | Median time | Model calls | Failed |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system, r in rows.items():
        lines.append(
            f"| {system} | {r['judge']:.2f} | {r['expected_cited']:.0%} | {r['no_citation']:.0%} "
            f"| {r['handoff']:.0%} | ${r['cost']:.4f} | {r['median_s']:.1f} s "
            f"| {r['calls']:.1f} | {r['failed']} |"
        )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    """Read the command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems", nargs="+", type=System, choices=list(System), default=DEFAULT_SYSTEMS
    )
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="questions at once")
    parser.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    return parser.parse_args()


def main() -> None:
    """Estimate the cost, then (unless --dry-run) run, grade and print the table."""
    args = _parse_args()
    items = _items(args.limit)
    systems: list[System] = args.systems
    print(f"{len(items)} questions × {len(systems)} systems: {', '.join(systems)}")
    print(f"Estimated cost: ${_estimate(systems, len(items)):.2f} (rough, from live demo runs)")
    print(f"Langfuse: {'experiment runs' if tracing.enabled() else 'off, local run'}\n")
    if args.dry_run:
        return

    store = vector_store(load_chunks())
    judge, judge_meter = _judge(), UsageMeter()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows: dict[System, dict[str, float]] = {}
    spent = 0.0  # what the answers actually cost, failed items included
    try:
        for system in systems:
            started = time.monotonic()
            print(f"Running {system} ...")
            scored = _run_system(system, items, store, judge, judge_meter, args.workers, stamp)
            rows[system] = _summarize(scored)
            system_cost = sum(s["cost"] for s in scored)
            spent += system_cost
            print(f"  done in {time.monotonic() - started:.0f} s, ${system_cost:.2f}")
            _check_failures(system, scored)
    finally:
        # Print what finished even on Ctrl-C or a stop: those runs are paid for.
        tracing.shutdown()
        if rows:
            print(f"\n{_table(rows)}\n")
            print(f"Judge: {judge_meter.summary()}")
            print(f"Total: ${spent + judge_meter.cost():.2f}")


if __name__ == "__main__":
    main()
