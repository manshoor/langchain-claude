"""Offline tests for the support workflow graph in 06_support_workflow.py.

Every role is a RunnableLambda fake, so no test calls a model.
"""

import importlib
from collections.abc import Iterable

import pytest
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import support_tools

wf = importlib.import_module("06_support_workflow")

REFUND_PASSAGE = "[refund-policy.md]\nLaptops can be returned within 30 days."
DRAFT = "Yes, within 30 days [refund-policy.md]."


# ---- fakes -------------------------------------------------------------------


def _triage(route: str, questions: Iterable[str] = (), priority: str = "normal") -> dict:
    """A triage result as the structured-output model would return it."""
    return {"route": route, "questions": list(questions), "priority": priority, "reason": "test"}


def _review(verdict: str, problems: Iterable[str] = ()) -> dict:
    return {"verdict": verdict, "problems": list(problems)}


def _research(payload: dict) -> dict:
    """Fake subagent: one search result, then notes."""
    question = payload["messages"][0].content
    return {
        "messages": [
            *payload["messages"],
            ToolMessage(REFUND_PASSAGE, name="search_policies", tool_call_id="t1"),
            AIMessage(
                f"Notes on {question} [refund-policy.md]",
                response_metadata={"model_name": "fake"},
            ),
        ]
    }


class _Writer:
    """Fake writer that records the system prompt of every call."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: list) -> AIMessage:
        self.prompts.append(next(m.content for m in messages if isinstance(m, SystemMessage)))
        return AIMessage(DRAFT)


def _team(triages: list[dict], reviews: Iterable[dict] = (), writer: _Writer | None = None):
    """Build a Team whose triage and reviewer replay the given results in order."""
    triage_iter = iter(triages)
    review_iter = iter(reviews)
    return wf.Team(
        triage=RunnableLambda(lambda _: next(triage_iter)),
        researcher=RunnableLambda(_research),
        writer=RunnableLambda(writer or _Writer()),
        reviewer=RunnableLambda(lambda _: next(review_iter)),
    )


def _graph(team):
    return wf.build_graph(team, InMemorySaver())


def _say(graph, config: dict, text: str) -> None:
    graph.invoke({"messages": [HumanMessage(text)]}, config)


@pytest.fixture
def config() -> dict:
    return {"configurable": {"thread_id": "test"}}


@pytest.fixture(autouse=True)
def _no_tickets(monkeypatch) -> None:
    monkeypatch.setattr(support_tools, "_tickets", [])


# ---- routers -----------------------------------------------------------------


def _state(**values) -> dict:
    base = {
        "messages": [HumanMessage("Can I return my laptop?")],
        "triage": _triage("answer", ["q"]),
        "findings": [],
        "draft": "",
        "review": None,
        "revisions": 0,
    }
    return base | values


def test_dispatch_handoff() -> None:
    assert wf._dispatch(_state(triage=_triage("handoff"))) == wf.Node.HANDOFF


def test_dispatch_chat() -> None:
    assert wf._dispatch(_state(triage=_triage("chat"))) == wf.Node.WRITE


def test_dispatch_one_send_per_question() -> None:
    sends = wf._dispatch(_state(triage=_triage("answer", ["refund?", "how long?"])))

    assert [s.node for s in sends] == [wf.Node.RESEARCH, wf.Node.RESEARCH]
    assert [s.arg["question"] for s in sends] == ["refund?", "how long?"]


def test_dispatch_no_questions_uses_message() -> None:
    sends = wf._dispatch(_state(triage=_triage("answer", ["", "  "])))

    assert [s.arg["question"] for s in sends] == ["Can I return my laptop?"]


def test_dispatch_caps_questions() -> None:
    sends = wf._dispatch(_state(triage=_triage("answer", ["a", "b", "c", "d", "e"])))

    assert [s.arg["question"] for s in sends] == ["a", "b", "c"]


def test_after_write() -> None:
    assert wf._after_write(_state(triage=_triage("chat"))) == wf.Node.REPLY
    assert wf._after_write(_state()) == wf.Node.REVIEW


@pytest.mark.parametrize(
    ("verdict", "revisions", "expected"),
    [
        ("pass", 0, "reply"),
        ("not_covered", 0, "handoff"),
        ("revise", 1, "write"),
        ("revise", 2, "handoff"),
    ],
)
def test_after_review(verdict: str, revisions: int, expected: str) -> None:
    state = _state(review=_review(verdict), revisions=revisions)

    assert wf._after_review(state) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("yes", True),
        ("Yes please", True),
        ("ok.", True),
        ("sure, go ahead", True),
        ("no", False),
        ("", False),
        # Review: these opened a ticket without consent.
        ("please don't", False),
        ("Please no", False),
        ("ok, no thanks", False),
        ("okay I guess not", False),
        ("sure, but not now", False),
    ],
)
def test_is_yes(answer: str, expected: bool) -> None:
    assert wf._is_yes(answer) is expected


def test_cited_docs() -> None:
    text = "Within 30 days [refund-policy.md], shipped [shipping.md] [refund-policy.md]."

    assert wf._cited_docs(text) == ["refund-policy.md", "shipping.md"]


# ---- whole graph -------------------------------------------------------------


def test_answer_path_runs_research_in_parallel(config) -> None:
    graph = _graph(_team([_triage("answer", ["refund?", "how long?"])], [_review("pass")]))

    _say(graph, config, "Refund for my laptop, and how long?")
    values = graph.get_state(config).values

    assert sorted(f["question"] for f in values["findings"]) == ["how long?", "refund?"]
    assert all(f["passages"] == REFUND_PASSAGE for f in values["findings"])
    assert values["messages"][-1].content == DRAFT
    assert isinstance(values["messages"][-1], AIMessage)


def test_revise_loop_sends_feedback(config) -> None:
    writer = _Writer()
    reviews = [_review("revise", ["Cite the timing."]), _review("pass")]
    graph = _graph(_team([_triage("answer", ["refund?"])], reviews, writer))

    _say(graph, config, "Refund?")

    assert len(writer.prompts) == 2
    assert "Cite the timing." not in writer.prompts[0]
    assert "Cite the timing." in writer.prompts[1]
    assert graph.get_state(config).values["revisions"] == 1


def test_failed_review_twice_hands_off(config) -> None:
    reviews = [_review("revise", ["x"]), _review("revise", ["y"])]
    graph = _graph(_team([_triage("answer", ["refund?"])], reviews))

    _say(graph, config, "Refund?")
    pending = graph.get_state(config).interrupts

    # The policies do cover it; the draft just failed review. Don't say "not covered".
    assert pending[0].value == wf.UNCONFIRMED_QUESTION


def test_handoff_yes_opens_ticket(config) -> None:
    graph = _graph(_team([_triage("answer", ["pickup?"])], [_review("not_covered")]))

    _say(graph, config, "Can I pick it up in Abu Dhabi?")
    assert graph.get_state(config).interrupts[0].value == wf.NOT_COVERED_QUESTION

    graph.invoke(Command(resume="yes"), config)
    last = graph.get_state(config).values["messages"][-1]

    assert "TCK-0001" in last.content
    assert support_tools._tickets == ["Can I pick it up in Abu Dhabi?"]


def test_handoff_no_skips_ticket(config) -> None:
    graph = _graph(_team([_triage("handoff")]))

    _say(graph, config, "I want a human.")
    graph.invoke(Command(resume="no"), config)

    assert support_tools._tickets == []
    assert "no ticket" in graph.get_state(config).values["messages"][-1].content.lower()


def test_second_turn_starts_clean(config) -> None:
    triages = [_triage("answer", ["refund?"]), _triage("chat")]
    graph = _graph(_team(triages, [_review("pass")]))

    _say(graph, config, "Refund?")
    _say(graph, config, "Thanks!")
    values = graph.get_state(config).values

    assert values["findings"] == []
    assert values["review"] is None
    assert values["revisions"] == 0
    assert len(values["messages"]) == 4


def test_reviewer_sees_every_question(config) -> None:
    # Live run: the writer dropped one of two questions and review still passed.
    seen: list[str] = []

    def reviewer(messages: list) -> dict:
        seen.append(messages[-1].content)
        return _review("pass")

    team = _team([_triage("answer", ["refund?", "how long?"])])
    team = wf.Team(team.triage, team.researcher, team.writer, RunnableLambda(reviewer))

    _say(_graph(team), config, "Refund, and how long?")

    assert "- refund?" in seen[0]
    assert "- how long?" in seen[0]


def test_chat_turn_trusts_earlier_replies(config) -> None:
    # Live run: with no notes, the writer retracted its own checked answer.
    writer = _Writer()
    triages = [_triage("answer", ["refund?"]), _triage("chat")]
    graph = _graph(_team(triages, [_review("pass")], writer))

    _say(graph, config, "Refund?")
    _say(graph, config, "What was my name?")

    assert wf.CHAT_NOTES not in writer.prompts[0]
    assert wf.CHAT_NOTES in writer.prompts[1]


def test_history_is_text_only(config) -> None:
    graph = _graph(_team([_triage("answer", ["refund?"])], [_review("pass")]))

    _say(graph, config, "Refund?")
    messages = graph.get_state(config).values["messages"]

    # Subagent tool calls and results stay out of the customer conversation.
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert isinstance(messages[-1].content, str)


def test_research_limit_falls_back_to_passages() -> None:
    # Review: the call-limit message became the notes, so the writer had no facts.
    def limited(payload: dict) -> dict:
        return {
            "messages": [
                *payload["messages"],
                ToolMessage(REFUND_PASSAGE, name="search_policies", tool_call_id="t1"),
                AIMessage("Model call limits exceeded: run limit (4/4)"),
            ]
        }

    team = _team([])
    team = wf.Team(team.triage, RunnableLambda(limited), team.writer, team.reviewer)

    update = wf._research({"question": "refund?"}, team=team)
    notes = update["findings"][0]["notes"]

    assert "limits exceeded" not in notes
    assert REFUND_PASSAGE in notes


def test_retry_continues_failed_turn(config) -> None:
    # Review: retrying sent the message again and re-ran paid research.
    research_runs: list[str] = []
    writer_calls: list[int] = []

    def research(payload: dict) -> dict:
        research_runs.append(payload["messages"][0].content)
        return _research(payload)

    def flaky_writer(messages: list) -> AIMessage:
        writer_calls.append(1)
        if len(writer_calls) == 1:
            raise RuntimeError("API overloaded")
        return AIMessage(DRAFT)

    team = _team([_triage("answer", ["refund?"])], [_review("pass")])
    team = wf.Team(
        team.triage, RunnableLambda(research), RunnableLambda(flaky_writer), team.reviewer
    )
    graph = _graph(team)

    with pytest.raises(RuntimeError):
        wf._run_turn(graph, config, "Refund?", lambda q: "no", [])
    reply = wf._run_turn(graph, config, None, lambda q: "no", [])

    messages = graph.get_state(config).values["messages"]
    assert reply == DRAFT
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert research_runs == ["refund?"]


class _Request:
    """Stand-in for a middleware ModelRequest."""

    def __init__(self, model: str) -> None:
        self.model = model

    def override(self, *, model: str) -> "_Request":
        return _Request(model)


def _handler(request: _Request) -> str:
    if request.model == "haiku":
        raise ModelRefusalError()
    return f"answered by {request.model}"


def test_refusal_retries_on_backup() -> None:
    fallback = wf._RefusalFallback("sonnet")

    assert fallback.wrap_model_call(_Request("haiku"), _handler) == "answered by sonnet"


def test_other_errors_do_not_fall_back() -> None:
    # Review: the stock fallback turned rate-limit errors into Sonnet calls.
    def overloaded(request: _Request) -> str:
        raise RuntimeError("429")

    with pytest.raises(RuntimeError):
        wf._RefusalFallback("sonnet").wrap_model_call(_Request("haiku"), overloaded)


def test_threads_do_not_collide_with_05() -> None:
    # Review: 06 on a thread 05 used read 05's tool calls.
    assert wf._thread_id("ali") == "06:ali"
