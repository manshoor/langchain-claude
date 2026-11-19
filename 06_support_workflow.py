"""Support workflow: one customer turn split across several Claude models.

    customer message
       │
       ▼
    triage ─────────────── Haiku 5.5: route, 1–3 sub-questions, priority
       │ answer            chat → write → reply · handoff → handoff
       ▼
    research × N ───────── Haiku 5.5 subagents, in parallel, each with search_policies
       │
       ▼
    write ◄──────┐ ─────── Opus 5.5: the reply, with [file.md] citations
       │         │ revise, once
       ▼         │
    review ──────┘ ─────── Sonnet 5.5: is every claim backed by a passage?
       │ pass              not covered, or failed twice → handoff
       ▼
    reply → END

    handoff ────────────── interrupt "Open a ticket?" → yes → create_ticket → END

    memory:  SqliteSaver → chat.db, one thread per customer

- The graph fixes the steps; models only fill them in. Routing is plain code,
  so it is testable without a model and the cost per turn has a ceiling.
- Only plain text crosses from one role to another. No model sees another
  model's thinking or tool calls.
- Graph state holds only plain values (dicts, strings, messages), so the
  checkpoint loads without registering custom classes.

    uv run 06_support_workflow.py                         # scripted demo, starts fresh
    uv run 06_support_workflow.py --chat                  # interactive, thread "workflow"
    uv run 06_support_workflow.py --chat --preset opus    # every role on Opus 5.5
    uv run 06_support_workflow.py --reset --thread ali    # forget one customer
"""

import argparse
import operator
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any, TypedDict

from anthropic import APIError
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRefusalMiddleware,
)
from langchain_core.exceptions import ModelRefusalError, OutputParserException
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Overwrite, Send, interrupt

import tracing
from common import SERVER_FALLBACK_MODELS, Effort, UsageMeter, make_model
from retrieval import load_chunks, vector_store
from support_tools import Priority, create_ticket, search_tool, today

DB_PATH = Path(__file__).parent / "chat.db"
DEMO_THREAD = "workflow-demo"
DEFAULT_THREAD = "workflow"
THREAD_PREFIX = "06:"  # keeps 06 threads apart from 04/05 threads in the same chat.db

# Cost ceiling per turn: at most 3 subagents, 4 model steps each, 1 rewrite.
# A refused Haiku step retries once on Sonnet inside the same step.
MAX_QUESTIONS = 3
RESEARCH_CALL_LIMIT = 4
MAX_REVISIONS = 1

# Roles see the last 12 conversation messages, starting on a customer message.
HISTORY_MESSAGES = 12

OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"

SEARCH_TOOL = "search_policies"

# Langfuse: one trace per turn, grouped into a session per thread.
TRACE_NAME = "support-turn"
WORKFLOW_TAG = "workflow"
CITATION = re.compile(r"\[([\w-]+\.md)\]")
YES_WORDS = frozenset({"y", "yes", "yeah", "yep", "sure", "ok", "okay", "please", "go"})
NO_WORDS = frozenset({"n", "no", "not", "don't", "dont", "nope", "nah", "never", "later", "cancel"})

DEMO_TURNS = [
    (
        "Hi, I'm Sara. I bought a laptop 20 days ago. Can I get my money back, "
        "and how long until it reaches my card?"
    ),
    "Can I pick the order up from your Abu Dhabi store instead?",
    "Thanks. What was my name again?",
]

TRIAGE_PROMPT = """You sort messages for the support team of Acme, a sample online store.

Classify the customer's latest message:
- answer: it needs facts from Acme's policies (refunds, returns, shipping,
  warranty, payments, pricing, subscriptions, accounts, privacy, support hours).
  Split it into 1-3 short, self-contained questions. Replace pronouns with
  what they refer to in the conversation.
- chat: greetings, thanks, or questions about this conversation itself.
- handoff: the customer asks for a human or a ticket.

Leave questions empty unless the route is answer."""

RESEARCH_PROMPT = """You research one question for Acme's support team.

Search Acme's policies with search_policies. If the results don't answer the
question, search again with other words. Use today for date maths.

Reply with short notes: the facts that answer the question, each followed by
its file, like [refund-policy.md]. If the policies don't answer it, reply
"Not covered." You write for a colleague, not the customer."""

WRITER_PROMPT = """You are the support assistant for Acme, a sample online store.
Today is {today}.

Reply to the customer's latest message. For anything about Acme's policies,
use only the research notes below and cite the file after each fact, like
[refund-policy.md]. Never invent policy, and add no general knowledge (for
example about banks or card issuers). If the notes don't say something,
leave it out. Answer every question in the notes. With no research notes, reply briefly from the conversation
alone.

Keep it under 80 words.

Research notes:
{notes}
{feedback}"""

REVIEW_PROMPT = """You check a support reply before the customer sees it.

Compare the draft with the passages:
- pass: the draft answers every listed question, and every policy fact in it
  is backed by a passage and cites that passage's file.
- revise: the passages answer the question, but the draft has a fixable
  problem (an unbacked fact, a missing or wrong citation, a missed part).
- not_covered: the passages don't answer the customer's question.

Only policy facts need a passage. Greetings, the company name Acme, offers to
help, and facts the customer gave (like a purchase date) do not.

List each problem in one line. Leave problems empty on pass."""

# Chat turns run no research. Without this, the writer read "no notes" as
# "nothing was checked" and retracted its own earlier, reviewed answers.
CHAT_NOTES = (
    "(none needed for this message. Earlier replies in this conversation were "
    "already checked against the policies; don't question or repeat them.)"
)

HANDOFF_QUESTION = "Shall I open a ticket so a human can help?"
NOT_COVERED_QUESTION = "Our policies don't cover that. " + HANDOFF_QUESTION
UNCONFIRMED_QUESTION = "I couldn't confirm a full answer from our policies. " + HANDOFF_QUESTION
TICKET_TEXT = "I've passed this to a human. {ticket}"
NO_TICKET_TEXT = "Okay, no ticket. Is there anything else I can help with?"
LIMIT_NOTES = "Research stopped at its call limit before writing notes. Raw passages:\n\n"
RETRY_INPUT = "(retry of the previous turn)"

# Errors the chat loop survives; the thread stays at its last checkpoint.
CHAT_ERRORS = (APIError, ModelRefusalError, OutputParserException)


# ---- names -------------------------------------------------------------------


class Role(StrEnum):
    """A job in the workflow. Each role can use its own model."""

    TRIAGE = "triage"
    RESEARCH = "research"
    WRITER = "writer"
    REVIEWER = "reviewer"


class Preset(StrEnum):
    """A named set of role-to-model choices."""

    MIXED = "mixed"
    OPUS = "opus"
    HAIKU = "haiku"


class Route(StrEnum):
    """What triage decided to do with the message."""

    ANSWER = "answer"
    CHAT = "chat"
    HANDOFF = "handoff"


class Verdict(StrEnum):
    """The reviewer's decision on a draft."""

    PASS = "pass"
    REVISE = "revise"
    NOT_COVERED = "not_covered"


class Node(StrEnum):
    """Graph node names."""

    TRIAGE = "triage"
    RESEARCH = "research"
    WRITE = "write"
    REVIEW = "review"
    REPLY = "reply"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class Seat:
    """The model and effort one role uses."""

    model: str
    effort: Effort


PRESETS: dict[Preset, dict[Role, Seat]] = {
    Preset.MIXED: {
        Role.TRIAGE: Seat(HAIKU, Effort.LOW),
        Role.RESEARCH: Seat(HAIKU, Effort.MEDIUM),  # low effort skips searches
        Role.WRITER: Seat(OPUS, Effort.LOW),
        Role.REVIEWER: Seat(SONNET, Effort.MEDIUM),
    },
    Preset.OPUS: {role: Seat(OPUS, Effort.LOW) for role in Role},
    Preset.HAIKU: {role: Seat(HAIKU, Effort.MEDIUM) for role in Role},
}


# ---- state -------------------------------------------------------------------


class Triage(TypedDict):
    """How to handle the customer's latest message."""

    route: Annotated[Route, ..., "answer, chat or handoff"]
    questions: Annotated[list[str], ..., "For answer: 1-3 policy questions. Else empty."]
    priority: Annotated[Priority, ..., "How urgent this is for the customer"]
    reason: Annotated[str, ..., "One short sentence"]


class Review(TypedDict):
    """The reviewer's decision and what to fix."""

    verdict: Annotated[Verdict, ..., "pass, revise or not_covered"]
    problems: Annotated[list[str], ..., "One line per problem. Empty on pass."]


class Finding(TypedDict):
    """What one research subagent found."""

    question: str
    notes: str
    passages: str


class ResearchTask(TypedDict):
    """The input one Send gives one research subagent."""

    question: str


class TurnState(TypedDict):
    """Graph state. messages persist per thread; the rest is reset each turn."""

    messages: Annotated[list[AnyMessage], add_messages]
    triage: Triage | None
    findings: Annotated[list[Finding], operator.add]  # parallel subagents append here
    draft: str
    review: Review | None
    revisions: int


@dataclass(frozen=True)
class Team:
    """The four role runnables. Tests swap in fakes."""

    triage: Runnable  # messages -> Triage
    researcher: Runnable  # {"messages": [...]} -> {"messages": [...]}
    writer: Runnable  # messages -> AIMessage
    reviewer: Runnable  # messages -> Review


# ---- team --------------------------------------------------------------------


def build_team(preset: Preset, store: InMemoryVectorStore) -> Team:
    """Create the real models for each role in the preset."""
    seats = PRESETS[preset]
    writer = seats[Role.WRITER]
    return Team(
        triage=_structured(seats[Role.TRIAGE], Triage),
        researcher=_researcher(seats[Role.RESEARCH], store),
        writer=make_model(writer.model, writer.effort),
        reviewer=_structured(seats[Role.REVIEWER], Review),
    )


def _structured(seat: Seat, schema: type) -> Runnable:
    """A model that returns `schema` as a dict, using Claude's structured output."""
    model = make_model(seat.model, seat.effort).with_structured_output(schema, method="json_schema")
    if seat.model in SERVER_FALLBACK_MODELS:
        return model

    # No server-side fallback (Haiku 5.5): retry a refusal on Sonnet ourselves.
    backup = make_model(SONNET, seat.effort).with_structured_output(schema, method="json_schema")
    return model.with_fallbacks([backup], exceptions_to_handle=(ModelRefusalError,))


def _researcher(seat: Seat, store: InMemoryVectorStore) -> Runnable:
    """A subagent with the search tool. It starts with an empty conversation."""
    middleware: list = [
        ModelCallLimitMiddleware(run_limit=RESEARCH_CALL_LIMIT, exit_behavior="end")
    ]

    # No server-side fallback: turn a refusal into an error, then retry on Sonnet.
    if seat.model not in SERVER_FALLBACK_MODELS:
        middleware += [
            _RefusalFallback(make_model(SONNET, seat.effort)),
            ModelRefusalMiddleware(),
        ]

    return create_agent(
        model=make_model(seat.model, seat.effort),
        tools=[search_tool(store), today],
        system_prompt=RESEARCH_PROMPT,
        middleware=middleware,
        checkpointer=False,  # its tool calls and thinking stay out of chat.db
        name="researcher",
    )


class _RefusalFallback(AgentMiddleware):
    """Retry a refused model call once on a backup model.

    LangChain's ModelFallbackMiddleware retries on any error, so a Haiku rate
    limit would become a Sonnet call at 20x the price. Only refusals retry here.
    """

    def __init__(self, backup: Any) -> None:
        super().__init__()
        self._backup = backup

    def wrap_model_call(self, request: Any, handler: Callable) -> Any:
        try:
            return handler(request)
        except ModelRefusalError:
            return handler(request.override(model=self._backup))


# ---- nodes -------------------------------------------------------------------


def _triage(state: TurnState, *, team: Team) -> dict:
    """Pick the route, and clear the previous turn's working fields."""
    triage = team.triage.invoke([SystemMessage(TRIAGE_PROMPT), *_recent(state["messages"])])
    return {
        "triage": triage,
        "findings": Overwrite([]),  # bypass the append reducer
        "draft": "",
        "review": None,
        "revisions": 0,
    }


def _research(task: ResearchTask, *, team: Team) -> dict:
    """Run one subagent on one question. Keep its notes and the raw passages."""
    result = team.researcher.invoke({"messages": [HumanMessage(task["question"])]})
    messages = result["messages"]
    passages = "\n\n".join(
        m.text for m in messages if isinstance(m, ToolMessage) and m.name == SEARCH_TOOL
    )

    # Hit the call limit before writing notes: give the writer the raw passages.
    final = messages[-1]
    notes = final.text if _is_model_reply(final) else LIMIT_NOTES + passages
    return {"findings": [Finding(question=task["question"], notes=notes, passages=passages)]}


def _is_model_reply(message: BaseMessage) -> bool:
    """A finished model reply. The call-limit stop message has no response metadata."""
    return (
        isinstance(message, AIMessage)
        and not message.tool_calls
        and bool(message.response_metadata)
    )


def _write(state: TurnState, *, team: Team) -> dict:
    """Draft the reply from the findings, plus reviewer feedback on a rewrite."""
    system = WRITER_PROMPT.format(
        today=date.today().isoformat(),
        notes=_format_findings(state["findings"]),
        feedback=_feedback(state),
    )
    reply = team.writer.invoke([SystemMessage(system), *_recent(state["messages"])])
    return {"draft": reply.text}


def _review(state: TurnState, *, team: Team) -> dict:
    """Check the draft against the passages the subagents found."""
    passages = "\n\n".join(f["passages"] for f in state["findings"] if f["passages"])
    questions = "\n".join(f"- {f['question']}" for f in state["findings"])
    prompt = (
        f"Customer message:\n{_latest_question(state['messages'])}\n\n"
        f"Questions the reply must answer:\n{questions}\n\n"
        f"Passages:\n{passages or '(none found)'}\n\n"
        f"Draft reply:\n{state['draft']}"
    )
    review = team.reviewer.invoke([SystemMessage(REVIEW_PROMPT), HumanMessage(prompt)])
    revisions = state["revisions"] + int(review["verdict"] == Verdict.REVISE)
    return {"review": review, "revisions": revisions}


def _reply(state: TurnState) -> dict:
    """Send the approved draft, as plain text, into the conversation."""
    return {"messages": [AIMessage(state["draft"])]}


def _handoff(state: TurnState) -> dict:
    """Ask the customer before opening a ticket. The graph pauses here."""
    answer = interrupt(_consent_question(state))
    if not _is_yes(answer):
        return {"messages": [AIMessage(NO_TICKET_TEXT)]}

    ticket = create_ticket.invoke(
        {"summary": _latest_question(state["messages"]), "priority": state["triage"]["priority"]}
    )
    return {"messages": [AIMessage(TICKET_TEXT.format(ticket=ticket))]}


# ---- routers -----------------------------------------------------------------


def _dispatch(state: TurnState) -> str | list[Send]:
    """After triage: one research subagent per question, or straight on."""
    triage = state["triage"]
    if triage["route"] == Route.HANDOFF:
        return Node.HANDOFF

    if triage["route"] == Route.CHAT:
        return Node.WRITE

    questions = [q.strip() for q in triage["questions"] if q.strip()]
    questions = questions[:MAX_QUESTIONS] or [_latest_question(state["messages"])]
    return [Send(Node.RESEARCH.value, ResearchTask(question=q)) for q in questions]


def _after_write(state: TurnState) -> str:
    """Chat replies skip review: they state no policy."""
    if state["triage"]["route"] == Route.CHAT:
        return Node.REPLY

    return Node.REVIEW


def _after_review(state: TurnState) -> str:
    """Pass → reply. Not covered, or still failing after a rewrite → handoff."""
    verdict = state["review"]["verdict"]
    if verdict == Verdict.PASS:
        return Node.REPLY

    if verdict == Verdict.REVISE and state["revisions"] <= MAX_REVISIONS:
        return Node.WRITE

    return Node.HANDOFF


# ---- graph -------------------------------------------------------------------


def build_graph(team: Team, checkpointer: BaseCheckpointSaver):
    """Wire the nodes as in the diagram at the top of this file."""
    builder = StateGraph(TurnState)
    builder.add_node(Node.TRIAGE, partial(_triage, team=team))
    builder.add_node(Node.RESEARCH, partial(_research, team=team))  # one run per Send
    builder.add_node(Node.WRITE, partial(_write, team=team))
    builder.add_node(Node.REVIEW, partial(_review, team=team))
    builder.add_node(Node.REPLY, _reply)
    builder.add_node(Node.HANDOFF, _handoff)

    builder.add_edge(START, Node.TRIAGE)
    builder.add_conditional_edges(Node.TRIAGE, _dispatch, [Node.RESEARCH, Node.WRITE, Node.HANDOFF])
    builder.add_edge(Node.RESEARCH, Node.WRITE)  # waits for every subagent
    builder.add_conditional_edges(Node.WRITE, _after_write, [Node.REVIEW, Node.REPLY])
    builder.add_conditional_edges(
        Node.REVIEW, _after_review, [Node.WRITE, Node.REPLY, Node.HANDOFF]
    )
    builder.add_edge(Node.REPLY, END)
    builder.add_edge(Node.HANDOFF, END)
    return builder.compile(checkpointer=checkpointer)


# ---- helpers -----------------------------------------------------------------


def _recent(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The last few conversation messages, starting on a customer message."""
    return trim_messages(
        messages,
        strategy="last",
        token_counter=len,
        max_tokens=HISTORY_MESSAGES,
        start_on="human",
    )


def _latest_question(messages: list[BaseMessage]) -> str:
    """The customer's latest message."""
    return next(m.text for m in reversed(messages) if isinstance(m, HumanMessage))


def _format_findings(findings: list[Finding]) -> str:
    """Research notes for the writer, one block per question."""
    if not findings:
        return CHAT_NOTES

    return "\n\n".join(f"Q: {f['question']}\n{f['notes']}" for f in findings)


def _feedback(state: TurnState) -> str:
    """Reviewer problems and the rejected draft, only on a rewrite."""
    review = state["review"]
    if not review or review["verdict"] != Verdict.REVISE:
        return ""

    problems = "\n".join(f"- {p}" for p in review["problems"])
    return f"\nA reviewer rejected your last draft. Fix every problem:\n{problems}\n\nLast draft:\n{state['draft']}"


def _consent_question(state: TurnState) -> str:
    """Tell the customer why we stopped: they asked, a gap in the docs, or a failed review."""
    if state["triage"]["route"] == Route.HANDOFF:
        return HANDOFF_QUESTION

    if state["review"]["verdict"] == Verdict.NOT_COVERED:
        return NOT_COVERED_QUESTION

    return UNCONFIRMED_QUESTION


def _is_yes(answer: Any) -> bool:
    """True only for a clear yes: a yes word and no word of refusal ("ok, no thanks")."""
    words = {w.strip(",.!?") for w in str(answer).lower().split()}
    return bool(words & YES_WORDS) and not words & NO_WORDS


def _cited_docs(text: str) -> list[str]:
    """The [file.md] citations in a reply, each once, in order."""
    return list(dict.fromkeys(CITATION.findall(text)))


# ---- CLI ---------------------------------------------------------------------

Ask = Callable[[str], str]


def _parse_args() -> argparse.Namespace:
    """Read the command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat", action="store_true", help="type your own messages")
    parser.add_argument("--thread", help="customer conversation name")
    parser.add_argument("--reset", action="store_true", help="delete the thread")
    parser.add_argument(
        "--preset", type=Preset, choices=list(Preset), default=Preset.MIXED, help="role models"
    )
    return parser.parse_args()


def _show(node: str, update: dict | None) -> None:
    """Print one finished step."""
    if not update:
        return

    if node == Node.TRIAGE:
        t = update["triage"]
        print(f"     triage   {t['route']} · {t['priority']} · {t['questions']}")
        return

    if node == Node.RESEARCH:
        for f in update["findings"]:
            print(f"     research {f['question']!r} → {', '.join(_cited_docs(f['notes'])) or '-'}")
        return

    if node == Node.WRITE:
        print(f"     write    draft, {len(update['draft'].split())} words")
        return

    if node == Node.REVIEW:
        r = update["review"]
        print(f"     review   {r['verdict']} {'; '.join(r['problems'])}".rstrip())


def _stream(graph, payload: Any, config: dict) -> None:
    """Run the graph until it ends or pauses, printing each step."""
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            _show(node, update)


def _resume(graph, config: dict, ask: Ask) -> None:
    """Answer any pending consent question and continue the graph."""
    while pending := graph.get_state(config).interrupts:
        answer = ask(pending[0].value)
        _stream(graph, Command(resume=answer), config)


def _run_turn(graph, config: dict, text: str | None, ask: Ask, tags: list[str]) -> str:
    """One customer message in, the bot's reply out, with a cost line and trace link.

    text=None continues a turn that failed, from its last checkpoint: finished
    steps (and their paid calls) are not run again, and no message is added.
    """
    meter = UsageMeter()
    started = time.monotonic()
    session = config["configurable"]["thread_id"]

    # Everything inside this block, consent pause included, is one Langfuse trace.
    payload = None if text is None else {"messages": [HumanMessage(text)]}
    with tracing.turn_trace(TRACE_NAME, session, tags, text or RETRY_INPUT) as trace:
        run_config = {**config, "callbacks": [meter, *trace.callbacks]}
        _stream(graph, payload, run_config)
        _resume(graph, run_config, ask)

        values = graph.get_state(config).values
        reply = values["messages"][-1].text
        trace.finish(reply)
        _score_turn(trace, values, reply)

    print(f"     cost     {meter.summary()} · {time.monotonic() - started:.1f}s")
    if url := trace.url():
        print(f"     trace    {url}")

    return reply


def _score_turn(trace: tracing.Trace, values: dict, reply: str) -> None:
    """Scores to filter and chart traces by in Langfuse."""
    trace.score("route", values["triage"]["route"])
    if values["review"]:
        trace.score("verdict", values["review"]["verdict"])

    trace.score("revisions", values["revisions"])
    trace.score("sources_cited", len(_cited_docs(reply)))


def _ask_customer(question: str) -> str:
    return input(f"bot> {question}\nyou> ")


def _auto_yes(question: str) -> str:
    print(f"bot> {question}\nyou> yes")
    return "yes"


def _chat(graph, config: dict, tags: list[str]) -> None:
    """Interactive loop. A failed turn can be continued from its checkpoint."""
    print(f"Thread '{config['configurable']['thread_id']}'. Empty line quits.\n")

    # A consent question left from last time: ask it, then show the outcome.
    if graph.get_state(config).interrupts:
        _resume(graph, config, _ask_customer)
        print(f"bot> {graph.get_state(config).values['messages'][-1].text}\n")

    while line := input("you> ").strip():
        _chat_turn(graph, config, line, tags)


def _chat_turn(graph, config: dict, text: str, tags: list[str]) -> None:
    """Run one turn. After an error, offer to continue it instead of resending it."""
    pending: str | None = text
    while True:
        try:
            print(f"bot> {_run_turn(graph, config, pending, _ask_customer, tags)}\n")
            return
        except CHAT_ERRORS as error:
            print(f"error> {error}")

        if not _is_yes(input("retry> Continue this turn where it stopped? ")):
            return

        pending = None


def _thread_id(name: str) -> str:
    """The checkpoint thread for a customer name, kept apart from 04 and 05."""
    return f"{THREAD_PREFIX}{name}"


def main() -> None:
    """Run the scripted demo, the interactive chat, or a reset."""
    args = _parse_args()
    thread_id = _thread_id(args.thread or (DEFAULT_THREAD if args.chat else DEMO_THREAD))
    config = {"configurable": {"thread_id": thread_id}}

    try:
        _run(args, thread_id, config)
    finally:
        tracing.shutdown()  # flush queued traces before exit


def _run(args: argparse.Namespace, thread_id: str, config: dict) -> None:
    """Open the checkpoint store and run the chosen mode."""
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        # Reset: delete the thread's history and stop, before loading any model.
        if args.reset:
            checkpointer.delete_thread(thread_id)
            print(f"Deleted thread '{thread_id}'.")
            return

        graph = build_graph(build_team(args.preset, vector_store(load_chunks())), checkpointer)
        seats = ", ".join(f"{r} {s.model} ({s.effort})" for r, s in PRESETS[args.preset].items())
        print(f"Preset {args.preset}: {seats}")
        print(f"Langfuse tracing: {'on' if tracing.enabled() else 'off'}\n")
        tags = [WORKFLOW_TAG, f"preset:{args.preset}"]

        if args.chat:
            _chat(graph, config, tags)
            return

        # Scripted: start fresh so every run shows the same conversation.
        checkpointer.delete_thread(thread_id)
        for turn in DEMO_TURNS:
            print(f"you> {turn}")
            print(f"bot> {_run_turn(graph, config, turn, _auto_yes, tags)}\n")


if __name__ == "__main__":
    main()
