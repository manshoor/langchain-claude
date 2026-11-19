"""Customer support bot: the earlier scripts combined into one agent.

    you ──► agent (02) ──► SummarizationMiddleware (04) ──► Claude
              │  tools
              ├── search_policies ──► vector search over docs/*.md (03b, retrieval.py)
              ├── create_ticket   ──► fake ticket desk (hand-off to a human)
              └── today
              │
              ▼
         SqliteSaver ──► chat.db   (memory per --thread, as in 04)

- RAG as a tool: the agent decides when to search, can search twice with
  different words, and must cite the file each fact came from.
- Vector search, not BM25 or hybrid: eval_retrieval.py measured it best on
  this corpus, mainly because customers paraphrase.
- When the docs have no answer, the bot says so and offers a ticket instead
  of guessing.

    uv run 05_support_bot.py                       # scripted demo, starts fresh
    uv run 05_support_bot.py --chat                # interactive, resumes thread "support"
    uv run 05_support_bot.py --chat --thread ali   # a separate customer
    uv run 05_support_bot.py --reset --thread ali  # forget one customer
"""

import argparse
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from common import make_model
from retrieval import load_chunks, vector_store
from support_tools import create_ticket, search_tool, today

DB_PATH = Path(__file__).parent / "chat.db"
DEMO_THREAD = "support-demo"
DEFAULT_THREAD = "support"

# Summarize once a thread passes 20 messages, keeping the last 6 word for word.
# Tool calls and results count as messages, so a support turn adds several.
SUMMARY_TRIGGER = ("messages", 20)
SUMMARY_KEEP = ("messages", 6)

SYSTEM_PROMPT = """You are the support assistant for Acme, a sample online store.

Answer policy questions only from search_policies results. Cite the file after
each fact, like [refund-policy.md]. Search again with other words if the first
results don't answer the question.

If the policies don't cover it, say so plainly, then offer to open a ticket.
Open a ticket only after the customer agrees.

Keep answers under 80 words."""

# One customer message each. Together they show retrieval on a
# paraphrase, a follow-up that needs memory, a gap in the docs, and a hand-off.
DEMO_TURNS = [
    "Hi, I'm Sara. I bought a laptop 20 days ago and want my money back. Possible?",
    "Ok. And how long until the money reaches my card?",
    "Can I pick the order up from your Abu Dhabi store instead?",
    "Yes please, open a ticket for that.",
    "Thanks. What was my name again?",
]


def build_bot(store: InMemoryVectorStore, checkpointer: BaseCheckpointSaver):
    """Build the single-agent support bot. The workflow eval reuses this."""
    return create_agent(
        model=make_model(),
        tools=[search_tool(store), create_ticket, today],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(model=make_model(), trigger=SUMMARY_TRIGGER, keep=SUMMARY_KEEP)
        ],
        checkpointer=checkpointer,
    )


def _parse_args() -> argparse.Namespace:
    """Read the command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat", action="store_true", help="type your own messages")
    parser.add_argument("--thread", help="customer conversation name")
    parser.add_argument("--reset", action="store_true", help="delete the thread")
    return parser.parse_args()


def _ask(bot, config: dict, text: str) -> str:
    """Send one customer turn, print the tool calls it caused, return the reply."""
    result = bot.invoke({"messages": [{"role": "user", "content": text}]}, config)
    messages = result["messages"]

    # Only show this turn's steps: everything after the last customer message.
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    for message in messages[last_human + 1 :]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"     call   {call['name']}({call['args']})")
            continue

        if isinstance(message, ToolMessage):
            sources = sorted(set(_cited_files(message.content)))
            print(f"     result {message.name}: {', '.join(sources) or message.content[:70]}")

    return messages[-1].text


def _cited_files(text: str) -> list[str]:
    """Pull the [file.md] tags out of search results."""
    return [
        line[1:-1] for line in text.splitlines() if line.startswith("[") and line.endswith(".md]")
    ]


def main() -> None:
    """Run the scripted demo, the interactive chat, or a reset."""
    args = _parse_args()
    thread_id = args.thread or (DEFAULT_THREAD if args.chat else DEMO_THREAD)
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        # Reset: delete the thread's history and stop, before loading any model.
        if args.reset:
            checkpointer.delete_thread(thread_id)
            print(f"Deleted thread '{thread_id}'.")
            return

        bot = build_bot(vector_store(load_chunks()), checkpointer)

        # Interactive: resume the thread and read lines until an empty one.
        if args.chat:
            print(f"Thread '{thread_id}'. Empty line quits.\n")
            while line := input("you> ").strip():
                print(f"bot> {_ask(bot, config, line)}\n")
            return

        # Scripted: start fresh so every run shows the same conversation.
        checkpointer.delete_thread(thread_id)
        for turn in DEMO_TURNS:
            print(f"you> {turn}")
            reply = _ask(bot, config, turn)
            print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
