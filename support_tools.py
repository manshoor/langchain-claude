"""Tools shared by the support bot (05) and the support workflow (06)."""

from datetime import date
from enum import StrEnum

from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore

from retrieval import format_chunks

TOP_K = 3


class Priority(StrEnum):
    """Ticket urgency, from the customer's point of view."""

    LOW = "low"
    NORMAL = "normal"
    URGENT = "urgent"


_tickets: list[str] = []


@tool
def create_ticket(summary: str, priority: Priority = Priority.NORMAL) -> str:
    """Open a support ticket for a human agent. Use only after the customer agrees."""
    # A real bot would call the help-desk API here.
    _tickets.append(summary)
    return f"Ticket TCK-{len(_tickets):04d} opened ({priority}). A human replies within one business day."


@tool
def today() -> str:
    """Return today's date, for questions about deadlines like '30 days from purchase'."""
    return date.today().isoformat()


def search_tool(store: InMemoryVectorStore) -> BaseTool:
    """Wrap the vector store as a tool the agent can call."""

    @tool
    def search_policies(query: str) -> str:
        """Search Acme's policy documents. Returns matching passages tagged with their file."""
        return format_chunks(store.similarity_search(query, k=TOP_K))

    return search_policies
