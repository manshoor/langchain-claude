"""Shared model setup for every script.

Reads ANTHROPIC_API_KEY from the environment (ChatAnthropic does this itself).
Override the model or effort per run:

    LC_MODEL=claude-sonnet-5-5 LC_EFFORT=medium uv run 01_chain.py

Code that needs a specific model per role passes it in:

    make_model("claude-haiku-5-5", Effort.MEDIUM)
"""

import os
import threading
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

# Defaults: current Opus, low effort so the demos stay fast and cheap.
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"
MAX_TOKENS = 4096
TOKENS_PER_MTOK = 1_000_000

# Server-side fallback: if a safety classifier refuses, the API retries on a
# suitable model instead of returning stop_reason "refusal".
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Models that accept fallbacks="default". Haiku 5.5 has no server-side
# fallback, so callers handle its refusals themselves.
SERVER_FALLBACK_MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-fable-5-1",
        "claude-fable-5-2",
    }
)


class Effort(StrEnum):
    """How much the model thinks before answering."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float


# Used to print a cost per turn. Haiku's price is for prompts up to 100K tokens.
PRICES = {
    "claude-opus-5-5": Price(input=4.00, output=20.00),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00),
    "claude-haiku-5-5": Price(input=0.10, output=0.50),
}


def make_model(model: str | None = None, effort: Effort | None = None) -> ChatAnthropic:
    """Return a chat model. With no arguments, LC_MODEL / LC_EFFORT or the defaults apply."""
    name = model or os.environ.get("LC_MODEL", DEFAULT_MODEL)
    level = effort or os.environ.get("LC_EFFORT", DEFAULT_EFFORT)

    # Plain model for anything without a server-side fallback.
    if name not in SERVER_FALLBACK_MODELS:
        return ChatAnthropic(model=name, max_tokens=MAX_TOKENS, reasoning_effort=str(level))

    return ChatAnthropic(
        model=name,
        max_tokens=MAX_TOKENS,
        reasoning_effort=str(level),
        betas=[FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},
    )


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """US dollars for one model's tokens. Unknown models count as 0."""
    price = PRICES.get(model)
    if price is None:
        return 0.0

    return (input_tokens * price.input + output_tokens * price.output) / TOKENS_PER_MTOK


class UsageMeter(BaseCallbackHandler):
    """Counts model calls, tokens and cost per model. Pass it in a run's callbacks.

    Thread-safe: the workflow's research subagents run in parallel.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._calls: Counter[str] = Counter()
        self._input: Counter[str] = Counter()
        self._output: Counter[str] = Counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record one finished model call."""
        generation = response.generations[0][0] if response.generations else None
        if not isinstance(generation, ChatGeneration):
            return

        message = generation.message
        usage = getattr(message, "usage_metadata", None) or {}
        model = message.response_metadata.get("model_name", "unknown")
        with self._lock:
            self._calls[model] += 1
            self._input[model] += usage.get("input_tokens", 0)
            self._output[model] += usage.get("output_tokens", 0)

    def calls(self) -> int:
        """Model calls so far."""
        return self._calls.total()

    def cost(self) -> float:
        """US dollars so far, across all models."""
        return sum(cost(m, self._input[m], self._output[m]) for m in self._calls)

    def summary(self) -> str:
        """One line: cost, calls per model."""
        calls = ", ".join(f"{m} ×{n}" for m, n in sorted(self._calls.items()))
        line = f"${self.cost():.4f} · {self.calls()} model calls ({calls or 'none'})"

        # A model missing from PRICES counts as $0; say so instead of hiding it.
        unpriced = sorted(m for m in self._calls if m not in PRICES)
        if unpriced:
            line += f" · unpriced: {', '.join(unpriced)}"

        return line
