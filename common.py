"""Shared model setup for every script.

Reads ANTHROPIC_API_KEY from the environment (ChatAnthropic does this itself).
Override the model or effort per run:

    LC_MODEL=claude-sonnet-5-5 LC_EFFORT=medium uv run 01_chain.py
"""

import os

from langchain_anthropic import ChatAnthropic

# Defaults: current Opus, low effort so the demos stay fast and cheap.
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"
MAX_TOKENS = 4096

# Server-side fallback: if a safety classifier refuses, the API retries on a
# suitable model instead of returning stop_reason "refusal".
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_model() -> ChatAnthropic:
    """Return the chat model all scripts share."""
    return ChatAnthropic(
        model=os.environ.get("LC_MODEL", DEFAULT_MODEL),
        max_tokens=MAX_TOKENS,
        reasoning_effort=os.environ.get("LC_EFFORT", DEFAULT_EFFORT),
        betas=[FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},
    )
