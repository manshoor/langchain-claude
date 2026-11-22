"""An agent that decides which local tools to call.

    uv run scripts/02_agent.py
"""

from datetime import date

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from acme_support.common import make_model

# Fake data so the script runs offline (apart from the model call).
EXCHANGE_RATES = {"USD": 3.6725, "EUR": 4.25, "GBP": 4.95}  # AED per unit
SERVER_STATUS = {"web-1": "healthy", "web-2": "high CPU (92%)", "db-1": "healthy"}


# The docstring and type hints become the tool's description and schema.
@tool
def convert_to_aed(amount: float, currency: str) -> str:
    """Convert an amount in USD, EUR or GBP to UAE dirhams (AED)."""
    rate = EXCHANGE_RATES.get(currency.upper())
    if rate is None:
        return f"Unknown currency: {currency}. Known: {', '.join(EXCHANGE_RATES)}"

    return f"{amount} {currency.upper()} = {amount * rate:.2f} AED"


@tool
def check_server(name: str) -> str:
    """Return the current status of a server by name, e.g. 'web-1'."""
    return SERVER_STATUS.get(name, f"No server named {name}")


@tool
def today() -> str:
    """Return today's date in ISO format."""
    return date.today().isoformat()


QUESTION = "What's today's date, is web-2 okay, and what is 1200 EUR in AED?"


def main() -> None:
    """Ask one question and print each tool call the agent makes."""
    # create_agent runs the loop: model -> tool calls -> results -> model ... -> answer.
    agent = create_agent(
        model=make_model(),
        tools=[convert_to_aed, check_server, today],
        system_prompt="You are a concise ops assistant. Use tools for facts.",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})

    # Print each step in one line so you can see which tools the model chose.
    print(f"You: {QUESTION}\n")
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"  call   {call['name']}({call['args']})")
            continue

        if isinstance(message, ToolMessage):
            print(f"  result {message.name}: {message.content}")

    # The last message is the model's final answer.
    print(f"\nAgent:\n{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
