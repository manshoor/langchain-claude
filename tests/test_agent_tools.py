"""Offline tests for the local tools in 02_agent.py."""

import importlib

agent = importlib.import_module("02_agent")


def test_convert_known_currency() -> None:
    assert (
        agent.convert_to_aed.invoke({"amount": 100, "currency": "usd"}) == "100.0 USD = 367.25 AED"
    )


def test_convert_unknown() -> None:
    reply = agent.convert_to_aed.invoke({"amount": 1, "currency": "JPY"})

    assert reply == "Unknown currency: JPY. Known: USD, EUR, GBP"


def test_check_server() -> None:
    assert agent.check_server.invoke({"name": "web-2"}) == "high CPU (92%)"
    assert agent.check_server.invoke({"name": "web-9"}) == "No server named web-9"
