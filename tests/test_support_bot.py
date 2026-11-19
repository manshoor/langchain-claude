"""Offline tests for the support bot's local helpers."""

import importlib

import support_tools

bot = importlib.import_module("05_support_bot")


def test_ticket_numbers(monkeypatch) -> None:
    monkeypatch.setattr(support_tools, "_tickets", [])

    first = support_tools.create_ticket.invoke({"summary": "Store pickup in Abu Dhabi"})
    second = support_tools.create_ticket.invoke({"summary": "Late parcel", "priority": "urgent"})

    assert first.startswith("Ticket TCK-0001 opened (normal)")
    assert second.startswith("Ticket TCK-0002 opened (urgent)")


def test_cited_files() -> None:
    text = "[refund-policy.md]\nCustomers can...\n\n[shipping.md]\nOrders inside..."

    assert bot._cited_files(text) == ["refund-policy.md", "shipping.md"]
