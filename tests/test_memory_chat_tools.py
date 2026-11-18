"""Offline tests for the tools in 04_memory_chat.py. HTTP calls are faked."""

import importlib
from urllib.error import URLError

import pytest

# The file name starts with a digit, so a plain import statement can't load it.
chat = importlib.import_module("04_memory_chat")

GEOCODE_REPLY = {
    "results": [{"name": "Dubai", "country": "UAE", "latitude": 25.2, "longitude": 55.3}]
}
FORECAST_REPLY = {
    "current": {
        "time": "2026-11-18T10:00",
        "temperature_2m": 29.5,
        "relative_humidity_2m": 55,
        "wind_speed_10m": 12.0,
        "weather_code": 1,
    },
    "daily": {"temperature_2m_max": [31.0], "temperature_2m_min": [22.0]},
}


def _fake_get_json(url: str, params: dict) -> dict:
    """Answer the geocode call, then the forecast call, without the network."""
    if url == chat.GEOCODE_URL:
        return GEOCODE_REPLY

    return FORECAST_REPLY


def test_weather_formats_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "_get_json", _fake_get_json)

    reply = chat.get_weather.invoke({"city": "Dubai"})

    assert reply.startswith("Dubai, UAE")
    assert "mainly clear" in reply
    assert "29.5°C" in reply
    assert "high 31.0°C, low 22.0°C" in reply


def test_weather_unknown_city(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "_get_json", lambda url, params: {})

    assert chat.get_weather.invoke({"city": "Atlantis"}) == "No place found named Atlantis."


def test_weather_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed HTTP call must come back as text the model can read, not crash the agent."""

    def fail(url: str, params: dict) -> dict:
        raise URLError("connection refused")

    monkeypatch.setattr(chat, "_get_json", fail)

    reply = chat.get_weather.invoke({"city": "Dubai"})

    assert reply.startswith("Weather lookup failed")
