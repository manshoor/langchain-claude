"""Multi-turn chat with memory that survives restarts and stays small.

Tools: today (local clock), web_search (DuckDuckGo), get_weather (Open-Meteo).

    you ──► agent ──► SummarizationMiddleware ──► Claude
              │         (too many messages? replace old ones with a summary)
              ▼
         SqliteSaver ──► chat.db   (one history per --thread name)

- Persistence: SqliteSaver stores each thread's messages in chat.db, so a
  later run with the same --thread continues the conversation.
- Size cap: once a thread reaches SUMMARY_TRIGGER messages, the older ones
  are replaced by a summary and only the last SUMMARY_KEEP stay word for word.

    uv run 04_memory_chat.py                      # scripted demo, starts fresh
    uv run 04_memory_chat.py --chat               # resume thread "main"
    uv run 04_memory_chat.py --chat --thread aws  # a separate conversation
    uv run 04_memory_chat.py --reset --thread aws # forget one thread
"""

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from common import make_model

DB_PATH = Path(__file__).parent / "chat.db"
SEARCH_RESULTS = 5

# Open-Meteo: free weather API, no key. City name -> coordinates -> weather.
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT_S = 10

# WMO weather codes Open-Meteo returns, in plain words.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
}
DEMO_THREAD = "demo"
DEFAULT_THREAD = "main"

# Small so the demo shows a summary after a few turns. For real use, count
# tokens instead, e.g. trigger=("tokens", 50_000), keep=("messages", 20).
SUMMARY_TRIGGER = ("messages", 6)
SUMMARY_KEEP = ("messages", 2)

DEMO_TURNS = [
    "Hi, I'm learning AWS. My first goal is passing the Solutions Architect exam.",
    "Which three services should I study first?",
    "How many hours a week should I plan for?",
    "Remind me: what was my goal again?",
]


def _get_json(url: str, params: dict) -> dict:
    """GET a URL with query parameters and return the parsed JSON body."""
    with urlopen(f"{url}?{urlencode(params)}", timeout=HTTP_TIMEOUT_S) as response:
        return json.load(response)


@tool
def today() -> str:
    """Return today's date and weekday."""
    return date.today().strftime("%A %Y-%m-%d")


@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) and return titles, links and snippets.

    Use for news and facts. Use get_weather for weather instead.
    """
    # ddgs scrapes DuckDuckGo and falls back to other engines when it blocks.
    try:
        results = DDGS().text(query, max_results=SEARCH_RESULTS)
    except DDGSException as error:
        return f"Search failed: {error}"

    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


@tool
def get_weather(city: str) -> str:
    """Return current weather and today's high/low for a city.

    Pass the full city name, e.g. 'Ras Al Khaimah' rather than 'RAK'.
    """
    # Step 1: city name -> coordinates.
    places = _get_json(GEOCODE_URL, {"name": city, "count": 1}).get("results")
    if not places:
        return f"No place found named {city}."

    place = places[0]

    # Step 2: coordinates -> current conditions and today's range.
    data = _get_json(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min",
            "forecast_days": 1,
            "timezone": "auto",
        },
    )
    now, daily = data["current"], data["daily"]
    code = now["weather_code"]
    sky = WEATHER_CODES.get(code, f"weather code {code}")
    return (
        f"{place['name']}, {place.get('country', '')} at {now['time']} local: "
        f"{sky}, {now['temperature_2m']}°C, humidity {now['relative_humidity_2m']}%, "
        f"wind {now['wind_speed_10m']} km/h. Today: high "
        f"{daily['temperature_2m_max'][0]}°C, low {daily['temperature_2m_min'][0]}°C."
    )


def _parse_args() -> argparse.Namespace:
    """Read the command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chat", action="store_true", help="type your own messages")
    parser.add_argument("--thread", help="conversation name")
    parser.add_argument("--reset", action="store_true", help="delete the thread")
    return parser.parse_args()


def _ask(chat, config: dict, text: str) -> str:
    """Send one user turn on the thread and return the reply text."""
    result = chat.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return result["messages"][-1].text


def _stored_count(chat, config: dict) -> int:
    """Return how many messages the thread holds in chat.db right now."""
    return len(chat.get_state(config).values.get("messages", []))


args = _parse_args()
default_thread = DEFAULT_THREAD if args.chat else DEMO_THREAD
thread_id = args.thread or default_thread
config = {"configurable": {"thread_id": thread_id}}

# The connection stays open for the whole run and closes on exit.
with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
    chat = create_agent(
        model=make_model(),
        tools=[today, web_search, get_weather],
        system_prompt=(
            "You are a friendly tutor. Keep answers under 80 words. "
            "Use the tools for today's date, weather and current facts."
        ),
        middleware=[
            SummarizationMiddleware(
                model=make_model(), trigger=SUMMARY_TRIGGER, keep=SUMMARY_KEEP
            )
        ],
        checkpointer=checkpointer,
    )

    # Reset: delete the thread's history and stop.
    if args.reset:
        checkpointer.delete_thread(thread_id)
        print(f"Deleted thread '{thread_id}'.")
        raise SystemExit(0)

    # Interactive: resume the thread and read lines until an empty one.
    if args.chat:
        print(f"Thread '{thread_id}': {_stored_count(chat, config)} messages stored.")
        print("Empty line quits.\n")
        while line := input("you> ").strip():
            print(f"bot> {_ask(chat, config, line)}")
            print(f"     ({_stored_count(chat, config)} messages stored)\n")
        raise SystemExit(0)

    # Scripted: start the demo thread fresh so every run shows the same thing.
    # The stored count drops when the summary replaces old messages; the last
    # turn only works if the goal survived in that summary.
    checkpointer.delete_thread(thread_id)
    for turn in DEMO_TURNS:
        print(f"you> {turn}")
        print(f"bot> {_ask(chat, config, turn)}")
        print(f"     ({_stored_count(chat, config)} messages stored)\n")
