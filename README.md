# langchain-claude

Small, runnable LangChain 1.x scripts that use Claude through `langchain-anthropic`.

| File | Shows |
| --- | --- |
| `01_chain.py` | Prompt template piped into the model; structured output into a Pydantic model |
| `02_agent.py` | `create_agent` with three local `@tool` functions |
| `03_rag.py` | RAG over `docs/*.md`: load, split, BM25 retrieval, answer with sources |
| `03b_rag_vector.py` | Same RAG with vector search: local fastembed embeddings, `InMemoryVectorStore`, similarity scores printed |
| `04_memory_chat.py` | Multi-turn chat whose memory is saved to `chat.db` (SQLite) per `--thread`, with old turns summarized to cap its size. Tools: `today`, `web_search` (DuckDuckGo), `get_weather` (Open-Meteo) |

`common.py` builds the shared model.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and `ANTHROPIC_API_KEY` in the environment.

```sh
uv sync
```

## Run

```sh
uv run 01_chain.py
uv run 02_agent.py
uv run 03_rag.py "Who approves refunds over 5000 AED?"
uv run 03b_rag_vector.py "Can I get my money back after a month?"
uv run 04_memory_chat.py                       # scripted demo, starts fresh each run
uv run 04_memory_chat.py --chat                # interactive, resumes thread "main"
uv run 04_memory_chat.py --chat --thread aws   # a separate conversation
uv run 04_memory_chat.py --reset --thread aws  # forget one conversation
```

Each run makes a few API calls (cents at most).

## Settings

| Env var | Default | Meaning |
| --- | --- | --- |
| `LC_MODEL` | `claude-opus-5-5` | Any Claude model ID |
| `LC_EFFORT` | `low` | `low`, `medium`, `high`, `xhigh`, `max` |

## Notes

- Structured output uses `method="json_schema"`. The default tool-calling method forces a tool call, which Claude Opus 5.5 rejects.
- No `temperature`: current Claude models reject sampling parameters.
- `common.py` turns on server-side fallback (`fallbacks: "default"`): if a safety check refuses a request, the API retries it on another model.
- `03_rag.py` uses BM25 (keyword ranking) through `rank-bm25` directly, so no embedding model is needed. It avoids `langchain-community`, which LangChain is winding down.
- `03b_rag_vector.py` embeds locally with fastembed (`BAAI/bge-small-en-v1.5`, about 70 MB, downloaded on first run, no API key). Anthropic has no embeddings API; to use a hosted one, replace `_LocalEmbeddings` with a LangChain embeddings class such as `VoyageAIEmbeddings`. For data that must persist, replace `InMemoryVectorStore` with Chroma, pgvector or similar.
- `04_memory_chat.py` summarizes once a thread holds 6 messages and keeps the last 2 word for word. That limit is small on purpose so the demo shows it; for real use, set `SUMMARY_TRIGGER` to a token count such as `("tokens", 50_000)`. Each summary costs one extra model call. Tool calls and results count as messages, so a turn that uses a tool fills the limit faster. Delete `chat.db` to wipe every thread.
- `web_search` uses the `ddgs` package, which scrapes DuckDuckGo (no official API) and falls back to other engines when DuckDuckGo blocks it. Results vary, and snippets often lack live numbers, which is why weather has its own tool.
- `get_weather` calls Open-Meteo (free, no key): city name to coordinates, then current conditions and today's high and low.
- With only two short sample docs, both RAG versions find the right answer. Vector search pays off on larger docs, where questions use different words than the text.
