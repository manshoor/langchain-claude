# langchain-claude

[![CI](https://github.com/manshoor/langchain-claude/actions/workflows/ci.yml/badge.svg)](https://github.com/manshoor/langchain-claude/actions/workflows/ci.yml)

LangChain 1.x on Claude, built up one idea per script and finished with a customer support bot. The bot answers from policy documents, cites its sources, remembers each customer and hands off to a human when the documents have no answer. Retrieval was chosen by measurement, not by default: an offline eval compares keyword, vector and hybrid search.

| File | Shows |
| --- | --- |
| `01_chain.py` | Prompt template piped into the model; structured output into a Pydantic model |
| `02_agent.py` | `create_agent` with three local `@tool` functions |
| `03_rag.py` | RAG over `docs/*.md`: load, split, BM25 retrieval, answer with sources |
| `03b_rag_vector.py` | Same RAG with vector search: local fastembed embeddings, `InMemoryVectorStore`, similarity scores printed |
| `04_memory_chat.py` | Multi-turn chat saved to `chat.db` (SQLite) per `--thread`, old turns summarized to cap its size. Tools: `today`, `web_search` (DuckDuckGo), `get_weather` (Open-Meteo) |
| `05_support_bot.py` | **Capstone.** 02 + 03b + 04 in one agent: RAG as a tool, citations, memory, ticket hand-off |
| `eval_retrieval.py` | Offline retrieval eval: BM25 vs vector vs hybrid on 24 labeled questions |

Shared code: `common.py` builds the model, `retrieval.py` holds loading, splitting and the three search methods.

## Support bot architecture

```
customer ──► agent (create_agent) ──► SummarizationMiddleware ──► Claude
                │   tools                (long threads → summary)
                ├── search_policies ──► vector search ──► docs/*.md (11 policies)
                ├── create_ticket   ──► help desk (stub)
                └── today
                │
                ▼
           SqliteSaver ──► chat.db   (one memory per customer thread)
```

The agent decides when to search, can search again with other words, and must cite the file behind each fact. If the policies don't cover a question, it says so and offers a ticket instead of guessing.

Sample run (`uv run 05_support_bot.py`, trimmed):

```
you> Hi, I'm Sara. I bought a laptop 20 days ago and want my money back. Possible?
     call   search_policies({'query': 'refund laptop return window'})
     result search_policies: refund-policy.md, returns-exchanges.md
bot> Hi Sara, yes, you're still eligible. You can ask for a refund within 30 days
     of purchase, and you're at day 20 [refund-policy.md]. ...

you> Can I pick the order up from your Abu Dhabi store instead?
     call   search_policies({'query': 'store pickup Abu Dhabi collect in store'})
     call   search_policies({'query': 'physical store locations drop off return in person'})
bot> Our policies don't mention an Abu Dhabi store or in-store pickup ... Should I
     open a ticket so an agent can check?

you> Yes please, open a ticket for that.
     call   create_ticket({'summary': 'Customer Sara wants a refund ...', 'priority': 'normal'})
bot> I've opened ticket TCK-0001 for you, Sara. ...

you> Thanks. What was my name again?
bot> Your name is Sara.
```

## Retrieval results

`uv run eval_retrieval.py`: 24 questions over 11 docs (21 chunks). Half reuse the docs' words ("Is shipping free on orders over 200 AED?"), half paraphrase ("How quickly will my parcel reach Riyadh?"). Each question names the doc that answers it.

| Method | Questions | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- | --- |
| BM25 | keyword (12) | 1.00 | 1.00 | 1.00 |
| BM25 | paraphrase (12) | 0.42 | 0.50 | 0.44 |
| vector | keyword (12) | 0.92 | 0.92 | 0.94 |
| vector | paraphrase (12) | **1.00** | **1.00** | **1.00** |
| hybrid (RRF) | keyword (12) | 0.92 | 0.92 | 0.94 |
| hybrid (RRF) | paraphrase (12) | 0.75 | 0.92 | 0.85 |
| **vector** | **all (24)** | **0.96** | **0.96** | **0.97** |
| hybrid (RRF) | all (24) | 0.83 | 0.92 | 0.90 |
| BM25 | all (24) | 0.71 | 0.75 | 0.72 |

hit@k: share of questions whose right doc is in the top k. MRR (mean reciprocal rank): 1.0 means the right doc always ranks first.

What the numbers show:

- **BM25 wins on exact words, fails on paraphrase.** "Riyadh" shares no word with "GCC orders (Saudi Arabia ...)", so keyword search scores nothing.
- **Vector search wins overall.** Its one miss is "Is shipping free on orders over 200 AED?", where the exact number matters more than meaning.
- **Hybrid did not beat vector here.** Reciprocal rank fusion gives both lists an equal vote, so BM25's confident wrong answers on paraphrases pull the right doc down. On a corpus with many codes, SKUs or names, hybrid would likely win. Only measuring tells.
- Two BM25 fixes came from reading failures, and each was measured: dropping chunks that share no word with the question (hybrid MRR 0.80 → 0.83), and removing stop words like "do" and "and" (BM25 0.70 → 0.72, hybrid 0.83 → 0.90).

Limits: 24 hand-written questions on 11 short sample docs. Enough to see clear gaps, too small to trust differences of a few points. The two BM25 fixes were found by reading failures on these same questions, so their gains are measured on tuning data and are optimistic. A held-out question set would give a fair number.

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
uv run 05_support_bot.py                       # scripted support demo
uv run 05_support_bot.py --chat --thread ali   # be the customer
uv run eval_retrieval.py                       # offline, no API key needed
```

Each script run makes a few API calls (cents at most). The eval and tests make none.

## Tests and CI

```sh
uv run pytest     # 17 offline tests: tools, retrieval, ranking fusion, eval scoring
uv run ruff check .
```

GitHub Actions runs lint, tests and the retrieval eval on every push, and writes the eval table to the job summary. Nothing in CI calls Claude, so it needs no secret and costs nothing. Model calls are kept out of tests on purpose: their output varies, and every run would cost money.

## Design decisions

- **Every script runs from `main()`** behind `if __name__ == "__main__"`, so tests can import the tools without starting a model or a chat.
- **Tools return errors as text.** `web_search` and `get_weather` catch network failures and tell the model, so one failed HTTP call can't crash a whole conversation. (`get_weather` once let them escape; a test now covers it.)
- **Structured output uses `method="json_schema"`.** The default tool-calling method forces a tool call (`tool_choice`), which Claude Opus 5.5 rejects with a 400.
- **No `temperature`.** Current Claude models reject sampling parameters.
- **Server-side fallback is on** (`fallbacks: "default"` in `common.py`): if a safety classifier refuses a request, the API retries it on another model instead of failing.
- **Effort defaults to `low`**, so demos stay fast and cheap. Raise it with `LC_EFFORT` for harder work.
- **No `langchain-community`.** BM25 uses `rank-bm25` directly, and the fastembed wrapper is 15 lines. LangChain is winding that package down.
- **Local embeddings** (fastembed, `BAAI/bge-small-en-v1.5`, about 70 MB, no API key). Anthropic has no embeddings API. For a hosted one, swap `LocalEmbeddings` for a class such as `VoyageAIEmbeddings`; for data that must persist, swap `InMemoryVectorStore` for Chroma or pgvector.
- **Web tools need no keys.** `web_search` uses `ddgs`, which scrapes DuckDuckGo (no official API) and falls back to other engines when blocked; results vary and snippets often lack live numbers, so weather has its own tool. `get_weather` calls Open-Meteo: city name to coordinates, then current conditions and today's high and low.
- **Memory is capped by summarization.** In 04 the limit is 6 messages so the demo shows it; the bot uses 20. Real apps should count tokens, e.g. `("tokens", 50_000)`. Each summary costs one extra model call. Delete `chat.db` to wipe every thread.
- **`.gitignore` is an allowlist.** New files stay out of git until added on purpose, so `.env`, keys and `chat.db` can't slip in.

## Settings

| Env var | Default | Meaning |
| --- | --- | --- |
| `LC_MODEL` | `claude-opus-5-5` | Any Claude model ID |
| `LC_EFFORT` | `low` | `low`, `medium`, `high`, `xhigh`, `max` |

## Record a demo

```sh
brew install asciinema
asciinema rec demo.cast -c "uv run 05_support_bot.py"
asciinema play demo.cast
```

`asciinema upload demo.cast` gives a shareable link for a CV. The upload is public, and the recording shows only what the terminal printed.
