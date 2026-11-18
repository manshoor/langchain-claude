"""Measure retrieval quality: BM25 vs vector vs hybrid, offline.

Each question in evals/retrieval_questions.json names the doc that answers it.
For every method, rank the docs for each question and score where the right
one lands:

    hit@1  right doc ranked first            (share of questions)
    hit@3  right doc in the top three        (share of questions)
    MRR    mean of 1 / rank of the right doc (1.0 = always first, 0.5 = always second)

No API calls, so it is free and gives the same numbers on every run.

    uv run eval_retrieval.py
"""

import json
from enum import StrEnum
from pathlib import Path

from retrieval import Search, bm25_search, hybrid_search, load_chunks, vector_search, vector_store

QUESTIONS_PATH = Path(__file__).parent / "evals" / "retrieval_questions.json"
HIT_DEPTHS = (1, 3)


class Style(StrEnum):
    """How a question is worded relative to the doc that answers it."""

    KEYWORD = "keyword"  # reuses the doc's own words
    PARAPHRASE = "paraphrase"  # same meaning, different words


def _source_rank(search: Search, question: str, source: str, depth: int) -> int | None:
    """Return the 1-based rank of a source file among retrieved chunks, or None.

    Several chunks can come from one file, so each file counts once, at the
    position of its best chunk.
    """
    seen: list[str] = []
    for doc in search(question, depth):
        name = doc.metadata["source"]
        if name not in seen:
            seen.append(name)

    return seen.index(source) + 1 if source in seen else None


def _scores(ranks: list[int | None]) -> dict[str, float]:
    """Turn a list of ranks into hit@k and MRR."""
    found = [rank for rank in ranks if rank is not None]
    scores = {
        f"hit@{depth}": sum(rank <= depth for rank in found) / len(ranks) for depth in HIT_DEPTHS
    }
    scores["MRR"] = sum(1 / rank for rank in found) / len(ranks)
    return scores


def _evaluate(methods: dict[str, Search], questions: list[dict], depth: int) -> list[str]:
    """Score every method on every question style; return markdown table rows."""
    groups = {"all": questions} | {
        style.value: [q for q in questions if q["style"] == style] for style in Style
    }

    rows = []
    for name, search in methods.items():
        for group, items in groups.items():
            ranks = [_source_rank(search, q["question"], q["source"], depth) for q in items]
            scores = _scores(ranks)
            cells = " | ".join(f"{value:.2f}" for value in scores.values())
            rows.append(f"| {name} | {group} ({len(items)}) | {cells} |")

    return rows


def main() -> None:
    """Build all three retrievers over the same chunks and print a results table."""
    questions = json.loads(QUESTIONS_PATH.read_text())
    chunks = load_chunks()

    # Rank every chunk, so a doc the method ranks last still gets a rank.
    depth = len(chunks)
    bm25 = bm25_search(chunks)
    vector = vector_search(vector_store(chunks))
    methods = {
        "BM25": bm25,
        "vector": vector,
        "hybrid (RRF)": hybrid_search([bm25, vector], depth=depth),
    }

    print(f"{len(questions)} questions, {len(chunks)} chunks\n")
    print("| Method | Questions | " + " | ".join(f"hit@{d}" for d in HIT_DEPTHS) + " | MRR |")
    print("| --- | --- |" + " --- |" * (len(HIT_DEPTHS) + 1))
    print("\n".join(_evaluate(methods, questions, depth)))


if __name__ == "__main__":
    main()
