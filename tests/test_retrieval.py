"""Offline tests for retrieval.py and the eval scoring. No embedding model needed."""

import importlib

from langchain_core.documents import Document

from retrieval import bm25_search, format_chunks, hybrid_search, load_chunks, tokenize

eval_retrieval = importlib.import_module("eval_retrieval")


def _doc(source: str, text: str = "") -> Document:
    return Document(page_content=text or source, metadata={"source": source})


def test_tokenize() -> None:
    assert tokenize("How do Refunds over 5000 AED work?") == ["refunds", "5000", "aed", "work"]


def test_chunks_keep_their_source() -> None:
    chunks = load_chunks()

    assert chunks
    assert all(chunk.metadata["source"].endswith(".md") for chunk in chunks)


def test_bm25_keyword_match() -> None:
    search = bm25_search(load_chunks())

    top = search("Who approves refunds over 5000 AED?", 1)

    assert top[0].metadata["source"] == "refund-policy.md"


def test_bm25_drops_zero_scores() -> None:
    search = bm25_search(load_chunks())

    assert search("zzz qqq", 5) == []


def test_hybrid_rrf_order() -> None:
    a, b, c = _doc("a.md"), _doc("b.md"), _doc("c.md")
    first = lambda question, k: [a, b, c][:k]
    second = lambda question, k: [b, c, a][:k]

    # b is 2nd and 1st, so 1/62 + 1/61 beats a's 1/61 + 1/63.
    merged = hybrid_search([first, second])("anything", 3)

    assert [d.metadata["source"] for d in merged] == ["b.md", "a.md", "c.md"]


def test_format_chunks() -> None:
    text = format_chunks([_doc("x.md", "one"), _doc("y.md", "two")])

    assert text == "[x.md]\none\n\n[y.md]\ntwo"


def test_source_rank() -> None:
    search = lambda question, k: [_doc("a.md", "1"), _doc("a.md", "2"), _doc("b.md")]

    assert eval_retrieval._source_rank(search, "q", "b.md", 3) == 2
    assert eval_retrieval._source_rank(search, "q", "c.md", 3) is None


def test_scores() -> None:
    scores = eval_retrieval._scores([1, 2, None, 4])

    assert scores == {"hit@1": 0.25, "hit@3": 0.5, "MRR": (1 + 0.5 + 0.25) / 4}


def test_rag_demo_finds_timing() -> None:
    """03_rag.py's default question must retrieve the '7 business days' chunk.

    It once ranked 4th, behind chunks that matched only 'do' and 'and', so
    the demo answered that refund timing wasn't documented.
    """
    rag = importlib.import_module("03_rag")
    search = bm25_search(load_chunks())

    top = search(rag.DEFAULT_QUESTION, rag.TOP_K)

    assert any("7 business days" in chunk.page_content for chunk in top)
