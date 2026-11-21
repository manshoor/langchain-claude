"""Retrieval shared by the RAG scripts, the eval and the support bot.

Three ways to rank chunks for a question:

    BM25    keyword match: counts shared words, weighs rare words higher
    vector  meaning match: cosine similarity of local fastembed embeddings
    hybrid  both lists merged with reciprocal rank fusion (RRF)

    docs/*.md ──► load ──► split ──► chunks ──┬──► BM25 index
                                              └──► embed ──► InMemoryVectorStore

Every search function has the same shape: (question, k) -> list[Document],
so callers can swap one for another.
"""

import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from acme_support.paths import ROOT

DOCS_DIR = ROOT / "docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# RRF constant from the original paper (Cormack et al., 2009). Larger values
# flatten the gap between rank 1 and rank 10.
RRF_K = 60

# Common words that match almost every chunk. Left in, they let a chunk that
# shares only "do" and "and" with the question outrank the one that answers it.
_STOP_WORDS_TEXT = (
    "a an and are as at be by can do does for from how i if in is it its long "
    "me my of on or over so that the their there this to was we what when "
    "where which who will with you your"
)
STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())

Search = Callable[[str, int], list[Document]]


def load_chunks(
    docs_dir: Path = DOCS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Load every .md file as a Document and split it into chunks.

    Each chunk keeps its file name in metadata["source"] for citations.
    """
    documents = [
        Document(page_content=path.read_text(), metadata={"source": path.name})
        for path in sorted(docs_dir.glob("*.md"))
    ]

    # Smaller chunks make retrieval more precise but carry less context.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without stop words.

    Example: 'How do refunds over 5000 AED work?' -> ['refunds', '5000', 'aed', 'work']
    """
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOP_WORDS]


def bm25_search(chunks: list[Document]) -> Search:
    """Build a BM25 index over the chunks and return its search function."""
    index = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])

    def search(question: str, k: int) -> list[Document]:
        # Drop chunks that share no words with the question (score 0). Their
        # order is arbitrary, so returning them would only add noise.
        scores = index.get_scores(tokenize(question))
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in ranked[:k] if scores[i] > 0]

    return search


class LocalEmbeddings(Embeddings):
    """LangChain Embeddings interface over a local fastembed model (ONNX, no API key).

    The first use downloads the model, about 70 MB, into the fastembed cache.
    """

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        # Imported here so BM25-only callers don't pay fastembed's import time.
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds the query prefix this model was trained with.
        return next(iter(self._model.query_embed(text))).tolist()


def vector_store(chunks: list[Document]) -> InMemoryVectorStore:
    """Embed every chunk once and keep the vectors in memory."""
    return InMemoryVectorStore.from_documents(chunks, LocalEmbeddings())


def vector_search(store: InMemoryVectorStore) -> Search:
    """Return a search function over an embedded store."""

    def search(question: str, k: int) -> list[Document]:
        return store.similarity_search(question, k=k)

    return search


def hybrid_search(searches: list[Search], depth: int = 20) -> Search:
    """Merge several rankings with reciprocal rank fusion.

    Each list gives a chunk 1 / (RRF_K + rank). A chunk ranked 1st by BM25 and
    3rd by vector scores 1/61 + 1/63. Scores add up, so chunks both methods
    like rise to the top. RRF uses ranks only, so BM25 scores and cosine
    similarities never need to be put on one scale.
    """

    def search(question: str, k: int) -> list[Document]:
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}
        for one_search in searches:
            for rank, doc in enumerate(one_search(question, depth), start=1):
                key = _chunk_key(doc)
                by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)

        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [by_key[key] for key in best]

    return search


def _chunk_key(doc: Document) -> str:
    """Identify a chunk by its source and text, since chunks have no ID."""
    return f"{doc.metadata['source']}\n{doc.page_content}"


def format_chunks(docs: list[Document]) -> str:
    """Join retrieved chunks, each tagged with its source file."""
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
