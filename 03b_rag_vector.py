"""RAG with vector search instead of BM25.

Same pipeline as 03_rag.py, but retrieval compares meaning, not words:
each chunk becomes an embedding (a list of numbers), and the question's
embedding is matched to the nearest chunks.

    question ──► embed ──┐
                         ├──► nearest chunks ──► prompt ──► Claude ──► answer
    docs ──► split ──► embed ──► InMemoryVectorStore

Embeddings run locally with fastembed (ONNX, no API key). The first run
downloads the model, about 70 MB, into the fastembed cache.

    uv run 03b_rag_vector.py "Can I get my money back after a month?"
"""

import sys
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import make_model

DOCS_DIR = Path(__file__).parent / "docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K = 3

# A paraphrase: shares few words with the docs, so keyword search finds less.
DEFAULT_QUESTION = "Can I get my money back if I bought it six weeks ago?"


class _LocalEmbeddings(Embeddings):
    """LangChain Embeddings interface over a local fastembed model."""

    def __init__(self, model_name: str) -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds the query prefix this model was trained with.
        return next(iter(self._model.query_embed(text))).tolist()


# Load and split, as in 03_rag.py.
documents = [
    Document(page_content=path.read_text(), metadata={"source": path.name})
    for path in sorted(DOCS_DIR.glob("*.md"))
]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)

# Embed every chunk once and keep the vectors in memory.
store = InMemoryVectorStore.from_documents(chunks, _LocalEmbeddings(EMBED_MODEL))
retriever = store.as_retriever(search_kwargs={"k": TOP_K})


def format_chunks(docs: list[Document]) -> str:
    """Join retrieved chunks, each tagged with its source file."""
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the context. Cite the source file in brackets "
            "after each fact. If the context does not contain the answer, say so.",
        ),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)
rag_chain = (
    {"context": retriever | format_chunks, "question": RunnablePassthrough()}
    | prompt
    | make_model()
    | StrOutputParser()
)

question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
print(f"Q: {question}\n")

# Show what retrieval picked and how close each chunk is (1.0 = same meaning).
print("Retrieved chunks:")
for doc, score in store.similarity_search_with_score(question, k=TOP_K):
    first_line = doc.page_content.splitlines()[0]
    print(f"  {score:.3f}  [{doc.metadata['source']}] {first_line[:70]}")

print()
print(rag_chain.invoke(question))
