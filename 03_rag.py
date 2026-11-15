"""RAG (retrieval-augmented generation) over local text files.

Loads docs/*.md, splits them into chunks, retrieves the chunks that best match
the question, and asks the model to answer only from those chunks.

Retrieval here is BM25 (keyword ranking) via the rank-bm25 package: no
embedding model, no extra API key. 03b_rag_vector.py is the vector search
version.

    uv run 03_rag.py "How long do refunds take?"
"""

import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from common import make_model

DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
DEFAULT_QUESTION = "How long do refunds take, and who approves refunds over 5000 AED?"

# Load: one Document per file, with the file name kept as metadata.
documents = [
    Document(page_content=path.read_text(), metadata={"source": path.name})
    for path in sorted(DOCS_DIR.glob("*.md"))
]

# Split: smaller chunks make retrieval more precise.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)


def _tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, e.g. 'Refunds over 5000 AED' -> ['refunds', 'over', '5000', 'aed']."""
    return re.findall(r"\w+", text.lower())


# Index: BM25 scores each chunk by how often it contains the question's words.
bm25 = BM25Okapi([_tokenize(chunk.page_content) for chunk in chunks])


def _search(question: str) -> list[Document]:
    """Return the TOP_K chunks with the highest BM25 score."""
    return bm25.get_top_n(_tokenize(question), chunks, n=TOP_K)


# Wrapping the function makes it a pipeline step, like a built-in retriever.
retriever = RunnableLambda(_search)


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

# The dict runs both branches on the same input: retrieve context, pass question.
rag_chain = (
    {"context": retriever | format_chunks, "question": RunnablePassthrough()}
    | prompt
    | make_model()
    | StrOutputParser()
)

question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
print(f"Q: {question}\n")
print(rag_chain.invoke(question))
