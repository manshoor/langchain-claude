"""RAG with vector search instead of BM25.

Same pipeline as 03_rag.py, but retrieval compares meaning, not words:
each chunk becomes an embedding (a list of numbers), and the question's
embedding is matched to the nearest chunks.

    question ──► embed ──┐
                         ├──► nearest chunks ──► prompt ──► Claude ──► answer
    docs ──► split ──► embed ──► InMemoryVectorStore

Embeddings run locally with fastembed (ONNX, no API key). The first run
downloads the model, about 70 MB, into the fastembed cache.

    uv run scripts/03b_rag_vector.py "Can I get my money back after a month?"
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from acme_support.common import make_model
from acme_support.retrieval import format_chunks, load_chunks, vector_store

TOP_K = 3

# A paraphrase: shares few words with the docs, so keyword search finds less.
DEFAULT_QUESTION = "Can I get my money back if I bought it six weeks ago?"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer only from the context. Cite the source file in brackets "
                "after each fact. If the context does not contain the answer, say so."
            ),
        ),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def main() -> None:
    """Show the retrieved chunks with scores, then answer from them."""
    store = vector_store(load_chunks())
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
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


if __name__ == "__main__":
    main()
