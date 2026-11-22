"""RAG (retrieval-augmented generation) over local text files.

Loads docs/*.md, splits them into chunks, retrieves the chunks that best match
the question, and asks the model to answer only from those chunks.

Retrieval here is BM25 (keyword ranking) via the rank-bm25 package: no
embedding model, no extra API key. 03b_rag_vector.py is the vector search
version. Both use the shared code in retrieval.py.

    uv run scripts/03_rag.py "How long do refunds take?"
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from acme_support.common import make_model
from acme_support.retrieval import bm25_search, format_chunks, load_chunks

TOP_K = 3
DEFAULT_QUESTION = "How long do refunds take, and who approves refunds over 5000 AED?"

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
    """Answer one question from the docs, citing sources."""
    search = bm25_search(load_chunks())

    # Wrapping the function makes it a pipeline step, like a built-in retriever.
    retriever = RunnableLambda(lambda question: search(question, TOP_K))

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


if __name__ == "__main__":
    main()
