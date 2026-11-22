"""Prompt template | model | parser, then structured output.

    uv run scripts/01_chain.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from acme_support.common import make_model

# Part A: a plain text chain. The "|" operator pipes each step into the next.
explain_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You explain technical ideas in at most three short sentences."),
        ("user", "Explain {topic} to a {audience}."),
    ]
)


# Part B: structured output. The model must return data matching this schema.
class JobPosting(BaseModel):
    """Fields pulled out of a job posting."""

    title: str = Field(description="Job title")
    company: str = Field(description="Hiring company")
    location: str = Field(description="City and country, or 'Remote'")
    skills: list[str] = Field(description="Required technical skills")


extract_prompt = ChatPromptTemplate.from_messages(
    [("user", "Extract the posting details.\n\n{posting}")]
)

POSTING = """Acme Logistics is hiring a Senior Backend Engineer in Dubai, UAE.
You will build Node.js and TypeScript services on AWS, with PostgreSQL and Redis.
Docker and Terraform experience is a plus."""


def main() -> None:
    """Run the plain chain, then the structured-output chain."""
    model = make_model()

    # Part A: prompt | model | parser.
    explain_chain = explain_prompt | model | StrOutputParser()
    print("== Plain chain ==")
    print(explain_chain.invoke({"topic": "a load balancer", "audience": "PHP developer"}))

    # Part B: method="json_schema" uses the API's native structured output. The
    # default (tool calling) would force a tool call, which Opus 5.5 rejects.
    extract_chain = extract_prompt | model.with_structured_output(
        JobPosting, method="json_schema"
    )
    print("\n== Structured output ==")
    job = extract_chain.invoke({"posting": POSTING})
    print(job.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
