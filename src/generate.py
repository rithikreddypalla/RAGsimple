"""
generate.py – Retrieve relevant context from the vector store and
generate a grounded answer using an OpenAI-compatible LLM.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from retrieve import retrieve

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

PROMPT_TEMPLATE = """You are a helpful assistant. Use only the context below to answer the question.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def generate(question: str) -> str:
    """Return an LLM-generated answer grounded in retrieved context."""
    docs = retrieve(question)
    if not docs:
        return "No relevant documents found. Please run src/ingest.py first."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    print(f"Question: {question}\n")
    answer = generate(question)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
