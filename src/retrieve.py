"""
retrieve.py – Query the ChromaDB vector store and return the top-k
most relevant document chunks for a given query string.
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def get_retriever(chroma_path: str = CHROMA_PATH, embed_model: str = EMBED_MODEL, k: int = 4):
    """Return a LangChain retriever backed by the persisted ChromaDB store."""
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vector_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int = 4):
    """Retrieve the top-*k* chunks relevant to *query*."""
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    docs = retrieve(query)
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} (source: {doc.metadata.get('source', 'unknown')}) ---")
        print(doc.page_content[:500])
