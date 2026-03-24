"""
ingest.py – Load documents from DATA_PATH, split them into chunks,
generate embeddings, and persist them in a ChromaDB vector store.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def load_documents(data_path: str):
    """Load PDF and plain-text documents from *data_path*."""
    loaders = [
        DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader),
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks, chroma_path: str, embed_model: str):
    """Embed chunks and persist them in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )
    return vector_store


def main():
    data_path = Path(DATA_PATH)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"No documents found in '{data_path}'. Add files and re-run.")
        return

    print(f"Loading documents from '{data_path}' …")
    documents = load_documents(str(data_path))
    print(f"  Loaded {len(documents)} page(s).")

    chunks = split_documents(documents)
    print(f"  Split into {len(chunks)} chunk(s).")

    print(f"Building vector store at '{CHROMA_PATH}' …")
    build_vector_store(chunks, CHROMA_PATH, EMBED_MODEL)
    print("  Done. Vector store persisted.")


if __name__ == "__main__":
    main()
