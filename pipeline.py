from build_db import build_vector_store, load_embedding_model
from chunking import chunk_documents
from docs import load_documents
from llm_call import build_prompt
from query import query_rag


def main():
    model = load_embedding_model()
    documents = load_documents()
    chunks = chunk_documents(documents, chunk_size=100)

    index, chunk_store = build_vector_store(chunks, model)

    question = "What was India's GDP growth last year?"
    retrieved_chunks, _ = query_rag(question, model, index, chunk_store, top_k=2)

    print("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        print("-", chunk)

    prompt = build_prompt(question, retrieved_chunks)
    print("\nPrompt to LLM:\n", prompt)


if __name__ == "__main__":
    main()