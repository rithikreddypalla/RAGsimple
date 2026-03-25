# RAGsimple
A simple Retrieval-Augmented Generation (RAG) proof of concept.

## What this project does
This project takes a user question, retrieves the most relevant text chunks from a small document set using embeddings + FAISS, and builds a final prompt that can be sent to an LLM.

Current flow:
1. Load documents
2. Chunk documents
3. Create embeddings and build FAISS index
4. Retrieve top-k chunks for a query
5. Build a context-aware prompt

---

## Project structure

- `pipeline.py` → Main orchestrator (entry point)
- `docs.py` → Document source
- `chunking.py` → Text chunking logic
- `build_db.py` → Embedding model loading + FAISS index construction
- `query.py` → Retrieval from vector index
- `llm_call.py` → Prompt construction for LLM call

---

## File-by-file explanation

### `pipeline.py`
Purpose: Coordinates the complete RAG pipeline from documents to prompt.

#### Methods

##### `main()`
- Loads embedding model via `load_embedding_model()`
- Loads raw documents via `load_documents()`
- Chunks docs via `chunk_documents()`
- Builds vector DB and chunk map via `build_vector_store()`
- Runs retrieval with `query_rag()`
- Prints retrieved chunks
- Builds final LLM prompt with `build_prompt()`
- Prints final prompt

Execution starts from:
```python
if __name__ == "__main__":
		main()
```

---

### `docs.py`
Purpose: Keeps all source documents in one place.

#### Methods

##### `load_documents()`
- Returns a Python list of document strings.
- Right now this is hardcoded sample content.
- In a real system, this can be replaced with file/database/API loading.

---

### `chunking.py`
Purpose: Splits long text into smaller pieces for embedding and retrieval.

#### Methods

##### `chunk_text(text, chunk_size=100)`
- Splits one document into word-based chunks.
- Uses fixed-size chunks (`chunk_size` words each).
- Returns a list of chunk strings.

##### `chunk_documents(documents, chunk_size=100)`
- Applies `chunk_text()` to every document.
- Flattens all chunks into one list.
- Returns full corpus chunk list.

---

### `build_db.py`
Purpose: Creates the vector retrieval backend (embeddings + FAISS index).

#### Methods

##### `load_embedding_model(model_name="all-MiniLM-L6-v2")`
- Loads a SentenceTransformer model.
- Default model is lightweight and commonly used for RAG demos.

##### `build_vector_store(chunks, model)`
- Encodes chunks into embeddings using `model.encode(chunks)`.
- Converts embeddings to NumPy array.
- Builds a FAISS `IndexFlatL2` index.
- Adds embeddings into index.
- Creates `chunk_store` mapping:
	- key: FAISS vector id (int)
	- value: original text chunk
- Returns `(index, chunk_store)`.

---

### `query.py`
Purpose: Retrieves the most relevant chunks for a user question.

#### Methods

##### `query_rag(query, model, index, chunk_store, top_k=2)`
- Encodes query text into embedding.
- Searches FAISS index for top-k nearest vectors.
- Uses returned ids to fetch chunk text from `chunk_store`.
- Returns:
	- `retrieved` (list of best chunks)
	- `distances` (FAISS distance scores)

---

### `llm_call.py`
Purpose: Builds the final prompt sent to an LLM.

#### Methods

##### `build_prompt(question, retrieved_chunks)`
- Joins retrieved chunks into one context string.
- Formats prompt as:
	- instruction
	- context
	- user question
- Returns prompt string.

Note: This file currently only builds the prompt (no actual API call yet).

---

## End-to-end workflow (complete)

1. `pipeline.py` starts `main()`.
2. `docs.load_documents()` returns raw documents.
3. `chunking.chunk_documents()` splits them into manageable chunks.
4. `build_db.load_embedding_model()` loads embedding model.
5. `build_db.build_vector_store()`:
	 - creates embeddings,
	 - builds FAISS index,
	 - prepares chunk id → text map.
6. User question is sent to `query.query_rag()`.
7. Query embedding is compared against all chunk embeddings in FAISS.
8. Top-k closest chunks are returned as retrieval context.
9. `llm_call.build_prompt()` combines context + question into final prompt.
10. Prompt is ready to send to any LLM API.

---

## How to run

### 1) Install dependencies
```bash
pip install sentence-transformers faiss-cpu numpy
```

If you use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install sentence-transformers faiss-cpu numpy
```

### 2) Run pipeline
```bash
python pipeline.py
```

Expected output:
- Retrieved chunks printed in terminal
- Final context-aware prompt printed in terminal

---
