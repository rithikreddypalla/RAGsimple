# RAGsimple

A simple Retrieval-Augmented Generation (RAG) pipeline built as a proof of concept.

## Overview

RAGsimple demonstrates a minimal RAG workflow:

1. **Ingest** – Load and chunk documents, generate embeddings, and store them in a vector database.
2. **Retrieve** – Given a user query, retrieve the most relevant document chunks.
3. **Generate** – Pass the retrieved context to an LLM to produce a grounded answer.

## Project Structure

```
RAGsimple/
├── src/
│   ├── ingest.py       # Document loading, chunking, and embedding
│   ├── retrieve.py     # Vector-store querying
│   └── generate.py     # LLM answer generation
├── data/               # Drop source documents here (PDF, TXT, etc.)
├── .env.example        # Template for required environment variables
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image for Python development
└── docker-compose.yml  # Compose file for one-command startup
```

## Prerequisites

- Python 3.11+  **or** Docker & Docker Compose
- An [OpenAI API key](https://platform.openai.com/api-keys) (or compatible LLM provider)

## Quick Start

### Local (Python venv)

```bash
# 1. Clone the repository
git clone https://github.com/rithikreddypalla/RAGsimple.git
cd RAGsimple

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and any other values)

# 5. Add documents
mkdir -p data
cp /path/to/your/document.pdf data/

# 6. Ingest documents
python src/ingest.py

# 7. Ask a question
python src/generate.py "What is the main topic of the document?"
```

### Docker

```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Build and start the container
docker compose up --build

# 3. Attach to the running container for interactive use
docker compose exec rag bash
# Inside the container:
python src/ingest.py
python src/generate.py "What is the main topic of the document?"
```

## Environment Variables

| Variable         | Description                              | Default   |
|-----------------|------------------------------------------|-----------|
| `OPENAI_API_KEY` | OpenAI (or compatible) API key          | *(required)* |
| `MODEL_NAME`     | LLM model to use                        | `gpt-4o-mini` |
| `EMBED_MODEL`    | Sentence-transformer embedding model    | `all-MiniLM-L6-v2` |
| `CHROMA_PATH`    | Directory for the ChromaDB vector store | `./chroma_db` |
| `DATA_PATH`      | Directory containing source documents   | `./data` |

## Development

### Running with hot-reload (docker compose watch)

```bash
docker compose watch
```

Changes to files inside `src/` are automatically synced into the container.

### Linting and formatting

```bash
pip install ruff
ruff check src/
ruff format src/
```

## Contributing

1. Fork the repo and create a feature branch.
2. Make your changes with tests where applicable.
3. Open a pull request describing your change.

## License

MIT
