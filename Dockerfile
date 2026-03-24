# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies needed by some Python packages (e.g. unstructured)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY .env.example .env.example

# Create data and vector-store directories
RUN mkdir -p data chroma_db

# Default: keep the container alive so developers can exec into it
CMD ["bash"]
