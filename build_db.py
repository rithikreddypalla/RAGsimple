import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
	return SentenceTransformer(model_name)


def build_vector_store(chunks, model):
	embeddings = model.encode(chunks)
	embeddings_np = np.array(embeddings)

	dim = embeddings_np.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(embeddings_np)

	chunk_store = {i: chunk for i, chunk in enumerate(chunks)}
	return index, chunk_store
