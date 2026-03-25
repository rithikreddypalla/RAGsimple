import numpy as np


def query_rag(query, model, index, chunk_store, top_k=2):
	query_embedding = model.encode([query])
	distances, indices = index.search(np.array(query_embedding), top_k)

	retrieved = [chunk_store[i] for i in indices[0]]
	return retrieved, distances
