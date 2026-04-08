import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
	def __init__(self):
		self.vectorizer = TfidfVectorizer()

	def fit(self, corpus):
		self.vectorizer.fit(corpus)

	def encode(self, texts):
		vectors = self.vectorizer.transform(texts)
		return vectors.toarray()


def load_embedding_model():
	return TfidfEmbedder()


def build_vector_store(chunks, model):
	model.fit(chunks)
	embeddings = model.encode(chunks)
	embeddings_np = np.array(embeddings)

	dim = embeddings_np.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(embeddings_np)

	chunk_store = {i: chunk for i, chunk in enumerate(chunks)}
	return index, chunk_store
