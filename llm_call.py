def build_prompt(question, retrieved_chunks):
	context = " ".join(retrieved_chunks)
	return f"Answer using context:\n{context}\n\nQuestion: {question}"
