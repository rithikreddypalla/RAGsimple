def chunk_text(text, chunk_size=100):
	words = text.split()
	return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_documents(documents, chunk_size=100):
	chunks = []
	for doc in documents:
		chunks.extend(chunk_text(doc, chunk_size=chunk_size))
	return chunks
