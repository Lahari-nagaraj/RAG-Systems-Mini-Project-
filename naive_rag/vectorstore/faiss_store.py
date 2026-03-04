import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []
        self.metadata = []

    def add(self, embeddings, chunks, metadata):
        self.index.add(np.array(embeddings))
        self.text_chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding, k=5):
        D, I = self.index.search(np.array(query_embedding), k)

        results = []
        for idx in I[0]:
            results.append({
                "text": self.text_chunks[idx],
                "metadata": self.metadata[idx]
            })
        return results