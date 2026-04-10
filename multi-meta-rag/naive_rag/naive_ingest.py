from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Separate storage for naive RAG
naive_index = faiss.IndexFlatL2(384)
naive_texts = []


def ingest_naive(text):
    chunks = [text[i:i+800] for i in range(0, len(text), 700)]

    embeddings = embedder.encode(chunks)

    naive_index.add(np.array(embeddings).astype("float32"))
    naive_texts.extend(chunks)

    return len(chunks)