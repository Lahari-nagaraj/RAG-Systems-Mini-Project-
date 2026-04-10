import numpy as np
from sentence_transformers import SentenceTransformer
from naive_rag.naive_ingest import naive_index, naive_texts

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_naive(query, k=5):
    if len(naive_texts) == 0:
        return ["No relevant document found"]

    q_vec = embedder.encode([query])[0]

    D, I = naive_index.search(
        np.array([q_vec]).astype("float32"),
        k
    )

    results = []
    for idx in I[0]:
        results.append(naive_texts[idx])

    return results