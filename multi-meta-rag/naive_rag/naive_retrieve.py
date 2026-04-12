import numpy as np
from sentence_transformers import SentenceTransformer
from naive_rag.naive_ingest import naive_index, naive_texts

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_naive(query: str, k: int = 10) -> list:
    """
    Returns a list of unique chunk dicts ordered by FAISS similarity (descending).

    Each dict has:
        text     : str   — chunk content
        score    : float — normalised similarity 1/(1+L2_dist)
        rank     : int   — 1-based rank after deduplication
        metadata : dict  — {"chunk_index": <faiss_idx>}

    Deduplication: if the same PDF was ingested multiple times, FAISS will
    return the same chunk text at multiple indices with identical scores.
    We keep only the first (highest-scoring) occurrence of each unique text
    so the LLM and metrics receive k *distinct* chunks, not k copies of 2.

    We retrieve k*5 from FAISS initially to ensure we still get k unique
    chunks after deduplication.
    """
    if len(naive_texts) == 0:
        return []

    q_vec = embedder.encode([query])[0]

    # Request more than k to absorb duplicates
    fetch_k = min(k * 5, len(naive_texts))

    D, I = naive_index.search(
        np.array([q_vec]).astype("float32"),
        fetch_k,
    )

    results  = []
    seen     = set()   # deduplication by exact text
    rank     = 1

    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        text = naive_texts[idx]

        # Skip duplicate texts — keep only highest-scoring copy
        if text in seen:
            continue
        seen.add(text)

        score = float(1 / (1 + dist))
        results.append({
            "text":     text,
            "score":    round(score, 4),
            "rank":     rank,
            "metadata": {"chunk_index": int(idx)},
        })
        rank += 1

        if len(results) >= k:
            break

    return results