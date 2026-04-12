import numpy as np
from sentence_transformers import SentenceTransformer
from retrieval.metadata_extractor import extract_metadata_filter

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def apply_metadata_filter(metadata_filter, texts, metadata_list):
    if not metadata_filter:
        return list(range(len(texts)))

    valid_indices = []
    for i, meta in enumerate(metadata_list):
        match = True
        for field, condition in metadata_filter.items():
            if "$in" in condition:
                values = condition["$in"]
                if field not in meta:
                    match = False
                    break
                meta_value = str(meta[field]).lower()
                if not any(v.lower() in meta_value for v in values):
                    match = False
                    break
        if match:
            valid_indices.append(i)

    return valid_indices


def retrieve(query: str, k_initial: int = 20) -> list:
    from ingestion.ingest import index, stored_texts, stored_metadata

    if len(stored_texts) == 0:
        print("DEBUG: stored_texts is empty!")
        return []

    print(f"DEBUG: Total stored chunks = {len(stored_texts)}")

    metadata_filter = extract_metadata_filter(query)

    valid_indices = apply_metadata_filter(metadata_filter, stored_texts, stored_metadata)
    if not valid_indices:
        valid_indices = list(range(len(stored_texts)))

    # ✅ Use FAISS search directly instead of reconstruct()
    query_vector = np.array([embedder.encode([query])[0]]).astype("float32")

    # Search all vectors, then filter to valid indices
    k = min(k_initial * 3, index.ntotal)
    if k == 0:
        return []

    distances, indices = index.search(query_vector, k)

    valid_set = set(valid_indices)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if idx in valid_set:
            text = stored_texts[idx]
            score = float(1 / (1 + dist))  # convert L2 distance to similarity score
            if "table" in text.lower():
                score += 0.1
            if "mrr" in text.lower() or "map@" in text.lower():
                score += 0.05
            results.append((score, idx))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:k_initial]
    top_score = results[0][0] if results else 0
    print(f"DEBUG: Retrieved {len(results)} chunks, top score = {top_score:.4f}")

    return [{
        "text": stored_texts[idx],
        "metadata": stored_metadata[idx],
        "score": score,
        "filter_applied": metadata_filter
    } for score, idx in results]