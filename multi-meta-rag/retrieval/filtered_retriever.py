import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.ingest import index, stored_texts, stored_metadata
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


def retrieve(query: str, k_initial: int = 30) -> list:
    if len(stored_texts) == 0:
        return []

    metadata_filter = extract_metadata_filter(query)

    valid_indices = apply_metadata_filter(
        metadata_filter,
        stored_texts,
        stored_metadata
    )

    if not valid_indices:
        valid_indices = list(range(len(stored_texts)))

    query_vector = embedder.encode([query])[0]

    filtered_embeddings = []
    for idx in valid_indices:
        vec = index.reconstruct(idx)
        filtered_embeddings.append(vec)

    filtered_embeddings = np.array(filtered_embeddings).astype("float32")

    scores = np.dot(filtered_embeddings, query_vector)

    results = []

    for i in range(len(scores)):
        text = stored_texts[valid_indices[i]]

        score = scores[i]

        # 🔥 Table boosting logic
        if "table" in text.lower():
            score += 0.3
        if "mrr" in text.lower() or "map@" in text.lower():
            score += 0.2

        results.append((score, valid_indices[i]))

    # sort descending
    results = sorted(results, key=lambda x: x[0], reverse=True)[:k_initial]

    final_results = []

    for rank, (score, idx) in enumerate(results):
        final_results.append({
            "text": stored_texts[idx],
            "metadata": stored_metadata[idx],
            "score": float(score),
            "filter_applied": metadata_filter
        })

    return final_results