from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        # paper §3.2 explicitly uses bge-reranker-large (BAAI/bge-reranker-large).
        _reranker = CrossEncoder("BAAI/bge-reranker-large")
    return _reranker


def rerank(query: str, candidates: list, top_k: int = 6) -> tuple:
    """
    Rerank candidates using the cross-encoder.

    Returns:
        top_chunks   : list of top_k chunk dicts with 'reranker_score' attached
        all_ranked   : full list of all candidates ranked by cross-encoder score,
                       each with 'reranker_score' attached — used for MRR/MAP/Hit
                       computation which need rank positions across ALL candidates,
                       not just the top-k passed to the LLM.
    """
    if not candidates:
        return [], []

    reranker = get_reranker()
    pairs    = [[query, c["text"]] for c in candidates]
    scores   = reranker.predict(pairs)

    # Attach cross-encoder score to every candidate dict (non-destructive copy)
    scored = []
    for score, doc in zip(scores, candidates):
        enriched = {**doc, "reranker_score": float(score)}
        scored.append((float(score), enriched))

    all_ranked = [doc for _, doc in sorted(scored, key=lambda x: x[0], reverse=True)]
    top_chunks = all_ranked[:top_k]

    return top_chunks, all_ranked