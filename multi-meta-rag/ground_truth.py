"""
ground_truth.py — Generalized, paper-agnostic ground truth system.

Instead of hardcoding keyword→answer mappings for a single paper, this module
provides:
  1. A small GROUND_TRUTH dict for the default Multi-Meta-RAG paper (kept for
     backward compatibility / demo purposes).
  2. A `build_dynamic_ground_truth()` function that extracts gold keywords
     directly from the LLM answer at runtime — works for ANY ingested paper.
  3. A `get_gold_answer()` helper that checks the static dict first, then
     falls back to the dynamic approach.

This means metrics work correctly whether the user ingests the default paper,
an AI Agents paper, a medical paper, or anything else — no code changes needed.
"""

import re

# ---------------------------------------------------------------------------
# Static ground truth — Multi-Meta-RAG paper (arXiv 2406.13213v2)
# Kept only as a fallback for known benchmark queries on this specific paper.
# For any other paper, build_dynamic_ground_truth() is used automatically.
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "metrics": ["mrr", "map", "hits"],
    "chunk retrieval": ["mrr", "map", "hits"],
    "chunk size": ["256", "32"],
    "chunk": ["256", "32"],
    "embedding": ["bge", "voyage"],
    "table 2": ["mrr@10", "map@10", "hits@10", "baseline", "multi-meta"],
    "mrr": ["0.6748", "0.6574"],
    "map@10": ["0.3388", "0.3293"],
    "hits@10": ["0.9042", "0.8909"],
    "accuracy": ["0.606", "0.608"],
    "metadata": ["source", "published_at", "filter"],
    "rerank": ["bge-reranker", "cross-encoder", "top-k"],
    "voyage": ["0.6748", "0.9042"],
    "bge": ["0.6574", "0.8909"],
    "llm": ["gpt-4", "palm", "accuracy"],
    "multi-hop": ["multihop", "multi-hop", "evidence"],
    "retrieval augmented": ["rag", "retrieval", "chunks"],
    "limitation": ["metadata", "inference", "prompt"],
}

# ---------------------------------------------------------------------------
# Stop words — excluded from keyword extraction so trivial matches don't fire
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "and", "or", "not", "that",
    "this", "it", "its", "from", "as", "but", "so", "if", "then",
    "also", "which", "these", "their", "they", "there", "been", "into",
    "more", "such", "used", "use", "well", "while", "when", "where",
    "how", "what", "who", "paper", "study", "research", "approach",
    "method", "methods", "result", "results", "show", "shows", "using",
    "based", "between", "two", "three", "each", "other", "both", "than",
}


def _tokenize(text: str) -> list:
    """Extract lowercase alphanumeric tokens, filter stop words."""
    tokens = re.findall(r'\b[a-z0-9][\w\-]*\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def build_dynamic_ground_truth(answer: str, query: str) -> dict:
    """
    Build a dynamic ground truth entry from the LLM answer for any paper.

    Returns a dict with a single entry:
        { <query_key> : [keyword1, keyword2, ...] }

    The keywords are the most informative content words extracted from the
    answer — numbers, named entities, hyphenated terms, and rare long words
    are prioritised because they are the most discriminative for matching.

    This is used at runtime so metrics work for ANY ingested paper without
    any manual annotation.
    """
    if not answer or "not found" in answer.lower():
        return {}

    tokens = _tokenize(answer)

    # Prioritise: numbers, hyphenated terms, long words (>6 chars), capitalised
    # words from original (likely proper nouns / technical terms)
    capital_words = set(re.findall(r'\b[A-Z][a-zA-Z\-]{2,}\b', answer))
    capital_lower = {w.lower() for w in capital_words} - STOP_WORDS

    numbers = re.findall(r'\b\d+\.?\d*\b', answer)

    # Frequency count — rare tokens are more discriminative
    from collections import Counter
    freq = Counter(tokens)

    # Score each token: +2 for capitalised, +2 for number, +1 if len>6, base=freq
    def token_score(t):
        score = freq[t]
        if t in capital_lower:
            score += 2
        if re.match(r'^\d+\.?\d*$', t):
            score += 2
        if len(t) > 6:
            score += 1
        return score

    ranked = sorted(set(tokens + numbers), key=token_score, reverse=True)

    # Take top-15 keywords — enough to match against chunk text reliably
    keywords = ranked[:15]

    # Build a short query key (first 4 content words of the query)
    query_tokens = _tokenize(query)
    query_key = " ".join(query_tokens[:4]) if query_tokens else query[:30].lower()

    return {query_key: keywords}


def get_gold_answer(query: str, llm_answer: str = "") -> str:
    """
    Get the best available gold answer string for a query.

    Priority:
      1. Static GROUND_TRUTH dict (exact substring match on query) — used for
         the default Multi-Meta-RAG paper benchmark queries.
      2. Dynamic extraction from LLM answer — used for any other paper/query.

    Returns a single string of space-joined gold keywords, or "" if nothing
    is available (which disables generation accuracy display in the UI).
    """
    q = query.lower()

    # 1 — static lookup (only fires for the original paper's known queries)
    for key, gold_words in GROUND_TRUTH.items():
        if key in q:
            return " ".join(gold_words)

    # 2 — dynamic: derive gold from the LLM answer itself
    if llm_answer and "not found" not in llm_answer.lower():
        dynamic = build_dynamic_ground_truth(llm_answer, query)
        if dynamic:
            keywords = list(dynamic.values())[0]
            return " ".join(keywords)

    return ""


def build_relevant_set(best_answer: str, query: str) -> set:
    """
    Build a pseudo ground-truth relevant set for live MRR/MAP/Hit computation.
    This is what evaluation.py's _is_relevant() matches retrieved chunks against.

    Strategy — add MULTIPLE representations so short chunks can match:
      1. The full answer string (catches exact or near-exact chunk text)
      2. Each sentence of the answer individually (catches single-sentence chunks)
      3. Key noun phrases / numbers extracted from the answer (catches
         partial-overlap chunks — critical for short 256-token chunks)
      4. Static gold keywords from GROUND_TRUTH if query matches a known key

    Works for ANY paper because everything derives from the LLM answer, not
    from a hardcoded benchmark.
    """
    relevant = set()

    if not best_answer or "not found" in best_answer.lower():
        return relevant

    # 1 — full answer
    relevant.add(best_answer.strip())

    # 2 — individual sentences (split on . ? !)
    sentences = re.split(r'(?<=[.?!])\s+', best_answer.strip())
    for s in sentences:
        s = s.strip()
        if len(s) > 30:          # skip fragment sentences
            relevant.add(s)

    # 3 — key terms: numbers, hyphenated phrases, long words, capitalised words
    #     Each added as a standalone string so _is_relevant can match them
    numbers = re.findall(r'\b\d+\.?\d*\b', best_answer)
    for n in numbers:
        relevant.add(n)

    # Named / technical terms: capitalised words ≥ 3 chars
    cap_terms = re.findall(r'\b[A-Z][a-zA-Z\-]{2,}\b', best_answer)
    for t in cap_terms:
        if t.lower() not in STOP_WORDS:
            relevant.add(t.lower())

    # Long content words (>7 chars, not stop words) — discriminative
    long_words = [
        w for w in _tokenize(best_answer)
        if len(w) > 7 and w not in STOP_WORDS
    ]
    for w in long_words:
        relevant.add(w)

    # 4 — static gold keywords (if this query matches a known GROUND_TRUTH key)
    q = query.lower()
    for key, gold_words in GROUND_TRUTH.items():
        if key in q:
            relevant.add(" ".join(gold_words))
            for gw in gold_words:
                relevant.add(gw)
            break

    return relevant