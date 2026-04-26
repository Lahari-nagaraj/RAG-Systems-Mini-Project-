import os
from groq import Groq
from dotenv import load_dotenv
from naive_rag.naive_retrieve import retrieve_naive

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)
MODEL  = "llama-3.1-8b-instant"


def naive_answer(query: str) -> dict:
    """
    Naive RAG baseline — plain FAISS kNN, no metadata filter, no reranker.

    Key behavioural difference from Multi-Meta-RAG:
      - Always attempts to answer from whatever top-5 chunks were retrieved.
      - Never refuses or says "not found" — it has no relevance gate.
      - This is intentional: naive RAG will hallucinate or give partial answers
        for out-of-scope queries, which is exactly the baseline behaviour we
        want to demonstrate vs Multi-Meta-RAG's precision.

    Returns:
        answer      : str  — LLM response (always generated, never refused)
        _chunks     : list — top-5 chunk dicts passed to LLM
        _all_ranked : list — all 10 retrieved chunk dicts (for MRR@10/Hit@10)
    """
    all_chunks = retrieve_naive(query, k=10)  # retrieve 10 for metric computation

    if not all_chunks:
        return {
            "answer":      "No documents have been ingested yet.",
            "_chunks":     [],
            "_all_ranked": [],
        }

    top5    = all_chunks[:5]   # only top-5 go to the LLM (paper baseline)
    context = "\n\n".join(c["text"] for c in top5)

    prompt = f"""You are a direct, confident research assistant. Read the context and answer the question in 2-4 sentences.

Rules:
- Always give a direct answer. Never say the context does not mention something.
- Never use phrases like "the context does not explicitly mention", "I cannot find", "it is possible that", or "I can only speculate".
- If the context is partially relevant, synthesize what you know and give a complete answer.
- Be factual and specific. Do not hedge.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400,
    )

    return {
        "answer":      response.choices[0].message.content.strip(),
        "_chunks":     top5,        # top-5 used for generation
        "_all_ranked": all_chunks,  # all 10 used for MRR@10 / Hit@10
    }