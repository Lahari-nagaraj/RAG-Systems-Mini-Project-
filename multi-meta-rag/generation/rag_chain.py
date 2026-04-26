import os
from groq import Groq
from retrieval.filtered_retriever import retrieve
from retrieval.reranker import rerank

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"


def answer_query(user_query: str) -> dict:
    try:
        # Retrieve 20 initial candidates (paper §3.2)
        candidates = retrieve(user_query, k_initial=20)

        if not candidates:
            return {
                "answer":        "Answer not found in the provided document.",
                "chunks_used":   0,
                "initial_chunks": 0,
                "filter_applied": {},
                "_chunks":       [],   # top-k sent to LLM
                "_all_ranked":   [],   # all 20 ranked — needed for MRR/MAP/Hit
            }

        # top_chunks  → top-6 sent to the LLM (paper Table 3)
        # all_ranked  → all 20 in reranker order — used for retrieval metrics
        top_chunks, all_ranked = rerank(user_query, candidates, top_k=6)

        context        = "\n\n".join([c["text"] for c in top_chunks])
        filter_applied = candidates[0].get("filter_applied", {})

        prompt = f"""You are a research assistant. Use ONLY the context below to answer the question clearly.
Answer in 2-4 sentences. If the topic is not mentioned at all in the context, say "Answer not found in the provided document."

Context:
{context}

Question: {user_query}
Answer:"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer":         answer,
            "filter_applied": filter_applied,
            "chunks_used":    len(top_chunks),
            "initial_chunks": len(candidates),
            "_chunks":        top_chunks,   # top-6 with reranker_score
            "_all_ranked":    all_ranked,   # all 20 with reranker_score for metrics
        }

    except Exception as e:
        return {
            "answer":         f"⚠️ API error: {str(e)}. Please wait and try again.",
            "chunks_used":    0,
            "initial_chunks": 0,
            "filter_applied": {},
            "_chunks":        [],
            "_all_ranked":    [],
        }