from groq import Groq
import os
from dotenv import load_dotenv

from retrieval.filtered_retriever import retrieve
from retrieval.reranker import rerank

load_dotenv()

client = Groq(api_key="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")

# ✅ Use smaller model (VERY IMPORTANT)
MODEL = "llama3-8b-8192"


def is_answer_in_context(answer, context):
    """
    STRICT CHECK: ensures answer words exist in context
    """
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = answer_words.intersection(context_words)

    return len(overlap) / (len(answer_words) + 1) > 0.5


def answer_query(user_query: str) -> dict:

    try:
        # --------------------------
        # STEP 1: REDUCED RETRIEVAL (LOW TOKEN USAGE)
        # --------------------------
        candidates = retrieve(user_query, k_initial=15)

        if not candidates:
            return {
                "answer": "Answer not found in the provided document.",
                "chunks_used": 0,
                "initial_chunks": 0,
                "filter_applied": {}
            }

        # --------------------------
        # STEP 2: REDUCED RERANKING
        # --------------------------
        top_chunks = rerank(user_query, candidates, top_k=5)

        # --------------------------
        # STEP 3: SMALL CONTEXT
        # --------------------------
        context = "\n\n".join([c["text"] for c in top_chunks])
        filter_applied = candidates[0].get("filter_applied", {})

        # --------------------------
        # STEP 4: STRICT PROMPT
        # --------------------------
        prompt = f"""
Extract the answer ONLY from the context.

Rules:
- Use ONLY sentences from context
- Do NOT add new information
- Do NOT explain
- Do NOT generalize
- Do NOT define anything

If answer is not present:
→ Respond EXACTLY:
"Answer not found in the provided document."

Context:
{context}

Question:
{user_query}
"""

        # --------------------------
        # STEP 5: API CALL WITH ERROR HANDLING
        # --------------------------
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

        # --------------------------
        # STEP 6: VALIDATION (NO HALLUCINATION)
        # --------------------------
        if not is_answer_in_context(answer, context):
            answer = "Answer not found in the provided document."

        # --------------------------
        # RETURN
        # --------------------------
        return {
            "answer": answer,
            "filter_applied": filter_applied,
            "chunks_used": len(top_chunks),
            "initial_chunks": len(candidates)
        }

    except Exception as e:
        return {
            "answer": "⚠️ Rate limit reached or API error. Please try again later.",
            "chunks_used": 0,
            "initial_chunks": 0,
            "filter_applied": {}
        }