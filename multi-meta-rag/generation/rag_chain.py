from groq import Groq
from retrieval.filtered_retriever import retrieve
from retrieval.reranker import rerank

client = Groq(api_key="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")
MODEL = "llama-3.1-8b-instant"


def answer_query(user_query: str) -> dict:
    try:
        # Increased to 20 initial candidates
        candidates = retrieve(user_query, k_initial=20)

        if not candidates:
            return {"answer": "Answer not found in the provided document.",
                    "chunks_used": 0, "initial_chunks": 0, "filter_applied": {}}

        # Rerank and keep top 5 (more context = better answers)
        top_chunks = rerank(user_query, candidates, top_k=5)
        context = "\n\n".join([c["text"][:500] for c in top_chunks])
        filter_applied = candidates[0].get("filter_applied", {})

        prompt = f"""You are a research assistant. Use the context below to answer the question clearly.
Answer in 2-4 sentences. If the topic is not mentioned at all in the context, say "Answer not found in the provided document."

Context:
{context}

Question: {user_query}
Answer:"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )

        answer = response.choices[0].message.content.strip()

        return {"answer": answer, "filter_applied": filter_applied,
                "chunks_used": len(top_chunks), "initial_chunks": len(candidates)}

    except Exception as e:
        return {"answer": f"⚠️ API error: {str(e)}. Please wait and try again.",
                "chunks_used": 0, "initial_chunks": 0, "filter_applied": {}}