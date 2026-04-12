from groq import Groq
import os
from dotenv import load_dotenv
from naive_rag.naive_retrieve import retrieve_naive

load_dotenv()

client = Groq(api_key="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")
MODEL = "llama-3.1-8b-instant"  # CHANGED from 70b to avoid rate limits


def naive_answer(query):
    docs = retrieve_naive(query)

    # Limit context size
    context = "\n\n".join(d[:500] for d in docs[:3])

    prompt = f"""Use the context if relevant. Otherwise use general knowledge. Be concise.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()