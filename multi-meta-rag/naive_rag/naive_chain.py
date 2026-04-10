from groq import Groq
import os
from dotenv import load_dotenv
from naive_rag.naive_retrieve import retrieve_naive

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def naive_answer(query):
    docs = retrieve_naive(query)

    context = "\n\n".join(docs)

    prompt = f"""
You are a helpful assistant.

Use the context if relevant.

If the context is not helpful:
→ Answer using your general knowledge.

ALWAYS provide a meaningful answer.

DO NOT output metadata or JSON.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()