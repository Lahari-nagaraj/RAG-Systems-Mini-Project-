from llm.ollama_client import generate_response
from retrieval.search import retrieve   # ⚠️ make sure this exists
from services.paper_service import search_papers


# -------------------------------
# EXISTING FLOW (kept safe)
# -------------------------------
def chat(query):
    docs = retrieve(query)

    if not docs:
        return "No relevant context found."

    context = "\n\n".join(docs)

    prompt = f"""
    Use ONLY this context to answer:

    {context}

    Question:
    {query}
    """

    return generate_response(prompt)


# -------------------------------
# NEW: Paper search (no QA)
# -------------------------------
def get_papers(query):
    return search_papers(query)


# -------------------------------
# NEW: QA after selecting paper
# -------------------------------
def chat_with_paper(query):
    docs = retrieve(query)

    if not docs:
        return "No relevant context found."

    context = "\n\n".join(docs)

    prompt = f"""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    return generate_response(prompt)