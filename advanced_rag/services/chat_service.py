from llm.ollama_client import generate_response
from retrieval.search import search_papers

def chat(query):
    docs = search_papers(query)

    context = "\n".join(docs[0])

    prompt = f"""
    Use ONLY this context to answer:

    {context}

    Question:
    {query}
    """

    return generate_response(prompt)