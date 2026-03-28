from llm.openai_client import generate_response

def extract_metadata(query):
    prompt = f"""
    Extract metadata from this query:
    {query}

    Return JSON with:
    - topic
    - keywords
    """

    response = generate_response(prompt)

    return response