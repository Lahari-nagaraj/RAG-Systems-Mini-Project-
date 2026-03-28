from llm.embedder import get_embedding
from vectorstore.chroma_db import query

def search_papers(query_text):
    embedding = get_embedding(query_text)

    results = query(embedding, k=5)

    return results["documents"]