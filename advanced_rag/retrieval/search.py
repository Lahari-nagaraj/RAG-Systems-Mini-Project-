from llm.embedder import get_embedding
from vectorstore.chroma_db import query_documents


def retrieve(query, k=3):
    # convert query → embedding
    query_embedding = get_embedding(query)

    # search vector DB
    results = query_documents(query_embedding, k)

    # return only documents (chunks)
    return results