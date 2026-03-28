import chromadb
from config import CHROMA_COLLECTION

client = chromadb.Client()
collection = client.get_or_create_collection(CHROMA_COLLECTION)


def add_documents(chunks, embeddings, metadata):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[metadata],
            ids=[f"{metadata['source']}_{i}"]
        )


def query(query_embedding, filter=None, k=5):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=filter
    )