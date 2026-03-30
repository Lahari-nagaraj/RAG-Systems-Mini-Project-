import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer
from retrieval.metadata_extractor import extract_metadata_filter

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "research_papers"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def build_qdrant_filter(metadata_filter: dict):
    if not metadata_filter:
        return None

    conditions = []
    for field, condition in metadata_filter.items():
        if isinstance(condition, dict) and "$in" in condition:
            values = condition["$in"]
            conditions.append(
                FieldCondition(key=field, match=MatchAny(any=values))
            )

    if not conditions:
        return None

    return Filter(must=conditions)


def retrieve(query: str, k_initial: int = 20) -> list:
    metadata_filter = extract_metadata_filter(query)
    qdrant_filter = build_qdrant_filter(metadata_filter)

    query_vector = embedder.encode(query).tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=k_initial,
        with_payload=True,
    )

    return [
        {
            "text": r.payload.get("text", ""),
            "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            "score": r.score,
            "filter_applied": metadata_filter,
        }
        for r in results
    ]