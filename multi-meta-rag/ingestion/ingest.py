import os
import uuid
import time
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.arxiv_loader import fetch_and_chunk_arxiv

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "research_papers"
EMBEDDING_DIM = 384
BATCH_SIZE = 50

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=60
)


def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    fields_to_index = ["source", "authors", "published_at", "arxiv_id", "categories"]
    existing_indexes = client.get_collection(COLLECTION_NAME).payload_schema.keys()

    for field in fields_to_index:
        if field not in existing_indexes:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass


def upsert_in_batches(points: list):
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        time.sleep(0.5)


def encode_in_batches(texts: list) -> list:
    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        batch_embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def ingest_arxiv_paper(arxiv_id: str) -> str:
    ensure_collection()
    chunks = fetch_and_chunk_arxiv(arxiv_id)

    texts = [c["text"] for c in chunks]
    embeddings = encode_in_batches(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": chunk["text"], **chunk["metadata"]}
        )
        for emb, chunk in zip(embeddings, chunks)
    ]

    upsert_in_batches(points)
    return f"Ingested {len(points)} chunks from arxiv:{arxiv_id}"


def ingest_pdf_file(text: str, metadata: dict) -> str:
    ensure_collection()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = encode_in_batches(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": chunk, **metadata, "chunk_index": i}
        )
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
    ]

    upsert_in_batches(points)
    return f"Ingested {len(points)} chunks"