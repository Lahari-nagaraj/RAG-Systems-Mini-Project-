import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_DIM = 384
INDEX_FILE = "faiss_index.bin"
META_FILE = "faiss_meta.npy"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load or create index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

# Load metadata
if os.path.exists(META_FILE):
    data = np.load(META_FILE, allow_pickle=True).item()
    stored_texts = data["texts"]
    stored_metadata = data["metadata"]
else:
    stored_texts = []
    stored_metadata = []


def save_index():
    faiss.write_index(index, INDEX_FILE)
    np.save(META_FILE, {
        "texts": stored_texts,
        "metadata": stored_metadata
    })


def ingest_pdf_file(text: str, metadata: dict) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # increased
        chunk_overlap=100,        # increased
        separators=["\n\n", "\n", ". ", " "]  # better structure
    )

    chunks = splitter.split_text(text)
    embeddings = embedder.encode(chunks)

    global stored_texts, stored_metadata

    index.add(np.array(embeddings).astype("float32"))

    stored_texts.extend(chunks)
    stored_metadata.extend([
        {**metadata, "chunk_index": i}
        for i in range(len(chunks))
    ])

    save_index()

    return f"Ingested {len(chunks)} chunks"


def ingest_arxiv_pdf_direct(arxiv_id: str, metadata: dict) -> str:
    import requests
    import fitz

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    response = requests.get(pdf_url, timeout=30)
    if response.status_code != 200:
        raise Exception("Failed to download PDF")

    doc = fitz.open(stream=response.content, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    return ingest_pdf_file(text, metadata)