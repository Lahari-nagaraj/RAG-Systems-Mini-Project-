import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

NAIVE_INDEX_FILE = "naive_faiss_index.bin"
NAIVE_TEXT_FILE  = "naive_texts.npy"

# FIX: persist to disk — original code kept everything in module-level variables
#      which are lost every time Streamlit re-runs the script (any widget click).
#      Multi-Meta-RAG already persisted; naive RAG was silently empty after reload.
if os.path.exists(NAIVE_INDEX_FILE):
    naive_index = faiss.read_index(NAIVE_INDEX_FILE)
    naive_texts = list(np.load(NAIVE_TEXT_FILE, allow_pickle=True))
else:
    naive_index = faiss.IndexFlatL2(384)
    naive_texts = []


def _save_naive():
    faiss.write_index(naive_index, NAIVE_INDEX_FILE)
    np.save(NAIVE_TEXT_FILE, naive_texts)


def ingest_naive(text: str) -> int:
    chunks = [text[i:i+800] for i in range(0, len(text), 700)]

    embeddings = embedder.encode(chunks)
    naive_index.add(np.array(embeddings).astype("float32"))
    naive_texts.extend(chunks)

    _save_naive()   # persist after every ingest
    return len(chunks)