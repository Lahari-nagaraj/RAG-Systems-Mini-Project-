from pypdf import PdfReader
from llm.embedder import get_embedding
from vectorstore.chroma_db import add_documents
import os
import requests


# -------------------------------
# EXISTING: Upload PDF pipeline
# -------------------------------
def process_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text)

    embeddings = [get_embedding(chunk) for chunk in chunks]

    metadata = {
        "source": os.path.basename(file_path),
        "type": "paper"
    }

    add_documents(chunks, embeddings, metadata)


# -------------------------------
# NEW: arXiv PDF pipeline
# -------------------------------
def process_arxiv_pdf(pdf_url):
    temp_path = "temp_arxiv.pdf"

    # download pdf
    response = requests.get(pdf_url)
    with open(temp_path, "wb") as f:
        f.write(response.content)

    reader = PdfReader(temp_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text)

    embeddings = [get_embedding(chunk) for chunk in chunks]

    metadata = {
        "source": pdf_url,
        "type": "arxiv"
    }

    add_documents(chunks, embeddings, metadata)

    # cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# -------------------------------
# COMMON: chunking logic
# -------------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks