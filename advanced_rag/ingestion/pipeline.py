from pypdf import PdfReader
from llm.embedder import get_embedding
from vectorstore.chroma_db import add_documents
import os

def process_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [text[i:i+800] for i in range(0, len(text), 800)]

    embeddings = [get_embedding(chunk) for chunk in chunks]

    metadata = {
        "source": os.path.basename(file_path),
        "type": "paper"
    }

    add_documents(chunks, embeddings, metadata)