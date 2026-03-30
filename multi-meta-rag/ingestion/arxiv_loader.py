import arxiv
import fitz
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def fetch_and_chunk_arxiv(arxiv_id: str) -> list:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))

    with tempfile.TemporaryDirectory() as tmpdir:
        paper.download_pdf(dirpath=tmpdir, filename="paper.pdf")
        pdf_path = os.path.join(tmpdir, "paper.pdf")
        text = extract_text_from_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=32,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_text(text)

    metadata_base = {
        "source": paper.title,
        "arxiv_id": arxiv_id,
        "authors": ", ".join(a.name for a in paper.authors[:5]),
        "published_at": str(paper.published.date()),
        "categories": ", ".join(paper.categories),
        "abstract": paper.summary[:500],
        "url": paper.entry_id,
    }

    return [
        {"text": chunk, "metadata": {**metadata_base, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]