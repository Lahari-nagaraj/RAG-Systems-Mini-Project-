import requests
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"


def search_papers(query: str, max_results: int = 5) -> list:
    """Search ArXiv using XML API — no rate limiting issues."""
    url = f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
    except Exception as e:
        return []

    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        try:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()

            # get arxiv ID from entry id URL
            entry_id = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
            arxiv_id = entry_id.split("/abs/")[-1]

            # build PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            # get authors
            authors = []
            for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                name = author.find("{http://www.w3.org/2005/Atom}name")
                if name is not None:
                    authors.append(name.text.strip())

            # get published date
            published = entry.find("{http://www.w3.org/2005/Atom}published")
            published_date = published.text[:10] if published is not None else "unknown"

            papers.append({
                "title": title,
                "summary": summary,
                "pdf_url": pdf_url,
                "arxiv_id": arxiv_id,
                "authors": ", ".join(authors[:3]),
                "published_at": published_date[:4],  # just the year
            })
        except Exception:
            continue

    return papers