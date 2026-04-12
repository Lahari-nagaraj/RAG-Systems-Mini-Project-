import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

ARXIV_API = "http://export.arxiv.org/api/query"


def search_papers(query: str, max_results: int = 5) -> list:
    """Search ArXiv using XML API."""
    try:
        # Properly encode the query
        encoded_query = quote(query)
        url = f"{ARXIV_API}?search_query=all:{encoded_query}&start=0&max_results={max_results}"

        headers = {"User-Agent": "ResearchPaperAssistant/1.0"}
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"ArXiv API error: {response.status_code}")
            return []

        root = ET.fromstring(response.content)
    except Exception as e:
        print(f"Search error: {e}")
        return []

    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        try:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()

            entry_id = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
            arxiv_id = entry_id.split("/abs/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            authors = []
            for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                name = author.find("{http://www.w3.org/2005/Atom}name")
                if name is not None:
                    authors.append(name.text.strip())

            published = entry.find("{http://www.w3.org/2005/Atom}published")
            published_date = published.text[:10] if published is not None else "unknown"

            papers.append({
                "title": title,
                "summary": summary,
                "pdf_url": pdf_url,
                "arxiv_id": arxiv_id,
                "authors": ", ".join(authors[:3]),
                "published_at": published_date[:4],
            })
        except Exception as e:
            print(f"Parse error: {e}")
            continue

    return papers