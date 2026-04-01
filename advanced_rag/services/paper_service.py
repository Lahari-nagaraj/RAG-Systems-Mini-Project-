import requests
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

def search_papers(query, max_results=5):
    url = f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url)
    root = ET.fromstring(response.content)

    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()

        pdf_link = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")

        papers.append({
            "title": title,
            "summary": summary,
            "pdf_url": pdf_link
        })

    return papers