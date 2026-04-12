import re


def extract_metadata_filter(query: str) -> dict:
    """
    Extract metadata filters using regex — NO API CALL needed.
    Saves tokens and avoids rate limits.
    """
    result = {}
    q = query.lower()

    # Extract year (4-digit number like 2023, 2024)
    year_match = re.findall(r'\b(20\d{2})\b', query)
    if year_match:
        result["published_at"] = {"$in": year_match}

    # Extract arxiv ID pattern like 2406.13213
    arxiv_match = re.findall(r'\b(\d{4}\.\d{4,5})\b', query)
    if arxiv_match:
        result["arxiv_id"] = {"$in": arxiv_match}

    # Extract quoted titles
    title_match = re.findall(r'"([^"]+)"', query)
    if title_match:
        result["source"] = {"$in": title_match}

    # Common author patterns: "by Smith", "Smith et al", "Smith and Lee"
    author_match = re.findall(
        r'\b(?:by|author|paper by)\s+([A-Z][a-z]+)', query
    )
    et_al = re.findall(r'([A-Z][a-z]+)\s+et\s+al', query)
    all_authors = author_match + et_al
    if all_authors:
        result["authors"] = {"$in": all_authors}

    return result