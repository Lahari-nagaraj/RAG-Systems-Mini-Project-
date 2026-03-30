import arxiv
import time
from dotenv import load_dotenv
load_dotenv()


def search_arxiv_by_query(query: str, max_results: int = 3) -> list:
    """Search arxiv for papers matching a query, return metadata list."""
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=3
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    try:
        for paper in client.results(search):
            papers.append({
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors[:5]),
                "published_at": str(paper.published.date()),
                "abstract": paper.summary[:500],
                "url": paper.entry_id,
                "categories": ", ".join(paper.categories),
            })
            time.sleep(1)
    except Exception as e:
        print(f"ArXiv search error: {e}")

    return papers