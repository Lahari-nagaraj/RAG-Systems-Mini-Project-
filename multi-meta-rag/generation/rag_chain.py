import os
from dotenv import load_dotenv
import time
load_dotenv()

from groq import Groq
from retrieval.filtered_retriever import retrieve
from retrieval.reranker import rerank
from ingestion.arxiv_search import search_arxiv_by_query
from ingestion.ingest import ingest_arxiv_paper

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a research assistant. Answer questions based ONLY on the provided research paper excerpts.
If the answer is not in the context, say "I couldn't find this in the available papers."
Always mention which paper/source your answer comes from."""


def answer_query(user_query: str) -> dict:
    # Step 1: try local retrieval first
    candidates = retrieve(user_query, k_initial=20)
    top_chunks = rerank(user_query, candidates, top_k=6)

    auto_ingested = []

    # Step 2: if nothing found locally, search arxiv automatically
    if not top_chunks:
        try:
            papers = search_arxiv_by_query(user_query, max_results=2)
            for paper in papers:
                try:
                    ingest_arxiv_paper(paper["arxiv_id"])
                    auto_ingested.append(paper["title"])
                    time.sleep(2)  # wait between ingestions
                except Exception as e:
                    print(f"Failed to ingest {paper['arxiv_id']}: {e}")
        except Exception as e:
            print(f"ArXiv search failed: {e}")

        # retry retrieval after ingestion
        if auto_ingested:
            candidates = retrieve(user_query, k_initial=20)
            top_chunks = rerank(user_query, candidates, top_k=6)

    # Step 3: if still nothing
    if not top_chunks:
        return {
            "answer": "No relevant papers found even after searching ArXiv. Try rephrasing your query.",
            "sources": [],
            "filter_applied": {},
            "chunks_used": 0,
            "auto_ingested": [],
        }

    # Step 4: build context and generate answer
    context = "\n\n---\n\n".join([
        f"[{c['metadata'].get('source', 'Unknown')}]\n{c['text']}"
        for c in top_chunks
    ])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    sources = list({c["metadata"].get("source", "Unknown") for c in top_chunks})
    filter_applied = candidates[0]["filter_applied"] if candidates else {}

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "filter_applied": filter_applied,
        "chunks_used": len(top_chunks),
        "auto_ingested": auto_ingested,
    }