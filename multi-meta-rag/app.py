import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import fitz

from generation.rag_chain import answer_query
from ingestion.ingest import ingest_arxiv_paper, ingest_pdf_file
from ingestion.arxiv_search import search_arxiv_by_query
from retrieval.filtered_retriever import retrieve
from retrieval.reranker import rerank

st.set_page_config(page_title="Multi-Meta-RAG", page_icon="📚", layout="wide")
st.title("📚 Multi-Meta-RAG — Research Paper Assistant")

tab1, tab2 = st.tabs(["💬 Ask Questions", "📥 Ingest Papers"])

with tab2:
    st.subheader("Add papers to the knowledge base")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Via ArXiv ID**")
        arxiv_id = st.text_input("ArXiv ID (e.g. 2406.13213)")
        if st.button("Ingest from ArXiv"):
            if not arxiv_id.strip():
                st.error("Please enter an ArXiv ID")
            else:
                with st.spinner("Downloading and indexing..."):
                    try:
                        msg = ingest_arxiv_paper(arxiv_id.strip())
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.markdown("**Via PDF Upload**")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        paper_title = st.text_input("Paper Title")
        paper_authors = st.text_input("Authors")
        paper_year = st.text_input("Year (e.g. 2024)")
        if st.button("Ingest PDF"):
            if not uploaded:
                st.error("Please upload a PDF file")
            elif not paper_title.strip():
                st.error("Please enter a paper title")
            else:
                with st.spinner("Processing PDF..."):
                    try:
                        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
                        text = "\n".join(p.get_text() for p in doc)
                        metadata = {
                            "source": paper_title.strip(),
                            "authors": paper_authors.strip() or "Unknown",
                            "published_at": paper_year.strip() or "unknown",
                        }
                        msg = ingest_pdf_file(text, metadata)
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab1:
    st.subheader("Ask anything about the ingested papers")

    st.markdown(
        "💡 **Tip:** Ask specific questions like "
        "*'What did the Multi-Meta-RAG paper say about filtering?'* "
        "or general ones like *'What is retrieval augmented generation?'*"
    )

    query = st.text_area(
        "Your question:",
        height=100,
        placeholder="e.g. What are the latest methods for multi-hop question answering?"
    )

    if st.button("Ask", type="primary") and query:
        auto_ingested = []

        with st.spinner("Searching knowledge base..."):
            try:
                candidates = retrieve(query, k_initial=20)
                top_chunks = rerank(query, candidates, top_k=6)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        # if nothing found locally, try ArXiv
        if not top_chunks:
            with st.spinner("Nothing found locally — searching ArXiv..."):
                try:
                    papers = search_arxiv_by_query(query, max_results=2)
                    if papers:
                        for paper in papers:
                            try:
                                with st.spinner(f"Ingesting: {paper['title'][:60]}..."):
                                    ingest_arxiv_paper(paper["arxiv_id"])
                                    auto_ingested.append(paper["title"])
                                    time.sleep(2)
                            except Exception as e:
                                st.warning(f"Could not ingest '{paper['title']}': {e}")
                        
                        # retry retrieval after ingestion
                        if auto_ingested:
                            try:
                                candidates = retrieve(query, k_initial=20)
                                top_chunks = rerank(query, candidates, top_k=6)
                            except Exception as e:
                                st.error(f"Retrieval error after ingestion: {e}")
                                st.stop()
                    else:
                        st.warning(
                            "⚠️ ArXiv returned no results — possibly rate limited. "
                            "Try again in 1 minute or upload a PDF manually."
                        )
                except Exception as e:
                    st.warning(f"⚠️ ArXiv search failed: {e}. Try uploading a PDF manually.")

        # show auto ingested notice
        if auto_ingested:
            st.warning(
                f"📥 No local results found. Auto-fetched from ArXiv: "
                f"**{', '.join(auto_ingested)}**"
            )

        if not top_chunks:
            st.error(
                "No relevant papers found. Please upload a paper first via the "
                "**Ingest Papers** tab."
            )
        else:
            with st.spinner("Generating answer..."):
                try:
                    result = answer_query(query)

                    # metadata filter box
                    if result["filter_applied"]:
                        st.info(
                            f"🔍 Metadata filter applied: `{result['filter_applied']}`"
                        )
                    else:
                        st.info("🔍 No metadata filter — full knowledge base searched")

                    # answer
                    st.markdown("### Answer")
                    st.write(result["answer"])

                    # sources and chunk count
                    if result["sources"]:
                        st.markdown(
                            f"**Sources used:** {', '.join(result['sources'])}"
                        )
                    st.caption(f"Retrieved from {result['chunks_used']} chunks")

                except Exception as e:
                    st.error(f"Error generating answer: {e}")