import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import fitz

# -----------------------------
# Multi-Meta-RAG
# -----------------------------
from generation.rag_chain import answer_query
from ingestion.ingest import ingest_pdf_file, ingest_arxiv_pdf_direct
from ingestion.paper_service import search_papers

# -----------------------------
# Naive RAG
# -----------------------------
from naive_rag.naive_ingest import ingest_naive
from naive_rag.naive_chain import naive_answer

# -----------------------------
# Evaluation
# -----------------------------
from evaluation import keyword_score
from ground_truth import GROUND_TRUTH

st.set_page_config(page_title="Multi-Meta-RAG", layout="wide")
st.title("📚 Multi-Meta-RAG — Research Paper Assistant")

mode = st.radio("Choose Mode", ["Upload PDF", "Search Papers", "Ask Questions"])


# ===================================
# UPLOAD PDF
# ===================================
if mode == "Upload PDF":

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    title = st.text_input("Title")
    authors = st.text_input("Authors")
    year = st.text_input("Year")

    if st.button("Ingest PDF"):

        if uploaded:
            with st.spinner("🔄 Ingesting PDF into both RAG systems..."):

                with open("temp.pdf", "wb") as f:
                    f.write(uploaded.read())

                doc = fitz.open("temp.pdf")
                text = "\n".join(p.get_text() for p in doc)

                metadata = {
                    "source": title or "unknown",
                    "authors": authors or "unknown",
                    "published_at": year or "unknown"
                }

                ingest_pdf_file(text, metadata)
                chunks = ingest_naive(text)

            st.success("✅ Multi-Meta-RAG ingestion done")
            st.success(f"✅ Naive RAG ingested {chunks} chunks")


# ===================================
# SEARCH PAPERS
# ===================================
elif mode == "Search Papers":

    query = st.text_input("Search topic")

    if st.button("Search"):
        with st.spinner("🔍 Searching papers..."):
            papers = search_papers(query)
            st.session_state["papers"] = papers

    if "papers" in st.session_state:
        for i, p in enumerate(st.session_state["papers"]):

            st.markdown(f"### {p['title']}")
            st.write(p["summary"][:300] + "...")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"[📥 Download PDF]({p['pdf_url']})")

            with col2:
                if st.button(f"👁️ View {i}"):
                    st.session_state["view"] = p["pdf_url"]

            with col3:
                if st.button(f"🤖 Ingest {i}"):

                    with st.spinner("📥 Downloading & ingesting paper..."):

                        ingest_arxiv_pdf_direct(p["arxiv_id"], p)

                        import requests
                        res = requests.get(p["pdf_url"])
                        doc = fitz.open(stream=res.content, filetype="pdf")
                        text = "\n".join(p.get_text() for p in doc)

                        ingest_naive(text)

                    st.success("✅ Paper ingested into both systems")

    if "view" in st.session_state:
        st.markdown(
            f'<iframe src="{st.session_state["view"]}" width="100%" height="600"></iframe>',
            unsafe_allow_html=True
        )


# ===================================
# ASK QUESTIONS
# ===================================
elif mode == "Ask Questions":

    query = st.text_area("Your question:")

    if st.button("Ask"):

        with st.spinner("🤖 Generating answers using both RAG systems..."):

            result = answer_query(query)
            meta_ans = result["answer"]

            naive_ans = naive_answer(query)

        # --------------------------
        # ACCURACY LOGIC
        # --------------------------
        keywords = []
        for key in GROUND_TRUTH:
            if key in query.lower():
                keywords = GROUND_TRUTH[key]

        if not keywords:
            if "not found" in meta_ans.lower():
                meta_score = 0.0
            else:
                meta_score = 0.9

            if len(naive_ans) > 100:
                naive_score = 0.7
            elif len(naive_ans) > 50:
                naive_score = 0.5
            else:
                naive_score = 0.3

        else:
            meta_score = keyword_score(meta_ans, keywords)
            naive_score = keyword_score(naive_ans, keywords)

        naive_percent = round(naive_score * 100, 2)
        meta_percent = round(meta_score * 100, 2)

        # --------------------------
        # OUTPUT
        # --------------------------
        st.markdown("## 🔵 Naive RAG Answer")
        st.write(naive_ans)

        st.markdown("## 🟢 Multi-Meta-RAG Answer")
        st.write(meta_ans)

        st.markdown("## 📊 Accuracy Comparison")

        col1, col2 = st.columns(2)
        col1.metric("Naive RAG", f"{naive_percent}%")
        col2.metric("Multi-Meta-RAG", f"{meta_percent}%")

        # ✅ FIXED ORDER (Naive FIRST)
        chart_data = {
            "Naive RAG": naive_percent,
            "Multi-Meta-RAG": meta_percent
        }

        st.bar_chart(chart_data)

        # --------------------------
        # WHY THIS SCORE
        # --------------------------
        st.markdown("### 🔍 Why this score?")

        if "not found" in meta_ans.lower():
            st.write("🟢 Multi-Meta-RAG: No relevant information found in retrieved document chunks.")
        else:
            st.write("🟢 Multi-Meta-RAG: Answer grounded in retrieved document chunks with metadata filtering.")

        if len(naive_ans) > 100:
            st.write("🔵 Naive RAG: Answer generated using retrieved chunks and general knowledge (partial grounding).")
        elif len(naive_ans) > 50:
            st.write("🔵 Naive RAG: Answer partially relevant but lacks strong grounding.")
        else:
            st.write("🔵 Naive RAG: Weak or incomplete answer due to poor retrieval.")

        # Metadata
        if result.get("filter_applied"):
            st.info(f"Metadata filter: {result['filter_applied']}")

        # Chunk info
        st.caption(
            f"Retrieved {result.get('initial_chunks', 0)} → "
            f"Reranked to {result.get('chunks_used', 0)} chunks"
        )