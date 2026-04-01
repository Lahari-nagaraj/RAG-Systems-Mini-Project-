import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from ingestion.pipeline import process_pdf, process_arxiv_pdf
from services.chat_service import chat, chat_with_paper, get_papers
from vectorstore.chroma_db import clear_collection

st.set_page_config(page_title="RAG Research Assistant", layout="wide")

st.title("📚 Advanced RAG Research Assistant")

# -----------------------------------
# MODE SELECTION
# -----------------------------------
mode = st.radio("Choose Mode", ["Upload PDF", "Search Papers"])


# ===================================
# 🔵 MODE 1: UPLOAD PDF (EXISTING)
# ===================================
if mode == "Upload PDF":
    st.subheader("Upload your own PDF")

    uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.success("PDF uploaded!")

        if st.button("Process PDF"):
            clear_collection()
            process_pdf("temp.pdf")
            st.success("PDF processed and stored!")

    query = st.text_input("Ask a question about the uploaded paper:")

    if query:
        with st.spinner("Thinking..."):
            answer = chat(query)
        st.write("### Answer:")
        st.write(answer)


# ===================================
# 🟢 MODE 2: SEARCH PAPERS (NEW)
# ===================================
elif mode == "Search Papers":
    st.subheader("Search Research Papers (arXiv)")

    search_query = st.text_input("Enter topic")

    if st.button("Search"):
        papers = get_papers(search_query)
        st.session_state["papers"] = papers

    # -------------------------------
    # DISPLAY PAPERS
    # -------------------------------
    if "papers" in st.session_state:
        for i, paper in enumerate(st.session_state["papers"]):
            st.markdown(f"### {paper['title']}")
            st.write(paper["summary"][:300] + "...")

            col1, col2, col3 = st.columns(3)

            # 📥 DOWNLOAD
            with col1:
                if paper["pdf_url"]:
                    st.markdown(f"[📥 Download PDF]({paper['pdf_url']})")

            # 👁️ VIEW
            with col2:
                if st.button(f"View Paper {i}"):
                    st.session_state["view_pdf"] = paper["pdf_url"]

            # 🤖 SELECT
            with col3:
                if st.button(f"Select Paper {i}"):
                    clear_collection()
                    process_arxiv_pdf(paper["pdf_url"])
                    st.session_state["selected_paper"] = True
                    st.success("Paper loaded! Now ask questions below.")

    # -------------------------------
    # PDF VIEWER
    # -------------------------------
    if "view_pdf" in st.session_state:
        st.subheader("📄 Paper Viewer")

        pdf_url = st.session_state["view_pdf"]

        # Try embed viewer
        st.markdown(
            f"""
            <iframe src="{pdf_url}" width="100%" height="600px"></iframe>
            """,
            unsafe_allow_html=True
        )

        # fallback
        st.markdown(f"[🔗 Open Full Paper in New Tab]({pdf_url})")

    # -------------------------------
    # QA FOR SELECTED PAPER
    # -------------------------------
    if st.session_state.get("selected_paper"):
        question = st.text_input("Ask a question about this paper:")

        if st.button("Ask"):
            with st.spinner("Thinking..."):
                answer = chat_with_paper(question)

            st.write("### Answer:")
            st.write(answer)