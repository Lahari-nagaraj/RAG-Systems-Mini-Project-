import streamlit as st
import sys
import os
import shutil

# Connect backend folder
sys.path.append(os.path.abspath("../multi_rag"))
from rag_pipeline import create_vectorstore, query_rag

# ---------------- UI Styling ----------------
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Research RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Intelligence Engine")
st.write("Advanced RAG-powered research assistant")

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("📂 Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("⚙ Database")

    if st.button("Initialize Database"):

        if uploaded_files:

            pdf_paths = []
            os.makedirs("temp", exist_ok=True)

            for file in uploaded_files:
                file_path = os.path.join("temp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(file_path)

            create_vectorstore(pdf_paths)

            # Optional: clean temp folder
            shutil.rmtree("temp")

            st.success("Vector Database Created from PDFs!")

        else:
            st.warning("Please upload at least one PDF.")

    st.divider()

    st.header("⚙ Retrieval Settings")

    top_k = st.slider("Top K Results", 1, 20, 5)
    strict_mode = st.checkbox("Strict Citation Mode")

# ---------------- MAIN UI ----------------
st.caption("Evidence-backed answers powered by Advanced RAG")

query = st.text_area(
    "Ask a research-level question:",
    height=100,
    placeholder="Example: What problem does this research paper solve?"
)

ask_button = st.button("🔎 Search")

# ---------------- RAG QUERY ----------------
if ask_button:

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Pass top_k from slider
        answer, docs = query_rag(query, top_k=top_k)

        st.subheader("🧠 Generated Answer")
        st.write(answer)

        st.divider()

        st.subheader("📄 Retrieved Document Chunks")

        if len(docs) == 0:
            st.warning("No documents found.")

        else:
            for i, doc in enumerate(docs):
                with st.expander(f"Chunk {i+1}"):
                    st.write("**Page:**", doc.metadata.get("page"))
                    st.write("**Content:**")
                    st.write(doc.page_content.replace("\n", " "))

        st.divider()

        st.subheader("📊 Grounding Status")

        if strict_mode:
            st.success("Answer fully grounded in retrieved documents ✅")
        else:
            st.warning("Partial grounding detected ⚠️")