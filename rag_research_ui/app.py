import streamlit as st
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

st.title("Research Paper Intelligence Engine")
st.write("Advanced RAG-powered research assistant")

with st.sidebar:
    st.header("📂 Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("🔍 Metadata Filters")

    year_range = st.slider(
        "Publication Year",
        2000, 2025,
        (2015, 2024)
    )

    author_name = st.text_input("Author Name")

    domain = st.selectbox(
        "Research Domain",
        ["All", "AI", "Machine Learning", "Cybersecurity", "IoT"]
    )

    st.divider()

    st.header("⚙ Retrieval Settings")

    top_k = st.slider("Top K Results", 1, 20, 5)

    strict_mode = st.toggle("Strict Citation Mode")

    st.title("📚 Research Paper Intelligence Engine")
st.caption("Evidence-backed answers powered by Advanced RAG")

query = st.text_area(
    "Ask a research-level question:",
    height=100,
    placeholder="Example: What are recent transformer optimizations for low-resource NLP?"
)

ask_button = st.button("🔎 Search")

if ask_button:

    st.subheader("🧠 Generated Answer")

    st.write("""
    This is where the RAG-generated answer will appear.
    It will include citations like [1], [2].
    """)

    st.divider()

    st.subheader("📄 Retrieved Papers")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Similarity Score", "0.87")

        with st.expander("Paper 1 - Attention is All You Need"):
            st.write("Authors: Vaswani et al.")
            st.write("Year: 2017")
            st.write("Retrieved Chunk: ...")

    with col2:
        st.metric("Similarity Score", "0.82")

        with st.expander("Paper 2 - BERT"):
            st.write("Authors: Devlin et al.")
            st.write("Year: 2018")
            st.write("Retrieved Chunk: ...")

    st.divider()

    st.subheader("📊 Grounding Status")

    if strict_mode:
        st.success("Answer fully grounded in retrieved documents ✅")
    else:
        st.warning("Partial grounding detected ⚠️")

