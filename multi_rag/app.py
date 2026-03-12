import streamlit as st
from rag_pipeline import create_vectorstore, query_rag

st.title("Multi-RAG with Metadata Filtering")

if st.button("Initialize Database"):
    create_vectorstore()
    st.success("Vector database created!")

domain = st.selectbox(
    "Select domain",
    ["AI", "cybersecurity", "IoT"]
)

question = st.text_input("Ask a question")

if st.button("Get Answer"):
    result = query_rag(question, domain)
    st.write("Retrieved Information:")
    st.write(result)