import streamlit as st
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from llm.ollama_client import OllamaLLM
from rag.naive_pipeline import NaiveRAG

@st.cache_resource
def initialize_system():
    embedding_model = EmbeddingModel()
    llm = OllamaLLM()
    dimension = embedding_model.encode(["test"]).shape[1]
    vectorstore = FAISSStore(dimension)

    return embedding_model, vectorstore, llm

embedding_model, vectorstore, llm = initialize_system()
rag_system = NaiveRAG(embedding_model, vectorstore, llm)

st.title("Naive RAG Research Baseline")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = chunk_text(text)

    embeddings = embedding_model.encode(chunks)

    metadata = [{"source": uploaded_file.name}] * len(chunks)

    vectorstore.add(embeddings, chunks, metadata)

    st.success("Document indexed successfully!")

query = st.text_input("Ask a question")

if st.button("Search"):
    answer = rag_system.run(query)
    st.write("### Answer")
    st.write(answer)