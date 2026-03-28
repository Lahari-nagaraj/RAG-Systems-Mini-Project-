import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from ingestion.pipeline import process_pdf
from services.chat_service import chat

st.title("📚 Advanced RAG Research Assistant")

st.write("Upload a PDF and ask questions!")

# Upload PDF
uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("PDF uploaded!")

    if st.button("Process PDF"):
        process_pdf("temp.pdf")
        st.success("PDF processed and stored!")

# Ask question
query = st.text_input("Ask a question about the paper:")

if query:
    with st.spinner("Thinking..."):
        answer = chat(query)
    st.write("### Answer:")
    st.write(answer)