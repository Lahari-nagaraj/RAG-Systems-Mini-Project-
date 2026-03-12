import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# ---------------- Base Directory ----------------
BASE_DIR = os.path.dirname(__file__)

# ---------------- Load LLM once ----------------
generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    device=-1
)

# ---------------- Load PDFs ----------------
def load_pdfs(pdf_paths):
    documents = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    return documents

# ---------------- Split Documents ----------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    # Add page number to metadata if missing
    for i, doc in enumerate(docs):
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1

    return docs

# ---------------- Create Vector Store ----------------
def create_vectorstore(pdf_paths):
    documents = load_pdfs(pdf_paths)
    docs = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = os.path.join(BASE_DIR, "db")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )

    vectordb.persist()
    return vectordb

# ---------------- Query RAG System ----------------
def query_rag(question, top_k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=os.path.join(BASE_DIR, "db"),
        embedding_function=embeddings
    )

    # Retrieve relevant chunks
    docs = vectordb.similarity_search(question, k=top_k)

    if not docs:
        return "No relevant documents found.", []

    # Clean and join context (limit to first 3 chunks to avoid LLM truncation)
    context = "\n\n".join([doc.page_content.replace("\n", " ") for doc in docs[:3]])

    # Structured prompt for concise 2-3 sentence summary
    prompt = f"""
Summarize the context below in 2-3 sentences to answer the question concisely.
Highlight the key points. Do NOT repeat the question or instructions.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt, max_length=200)
    answer = result[0]["generated_text"].strip()

    return answer, docs