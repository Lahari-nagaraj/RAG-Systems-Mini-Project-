from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from llm.ollama_client import OllamaLLM
from rag.naive_pipeline import NaiveRAG

# Step 1: Load document
text = load_pdf("sample.pdf")   # Put a small research PDF in project folder

# Step 2: Chunk it

chunks = chunk_text(text)

# Step 3: Create embedding model
embedding_model = EmbeddingModel()

# Step 4: Create vector store
dimension = embedding_model.encode(["test"]).shape[1]
vectorstore = FAISSStore(dimension)

# Step 5: Embed chunks
embeddings = embedding_model.encode(chunks)

# Step 6: Add to FAISS
metadata = [{"source": "sample.pdf"}] * len(chunks)
vectorstore.add(embeddings, chunks, metadata)

# Step 7: Create LLM
llm = OllamaLLM(model="mistral")

# Step 8: Create RAG pipeline
rag = NaiveRAG(embedding_model, vectorstore, llm)

# Step 9: Ask question
questions = [
    "What are the main contributions of the paper?",
    "What are the three limitations of rule-based virtual assistants identified in the paper?",
    "What are the three major workflows of the RAGVA framework?",
    "What challenges are identified in engineering RAG-based systems?",
    "Who founded Google?"
]

for q in questions:
    print("\n==============================")
    print(f"QUESTION: {q}")
    print("==============================")

    answer = rag.run(q)

    print("\n=== ANSWER ===\n")
    print(answer)