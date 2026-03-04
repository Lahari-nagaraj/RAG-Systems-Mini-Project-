from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from llm.ollama_client import OllamaLLM
from rag.naive_pipeline import NaiveRAG

# 1️⃣ Load PDF
text = load_pdf("sample.pdf")

# 2️⃣ Break into chunks
chunks = chunk_text(text)

# 3️⃣ Create embedding model
embedding_model = EmbeddingModel()

# 4️⃣ Create FAISS vector store
dimension = embedding_model.encode(["test"]).shape[1]
vectorstore = FAISSStore(dimension)

# 5️⃣ Convert chunks into embeddings
embeddings = embedding_model.encode(chunks)

# 6️⃣ Store embeddings
metadata = [{"source": "sample.pdf"}] * len(chunks)
vectorstore.add(embeddings, chunks, metadata)

# 7️⃣ Create LLM (Mistral)
llm = OllamaLLM(model="mistral")

# 8️⃣ Create Naive RAG pipeline
rag = NaiveRAG(embedding_model, vectorstore, llm)

# 9️⃣ Ask question
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

print("\n=== ANSWER ===\n")
print(answer)