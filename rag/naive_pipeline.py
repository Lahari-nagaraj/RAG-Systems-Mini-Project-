class NaiveRAG:
    def __init__(self, embedding_model, vectorstore, llm):
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.llm = llm

    def run(self, query, k=5):
        print("\n🔎 Encoding query...")
        query_embedding = self.embedding_model.encode([query])

        print("🔎 Searching vectorstore...")
        retrieved = self.vectorstore.search(query_embedding, k)

        print("\n--- Retrieved Chunks ---\n")
        for i, r in enumerate(retrieved):
            print(f"Chunk {i+1}:")
            print(r["text"][:300])   # show first 300 characters
            print("------")

        context = "\n\n".join([r["text"] for r in retrieved])

        prompt = f"""
    Answer the question using ONLY the provided context.
    If answer not found, say 'Insufficient information'.

    Context:
    {context}

    Question:
    {query}
    """

        print("\n🤖 Sending prompt to LLM...")
        answer = self.llm.generate(prompt)

        print("✅ LLM responded.\n")

        return answer