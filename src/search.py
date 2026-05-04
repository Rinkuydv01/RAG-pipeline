class RAGSearch:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
Answer strictly based on the context.
Define the concept clearly.
Do not include unrelated concepts.
Keep the answer concise. '{query}'

Context:
{context}

Summary:
"""

        response = self.llm.invoke(prompt)
        return response.content