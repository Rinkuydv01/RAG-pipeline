import os
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch
from langchain_groq import ChatGroq

load_dotenv()

if __name__ == "__main__":

    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")

    if not os.path.exists("faiss_store/faiss.index"):
        print("Building FAISS index...")
        store.build_from_documents(docs)
    else:
        print("Loading existing FAISS index...")
        store.load()

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"   
    )

    rag = RAGSearch(store, llm)

    query = "What is regression in the art of seduction"
    result = rag.search_and_summarize(query)

    print("Summary:", result)