import os
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch
from langchain_groq import ChatGroq

load_dotenv()

if __name__ == "__main__":

    store = FaissVectorStore("faiss_store")

    if not os.path.exists("faiss_store/faiss.index"):
        print("[INFO] Building FAISS index (first time)...")
        docs = load_all_documents("data")   #Only load when needed
        store.build_from_documents(docs)
    else:
        print("[INFO] Loading existing FAISS index...")
        store.load()

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    rag = RAGSearch(store, llm)

    query = "Explain the Consulting Room Games from Games people play book"
    result = rag.search_and_summarize(query)

    print("Summary:", result)