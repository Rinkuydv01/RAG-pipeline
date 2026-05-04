import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # ================= BUILD =================
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # ✅ FIX: include source + page info
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", None)
            })

        self.add_embeddings(
            np.array(embeddings).astype("float32"),
            metadatas
        )

        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    # ================= ADD =================
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS.")

    # ================= SAVE =================
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved index + metadata to {self.persist_dir}")

    # ================= LOAD =================
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError("FAISS index not found. Build first.")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded FAISS index and metadata")

    # ================= SEARCH =================
    def search(self, query_embedding: np.ndarray, top_k: int = 5, source_filter: str = None):
        D, I = self.index.search(query_embedding, top_k * 3)  
        # ↑ get more results, we'll filter later

        results = []

        for idx, dist in zip(I[0], D[0]):
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            # ✅ FILTER (this is the real fix)
            if source_filter and source_filter not in meta["source"]:
                continue

            results.append({
                "index": idx,
                "distance": float(dist),
                "metadata": meta
            })

            if len(results) >= top_k:
                break

        return results

    # ================= QUERY =================
    def query(self, query_text: str, top_k: int = 5, source_filter: str = None):
        print(f"[INFO] Query: '{query_text}'")

        query_emb = self.model.encode([query_text]).astype("float32")

        return self.search(
            query_emb,
            top_k=top_k,
            source_filter=source_filter
        )


# ================= TEST =================
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")

    store.build_from_documents(docs)
    store.load()

    results = store.query(
        "What is a Rake in 'The Art of Seduction' by Robert Greene?",
        top_k=3
    )

    for r in results:
        print("\nSOURCE:", r["metadata"]["source"])
        print("TEXT:", r["metadata"]["text"][:200])