# RAG-based Document QA System

A Retrieval-Augmented Generation (RAG) pipeline to query unstructured documents (PDFs, text files) using semantic search and LLM-based answer generation.

## Tech Stack
Python, FAISS, LangChain, Sentence Transformers, Groq LLM

## Features
- Document ingestion, chunking, and embedding generation  
- FAISS-based vector store for semantic retrieval  
- Context-aware answer generation using Groq LLM  
- Modular pipeline (data loading, embeddings, search, generation)

## Setup
git clone https://github.com/Rinkuydv01/RAG-pipeline.git  
cd RAG-pipeline  
python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

Create a `.env` file:
GROQ_API_KEY=your_api_key

## Run
python app.py

## Notes
- Place input documents in the `data/` folder  
- FAISS index is created automatically on first run  
