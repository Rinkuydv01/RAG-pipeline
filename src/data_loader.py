import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Any

def load_all_documents(pdf_directory: str) -> List[Any]:
    """Process all PDFs in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            # Add source info to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)

            print(f"✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")

    return all_documents
