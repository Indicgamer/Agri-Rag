"""
Build FAISS Vector Store for Agri-RAG
Run this to build the FAISS index from documents.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader
from src.retrieval.faiss_retriever import FAISSVectorStore

print("="*60)
print("Building FAISS Vector Store")
print("="*60)

# Load documents
print("Loading documents...")
loader = DataLoader()
docs = loader.load_directory('data')
print(f"Loaded {len(docs)} documents")

# Build FAISS
print("Building FAISS index...")
faiss = FAISSVectorStore()

docs_list = [{'content': d.content, 'metadata': d.metadata} for d in docs]
added = faiss.add_documents(docs_list, 'Agriculture-CPG-2020.pdf')

print(f"Added {len(docs_list)} documents to FAISS")

# Stats
stats = faiss.get_stats()
print(f"\nFAISS Stats:")
print(f"  Total documents: {stats['total_documents']}")
print(f"  Persistence dir: {stats['persistence_dir']}")

print("\nFAISS Vector Store ready!")