"""
FAISS Vector Search Module
Handles semantic search over agricultural documents using FAISS (Facebook AI Similarity Search)
More efficient than ChromaDB for production use.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a vector search result"""
    content: str
    document: str
    score: float
    page: Optional[int] = None
    metadata: Optional[Dict] = None


class FAISSVectorStore:
    """
    Vector database for semantic search using FAISS
    Stores document embeddings and enables similarity-based retrieval
    More memory-efficient than ChromaDB for large datasets.
    """
    
    def __init__(
        self,
        collection_name: str = "agriculture_corpus",
        embedding_model: str = "all-MiniLM-L6-v2",
        persistence_dir: str = None,
        index_type: str = "Flat"  # Flat, IVF, HNSW
    ):
        """
        Initialize FAISS Vector Store
        
        Args:
            collection_name: Name for the vector store
            embedding_model: Sentence-transformer model for embeddings
            persistence_dir: Directory for persistent storage
            index_type: FAISS index type (Flat=exact, IVF=clustered, HNSW=graph-based)
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        
        # Setup persistence directory
        self.persistence_dir = Path(persistence_dir or "data/faiss_db")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Files for persistence
        self.index_path = self.persistence_dir / f"{collection_name}.index"
        self.documents_path = self.persistence_dir / f"{collection_name}_docs.pkl"
        self.metadata_path = self.persistence_dir / f"{collection_name}_meta.pkl"
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index
        self.documents = []  # Store original documents
        self.metadatas = []  # Store metadata
        self.index = None
        self._initialize_index()
        
        logger.info(f"FAISS VectorStore initialized: {collection_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Documents in index: {len(self.documents)}")
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        if self.index_path.exists():
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            
            # Load documents and metadata
            if self.documents_path.exists():
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadatas = pickle.load(f)
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
    
    def _embed_documents(self, texts: List[str]) -> "np.ndarray":
        """Generate embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings
    
    def add_documents(
        self,
        documents: List[Dict],
        document_source: str = "unknown",
        batch_size: int = 32
    ) -> int:
        """
        Add documents to FAISS vector store
        
        Args:
            documents: List of document dictionaries with 'content' and optional metadata
            document_source: Source of the documents
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents provided to add")
            return 0
        
        # Prepare documents
        contents = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            # Extract content
            if isinstance(doc, dict):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
            else:
                # Assume it's a Document object from data_loader
                content = doc.content if hasattr(doc, 'content') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            contents.append(content)
            
            # Prepare metadata
            full_metadata = {
                "source": document_source,
                "source_doc": getattr(doc, 'source', document_source) if hasattr(doc, 'source') else document_source,
                "page": str(getattr(doc, 'page', 'unknown')) if hasattr(doc, 'page') else "unknown",
                **metadata
            }
            metadatas.append(full_metadata)
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(contents)} documents...")
        all_embeddings = self._embed_documents(contents)
        
        # Add to FAISS index
        self.index.add(all_embeddings.astype(np.float32))
        
        # Store documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(contents)
        self.metadatas.extend(metadatas)
        
        # Persist to disk
        self._persist()
        
        logger.info(f"Total documents in FAISS index: {len(self.documents)}")
        return len(contents)
    
    def _persist(self):
        """Save FAISS index, documents, and metadata to disk"""
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadatas, f)
        
        logger.info(f"Persisted FAISS index to {self.persistence_dir}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            top_k: Number of top results to return
            where_filter: Optional metadata filter (not directly supported by FAISS, applied post-hoc)
            
        Returns:
            List of SearchResult objects
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query provided")
            return []
        
        if len(self.documents) == 0:
            logger.warning("No documents in index")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._embed_documents([query])
            
            # Search FAISS index
            top_k = min(top_k, len(self.documents))
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            search_results = []
            
            for idx, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
                if doc_idx < 0:  # FAISS returns -1 for empty slots
                    continue
                
                # Apply metadata filter if provided
                if where_filter and not self._matches_filter(self.metadatas[doc_idx], where_filter):
                    continue
                
                content = self.documents[doc_idx]
                metadata = self.metadatas[doc_idx]
                
                # Convert FAISS score (inner product) to similarity (0-1)
                similarity_score = float(score)
                
                result = SearchResult(
                    content=content,
                    document=metadata.get('source_doc', 'unknown'),
                    score=similarity_score,
                    page=metadata.get('page'),
                    metadata=metadata
                )
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Error during FAISS search: {str(e)}")
            return []
    
    def _matches_filter(self, metadata: Dict, where_filter: Dict) -> bool:
        """Check if metadata matches the filter"""
        for key, value in where_filter.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                # Handle operators like $eq
                if "$eq" in value:
                    if metadata[key] != value["$eq"]:
                        return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def search_by_source(
        self,
        query: str,
        source: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search within a specific document source
        
        Args:
            query: Query text
            source: Source document name
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        where_filter = {"source_doc": source}
        return self.search(query, top_k=top_k, where_filter=where_filter)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "index_type": type(self.index).__name__,
            "persistence_dir": str(self.persistence_dir)
        }
    
    def clear(self):
        """Clear all documents from the index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        self.metadatas = []
        self._persist()
        logger.info("Cleared FAISS index")
