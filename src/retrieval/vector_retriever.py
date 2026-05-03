"""
ChromaDB Vector Search Module
Handles semantic search over agricultural documents using embeddings
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

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


class VectorStore:
    """
    Vector database for semantic search using ChromaDB
    Stores document embeddings and enables similarity-based retrieval
    """
    
    def __init__(
        self,
        collection_name: str = "agriculture_corpus",
        embedding_model: str = "all-MiniLM-L6-v2",
        persistence_dir: str = None,
        persist: bool = True
    ):
        """
        Initialize ChromaDB Vector Store
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence-transformer model for embeddings
            persistence_dir: Directory for persistent storage
            persist: Whether to persist data to disk
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist = persist
        
        # Setup persistence directory
        self.persistence_dir = Path(persistence_dir or settings.chromadb.persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._initialize_client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"VectorStore initialized: {collection_name}")
        logger.info(f"Embedding model: {embedding_model}")
        logger.info(f"Persistence: {self.persistence_dir}")
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence"""
        if chromadb is None:
            raise ImportError(
                "chromadb is required for VectorStore. Install dependencies with "
                "`pip install -r requirements.txt`, or run `python local_demo.py` "
                "for the dependency-light demo."
            )

        if self.persist:
            self.client = chromadb.PersistentClient(
                path=str(self.persistence_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client()
        
        logger.info("ChromaDB client initialized")
    
    def add_documents(
        self,
        documents: List[Dict],
        document_source: str = "unknown",
        batch_size: int = 100
    ) -> int:
        """
        Add documents to vector store
        
        Args:
            documents: List of document dictionaries with 'content' and optional metadata
            document_source: Source of the documents
            batch_size: Batch size for adding documents
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents provided to add")
            return 0
        
        # Prepare documents for ChromaDB
        ids = []
        contents = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            doc_id = f"{document_source}_{idx}"
            ids.append(doc_id)
            
            # Extract content
            if isinstance(doc, dict):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
            else:
                # Assume it's a Document object from data_loader
                content = doc.content
                metadata = doc.metadata or {}
            
            contents.append(content)
            
            # Prepare metadata
            full_metadata = {
                "source": document_source,
                "source_doc": doc.source if hasattr(doc, 'source') else document_source,
                "page": str(doc.page) if hasattr(doc, 'page') and doc.page else "unknown",
                **metadata
            }
            metadatas.append(full_metadata)
        
        # Add documents in batches
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_contents,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                logger.info(f"Added {len(batch_ids)} documents (batch {i//batch_size + 1})")
                
            except Exception as e:
                logger.error(f"Error adding document batch: {str(e)}")
                continue
        
        logger.info(f"Total documents added to vector store: {total_added}")
        return total_added
    
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
            where_filter: Optional ChromaDB where filter
            
        Returns:
            List of SearchResult objects
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query provided")
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            search_results = []
            
            # Process results
            if results and results['ids'] and len(results['ids']) > 0:
                for idx, doc_id in enumerate(results['ids'][0]):
                    content = results['documents'][0][idx] if results['documents'] else ""
                    distance = results['distances'][0][idx] if results['distances'] else 0
                    metadata = results['metadatas'][0][idx] if results['metadatas'] else {}
                    
                    # Convert distance to similarity score (0-1, higher is better)
                    # ChromaDB with cosine uses distances, so convert
                    similarity_score = 1 - (distance / 2) if distance is not None else 0.5
                    
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
            logger.error(f"Error during search: {str(e)}")
            return []
    
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
        where_filter = {"source_doc": {"$eq": source}}
        return self.search(query, top_k=top_k, where_filter=where_filter)
    
    def search_multi_query(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """
        Perform multiple searches
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k=top_k)
        
        return results
    
    def delete_documents(self, document_source: str) -> bool:
        """
        Delete all documents from a specific source
        
        Args:
            document_source: Source document name
            
        Returns:
            True if successful
        """
        try:
            # Note: ChromaDB doesn't have direct delete by filter,
            # so we need to get all docs from source and delete them
            where_filter = {"source_doc": {"$eq": document_source}}
            results = self.collection.get(where=where_filter)
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents from source: {document_source}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection
        
        Returns:
            Collection information
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "persistence_dir": str(self.persistence_dir),
                "persistent": self.persist
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def persist_collection(self):
        """Persist collection to disk"""
        if self.persist:
            try:
                self.client.persist()
                logger.info("Collection persisted to disk")
            except Exception as e:
                logger.error(f"Error persisting collection: {str(e)}")
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from collection
        
        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def get_documents(self, limit: int = None) -> List[Dict]:
        """
        Get documents from collection
        
        Args:
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        try:
            results = self.collection.get(limit=limit)
            
            documents = []
            if results and results['ids']:
                for idx, doc_id in enumerate(results['ids']):
                    doc = {
                        'id': doc_id,
                        'content': results['documents'][idx] if results['documents'] else "",
                        'metadata': results['metadatas'][idx] if results['metadatas'] else {}
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def export_to_json(self, output_path: str) -> bool:
        """
        Export collection to JSON file
        
        Args:
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            documents = self.get_documents()
            
            export_data = {
                "collection_info": self.get_collection_info(),
                "documents": documents
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collection exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting collection: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="agriculture_corpus",
        persistence_dir="data/chroma_db"
    )
    
    # Example documents
    sample_docs = [
        {
            'content': 'Urea is a nitrogen-rich fertilizer used for rice. However, it is contraindicated for fungal blast disease.',
            'metadata': {'crop': 'rice', 'disease': 'fungal_blast'}
        },
        {
            'content': 'Proper drainage and resistant varieties prevent fungal blast in rice.',
            'metadata': {'crop': 'rice', 'practice': 'prevention'}
        },
        {
            'content': 'Potassium application improves cotton yield and disease resistance.',
            'metadata': {'crop': 'cotton', 'nutrient': 'potassium'}
        }
    ]
    
    # Add documents
    # vector_store.add_documents(sample_docs, document_source="example_doc.pdf")
    
    # Search
    # results = vector_store.search("What fertilizer for rice?", top_k=2)
    # for result in results:
    #     print(f"Score: {result.score:.2f} - {result.content[:100]}...")
    
    # Get info
    print(json.dumps(vector_store.get_collection_info(), indent=2))
