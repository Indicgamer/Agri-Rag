"""
Hybrid Retriever Module
Combines Neo4j graph traversal and ChromaDB vector search for comprehensive retrieval
"""

import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import re

from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.vector_retriever import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Represents a hybrid retrieval result"""
    content: str
    source: str
    score: float
    source_type: str  # "graph" or "vector"
    metadata: Optional[Dict] = None
    relationships: Optional[List[Dict]] = None  # For graph results


class HybridRetriever:
    """
    Combines graph-based and vector-based retrieval
    Provides two-stage filtering:
    1. Retrieve from both Neo4j and ChromaDB
    2. Fuse and rank results
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        vector_weight: float = 0.4,
        graph_weight: float = 0.6,
        top_k_vector: int = 5,
        top_k_graph: int = 10,
        fusion_method: str = "weighted_average"
    ):
        """
        Initialize Hybrid Retriever
        
        Args:
            vector_store: ChromaDB vector store
            knowledge_graph: Neo4j knowledge graph
            vector_weight: Weight for vector search results (0.0-1.0)
            graph_weight: Weight for graph results (0.0-1.0)
            top_k_vector: Top K results from vector search
            top_k_graph: Top K results from graph search
            fusion_method: Method for fusing results ("weighted_average", "rrf", "max")
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        
        # Normalize weights
        total_weight = vector_weight + graph_weight
        self.vector_weight = vector_weight / total_weight
        self.graph_weight = graph_weight / total_weight
        
        self.top_k_vector = top_k_vector
        self.top_k_graph = top_k_graph
        self.fusion_method = fusion_method
        
        logger.info(f"HybridRetriever initialized")
        logger.info(f"Weights - Vector: {self.vector_weight:.2f}, Graph: {self.graph_weight:.2f}")
        logger.info(f"Fusion method: {fusion_method}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        return_raw: bool = False
    ) -> List[HybridResult]:
        """
        Retrieve documents using both vector and graph retrieval
        
        Args:
            query: Query text
            top_k: Number of top results to return
            return_raw: Return unranked combined results
            
        Returns:
            List of HybridResult objects ranked by fusion score
        """
        logger.info(f"Hybrid retrieval for query: {query[:50]}...")
        
        # Stage 1: Retrieve from both sources
        vector_results = self._retrieve_vector(query)
        graph_results = self._retrieve_graph(query)
        
        logger.info(f"Vector search: {len(vector_results)} results")
        logger.info(f"Graph search: {len(graph_results)} results")
        
        # Stage 2: Convert to HybridResult format
        hybrid_results = []
        
        # Add vector results
        for i, result in enumerate(vector_results):
            hybrid_results.append(HybridResult(
                content=result.content,
                source=result.document,
                score=result.score,
                source_type="vector",
                metadata=result.metadata
            ))
        
        # Add graph results
        for i, result in enumerate(graph_results):
            hybrid_results.append(HybridResult(
                content=result.get('content', ''),
                source=result.get('entity', 'unknown'),
                score=result.get('score', 0.5),
                source_type="graph",
                metadata=result.get('metadata'),
                relationships=result.get('relationships')
            ))
        
        if return_raw:
            return hybrid_results
        
        # Stage 3: Fuse and rank results
        fused_results = self._fuse_results(hybrid_results, query, top_k)
        
        return fused_results
    
    def _retrieve_vector(self, query: str) -> List[SearchResult]:
        """
        Retrieve from vector store
        
        Args:
            query: Query text
            
        Returns:
            List of SearchResult objects
        """
        try:
            results = self.vector_store.search(query, top_k=self.top_k_vector)
            return results
        except Exception as e:
            logger.error(f"Error in vector retrieval: {str(e)}")
            return []
    
    def _retrieve_graph(self, query: str) -> List[Dict]:
        """
        Retrieve from graph using multiple strategies
        
        Args:
            query: Query text
            
        Returns:
            List of graph results
        """
        graph_results = []
        
        try:
            # Query-focused relationship lookup. Subgraph expansion is avoided
            # during normal QA because it is slow and produces weak evidence.
            terms = self._query_terms(query)
            triplets = self.knowledge_graph.query_triplets(
                terms=terms,
                limit=self.top_k_graph * 3
            )
            
            for triplet in self._rank_triplets(triplets, terms)[:self.top_k_graph]:
                result = {
                    'entity': triplet.get('subject', 'unknown'),
                    'content': f"{triplet['subject']} - {triplet['relation']} - {triplet['object']}",
                    'score': 0.7,  # Generic triplet gets moderate score
                    'relationships': [triplet],
                    'metadata': {
                        'type': 'graph_triplet',
                        'relation': triplet.get('relation')
                    }
                }
                graph_results.append(result)
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {str(e)}")
            return []

    def _query_terms(self, query: str) -> List[str]:
        """Extract content terms for graph filtering."""
        stopwords = {
            "a", "an", "and", "are", "as", "can", "for", "from", "how", "in",
            "is", "it", "my", "of", "on", "or", "the", "to", "what", "when",
            "where", "which", "who", "why", "with", "should", "do", "does",
        }
        return [
            token for token in re.findall(r"[a-z0-9]+", query.lower())
            if token not in stopwords and len(token) > 2
        ][:8]

    def _rank_triplets(self, triplets: List[Dict], terms: List[str]) -> List[Dict]:
        """Rank graph triplets by simple query-term overlap."""
        if not terms:
            return triplets

        term_set = set(terms)

        def score(triplet: Dict) -> int:
            text = " ".join([
                str(triplet.get("subject", "")),
                str(triplet.get("relation", "")),
                str(triplet.get("object", "")),
            ]).lower()
            return sum(1 for term in term_set if term in text)

        return sorted(triplets, key=score, reverse=True)
    
    def _extract_entities(self, query: str) -> Set[str]:
        """
        Extract entities from query (simple heuristic)
        In a real system, this would use NER or domain-specific extraction
        
        Args:
            query: Query text
            
        Returns:
            Set of entity strings
        """
        # Simple extraction: capitalized words, known agricultural terms
        agricultural_terms = {
            'rice', 'wheat', 'cotton', 'maize', 'sugarcane',
            'urea', 'nitrogen', 'fertilizer', 'disease', 'pest',
            'fungal', 'blast', 'drainage', 'yield', 'resistance'
        }
        
        tokens = query.lower().split()
        entities = set()
        
        for token in tokens:
            # Remove punctuation
            token = token.strip('.,!?;:')
            if token in agricultural_terms:
                entities.add(token.capitalize())
        
        return entities
    
    def _fuse_results(
        self,
        hybrid_results: List[HybridResult],
        query: str,
        top_k: int
    ) -> List[HybridResult]:
        """
        Fuse results from multiple sources
        
        Args:
            hybrid_results: Combined results from both sources
            query: Original query
            top_k: Number of top results
            
        Returns:
            Ranked and deduplicated results
        """
        if not hybrid_results:
            return []
        
        # Group by content to deduplicate
        seen_content = {}
        
        for result in hybrid_results:
            content_key = result.content[:100].lower()  # Use first 100 chars as key
            
            if content_key not in seen_content:
                seen_content[content_key] = result
            else:
                # Merge results from same content
                existing = seen_content[content_key]
                
                # Combine scores based on fusion method
                if self.fusion_method == "weighted_average":
                    if result.source_type == "vector":
                        existing.score = (existing.score * self.graph_weight +
                                        result.score * self.vector_weight)
                    else:
                        existing.score = (existing.score * self.vector_weight +
                                        result.score * self.graph_weight)
                
                elif self.fusion_method == "max":
                    existing.score = max(existing.score, result.score)
        
        # Sort by score
        ranked_results = sorted(
            seen_content.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # Return top K
        return ranked_results[:top_k]
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 10,
        expand_context: bool = True
    ) -> Dict:
        """
        Retrieve documents with expanded context
        
        Args:
            query: Query text
            top_k: Number of results
            expand_context: Expand context with related entities
            
        Returns:
            Dictionary with results and metadata
        """
        # Get primary results
        primary_results = self.retrieve(query, top_k=top_k)
        
        # Optionally expand context
        expanded_results = primary_results
        if expand_context and len(primary_results) > 0:
            # Get related entities from first result
            main_result = primary_results[0]
            entities = self._extract_entities(main_result.content)
            
            for entity in list(entities)[:3]:  # Limit expansion
                neighbors = self.knowledge_graph.get_neighbors(entity, hops=1)
                logger.info(f"Expanded context with {len(neighbors)} neighbor relationships")
        
        return {
            "query": query,
            "num_results": len(primary_results),
            "results": primary_results,
            "retrieval_config": {
                "vector_weight": self.vector_weight,
                "graph_weight": self.graph_weight,
                "fusion_method": self.fusion_method
            }
        }
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get retrieval statistics
        
        Returns:
            Statistics dictionary
        """
        vector_info = self.vector_store.get_collection_info()
        graph_stats = self.knowledge_graph.get_statistics()
        
        return {
            "vector_store": vector_info,
            "knowledge_graph": graph_stats,
            "retrieval_config": {
                "vector_weight": self.vector_weight,
                "graph_weight": self.graph_weight,
                "top_k_vector": self.top_k_vector,
                "top_k_graph": self.top_k_graph,
                "fusion_method": self.fusion_method
            }
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retriever (requires vector_store and kg to be initialized)
    # retriever = HybridRetriever(vector_store, knowledge_graph)
    
    # Retrieve results
    # results = retriever.retrieve("How to prevent fungal blast in rice?", top_k=5)
    # for result in results:
    #     print(f"[{result.source_type.upper()}] Score: {result.score:.2f}")
    #     print(f"  {result.content[:100]}...")
    
    print("HybridRetriever module ready")
