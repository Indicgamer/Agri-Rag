"""
Unit tests for Hybrid Retriever Module
Tests hybrid retrieval logic combining vector and graph search
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHybridResult:
    """Test HybridResult dataclass"""
    
    def test_create_result(self):
        from src.retrieval.hybrid_retriever import HybridResult
        
        result = HybridResult(
            content="Test content",
            source="test.pdf",
            score=0.9,
            source_type="vector",
            metadata={"page": 1}
        )
        
        assert result.content == "Test content"
        assert result.source == "test.pdf"
        assert result.score == 0.9
        assert result.source_type == "vector"
        assert result.metadata["page"] == 1


class TestHybridRetrieverInit:
    """Test HybridRetriever initialization"""
    
    def test_init_defaults(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        assert retriever.vector_store is vector_store
        assert retriever.knowledge_graph is knowledge_graph
    
    def test_weight_normalization(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            vector_weight=0.4,
            graph_weight=0.6
        )
        
        # Weights should be normalized to sum to 1
        assert abs(retriever.vector_weight + retriever.graph_weight - 1.0) < 0.01


class TestQueryTerms:
    """Test query term extraction"""
    
    def test_extract_terms_basic(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        terms = retriever._query_terms("How to prevent fungal blast in rice?")
        
        assert "fungal" in terms
        assert "blast" in terms
        assert "rice" in terms
    
    def test_filter_stopwords(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        terms = retriever._query_terms("How to do this")
        
        # "how" and "do" should be filtered
        assert "how" not in terms
        assert "do" not in terms
    
    def test_terms_limit(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        # Generate more than 8 terms
        terms = retriever._query_terms(
            "prevent fungal blast brown spot leaf blight bacterial panicle rice wheat maize"
        )
        
        # Should be limited to 8
        assert len(terms) <= 8


class TestExtractEntities:
    """Test entity extraction"""
    
    def test_extract_agricultural_terms(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        entities = retriever._extract_entities("How to treat fungal blast in rice?")
        
        assert "Rice" in entities or "Fungal" in entities
    
    def test_no_entities(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        entities = retriever._extract_entities("How does this work?")
        
        assert len(entities) == 0


class TestRankTriplets:
    """Test triplet ranking"""
    
    def test_rank_triplets_empty(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        ranked = retriever._rank_triplets([], ["fungal"])
        
        assert ranked == []
    
    def test_rank_triplets_by_overlap(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        triplets = [
            {"subject": "rice", "relation": "affected_by", "object": "fungal blast"},
            {"subject": "urea", "relation": "provides", "object": "nitrogen"},
            {"subject": "fungal blast", "relation": "causes", "object": "yield loss"}
        ]
        
        ranked = retriever._rank_triplets(triplets, ["fungal", "blast", "rice"])
        
        # Should be sorted by term overlap
        assert ranked[0]["subject"] in ["rice", "fungal blast"]


class TestFusionResults:
    """Test result fusion"""
    
    def test_fuse_empty(self):
        from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        fused = retriever._fuse_results([], "test query", 5)
        
        assert len(fused) == 0
    
    def test_fuse_deduplication(self):
        from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            fusion_method="weighted_average"
        )
        
        # Identical content (first 100 chars are same)
        results = [
            HybridResult("Test content is here and more text added", "doc1", 0.9, "vector"),
            HybridResult("Test content is here and more text added", "doc2", 0.8, "graph")
        ]
        
        fused = retriever._fuse_results(results, "test", 5)
        
        # Should be deduplicated to 1 (first 100 chars are identical)
        assert len(fused) == 1
    
    def test_fuse_top_k(self):
        from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        results = [
            HybridResult(f"Content {i}", f"doc{i}", 0.9 - i*0.1, "vector")
            for i in range(10)
        ]
        
        fused = retriever._fuse_results(results, "test", 3)
        
        assert len(fused) == 3
    
    def test_fuse_weighted_average(self):
        from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            vector_weight=0.4,
            graph_weight=0.6,
            fusion_method="weighted_average"
        )
        
        results = [
            HybridResult("Test content", "doc1", 0.5, "vector"),
            HybridResult("Test content", "doc2", 0.9, "graph")
        ]
        
        fused = retriever._fuse_results(results, "test", 2)
        
        # Score should be weighted combination
        assert fused[0].score > 0.5


class TestRetrieve:
    """Test retrieve method"""
    
    def test_retrieve_empty_query(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        vector_store.search = Mock(return_value=[])
        knowledge_graph.query_triplets = Mock(return_value=[])
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        results = retriever.retrieve("", top_k=3)
        
        assert isinstance(results, list)
    
    def test_retrieve_with_mocks(self):
        from src.retrieval.hybrid_retriever import HybridRetriever, SearchResult
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        # Mock vector search results
        vector_store.search = Mock(return_value=[
            SearchResult("fact about rice", "doc1", 0.9, {})
        ])
        
        # Mock graph results
        knowledge_graph.query_triplets = Mock(return_value=[
            {"subject": "rice", "relation": "needs", "object": "nitrogen"}
        ])
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            top_k_vector=3,
            top_k_graph=3
        )
        
        results = retriever.retrieve("rice nitrogen", top_k=3)
        
        # Should get results from both sources
        assert len(results) >= 1


class TestRetrieveWithContext:
    """Test retrieve with context"""
    
    def test_retrieve_with_context_basic(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        vector_store.search = Mock(return_value=[])
        knowledge_graph.query_triplets = Mock(return_value=[])
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        result = retriever.retrieve_with_context("test query")
        
        assert "query" in result
        assert "num_results" in result
        assert "results" in result
        assert "retrieval_config" in result


class TestGetRetrievalStats:
    """Test retrieval statistics"""
    
    def test_get_stats(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        vector_store.get_collection_info = Mock(return_value={"count": 100})
        knowledge_graph.get_statistics = Mock(return_value={"nodes": 50})
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph
        )
        
        stats = retriever.get_retrieval_stats()
        
        assert "vector_store" in stats
        assert "knowledge_graph" in stats
        assert "retrieval_config" in stats


class TestIntegration:
    """Integration tests"""
    
    def test_full_retrieval_pipeline(self):
        from src.retrieval.hybrid_retriever import HybridRetriever, SearchResult
        
        vector_store = Mock()
        knowledge_graph = Mock()
        
        # Mock vector results
        vector_store.search = Mock(return_value=[
            SearchResult("Urea provides nitrogen for rice", "doc1", 0.9, {}),
            SearchResult("Fungal blast affects rice", "doc2", 0.85, {}),
            SearchResult("Proper drainage prevents disease", "doc3", 0.8, {})
        ])
        
        # Mock graph results
        knowledge_graph.query_triplets = Mock(return_value=[
            {"subject": "rice", "relation": "needs", "object": "nitrogen"},
            {"subject": "urea", "relation": "contains", "object": "nitrogen"},
            {"subject": "fungal blast", "relation": "affects", "object": "rice"}
        ])
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            vector_weight=0.4,
            graph_weight=0.6,
            fusion_method="weighted_average"
        )
        
        # Retrieve
        results = retriever.retrieve("nitrogen rice fungal blast", top_k=5)
        
        # Should have results
        assert len(results) >= 1
        
        # Verify sources
        sources = set(r.source_type for r in results)
        assert "vector" in sources or "graph" in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])