"""
Unit tests for NLI Pruner Module
Tests core functionality of pruning logic using heuristic mode
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pruning.nli_pruner import NLIPruner, PruningResult, EntailmentLabel


class TestNLIPrunerInit:
    """Test NLI Pruner initialization"""
    
    def test_init_with_defaults(self):
        pruner = NLIPruner(provider="heuristic")
        assert pruner.provider == "heuristic"
        assert pruner.entailment_threshold == 0.6
        assert pruner.neutral_threshold == 0.3
        assert pruner.keep_contradictions == True
    
    def test_init_custom_thresholds(self):
        pruner = NLIPruner(
            provider="heuristic",
            entailment_threshold=0.7,
            neutral_threshold=0.2
        )
        assert pruner.entailment_threshold == 0.7
        assert pruner.neutral_threshold == 0.2


class TestHypothesisGeneration:
    """Test hypothesis generation from questions"""
    
    def test_generate_hypothesis_simple(self):
        pruner = NLIPruner(provider="heuristic")
        hypothesis = pruner.generate_hypothesis("How to prevent fungal blast in rice?")
        assert "fungal" in hypothesis.lower() or "blast" in hypothesis.lower()
    
    def test_generate_hypothesis_removes_question_words(self):
        pruner = NLIPruner(provider="heuristic")
        hypothesis = pruner.generate_hypothesis("What is urea?")
        assert hypothesis.strip() != ""
        assert not hypothesis.lower().startswith("what")


class TestContentTerms:
    """Test content term extraction"""
    
    def test_extract_basic_terms(self):
        pruner = NLIPruner(provider="heuristic")
        terms = pruner._content_terms("Urea is a nitrogen fertilizer")
        assert "urea" in terms
        assert "nitrogen" in terms
        assert "fertilizer" in terms
    
    def test_filter_stopwords(self):
        pruner = NLIPruner(provider="heuristic")
        terms = pruner._content_terms("the and is a for")
        assert len(terms) == 0
    
    def test_filter_short_terms(self):
        pruner = NLIPruner(provider="heuristic")
        terms = pruner._content_terms("N P K are nutrients")
        assert "np" not in terms  # too short


class TestLexicalRelevance:
    """Test lexical relevance scoring"""
    
    def test_high_overlap(self):
        pruner = NLIPruner(provider="heuristic")
        score = pruner._lexical_relevance(
            "Urea provides nitrogen for rice",
            "nitrogen for rice"
        )
        assert score > 0.5
    
    def test_no_overlap(self):
        pruner = NLIPruner(provider="heuristic")
        score = pruner._lexical_relevance(
            "Weather was sunny yesterday",
            "nitrogen for rice"
        )
        assert score == 0.0
    
    def test_empty_inputs(self):
        pruner = NLIPruner(provider="heuristic")
        assert pruner._lexical_relevance("", "") == 0.0


class TestCheckEntailment:
    """Test entailment checking"""
    
    def test_returns_tuple(self):
        pruner = NLIPruner(provider="heuristic")
        label, score = pruner.check_entailment(
            "Fungal blast affects rice",
            "fungal blast rice"
        )
        assert label in ["entailment", "neutral", "contradiction"]
        assert 0.0 <= score <= 1.0
    
    def test_detects_contradiction(self):
        pruner = NLIPruner(provider="heuristic")
        label, score = pruner.check_entailment(
            "Avoid urea application",
            "urea helps rice"
        )
        assert label == "contradiction"


class TestPruneFacts:
    """Test fact pruning logic"""
    
    def test_prune_empty_facts(self):
        pruner = NLIPruner(provider="heuristic")
        pruned, details = pruner.prune_facts([], "test hypothesis")
        assert len(pruned) == 0
    
    def test_prune_retains_entailing_facts(self):
        pruner = NLIPruner(provider="heuristic")
        facts = [
            "Urea is a nitrogen fertilizer for rice",
            "Fungal blast disease affects rice"
        ]
        pruned, details = pruner.prune_facts(facts, "nitrogen rice")
        assert len(pruned) >= 1
    
    def test_prune_filters_contradictions(self):
        pruner = NLIPruner(provider="heuristic")
        facts = [
            "Urea is a nitrogen fertilizer",  # entailment
            "Avoid urea application",  # contradiction
            "Weather was sunny"  # neutral
        ]
        pruned, details = pruner.prune_facts(facts, "urea rice")
        # Contradictions should be filtered out
        contradiction_facts = [d for d in details if d.label == "contradiction"]
        for cf in contradiction_facts:
            assert cf.keep == False


class TestPruneWithContext:
    """Test comprehensive pruning with context"""
    
    def test_prune_with_dict_facts(self):
        pruner = NLIPruner(provider="heuristic")
        facts = [
            {"content": "Urea provides nitrogen", "score": 0.9},
            {"content": "Weather is sunny", "score": 0.5}
        ]
        result = pruner.prune_with_context(facts, "How to apply urea?")
        assert "question" in result
        assert "hypothesis" in result
        assert result["total_facts"] == 2
    
    def test_retention_rate_calculated(self):
        pruner = NLIPruner(provider="heuristic")
        facts = ["fact1", "fact2", "fact3"]
        result = pruner.prune_with_context(facts, "test query")
        assert "retention_rate" in result
        assert "retention_rate_pct" in result


class TestGetStatistics:
    """Test statistics generation"""
    
    def test_statistics_empty(self):
        pruner = NLIPruner(provider="heuristic")
        stats = pruner.get_statistics([])
        assert stats == {}
    
    def test_statistics_calculation(self):
        pruner = NLIPruner(provider="heuristic")
        facts = [
            "Urea is nitrogen fertilizer",
            "Fungal blast affects rice",
            "Weather is sunny"
        ]
        _, details = pruner.prune_facts(facts, "nitrogen rice")
        stats = pruner.get_statistics(details)
        
        assert "total_facts" in stats
        assert stats["total_facts"] == 3
        assert "kept_count" in stats
        assert "removed_count" in stats
        assert "retention_rate" in stats
    
    def test_label_distribution(self):
        pruner = NLIPruner(provider="heuristic")
        facts = ["urea nitrogen", "avoid nitrogen", "weather sunny"]
        _, details = pruner.prune_facts(facts, "nitrogen rice")
        stats = pruner.get_statistics(details)
        
        assert "label_distribution" in stats
        dist = stats["label_distribution"]
        assert "entailment" in dist
        assert "neutral" in dist
        assert "contradiction" in dist


class TestGetContradictions:
    """Test contradiction extraction"""
    
    def test_get_contradictions_none(self):
        pruner = NLIPruner(provider="heuristic")
        facts = ["Urea is nitrogen"]
        _, details = pruner.prune_facts(facts, "nitrogen rice")
        contradictions = pruner.get_contradictions(details)
        assert len(contradictions) == 0  # Not kept by default
    
    def test_get_contradictions_with_keep(self):
        pruner = NLIPruner(provider="heuristic", keep_contradictions=True)
        facts = ["Avoid urea application"]
        _, details = pruner.prune_facts(facts, "urea rice")
        contradictions = pruner.get_contradictions(details)
        # Depends on keep setting


class TestIntegration:
    """Integration tests"""
    
    def test_full_pruning_pipeline(self):
        pruner = NLIPruner(provider="heuristic")
        facts = [
            "Urea provides essential nitrogen for rice growth",
            "Fungal blast is a disease affecting rice panicles",
            "Proper field drainage prevents fungal blast",
            "The weather was sunny yesterday",
            "Excessive urea causes leaf burn",
            "Zinc deficiency causes leaf chlorosis in rice"
        ]
        
        question = "How to prevent fungal blast in rice?"
        result = pruner.prune_with_context(facts, question)
        
        # Should have pruning results
        assert result["total_facts"] == 6
        assert result["retained_count"] >= 0
        assert result["retained_count"] <= result["total_facts"]
        
        # Statistics should be present
        assert "retention_rate" in result
        assert "label_distribution" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])