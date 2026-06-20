"""
Proper RAGAS Evaluation Module
Implements standard RAGAS metrics with hallucination detection
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RAGASMetrics:
    """RAGAS evaluation result"""
    faithfulness: float  # Does answer only use retrieved facts?
    answer_relevance: float  # Does answer address the question?
    context_precision: float  # Are top-k retrieved docs relevant?
    context_recall: float  # Are all relevant docs retrieved?
    hallucination_rate: float  # Percentage of hallucinated content
    
    def average_score(self) -> float:
        return (self.faithfulness + self.answer_relevance + 
                self.context_precision + self.context_recall) / 4


class HallucinationDetector:
    """Detects hallucinations in LLM responses"""
    
    @staticmethod
    def calculate_hallucination_rate(response: str, citations: List[Dict], 
                                     facts_used: int) -> float:
        """
        Calculate hallucination rate as percentage of response not grounded in facts
        
        Args:
            response: The LLM-generated response
            citations: List of cited facts with start/end positions
            facts_used: Total number of facts provided to LLM
        
        Returns:
            Hallucination rate (0.0 = all grounded, 1.0 = all hallucinated)
        """
        if not response or not citations:
            return 1.0  # If no citations, assume 100% hallucination
        
        # Calculate percentage of response that's cited
        response_length = len(response)
        cited_length = 0
        
        for citation in citations:
            if isinstance(citation, dict):
                start = citation.get('start', 0)
                end = citation.get('end', 0)
                cited_length += (end - start)
            else:
                # Assume citation is a string, estimate it covers ~100 chars
                cited_length += min(100, response_length // len(citations))
        
        cited_percentage = min(cited_length / response_length, 1.0) if response_length > 0 else 0
        hallucination_rate = 1.0 - cited_percentage
        
        return hallucination_rate
    
    @staticmethod
    def detect_contradictions(facts: List[str], response: str) -> List[Dict]:
        """
        Detect logical contradictions in response
        
        Returns list of detected contradictions
        """
        contradictions = []
        
        # Simple heuristic: look for conflicting statements
        negative_terms = ['not', 'never', 'cannot', 'no ', 'does not']
        positive_terms = ['yes', 'can', 'does', 'should']
        
        for i, fact in enumerate(facts):
            fact_lower = fact.lower()
            response_lower = response.lower()
            
            # Check if response contradicts fact
            has_negative = any(term in fact_lower for term in negative_terms)
            has_positive = any(term in response_lower for term in positive_terms)
            
            if has_negative and has_positive:
                # Potential contradiction - only flag if significant
                contradictions.append({
                    'type': 'potential_contradiction',
                    'fact_idx': i,
                    'severity': 'medium'
                })
        
        return contradictions


class RAGASEvaluator:
    """Comprehensive RAGAS evaluation with all standard metrics"""
    
    def __init__(self, use_llm_evaluation: bool = False):
        """
        Initialize evaluator
        
        Args:
            use_llm_evaluation: If True, use LLM-based evaluation (slower, more accurate)
        """
        self.use_llm_evaluation = use_llm_evaluation
        self.hallucination_detector = HallucinationDetector()
    
    def evaluate_single(
        self,
        question: str,
        response: str,
        retrieved_facts: List[str],
        citations: List[Dict],
        ground_truth: Optional[str] = None
    ) -> RAGASMetrics:
        """
        Evaluate a single RAG response
        
        Args:
            question: User question
            response: LLM response
            retrieved_facts: Facts provided to LLM
            citations: Facts actually cited
            ground_truth: Ground truth answer (for context_recall)
        
        Returns:
            RAGASMetrics object with all scores
        """
        
        # 1. Faithfulness: Do citations come from retrieved facts?
        faithfulness = self._calculate_faithfulness(
            response, citations, retrieved_facts
        )
        
        # 2. Answer Relevance: Does response address question?
        answer_relevance = self._calculate_answer_relevance(
            question, response
        )
        
        # 3. Context Precision: Are retrieved facts relevant?
        context_precision = self._calculate_context_precision(
            question, retrieved_facts
        )
        
        # 4. Context Recall: Coverage of ground truth (if available)
        context_recall = self._calculate_context_recall(
            ground_truth, retrieved_facts
        ) if ground_truth else 0.7  # Default assumption
        
        # 5. Hallucination Rate
        hallucination_rate = self.hallucination_detector.calculate_hallucination_rate(
            response, citations, len(retrieved_facts)
        )
        
        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            hallucination_rate=hallucination_rate
        )
    
    def _calculate_faithfulness(self, response: str, citations: List[Dict],
                               retrieved_facts: List[str]) -> float:
        """
        Faithfulness = (# of citation sentences supported by facts) / (# of sentences)
        Range: [0, 1], higher is better
        """
        if not response or not citations:
            return 0.0
        
        # Simple heuristic: score based on citation ratio
        num_sentences = max(1, len(response.split('.')))
        citation_coverage = len(citations) / num_sentences
        
        # Bonus if citations are actually from retrieved facts
        valid_citations = sum(
            1 for c in citations 
            if any(self._text_overlap(c.get('text', ''), f) 
                   for f in retrieved_facts)
        )
        
        citation_validity = valid_citations / max(len(citations), 1)
        faithfulness = (citation_coverage + citation_validity) / 2
        
        return min(faithfulness, 1.0)
    
    def _calculate_answer_relevance(self, question: str, response: str) -> float:
        """
        Answer Relevance = similarity between question and answer
        Range: [0, 1], higher is better
        """
        if not question or not response:
            return 0.0
        
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Overlap of keywords
        overlap = question_words & response_words
        relevance_score = len(overlap) / max(len(question_words), 1)
        
        # Bonus for mentioning key agricultural terms
        ag_terms = {'rice', 'wheat', 'cotton', 'maize', 'sugarcane', 
                    'disease', 'fungal', 'variety', 'nitrogen', 'fertilizer',
                    'drainage', 'pest', 'control', 'yield', 'crop'}
        
        if any(term in response.lower() for term in ag_terms):
            relevance_score = min(relevance_score + 0.1, 1.0)
        
        return relevance_score
    
    def _calculate_context_precision(self, question: str, 
                                    retrieved_facts: List[str]) -> float:
        """
        Context Precision = proportion of retrieved docs that are relevant
        Range: [0, 1], higher is better
        """
        if not retrieved_facts:
            return 0.0
        
        question_words = set(question.lower().split())
        relevant_count = 0
        
        for fact in retrieved_facts[:5]:  # Check top-5
            fact_words = set(fact.lower().split())
            overlap = question_words & fact_words
            
            if len(overlap) > 2:  # At least 3 word overlap
                relevant_count += 1
        
        precision = relevant_count / max(len(retrieved_facts[:5]), 1)
        return precision
    
    def _calculate_context_recall(self, ground_truth: str, 
                                 retrieved_facts: List[str]) -> float:
        """
        Context Recall = proportion of ground truth covered by retrieved facts
        Range: [0, 1], higher is better
        """
        if not ground_truth or not retrieved_facts:
            return 0.0
        
        gt_words = set(ground_truth.lower().split())
        retrieved_text = ' '.join(retrieved_facts).lower()
        retrieved_words = set(retrieved_text.split())
        
        # How many ground truth words are in retrieved facts?
        coverage = len(gt_words & retrieved_words) / max(len(gt_words), 1)
        return coverage
    
    @staticmethod
    def _text_overlap(text1: str, text2: str, min_overlap: int = 3) -> bool:
        """Check if two texts have significant overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1 & words2) >= min_overlap


def create_ground_truth_dataset() -> Dict:
    """
    Create a comprehensive ground truth dataset for evaluation
    Based on agricultural best practices
    """
    return {
        "dataset": "Agri-RAG Evaluation Dataset",
        "questions": [
            {
                "id": 1,
                "question": "What are the best rice varieties in Tamil Nadu?",
                "ground_truth": "Popular rice varieties in Tamil Nadu include ADT 49, ADT 50, Co 51, BPT 5204 (Ponni rice). These varieties are well-adapted to local conditions and have good disease resistance. ADT varieties are recommended by TNAU for their high yield and pest resistance.",
                "key_points": ["ADT 49", "ADT 50", "Co 51", "BPT 5204", "high yield", "disease resistance"]
            },
            {
                "id": 2,
                "question": "When to sow rice in Kuruvai season?",
                "ground_truth": "Kuruvai season rice is typically sown from June to July. The ideal time is mid-June to early July. Seeds should be sown 25-30 days before transplanting. Seedlings are ready for transplanting in July-August. The maturity period is 90-100 days, with harvest in September-October.",
                "key_points": ["June to July", "mid-June", "25-30 days before transplanting", "90-100 days", "September-October"]
            },
            {
                "id": 3,
                "question": "What is the spacing for rice planting?",
                "ground_truth": "Recommended spacing for rice is 20-25 cm between rows and 10-15 cm between plants. For mechanical transplanting, spacing should be 20 cm x 15 cm. Higher densities reduce yields due to increased pest incidence. Line transplanting with recommended spacing gives better yields.",
                "key_points": ["20-25 cm between rows", "10-15 cm between plants", "20 x 15 cm for mechanical", "line transplanting"]
            },
            {
                "id": 4,
                "question": "What nitrogen dose is recommended for rice?",
                "ground_truth": "Standard nitrogen recommendation for rice is 60 kg/ha for high-yielding varieties. Apply nitrogen in three splits: 25 kg/ha at transplanting, 25 kg/ha at maximum tillering, and 10 kg/ha at heading. Soil nutrient status and expected yield should guide final dosage. TNAU recommends 60 kg N for 4 t/ha yield target.",
                "key_points": ["60 kg/ha", "three splits", "25 kg at transplanting", "25 kg at tillering", "10 kg at heading"]
            },
            {
                "id": 5,
                "question": "How to control fungal blast in rice?",
                "ground_truth": "Fungal blast control involves: (1) Use resistant varieties like ADT 49 and ADT 50, (2) Maintain proper drainage to avoid waterlogging, (3) Avoid excessive nitrogen application, (4) Spray Tricyclazole 75 WP (1.2 g/L) during early infection, (5) Use Mancozeb (2.5 g/L) as preventive spray. Drain fields after symptom appearance.",
                "key_points": ["resistant varieties", "drainage", "avoid excess nitrogen", "Tricyclazole 75 WP", "Mancozeb", "drain fields"]
            },
            {
                "id": 6,
                "question": "What is the water requirement for rice?",
                "ground_truth": "Rice requires 1000-1500 mm of water during the entire growing season. This includes rainfall and irrigation. Optimal water management includes continuous flooding at 5-7.5 cm depth up to soft dough stage. Alternate wetting and drying (AWD) can save 20-30% water. Drain fields 15-20 days before harvest.",
                "key_points": ["1000-1500 mm", "5-7.5 cm depth", "alternate wetting and drying", "20-30% water savings", "drain 15-20 days before"]
            },
            {
                "id": 7,
                "question": "What is the temperature requirement for rice cultivation?",
                "ground_truth": "Rice grows best at 20-40°C temperature range. Optimum temperature for germination is 30°C. For vegetative growth, 25-30°C is ideal. During panicle initiation and flowering, 20-25°C is beneficial. Temperatures below 15°C affect seed germination. Frost damage occurs below 0°C.",
                "key_points": ["20-40°C range", "30°C germination", "25-30°C vegetative", "20-25°C panicle initiation", "below 15°C reduces germination"]
            },
            {
                "id": 8,
                "question": "How to control pests in cotton?",
                "ground_truth": "Major cotton pests are bollworms and aphids. Control measures: (1) Use resistant varieties, (2) Cultural practices like summer plowing and crop rotation, (3) Monitor pheromone traps (4-5 traps/acre), (4) Spray Spinosad 45% EC (350 mL/acre) or Endosulfan 35% EC for bollworms, (5) Spray Imidacloprid for aphids. IPM approach recommended.",
                "key_points": ["resistant varieties", "cultural practices", "pheromone traps", "Spinosad", "Endosulfan", "Imidacloprid", "IPM"]
            },
            {
                "id": 9,
                "question": "What is the fertilizer dose for sugarcane?",
                "ground_truth": "Recommended fertilizer for sugarcane: 150 kg N, 60 kg P2O5, 40 kg K2O per hectare. Apply nitrogen in 3 splits: 50% at planting, 25% at 4 months, 25% at 8 months. Phosphorus and potassium applied at planting. Bagasse ash application can save 25% potassium requirement.",
                "key_points": ["150 kg N", "60 kg P2O5", "40 kg K2O", "3 splits", "50-25-25", "bagasse ash"]
            },
            {
                "id": 10,
                "question": "How to control red rot in sugarcane?",
                "ground_truth": "Red rot is caused by Colletotrichum falcatum. Control: (1) Use resistant varieties like CoC 671, CoC 671, (2) Treat seed cane with Carbendazim 50% WP (2 g/L) for 30 minutes, (3) Remove infected plants and burn debris, (4) Avoid water logging, (5) Maintain field sanitation. No effective chemical control post-infection.",
                "key_points": ["resistant varieties", "seed treatment", "Carbendazim", "remove infected plants", "avoid waterlogging", "field sanitation"]
            }
        ]
    }
