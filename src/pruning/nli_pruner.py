"""
NLI-Gated Pruner Module
Filters retrieved facts using Natural Language Inference (DeBERTa-v3)
Core innovation: Logic-based pruning instead of similarity-based retrieval
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

AutoTokenizer = None
AutoModelForSequenceClassification = None
torch = None

from configs.settings import settings

logger = logging.getLogger(__name__)


class EntailmentLabel(Enum):
    """NLI labels"""
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass
class PruningResult:
    """Result from NLI-based pruning"""
    original_fact: str
    label: str  # "entailment", "neutral", "contradiction"
    score: float
    keep: bool
    reasoning: Optional[str] = None


class NLIPruner:
    """
    Prunes retrieved facts using Natural Language Inference
    Uses DeBERTa-v3 Cross-Encoder to determine if facts entail the hypothesis
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        provider: str = "transformers",
        device: str = "cuda",
        hf_token: str = None,
        entailment_threshold: float = 0.6,  # Balanced
        neutral_threshold: float = 0.3,  # Balanced
        keep_contradictions: bool = True
    ):
        """
        Initialize NLI Pruner
        
        Args:
            model_name: HuggingFace model ID for NLI
            device: Device to use ("cuda" or "cpu")
            entailment_threshold: Score threshold for entailment (0.0-1.0)
            neutral_threshold: Score threshold for neutral (0.0-1.0)
            keep_contradictions: Whether to keep contradictions as warnings
        """
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.entailment_threshold = entailment_threshold
        self.neutral_threshold = neutral_threshold
        self.keep_contradictions = keep_contradictions
        
        self.tokenizer = None
        self.model = None
        if self.provider == "transformers":
            self._load_model()
        else:
            logger.info("Using heuristic NLI pruner; transformer model loading skipped")
        
        logger.info(f"NLI Pruner initialized")
        logger.info(f"Provider: {self.provider}, Model: {model_name} on {self.device}")
        logger.info(f"Thresholds - Entailment: {entailment_threshold}, Neutral: {neutral_threshold}")
    
    def _load_model(self):
        """Load DeBERTa-v3 model and tokenizer"""
        global AutoTokenizer, AutoModelForSequenceClassification, torch

        try:
            from transformers import AutoTokenizer as _AutoTokenizer
            from transformers import AutoModelForSequenceClassification as _AutoModelForSequenceClassification
            import torch as _torch

            AutoTokenizer = _AutoTokenizer
            AutoModelForSequenceClassification = _AutoModelForSequenceClassification
            torch = _torch
        except ImportError:
            raise ImportError("transformers and torch required for NLI. Install: pip install transformers torch")

        self.device = self.device if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate_hypothesis(self, question: str) -> str:
        """
        Generate a hypothesis from the question
        In a real system, this could use more sophisticated methods
        
        Args:
            question: Original question
            
        Returns:
            Hypothesis statement
        """
        # Simple heuristic: convert question to statement
        question = question.strip()
        
        if question.endswith('?'):
            question = question[:-1]
        
        # Remove question words
        question_words = ['how', 'what', 'when', 'where', 'why', 'who', 'which']
        tokens = question.split()
        
        if tokens and tokens[0].lower() in question_words:
            tokens = tokens[1:]
        
        hypothesis = ' '.join(tokens)
        
        # Convert to present tense
        if hypothesis.startswith('can'):
            hypothesis = hypothesis.replace('can ', '', 1)
        
        logger.info(f"Generated hypothesis: {hypothesis}")
        return hypothesis
    
    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Check if premise entails hypothesis using DeBERTa-v3
        
        Args:
            premise: Premise text (retrieved fact)
            hypothesis: Hypothesis to check against
            
        Returns:
            Tuple of (label, score)
            label: "entailment", "neutral", or "contradiction"
            score: Confidence score (0.0-1.0)
        """
        if self.provider != "transformers":
            return self._check_entailment_heuristic(premise, hypothesis)

        try:
            # Tokenize input
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            
            label_idx = torch.argmax(probabilities, dim=-1).item()
            score = probabilities[0, label_idx].item()
            
            label = self._normalize_model_label(label_idx)
            
            logger.debug(f"Entailment check: {label} ({score:.3f})")
            
            return label, score
            
        except Exception as e:
            logger.error(f"Error checking entailment: {str(e)}")
            return "neutral", 0.5

    def _normalize_model_label(self, label_idx: int) -> str:
        """Map model-specific label names into entailment/neutral/contradiction."""
        raw_label = str(self.model.config.id2label.get(label_idx, label_idx)).lower()
        if "entail" in raw_label:
            return "entailment"
        if "contrad" in raw_label:
            return "contradiction"
        return "neutral"

    def _check_entailment_heuristic(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Lightweight lexical fallback for demos when HF NLI is unavailable."""
        premise_terms = self._content_terms(premise)
        hypothesis_terms = self._content_terms(hypothesis)

        if not premise_terms or not hypothesis_terms:
            return "neutral", 0.5

        overlap = premise_terms & hypothesis_terms
        score = len(overlap) / max(len(hypothesis_terms), 1)

        contradiction_markers = {"avoid", "not", "never", "contraindicated", "harmful", "excessive"}
        if overlap and contradiction_markers & premise_terms:
            return "contradiction", min(0.95, 0.55 + score)

        if score >= 0.25:
            return "entailment", min(0.95, 0.55 + score)

        return "neutral", max(0.1, 1 - score)

    def _content_terms(self, text: str) -> set:
        """Extract simple content terms for heuristic pruning."""
        stopwords = {
            "a", "an", "and", "are", "as", "be", "by", "can", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
            "when", "where", "which", "who", "why", "with", "my", "i",
        }
        return {
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in stopwords and len(token) > 2
        }

    def _lexical_relevance(self, fact: str, hypothesis: str) -> float:
        """Estimate whether a fact is about the same entities/actions as the query."""
        fact_terms = self._content_terms(fact)
        hypothesis_terms = self._content_terms(hypothesis)
        if not fact_terms or not hypothesis_terms:
            return 0.0
        return len(fact_terms & hypothesis_terms) / len(hypothesis_terms)
    
    def prune_facts(
        self,
        facts: List[str],
        hypothesis: str
    ) -> Tuple[List[str], List[PruningResult]]:
        """
        Prune facts based on entailment with hypothesis
        
        Args:
            facts: List of retrieved facts
            hypothesis: Hypothesis to check against
            
        Returns:
            Tuple of (pruned_facts, pruning_details)
            pruned_facts: List of facts to keep
            pruning_details: Detailed pruning results
        """
        pruned_facts = []
        pruning_details = []
        
        logger.info(f"Pruning {len(facts)} facts against hypothesis")
        
        for fact in facts:
            if not fact or len(fact.strip()) == 0:
                continue
            
            # Check entailment
            label, score = self.check_entailment(fact, hypothesis)
            lexical_score = self._lexical_relevance(fact, hypothesis)
            
            # Determine if we keep the fact
            keep = False
            reasoning = ""
            
            if label == "entailment":
                if score >= self.entailment_threshold and lexical_score >= 0.3:
                    keep = True
                    reasoning = (
                        f"Entails hypothesis ({score:.3f}) "
                        f"and lexical relevance ({lexical_score:.3f})"
                    )
                elif score >= 0.5:
                    keep = True
                    reasoning = (
                        f"Strong entailment ({score:.3f})"
                    )
                else:
                    keep = False
                    reasoning = f"Weak entailment ({score:.3f})"
            
            elif label == "contradiction":
                # Discard contradictions - they cause hallucinations
                keep = False
                reasoning = "Contradiction discarded (causes hallucination)"
            
            else:  # neutral
                # Be more selective with neutral facts
                if lexical_score >= 0.55:
                    keep = True
                    reasoning = (
                        f"Neutral but lexically relevant "
                        f"({lexical_score:.3f})"
                    )
                else:
                    keep = False
                    reasoning = f"Not relevant (lexical: {lexical_score:.3f})"
            
            # Store result
            result = PruningResult(
                original_fact=fact,
                label=label,
                score=score,
                keep=keep,
                reasoning=reasoning
            )
            pruning_details.append(result)
            
            # Add to output if kept
            if keep:
                pruned_facts.append(fact)
            
            # Fallback: ensure minimum facts
            if not pruned_facts and facts:
                # Pruning too aggressive - keep top 5 by lexical relevance
                logger.warning(f"Pruning filtered all facts ({len(facts)}) - using fallback")
                pruned_facts = facts[:5]
            
            logger.debug(f"Fact: {fact[:60]}... -> {label} -> Keep: {keep}")
        
        logger.info(f"Pruning complete: {len(pruned_facts)}/{len(facts)} facts retained")
        
        return pruned_facts, pruning_details
    
    def prune_with_context(
        self,
        facts: List[Dict],
        question: str
    ) -> Dict:
        """
        Comprehensive pruning with full context
        
        Args:
            facts: List of fact dictionaries with 'content' and optional 'score'
            question: Original question
            
        Returns:
            Dictionary with pruning results
        """
        # Generate hypothesis
        hypothesis = self.generate_hypothesis(question)
        
        # Extract fact texts
        fact_texts = [f.get('content', f) if isinstance(f, dict) else f for f in facts]
        
        # Prune
        pruned_texts, pruning_details = self.prune_facts(fact_texts, hypothesis)
        
        # Group by label for statistics
        label_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
        for detail in pruning_details:
            label_counts[detail.label] += 1
        
        result = {
            "question": question,
            "hypothesis": hypothesis,
            "total_facts": len(facts),
            "pruned_facts": pruned_texts,
            "retained_count": len(pruned_texts),
            "removed_count": len(facts) - len(pruned_texts),
            "retention_rate": len(pruned_texts) / len(facts) if facts else 0,
            "label_distribution": label_counts,
            "pruning_details": pruning_details,
            "retention_rate_pct": f"{(len(pruned_texts) / len(facts) * 100):.1f}%" if facts else "0%"
        }
        
        logger.info(f"Pruning summary: {label_counts}")
        
        return result
    
    def get_contradictions(self, pruning_details: List[PruningResult]) -> List[str]:
        """
        Extract contradictory facts as warnings
        
        Args:
            pruning_details: Results from pruning
            
        Returns:
            List of contradictory facts
        """
        contradictions = [
            detail.original_fact for detail in pruning_details
            if detail.label == "contradiction" and detail.keep
        ]
        
        return contradictions
    
    def get_statistics(self, pruning_details: List[PruningResult]) -> Dict:
        """
        Get statistics from pruning
        
        Args:
            pruning_details: Results from pruning
            
        Returns:
            Statistics dictionary
        """
        if not pruning_details:
            return {}
        
        labels = [detail.label for detail in pruning_details]
        scores = [detail.score for detail in pruning_details]
        
        kept = [d for d in pruning_details if d.keep]
        removed = [d for d in pruning_details if not d.keep]
        
        return {
            "total_facts": len(pruning_details),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "retention_rate": len(kept) / len(pruning_details) if pruning_details else 0,
            "label_distribution": {
                "entailment": labels.count("entailment"),
                "neutral": labels.count("neutral"),
                "contradiction": labels.count("contradiction")
            },
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pruner (requires transformers and torch)
    # pruner = NLIPruner()
    
    # Example facts
    # facts = [
    #     "Urea is a nitrogen-rich fertilizer used for rice cultivation",
    #     "Fungal blast is a disease that affects rice",
    #     "Proper drainage prevents fungal blast in rice",
    #     "The weather was sunny yesterday",  # Neutral
    #     "Urea application causes fungal blast"  # Contradiction
    # ]
    
    # Prune facts
    # question = "How to prevent fungal blast in rice?"
    # result = pruner.prune_with_context(facts, question)
    
    # print(f"Retained: {result['retained_count']}/{result['total_facts']}")
    # for detail in result['pruning_details']:
    #     print(f"{detail.label}: {detail.original_fact[:50]}...")
    
    print("NLI Pruner module ready (requires transformers/torch)")
