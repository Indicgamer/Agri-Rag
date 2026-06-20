"""
Complete Working Evaluation Script
Compares NLI-Gated vs Standard RAG with proper RAGAS metrics and hallucination detection
"""

import sys
from pathlib import Path
import logging
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.faiss_retriever import FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pruning.nli_pruner import NLIPruner
from src.models.response_generator import ResponseGenerator
from src.models.rag_pipeline import AgriRAGPipeline, PipelineResult
from src.evaluation.ragas_evaluator import RAGASEvaluator, HallucinationDetector
from configs.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_nli_pipeline():
    """Initialize NLI-Gated pipeline"""
    logger.info("Initializing NLI-Gated Pipeline...")
    try:
        neo4j = KnowledgeGraph(
            uri=settings.neo4j.uri,
            username=settings.neo4j.username,
            password=settings.neo4j.password,
            database=settings.neo4j.database
        )
        
        faiss = FAISSVectorStore(
            collection_name=settings.faiss.collection_name,
            embedding_model=settings.faiss.embedding_model,
            persistence_dir=str(settings.faiss.persistence_dir)
        )
        
        hybrid = HybridRetriever(
            vector_store=faiss,
            knowledge_graph=neo4j,
            vector_weight=0.35,
            graph_weight=0.65,
            top_k_vector=5,
            top_k_graph=10
        )
        
        nli_pruner = NLIPruner(
            model_name="cross-encoder/nli-deberta-v3-base",
            provider="transformers",
            device="cpu",
            entailment_threshold=0.35,
            neutral_threshold=0.20
        )
        
        response_gen = ResponseGenerator(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1000
        )
        
        pipeline = AgriRAGPipeline(
            hybrid_retriever=hybrid,
            nli_pruner=nli_pruner,
            response_generator=response_gen
        )
        logger.info("✓ NLI-Gated Pipeline initialized")
        return pipeline
    except Exception as e:
        logger.error(f"✗ Failed to initialize NLI pipeline: {e}")
        return None


def initialize_baseline_pipeline():
    """Initialize baseline RAG pipeline (no pruning)"""
    logger.info("Initializing Baseline RAG Pipeline...")
    try:
        neo4j = KnowledgeGraph(
            uri=settings.neo4j.uri,
            username=settings.neo4j.username,
            password=settings.neo4j.password,
            database=settings.neo4j.database
        )
        
        faiss = FAISSVectorStore(
            collection_name=settings.faiss.collection_name,
            embedding_model=settings.faiss.embedding_model,
            persistence_dir=str(settings.faiss.persistence_dir)
        )
        
        hybrid = HybridRetriever(
            vector_store=faiss,
            knowledge_graph=neo4j,
            vector_weight=0.35,
            graph_weight=0.65,
            top_k_vector=5,
            top_k_graph=10
        )
        
        response_gen = ResponseGenerator(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1000
        )
        
        # Baseline: just retrieval + generation, no pruning
        class BaselineRAG:
            def __init__(self, retriever, generator):
                self.retriever = retriever
                self.generator = generator
            
            def process(self, query, top_k=10):
                facts = self.retriever.retrieve(query, top_k=top_k)
                fact_texts = [f.content if hasattr(f, 'content') else str(f) for f in facts]
                
                response = self.generator.generate(query, fact_texts)
                
                return PipelineResult(
                    query=query,
                    response=response.response if hasattr(response, 'response') else str(response),
                    confidence=0.85,
                    citations=[{"text": f, "idx": i} for i, f in enumerate(fact_texts[:3])],
                    warnings=[],
                    pipeline_steps={'retrieval': {'num_results': len(facts)}},
                    execution_time=0.5,
                    timestamp=datetime.now().isoformat(),
                    retrieved_facts=fact_texts
                )
        
        pipeline = BaselineRAG(hybrid, response_gen)
        logger.info("✓ Baseline RAG Pipeline initialized")
        return pipeline
    except Exception as e:
        logger.error(f"✗ Failed to initialize baseline pipeline: {e}")
        return None


def evaluate_single_query(nli_pipeline, baseline_pipeline, question, ground_truth):
    """Evaluate one question with both approaches"""
    evaluator = RAGASEvaluator()
    hallucination_detector = HallucinationDetector()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*70}")
    
    # Get responses
    try:
        nli_result = nli_pipeline.process(question, top_k_retrieval=10)
        baseline_result = baseline_pipeline.process(question, top_k=10)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return None
    
    # Extract facts used
    nli_facts = getattr(nli_result, 'retrieved_facts', [])
    baseline_facts = getattr(baseline_result, 'retrieved_facts', [])
    
    # Calculate RAGAS metrics for NLI
    nli_metrics = evaluator.evaluate_single(
        question=question,
        response=nli_result.response,
        retrieved_facts=nli_facts,
        citations=nli_result.citations or [],
        ground_truth=ground_truth
    )
    
    # Calculate RAGAS metrics for Baseline
    baseline_metrics = evaluator.evaluate_single(
        question=question,
        response=baseline_result.response,
        retrieved_facts=baseline_facts,
        citations=baseline_result.citations or [],
        ground_truth=ground_truth
    )
    
    # Calculate hallucination rates
    nli_hallucination = hallucination_detector.calculate_hallucination_rate(
        nli_result.response,
        nli_result.citations or [],
        len(nli_facts)
    )
    
    baseline_hallucination = hallucination_detector.calculate_hallucination_rate(
        baseline_result.response,
        baseline_result.citations or [],
        len(baseline_facts)
    )
    
    result = {
        "question": question,
        "ground_truth": ground_truth,
        "nli_response": nli_result.response[:500],
        "baseline_response": baseline_result.response[:500],
        "nli_metrics": {
            "faithfulness": nli_metrics.faithfulness,
            "answer_relevance": nli_metrics.answer_relevance,
            "context_precision": nli_metrics.context_precision,
            "context_recall": nli_metrics.context_recall,
            "hallucination_rate": nli_hallucination,
            "average_score": nli_metrics.average_score()
        },
        "baseline_metrics": {
            "faithfulness": baseline_metrics.faithfulness,
            "answer_relevance": baseline_metrics.answer_relevance,
            "context_precision": baseline_metrics.context_precision,
            "context_recall": baseline_metrics.context_recall,
            "hallucination_rate": baseline_hallucination,
            "average_score": baseline_metrics.average_score()
        },
        "nli_better": nli_metrics.average_score() > baseline_metrics.average_score(),
        "hallucination_reduction": baseline_hallucination - nli_hallucination
    }
    
    # Log results
    logger.info(f"\nNLI-Gated Metrics:")
    logger.info(f"  Faithfulness:       {nli_metrics.faithfulness:.2%}")
    logger.info(f"  Answer Relevance:   {nli_metrics.answer_relevance:.2%}")
    logger.info(f"  Context Precision:  {nli_metrics.context_precision:.2%}")
    logger.info(f"  Context Recall:     {nli_metrics.context_recall:.2%}")
    logger.info(f"  Hallucination Rate: {nli_hallucination:.2%}")
    logger.info(f"  Average Score:      {nli_metrics.average_score():.2%}")
    
    logger.info(f"\nBaseline RAG Metrics:")
    logger.info(f"  Faithfulness:       {baseline_metrics.faithfulness:.2%}")
    logger.info(f"  Answer Relevance:   {baseline_metrics.answer_relevance:.2%}")
    logger.info(f"  Context Precision:  {baseline_metrics.context_precision:.2%}")
    logger.info(f"  Context Recall:     {baseline_metrics.context_recall:.2%}")
    logger.info(f"  Hallucination Rate: {baseline_hallucination:.2%}")
    logger.info(f"  Average Score:      {baseline_metrics.average_score():.2%}")
    
    winner = "🏆 NLI-GATED WINS" if result["nli_better"] else "🏆 BASELINE WINS"
    logger.info(f"\n{winner}")
    logger.info(f"Hallucination Reduction: {result['hallucination_reduction']:+.2%}")
    
    return result


def main():
    """Run complete evaluation"""
    logger.info("="*70)
    logger.info("AGRI-RAG EVALUATION: NLI-GATED vs STANDARD RAG")
    logger.info("="*70)
    
    # Load ground truth
    logger.info("\nLoading ground truth dataset...")
    gt_file = Path("data/ragas_ground_truth.json")
    
    if not gt_file.exists():
        logger.error(f"Ground truth file not found: {gt_file}")
        return
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    questions = gt_data.get('questions', [])[:10]  # Use first 10 for faster evaluation
    logger.info(f"Loaded {len(questions)} questions")
    
    # Initialize pipelines
    logger.info("\nInitializing pipelines...")
    nli_pipeline = initialize_nli_pipeline()
    baseline_pipeline = initialize_baseline_pipeline()
    
    if not nli_pipeline or not baseline_pipeline:
        logger.error("Failed to initialize pipelines")
        return
    
    # Evaluate each question
    results = []
    for q_data in questions:
        try:
            result = evaluate_single_query(
                nli_pipeline,
                baseline_pipeline,
                q_data['question'],
                q_data.get('ground_truth_answer', '')
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
    
    # Aggregate results
    logger.info("\n" + "="*70)
    logger.info("AGGREGATED RESULTS")
    logger.info("="*70)
    
    if results:
        nli_wins = sum(1 for r in results if r["nli_better"])
        baseline_wins = len(results) - nli_wins
        
        avg_nli_faith = sum(r['nli_metrics']['faithfulness'] for r in results) / len(results)
        avg_baseline_faith = sum(r['baseline_metrics']['faithfulness'] for r in results) / len(results)
        
        avg_nli_hall = sum(r['nli_metrics']['hallucination_rate'] for r in results) / len(results)
        avg_baseline_hall = sum(r['baseline_metrics']['hallucination_rate'] for r in results) / len(results)
        
        avg_nli_score = sum(r['nli_metrics']['average_score'] for r in results) / len(results)
        avg_baseline_score = sum(r['baseline_metrics']['average_score'] for r in results) / len(results)
        
        logger.info(f"\nQuestions: {len(results)}")
        logger.info(f"NLI-Gated Wins: {nli_wins} ({nli_wins/len(results)*100:.0f}%)")
        logger.info(f"Baseline Wins:  {baseline_wins} ({baseline_wins/len(results)*100:.0f}%)")
        
        logger.info(f"\n{'Metric':<30} {'NLI-Gated':<15} {'Baseline':<15} {'Diff':<15}")
        logger.info("-"*75)
        logger.info(f"{'Avg Faithfulness':<30} {avg_nli_faith:>14.2%} {avg_baseline_faith:>14.2%} {avg_nli_faith-avg_baseline_faith:>+14.2%}")
        logger.info(f"{'Avg Hallucination Rate':<30} {avg_nli_hall:>14.2%} {avg_baseline_hall:>14.2%} {avg_baseline_hall-avg_nli_hall:>+14.2%}")
        logger.info(f"{'Avg Overall Score':<30} {avg_nli_score:>14.2%} {avg_baseline_score:>14.2%} {avg_nli_score-avg_baseline_score:>+14.2%}")
        
        # Save results
        output_file = Path("evaluation_results_new.json")
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(results),
                "nli_wins": nli_wins,
                "baseline_wins": baseline_wins,
                "aggregated_metrics": {
                    "nli": {
                        "avg_faithfulness": float(avg_nli_faith),
                        "avg_hallucination_rate": float(avg_nli_hall),
                        "avg_overall_score": float(avg_nli_score)
                    },
                    "baseline": {
                        "avg_faithfulness": float(avg_baseline_faith),
                        "avg_hallucination_rate": float(avg_baseline_hall),
                        "avg_overall_score": float(avg_baseline_score)
                    }
                },
                "individual_results": results
            }, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")
    else:
        logger.warning("No results generated")


if __name__ == "__main__":
    main()
