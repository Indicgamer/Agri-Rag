"""
Phase 2: Query & Evaluation Script for Agri-RAG
Lightweight: Querying, UI, and RAGAS metrics
Run this after Phase 1 completes.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.settings import settings
from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.faiss_retriever import FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pruning.nli_pruner import NLIPruner
from src.models.response_generator import ResponseGenerator
from src.models.rag_pipeline import AgriRAGPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_pipeline() -> AgriRAGPipeline:
    """Initialize the RAG pipeline with FAISS and Neo4j"""
    logger.info("Initializing RAG Pipeline...")
    
    # Initialize Neo4j
    neo4j = KnowledgeGraph(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database
    )
    
    # Initialize FAISS
    faiss_store = FAISSVectorStore(
        collection_name=settings.faiss.collection_name,
        embedding_model=settings.faiss.embedding_model,
        persistence_dir=str(settings.faiss.persistence_dir)
    )
    
    # Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        vector_store=faiss_store,
        knowledge_graph=neo4j,
        vector_weight=settings.retrieval.vector_weight,
        graph_weight=settings.retrieval.graph_weight
    )
    
    # Initialize NLI Pruner with FP16 optimization for VRAM
    import torch
    nli_pruner = NLIPruner(
        model_name=settings.nli.model_name,
        device="cuda" if torch.cuda.is_available() and settings.nli.device == "cuda" else "cpu",
        entailment_threshold=settings.nli.entailment_threshold,
        neutral_threshold=settings.nli.neutral_threshold,
        keep_contradictions=True  # Keep contradictions for safety warnings
    )
    
    # Load NLI model in FP16 if on CUDA
    if nli_pruner.model is not None and torch.cuda.is_available():
        nli_pruner.model = nli_pruner.model.half()
        logger.info("NLI model loaded in FP16 mode for VRAM optimization")
    
    # Initialize Response Generator
    response_generator = ResponseGenerator(
        provider=settings.llm.provider,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens
    )
    
    # Create Pipeline
    pipeline = AgriRAGPipeline(
        hybrid_retriever=hybrid_retriever,
        nli_pruner=nli_pruner,
        response_generator=response_generator
    )
    
    logger.info("Pipeline initialized successfully!")
    return pipeline


def run_sample_queries(pipeline: AgriRAGPipeline):
    """Run sample queries and display results with contradiction warnings"""
    sample_questions = [
        "How can I prevent fungal blast in rice?",
        "What is the best fertilizer for wheat?",
        "How to control bollworms in cotton?",
        "When should I sow maize seeds?",
        "What are the common diseases in sugarcane?"
    ]
    
    logger.info("Running sample queries...")
    
    for question in sample_questions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Question: {question}")
        logger.info(f"{'='*60}")
        
        try:
            result = pipeline.process(question, top_k_retrieval=12)
            
            logger.info(f"\nResponse:\n{result.response}")
            logger.info(f"\nConfidence: {result.confidence:.2%}")
            logger.info(f"Execution time: {result.execution_time:.2f}s")
            
            # SAFETY ALERT: Show contradictions found during pruning
            if result.warnings:
                logger.warning(f"\n⚠️ SAFETY ALERT: Found {len(result.warnings)} contradictory facts!")
                for warning in result.warnings:
                    logger.warning(f"  - Conflict: {warning}")
            
            if result.citations:
                logger.info(f"Citations: {len(result.citations)} sources")
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")


def calculate_ragas_metrics(pipeline: AgriRAGPipeline) -> dict:
    """
    Calculate RAGAS metrics using Groq as the evaluator LLM
    Uses ground truth Q&A pairs from TNAU manuals
    """
    logger.info("Calculating RAGAS metrics...")
    
    try:
        # Clear any OpenAI env vars to avoid conflicts with Groq
        for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"]:
            os.environ.pop(key, None)
        
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, answer_relevancy
        from datasets import Dataset
        
        # Create Groq evaluator (overrides RAGAS default gpt-4o-mini)
        groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        
        # Override RAGAS default LLM on each metric
        faithfulness.llm = groq_llm
        context_precision.llm = groq_llm
        answer_relevancy.llm = groq_llm
        
        # Load ground truth dataset if available
        ground_truth_file = Path("data/ragas_ground_truth.json")
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                questions_data = gt_data.get('questions', [])
                logger.info(f"Loaded {len(questions_data)} ground truth Q&A pairs")
        else:
            # Fallback to hardcoded questions with TNAU-based answers
            questions_data = [
                {"question": "How can I prevent fungal blast in rice?", "answer": "Use proper drainage, resistant varieties, and apply Tricyclazole 75WP fungicide."},
                {"question": "What fertilizer for wheat?", "answer": "Apply NPK fertilizer based on soil test, typically 120kg N, 60kg P2O5, 40kg K2O per hectare."},
                {"question": "How to control cotton bollworms?", "answer": "Use integrated pest management with pheromone traps and approved insecticides like Spinosad."}
            ]
            logger.info("Using fallback ground truth questions")
        
        # Prepare dataset
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for q_data in questions_data[:5]:  # Limit to 5 for faster evaluation
            q = q_data['question']
            eval_data["question"].append(q)
            eval_data["ground_truth"].append(q_data['answer'])
            
            result = pipeline.process(q, top_k_retrieval=12)
            eval_data["answer"].append(result.response)
            eval_data["contexts"].append(result.retrieved_facts[:10])  # Limit contexts
        
        ds = Dataset.from_dict(eval_data)
        
        logger.info("Running RAGAS evaluation (this may take a while)...")
        metrics = evaluate(ds, metrics=[faithfulness, context_precision, answer_relevancy])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RAGAS Results:")
        logger.info(f"{'='*60}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.3f}")
        
        return metrics
        
    except ImportError as e:
        logger.warning(f"Missing: {e}")
        return {}
    except Exception as e:
        logger.error(f"RAGAS error: {e}")
        return {}


def main():
    """Run Phase 2: Query & Evaluation"""
    logger.info("="*60)
    logger.info("  Agri-RAG Phase 2: Query & Evaluation")
    logger.info("="*60)
    
    try:
        # Initialize pipeline
        pipeline = initialize_pipeline()
        
        # Run sample queries
        run_sample_queries(pipeline)
        
        # Calculate RAGAS metrics
        metrics = calculate_ragas_metrics(pipeline)
        
        logger.info("\n" + "="*60)
        logger.info("  Phase 2 Complete!")
        logger.info("="*60)
        logger.info("Next: Run 'streamlit run ui/app.py' for the web interface")
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
