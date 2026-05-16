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
    
    # Initialize NLI Pruner
    nli_pruner = NLIPruner(
        model_name=settings.nli.model_name,
        device="cuda" if settings.nli.device == "cuda" else "cpu",
        entailment_threshold=settings.nli.entailment_threshold,
        neutral_threshold=settings.nli.neutral_threshold
    )
    
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
    """Run sample queries and display results"""
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
            result = pipeline.process(question, top_k_retrieval=6)
            
            logger.info(f"\nResponse:\n{result.response}")
            logger.info(f"\nConfidence: {result.confidence:.2%}")
            logger.info(f"Execution time: {result.execution_time:.2f}s")
            
            if result.warnings:
                logger.warning(f"Warnings: {result.warnings}")
            
            if result.citations:
                logger.info(f"Citations: {len(result.citations)} sources")
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")


def calculate_ragas_metrics(pipeline: AgriRAGPipeline) -> dict:
    """
    Calculate RAGAS metrics using GroQ as the evaluator LLM
    """
    logger.info("Calculating RAGAS metrics...")
    
    try:
        # Clear any OpenAI env vars to avoid conflicts with GroQ
        for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"]:
            os.environ.pop(key, None)
        
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, answer_relevancy
        from datasets import Dataset
        
        # Create GroQ evaluator (overrides RAGAS default gpt-4o-mini)
        groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        
        # Override RAGAS default LLM on each metric
        faithfulness.llm = groq_llm
        context_precision.llm = groq_llm
        answer_relevancy.llm = groq_llm
        
        # Prepare dataset
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [
                "Use proper drainage and resistant varieties to prevent fungal blast in rice.",
                "Apply NPK fertilizer based on soil test for wheat.",
                "Use integrated pest management and approved insecticides for cotton bollworms."
            ]
        }
        
        questions = [
            "How can I prevent fungal blast in rice?",
            "What fertilizer for wheat?",
            "How to control cotton bollworms?"
        ]
        
        for q in questions:
            eval_data["question"].append(q)
            result = pipeline.process(q, top_k_retrieval=6)
            eval_data["answer"].append(result.response)
            eval_data["contexts"].append(result.retrieved_facts)
        
        ds = Dataset.from_dict(eval_data)
        metrics = evaluate(ds, metrics=[faithfulness, context_precision])
        
        logger.info(f"RAGAS Results:")
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
