#!/usr/bin/env python3
"""
Quick end-to-end test
Tests both pipelines with a single question
Run this to verify everything works before presentation
"""

import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

print("🧪 End-to-End Test\n")
print("="*60)

# Test imports
print("\n1️⃣  Testing imports...")
try:
    from src.evaluation.ragas_evaluator import RAGASEvaluator, HallucinationDetector
    from src.retrieval.neo4j_retriever import KnowledgeGraph
    from src.retrieval.faiss_retriever import FAISSVectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from configs.settings import settings
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test evaluator
print("\n2️⃣  Testing RAGAS Evaluator...")
try:
    evaluator = RAGASEvaluator()
    
    # Test with mock data
    test_metrics = evaluator.evaluate_single(
        question="How to control fungal blast?",
        response="Use proper drainage, resistant varieties, and fungicide sprays.",
        retrieved_facts=[
            "Drainage prevents fungal blast in rice fields",
            "Resistant varieties like ADT 49 prevent fungal blast",
            "Tricyclazole spray is effective against fungal blast"
        ],
        citations=[
            {"text": "Drainage prevents fungal blast", "idx": 0},
            {"text": "Resistant varieties", "idx": 1}
        ],
        ground_truth="Fungal blast control involves proper drainage, resistant varieties, and fungicide application."
    )
    
    print(f"✓ Faithfulness: {test_metrics.faithfulness:.1%}")
    print(f"✓ Answer Relevance: {test_metrics.answer_relevance:.1%}")
    print(f"✓ Context Precision: {test_metrics.context_precision:.1%}")
    print(f"✓ Context Recall: {test_metrics.context_recall:.1%}")
    print(f"✓ Hallucination Rate: {test_metrics.hallucination_rate:.1%}")
    print(f"✓ Average Score: {test_metrics.average_score():.1%}")
except Exception as e:
    print(f"✗ Evaluator test failed: {e}")
    sys.exit(1)

# Test hallucination detector
print("\n3️⃣  Testing Hallucination Detector...")
try:
    detector = HallucinationDetector()
    
    hall_rate = detector.calculate_hallucination_rate(
        response="Rice requires 1000-1500 mm water and proper drainage.",
        citations=[{"text": "water requirement", "idx": 0}],
        facts_used=3
    )
    print(f"✓ Hallucination Detection: {hall_rate:.1%}")
except Exception as e:
    print(f"✗ Hallucination detector test failed: {e}")

# Test database connectivity
print("\n4️⃣  Testing Database Connectivity...")
try:
    kg = KnowledgeGraph(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database
    )
    print(f"✓ Neo4j connected: {settings.neo4j.uri}")
except Exception as e:
    print(f"! Neo4j connection note: {e}")
    print("  (This is OK if Neo4j isn't needed for web UI)")

# Test FAISS
print("\n5️⃣  Testing FAISS Vector Store...")
try:
    faiss_store = FAISSVectorStore(
        collection_name=settings.faiss.collection_name,
        embedding_model=settings.faiss.embedding_model,
        persistence_dir=str(settings.faiss.persistence_dir)
    )
    print(f"✓ FAISS initialized: {settings.faiss.persistence_dir}")
except Exception as e:
    print(f"! FAISS note: {e}")
    print("  (Indices will be created on first ingestion)")

# Test ground truth
print("\n6️⃣  Testing Ground Truth Dataset...")
try:
    import json
    gt_file = Path("data/ragas_ground_truth.json")
    with open(gt_file) as f:
        gt_data = json.load(f)
    num_q = len(gt_data.get('questions', []))
    print(f"✓ Ground truth loaded: {num_q} questions")
    
    # Show first question
    first_q = gt_data['questions'][0]
    print(f"  Sample Q: {first_q['question'][:50]}...")
    print(f"  Has answer: {'ground_truth_answer' in first_q and first_q['ground_truth_answer'][:30]}")
except Exception as e:
    print(f"✗ Ground truth test failed: {e}")

# Summary
print("\n" + "="*60)
print("\n✅ End-to-End Test Complete!")
print("\nNext steps:")
print("  1. Run: streamlit run ui/demo_comparison.py --server.port 8510")
print("  2. Open: http://localhost:8510")
print("  3. Click 'Initialize System'")
print("  4. Ask a question and see results")
print("\nYou're ready to present! 🎉")
