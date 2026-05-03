"""
Demo/Testing Script for Agri-RAG
Quick start guide to test individual components
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import DataLoader
from src.models.triplet_extractor import TripletExtractor
from src.retrieval.vector_retriever import VectorStore
from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pruning.nli_pruner import NLIPruner
from src.models.response_generator import ResponseGenerator
from src.models.rag_pipeline import AgriRAGPipeline


def demo_data_loader():
    """Demo: Load agricultural data"""
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Data Loader")
    logger.info("="*80)
    
    loader = DataLoader(chunk_size=512, chunk_overlap=50)
    
    # Sample text
    sample_text = """
    Rice cultivation in India requires proper water management. 
    Urea is commonly used as a nitrogen fertilizer for rice. 
    However, excessive urea application can lead to disease susceptibility.
    Fungal blast is a major disease in rice cultivation areas with poor drainage.
    Proper water management and resistant varieties prevent fungal blast.
    """
    
    logger.info("Sample text loaded")
    logger.info(f"Text length: {len(sample_text)} characters")
    
    # Chunk the text
    chunks = loader._chunk_text(sample_text)
    logger.info(f"Number of chunks: {len(chunks)}")
    
    for idx, chunk in enumerate(chunks):
        logger.info(f"  Chunk {idx+1}: {chunk[:60]}...")
    
    return loader


def demo_triplet_extractor():
    """Demo: Extract triplets (requires Ollama)"""
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Triplet Extractor (Llama 3 via Ollama)")
    logger.info("="*80)
    
    try:
        extractor = TripletExtractor()
        
        sample_text = """
        Urea is a nitrogen-rich fertilizer commonly used for rice cultivation.
        Fungal blast is a disease that affects rice in humid conditions.
        Proper drainage prevents fungal blast disease.
        Resistant rice varieties improve disease tolerance.
        """
        
        logger.info("Extracting triplets from sample text...")
        triplets = extractor.extract_triplets(sample_text, "demo.pdf", page=1)
        
        logger.info(f"Extracted {len(triplets)} triplets:")
        for triplet in triplets:
            logger.info(f"  {triplet.subject} --[{triplet.relation}]--> {triplet.object}")
        
        return extractor
        
    except Exception as e:
        logger.warning(f"Triplet extraction demo skipped: {str(e)}")
        logger.info("(Requires Ollama running with the llama3 model)")
        return None


def demo_vector_store():
    """Demo: Vector search with ChromaDB"""
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: ChromaDB Vector Store")
    logger.info("="*80)
    
    vector_store = VectorStore(persistence_dir="data/chroma_db")
    
    # Sample documents
    sample_docs = [
        {
            'content': 'Urea is a nitrogen-rich fertilizer used for rice cultivation',
            'metadata': {'crop': 'rice', 'nutrient': 'nitrogen'}
        },
        {
            'content': 'Fungal blast is a disease that affects rice in humid conditions',
            'metadata': {'crop': 'rice', 'disease': 'fungal_blast'}
        },
        {
            'content': 'Proper drainage prevents fungal blast in rice fields',
            'metadata': {'crop': 'rice', 'practice': 'drainage'}
        },
        {
            'content': 'Cotton requires potassium for better yield and disease resistance',
            'metadata': {'crop': 'cotton', 'nutrient': 'potassium'}
        }
    ]
    
    # Add documents
    logger.info(f"Adding {len(sample_docs)} documents to vector store...")
    added = vector_store.add_documents(sample_docs, document_source="demo.pdf")
    logger.info(f"Added {added} documents")
    
    # Search
    query = "How to prevent fungal blast in rice?"
    logger.info(f"Searching for: {query}")
    
    results = vector_store.search(query, top_k=2)
    logger.info(f"Found {len(results)} results:")
    
    for idx, result in enumerate(results, 1):
        logger.info(f"  [{idx}] Score: {result.score:.3f}")
        logger.info(f"       {result.content[:70]}...")
    
    return vector_store


def demo_knowledge_graph():
    """Demo: Neo4j Knowledge Graph (requires Neo4j running)"""
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Neo4j Knowledge Graph")
    logger.info("="*80)
    
    try:
        kg = KnowledgeGraph()
        
        # Sample triplets
        triplets = [
            {'subject': 'Urea', 'relation': 'AFFECTS', 'object': 'Rice Yield'},
            {'subject': 'Fungal Blast', 'relation': 'OCCURS_IN', 'object': 'Rice'},
            {'subject': 'Drainage', 'relation': 'PREVENTS', 'object': 'Fungal Blast'},
            {'subject': 'Resistant Variety', 'relation': 'PREVENTS', 'object': 'Fungal Blast'},
        ]
        
        logger.info(f"Adding {len(triplets)} triplets to knowledge graph...")
        added = kg.add_triplets_batch(triplets)
        logger.info(f"Added {added} triplets")
        
        # Query
        logger.info("Querying subgraph for 'Fungal Blast'...")
        subgraph = kg.get_subgraph('Fungal Blast', max_hops=2)
        
        logger.info(f"Subgraph: {subgraph.get('node_count')} nodes, {subgraph.get('edge_count')} edges")
        
        # Get statistics
        stats = kg.get_statistics()
        logger.info(f"Graph stats: {stats}")
        
        return kg
        
    except Exception as e:
        logger.warning(f"Knowledge graph demo skipped: {str(e)}")
        logger.info("(Requires Neo4j running at configured URI)")
        return None


def demo_nli_pruner():
    """Demo: NLI-based Pruning (requires transformers)"""
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: NLI-Gated Pruner (DeBERTa-v3)")
    logger.info("="*80)
    
    try:
        pruner = NLIPruner()
        
        facts = [
            "Urea is a nitrogen-rich fertilizer used for rice cultivation",
            "Fungal blast is a disease in rice that requires proper drainage",
            "Resistant rice varieties improve disease tolerance",
            "The weather was sunny yesterday",  # Neutral
            "Urea application causes fungal blast"  # Contradiction
        ]
        
        question = "How to prevent fungal blast in rice?"
        
        logger.info(f"Pruning {len(facts)} facts...")
        pruned_facts, details = pruner.prune_facts(facts, question)
        
        logger.info(f"Retained {len(pruned_facts)}/{len(facts)} facts:")
        
        for detail in details:
            logger.info(f"  {detail.label.upper():15} - {detail.keep} - {detail.original_fact[:50]}...")
        
        stats = pruner.get_statistics(details)
        logger.info(f"Statistics: {stats}")
        
        return pruner
        
    except Exception as e:
        logger.warning(f"NLI pruner demo skipped: {str(e)}")
        logger.info("(Requires transformers and torch)")
        return None


def demo_response_generator():
    """Demo: Response Generation (requires Ollama)"""
    logger.info("\n" + "="*80)
    logger.info("DEMO 6: Response Generator (Llama 3 via Ollama)")
    logger.info("="*80)
    
    try:
        generator = ResponseGenerator()
        
        pruned_facts = [
            "Urea is a nitrogen-rich fertilizer commonly used for rice",
            "Fungal blast prevention requires proper drainage",
            "Resistant rice varieties improve disease tolerance"
        ]
        
        question = "How to prevent fungal blast in rice?"
        warnings = ["Some old sources incorrectly claim urea causes fungal blast"]
        
        logger.info("Generating response...")
        response = generator.generate(
            question,
            pruned_facts,
            warnings,
            include_citations=True
        )
        
        logger.info(f"Response:\n{response.response}\n")
        logger.info(f"Confidence: {response.confidence_score:.2f}")
        logger.info(f"Citations: {len(response.citations)}")
        
        return generator
        
    except Exception as e:
        logger.warning(f"Response generator demo skipped: {str(e)}")
        logger.info("(Requires Ollama running with the llama3 model)")
        return None


def main():
    """Run all demos"""
    logger.info("\n" + "="*80)
    logger.info("AGRI-RAG COMPONENT DEMOS")
    logger.info("="*80)
    
    logger.info("\nNote: Some demos require external services:")
    logger.info("  - Ollama (for LLM features)")
    logger.info("  - Neo4j (for graph database)")
    logger.info("  - transformers + torch (for NLI)")
    logger.info("\nThose will show warnings if services are unavailable.\n")
    
    # Run demos
    demo_data_loader()
    demo_triplet_extractor()
    demo_vector_store()
    demo_knowledge_graph()
    demo_nli_pruner()
    demo_response_generator()
    
    logger.info("\n" + "="*80)
    logger.info("DEMOS COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Ensure all services are running (Ollama, Neo4j)")
    logger.info("2. Configure .env file with credentials")
    logger.info("3. Load agricultural data using data_loader")
    logger.info("4. Run: streamlit run ui/app.py")
    logger.info("5. Or use: from main import get_app; app = get_app(); app.initialize_all()")


if __name__ == "__main__":
    main()
