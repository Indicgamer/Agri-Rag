"""
Main Application Entry Point for Agri-RAG
Initializes and manages the complete system
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.settings import settings
from src.utils.data_loader import DataLoader
from src.models.triplet_extractor import TripletExtractor
from src.retrieval.vector_retriever import VectorStore
from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pruning.nli_pruner import NLIPruner
from src.models.response_generator import ResponseGenerator
from src.models.rag_pipeline import AgriRAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AgriRAGApplication:
    """Main application class for Agri-RAG system"""
    
    def __init__(self):
        """Initialize all components"""
        self.data_loader = None
        self.triplet_extractor = None
        self.vector_store = None
        self.knowledge_graph = None
        self.hybrid_retriever = None
        self.nli_pruner = None
        self.response_generator = None
        self.pipeline = None
        
        logger.info("Agri-RAG Application initializing...")
    
    def initialize_retrieval(self):
        """Initialize retrieval components (Vector Store + Knowledge Graph)"""
        try:
            logger.info("Initializing retrieval components...")
            
            # Initialize Vector Store (ChromaDB)
            self.vector_store = VectorStore(
                collection_name=settings.chromadb.collection_name,
                embedding_model=settings.chromadb.embedding_model,
                persistence_dir=str(settings.chromadb.persistence_dir)
            )
            
            # Initialize Knowledge Graph (Neo4j)
            self.knowledge_graph = KnowledgeGraph(
                uri=settings.neo4j.uri,
                username=settings.neo4j.username,
                password=settings.neo4j.password,
                database=settings.neo4j.database
            )
            
            logger.info("OK Retrieval components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval: {str(e)}")
            return False
    
    def initialize_extraction(self):
        """Initialize data loading and triplet extraction"""
        try:
            logger.info("Initializing extraction components...")
            
            # Initialize Data Loader with safer chunk sizes for large PDF ingestion
            self.data_loader = DataLoader(
                chunk_size=500,
                chunk_overlap=25,
                min_chunk_length=100,
                chunk_strategy="paragraph"
            )
            
            # Initialize Triplet Extractor (Llama 3 via Ollama)
            self.triplet_extractor = TripletExtractor(
                provider=settings.triplet_extractor.provider,
                model_name=settings.triplet_extractor.model_name,
                openai_model=settings.triplet_extractor.openai_model,
                openai_base_url=settings.triplet_extractor.openai_base_url,
                api_key=settings.triplet_extractor.api_key,
                allow_rule_fallback=settings.triplet_extractor.allow_rule_fallback,
                base_url=settings.triplet_extractor.base_url,
                temperature=settings.triplet_extractor.temperature,
                max_tokens=settings.triplet_extractor.max_tokens
            )
            
            logger.info("OK Extraction components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize extraction: {str(e)}")
            return False
    
    def initialize_pruning_and_generation(self):
        """Initialize NLI pruner and response generator"""
        try:
            logger.info("Initializing pruning and generation components...")
            
            # Initialize NLI Pruner (DeBERTa-v3)
            self.nli_pruner = NLIPruner(
                provider=settings.nli.provider,
                model_name=settings.nli.model_name,
                device=settings.nli.device,
                hf_token=settings.nli.hf_token,
                entailment_threshold=settings.nli.entailment_threshold,
                neutral_threshold=settings.nli.neutral_threshold
            )
            
            # Initialize Response Generator (Llama 3 via Ollama)
            self.response_generator = ResponseGenerator(
                provider=settings.llm.provider,
                model_name=settings.llm.model_name,
                openai_model=settings.llm.openai_model,
                openai_base_url=settings.llm.openai_base_url,
                api_key=settings.llm.api_key,
                allow_response_fallback=settings.llm.allow_response_fallback,
                base_url=settings.llm.base_url,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens
            )
            
            logger.info("OK Pruning and generation components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pruning/generation: {str(e)}")
            return False
    
    def initialize_pipeline(self):
        """Initialize the complete RAG pipeline"""
        try:
            if not all([self.hybrid_retriever, self.nli_pruner, self.response_generator]):
                raise RuntimeError("Prerequisites not initialized")
            
            logger.info("Initializing RAG pipeline...")
            
            self.pipeline = AgriRAGPipeline(
                hybrid_retriever=self.hybrid_retriever,
                nli_pruner=self.nli_pruner,
                response_generator=self.response_generator
            )
            
            logger.info("OK RAG pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            return False
    
    def setup_hybrid_retriever(self):
        """Setup hybrid retriever"""
        try:
            if not all([self.vector_store, self.knowledge_graph]):
                raise RuntimeError("Vector store and knowledge graph required")
            
            logger.info("Setting up hybrid retriever...")
            
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                knowledge_graph=self.knowledge_graph,
                vector_weight=settings.retrieval.vector_weight,
                graph_weight=settings.retrieval.graph_weight,
                top_k_vector=settings.retrieval.top_k_vector,
                top_k_graph=settings.retrieval.top_k_graph
            )
            
            logger.info("OK Hybrid retriever setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid retriever: {str(e)}")
            return False
    
    def initialize_all(self):
        """Initialize entire application"""
        logger.info("="*80)
        logger.info("AGRI-RAG APPLICATION INITIALIZATION")
        logger.info("="*80)
        
        steps = [
            ("Retrieval Components", self.initialize_retrieval),
            ("Extraction Components", self.initialize_extraction),
            ("Hybrid Retriever", self.setup_hybrid_retriever),
            ("Pruning & Generation", self.initialize_pruning_and_generation),
            ("RAG Pipeline", self.initialize_pipeline)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n[{steps.index((step_name, step_func))+1}/{len(steps)}] {step_name}")
            if not step_func():
                logger.error(f"Initialization failed at: {step_name}")
                return False
            logger.info("-"*40)
        
        logger.info("\n" + "="*80)
        logger.info("OK ALL COMPONENTS INITIALIZED SUCCESSFULLY")
        logger.info("="*80 + "\n")
        
        return True
    
    def load_data(self, data_path: str):
        """Load agricultural data"""
        if not self.data_loader:
            logger.error("Data loader not initialized")
            return None
        
        logger.info(f"Loading data from: {data_path}")
        documents = self.data_loader.load_directory(data_path)
        logger.info(f"Loaded {len(documents)} documents")
        
        return documents
    
    def extract_triplets(self, documents):
        """Extract triplets from documents"""
        if not self.triplet_extractor:
            logger.error("Triplet extractor not initialized")
            return []
        
        logger.info(f"Extracting triplets from {len(documents)} documents...")
        triplets = []

        for idx, document in enumerate(documents, 1):
            logger.info(
                "Extracting triplets from chunk %s/%s (%s)",
                idx,
                len(documents),
                getattr(document, "source", "unknown")
            )
            extracted = self.triplet_extractor.extract_triplets(
                text=document.content,
                document_source=document.source,
                document_page=document.page
            )

            for triplet in extracted:
                triplet_dict = triplet.to_dict()
                triplet_dict["metadata"] = {
                    "confidence": triplet.confidence,
                    "source_doc": triplet.source_doc,
                    "source_text": triplet.source_text,
                    "chunk_id": document.chunk_id,
                    "page": document.page,
                }
                triplets.append(triplet_dict)

            if (
                hasattr(self.triplet_extractor, "is_rate_limited")
                and self.triplet_extractor.is_rate_limited()
            ):
                remaining = self.triplet_extractor.rate_limit_remaining_seconds()
                logger.warning(
                    "Stopping triplet extraction at chunk %s/%s because the provider is rate-limited "
                    "for another %.0f seconds. Ingested triplets collected so far will still be saved.",
                    idx,
                    len(documents),
                    remaining
                )
                break
        
        logger.info(f"Triplet extraction complete: {len(triplets)} triplets")
        return triplets
    
    def add_to_knowledge_graph(self, triplets):
        """Add triplets to knowledge graph"""
        if not self.knowledge_graph:
            logger.error("Knowledge graph not initialized")
            return False
        
        logger.info(f"Adding {len(triplets)} triplets to knowledge graph...")
        added = self.knowledge_graph.add_triplets_batch(triplets)
        logger.info(f"Added {added} triplets")
        
        return added > 0
    
    def add_to_vector_store(self, documents):
        """Add documents to vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return False
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        added = self.vector_store.add_documents(documents, "agricultural_data")
        logger.info(f"Added {added} documents")
        
        return added > 0

    def ingest_data(self, data_path: str):
        """Load documents, index them, extract triplets, and populate the KG."""
        documents = self.load_data(data_path)
        if not documents:
            return {
                "documents": 0,
                "vector_added": 0,
                "triplets": 0,
                "graph_added": 0,
                "success": False,
                "message": "No documents found. Add .txt or .pdf files to the data directory."
            }

        vector_added = self.vector_store.add_documents(documents, "agricultural_data")
        triplets = self.extract_triplets(documents)
        graph_added = self.knowledge_graph.add_triplets_batch(triplets) if triplets else 0

        return {
            "documents": len(documents),
            "vector_added": vector_added,
            "triplets": len(triplets),
            "graph_added": graph_added,
            "success": vector_added > 0 or graph_added > 0,
            "message": "Ingestion complete"
        }

    def reset_indexes(self):
        """Clear ChromaDB collection and Neo4j graph."""
        vector_cleared = self.vector_store.clear_collection() if self.vector_store else False
        graph_cleared = self.knowledge_graph.clear_graph() if self.knowledge_graph else False
        return {
            "vector_cleared": vector_cleared,
            "graph_cleared": graph_cleared,
            "success": vector_cleared and graph_cleared
        }
    
    def query(self, question: str):
        """Process a query through the pipeline"""
        if not self.pipeline:
            logger.error("Pipeline not initialized")
            return None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"USER QUERY: {question}")
        logger.info("="*80 + "\n")
        
        result = self.pipeline.process(question)
        
        return result
    
    def get_system_status(self):
        """Get system status"""
        status = {
            "components": {
                "data_loader": self.data_loader is not None,
                "triplet_extractor": self.triplet_extractor is not None,
                "vector_store": self.vector_store is not None,
                "knowledge_graph": self.knowledge_graph is not None,
                "hybrid_retriever": self.hybrid_retriever is not None,
                "nli_pruner": self.nli_pruner is not None,
                "response_generator": self.response_generator is not None,
                "pipeline": self.pipeline is not None
            },
            "configuration": {
                "crops": settings.crops,
                "retrieval_weights": {
                    "vector": settings.retrieval.vector_weight,
                    "graph": settings.retrieval.graph_weight
                },
                "models": {
                    "provider": settings.llm.provider,
                    "nli_provider": settings.nli.provider,
                    "triplet_extractor": settings.triplet_extractor.model_name,
                    "triplet_extractor_api": settings.triplet_extractor.openai_model,
                    "nli": settings.nli.model_name,
                    "llm": settings.llm.model_name,
                    "llm_api": settings.llm.openai_model
                }
            }
        }
        
        # Add database stats if available
        if self.knowledge_graph:
            status["database"] = self.knowledge_graph.get_statistics()
        
        if self.vector_store:
            status["vector_store"] = self.vector_store.get_collection_info()
        
        return status


# Global application instance
_app = None


def get_app() -> AgriRAGApplication:
    """Get or create application instance"""
    global _app
    if _app is None:
        _app = AgriRAGApplication()
    return _app


def main():
    """Main entry point"""
    app = get_app()
    
    # Initialize all components
    if not app.initialize_all():
        logger.error("Failed to initialize application")
        sys.exit(1)
    
    # Print system status
    logger.info("\nSystem Status:")
    import json
    status = app.get_system_status()
    logger.info(json.dumps(status, indent=2, default=str))
    
    # Ready for queries
    logger.info("\nOK Agri-RAG system is ready!")
    logger.info("Call app.query(question) to process a query")


if __name__ == "__main__":
    main()
