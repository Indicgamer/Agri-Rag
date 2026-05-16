"""
Agri-RAG Phase 1: Advanced Ingestion Engine
Author: Sachidanand C G (USN: 1RV24SCS11)
Features: Table-aware extraction, Verbalized Triplets, Incremental Neo4j Merging.
"""

import sys
import time
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.settings import settings
from src.utils.data_loader import DataLoader
from src.models.triplet_extractor import TripletExtractor
from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.faiss_retriever import FAISSVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AgriRAG - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
OUTPUT_TRIPLETS = Path("data/exports/triplets_verbalized.jsonl")
GROUND_TRUTH_FILE = Path("data/exports/ragas_ground_truth.jsonl")
OUTPUT_TRIPLETS.parent.mkdir(parents=True, exist_ok=True)

def setup_neo4j_schema(kg: KnowledgeGraph):
    """Ensure unique constraints for incremental loading"""
    logger.info("Configuring Neo4j Schema (Constraints & Indices)...")
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Crop) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pest) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chemical) REQUIRE ch.name IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.name)"
    ]
    for cmd in constraints:
        try:
            kg.query(cmd)
        except Exception as e:
            logger.debug(f"Constraint might already exist: {e}")

def get_label(entity_name: str, context: str) -> str:
    """Heuristic to map entity to specific Neo4j Labels"""
    crops = ['rice', 'wheat', 'maize', 'sugarcane', 'cotton']
    chemicals = ['urea', 'npk', 'pesticide', 'fungicide', 'insecticide', 'dosage']
    
    name_low = entity_name.lower()
    if any(c in name_low for c in crops): return "Crop"
    if any(ch in name_low for ch in chemicals): return "Chemical"
    if "symptom" in context.lower() or "look" in context.lower(): return "Symptom"
    if "pest" in context.lower() or "worm" in context.lower(): return "Pest"
    return "Entity"

def step1_load_documents(data_dir: str = "data") -> list:
    """Load documents with table-preservation settings"""
    logger.info("STEP 1: Loading Documents...")
    # Use a slightly larger chunk size to keep tables together
    loader = DataLoader(chunk_size=600, chunk_overlap=100)
    documents = loader.load_directory(data_dir)
    logger.info(f"Loaded {len(documents)} document chunks (Table-Aware).")
    return documents

def step2_extract_and_verbalize(documents: list) -> int:
    """Extract S-R-O and generate Verbalized Premises for NLI"""
    logger.info("STEP 2: Triplet Extraction & Verbalization...")
    
    # Custom prompt for Llama-3 to handle tables and verbalization
    extractor = TripletExtractor(
        provider="openrouter", # Using MiniMax or Llama-3 via OpenRouter
        custom_prompt="""
        Extract agricultural triplets from the text/table below.
        For each triplet, provide:
        1. Subject, Relation, Object.
        2. A 'verbalized' full sentence describing the fact (Premise).
        3. Category: Crop, Disease, Pest, Chemical, or Symptom.
        
        Example JSON Output:
        {"triplets": [{"subject": "Rice", "relation": "requires_nitrogen", "object": "120kg/ha", "verbalized": "Rice requires nitrogen at a dosage of 120kg per hectare.", "category": "Chemical"}]}
        """
    )
    
    kg = KnowledgeGraph(uri=settings.neo4j.uri, username=settings.neo4j.username, password=settings.neo4j.password)
    setup_neo4j_schema(kg)
    
    total_triplets = 0
    for idx, doc in enumerate(documents, 1):
        try:
            # Extract
            raw_content = doc.content if hasattr(doc, 'content') else str(doc)
            res = extractor.extract_triplets(raw_content) # Assuming this returns a list of objects
            
            for t in res:
                # Add metadata
                t_data = {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                    "verbalized": getattr(t, 'verbalized', f"{t.subject} {t.relation} {t.object}"),
                    "label": get_label(t.subject, raw_content),
                    "source": doc.metadata.get('source', 'TNAU_Manual')
                }
                
                # Incremental Save to File (For RAGAS/NLI)
                with open(OUTPUT_TRIPLETS, 'a') as f:
                    f.write(json.dumps(t_data) + "\n")
                
                # Incremental MERGE to Neo4j (Symbolic Storage)
                # This Cypher ensures no duplicates are created
                query = f"""
                MERGE (s:{t_data['label']} {{name: $sub}})
                MERGE (o:Entity {{name: $obj}})
                MERGE (s)-[r:{t_data['relation'].upper()}]->(o)
                SET r.verbalized = $verb, r.source = $src
                """
                kg.query(query, parameters={
                    "sub": t_data['subject'], "obj": t_data['object'], 
                    "verb": t_data['verbalized'], "src": t_data['source']
                })
                total_triplets += 1
            
            if idx % 10 == 0: logger.info(f"Processed {idx}/{len(documents)} chunks...")
                
        except Exception as e:
            logger.error(f"Error at doc {idx}: {e}")
            continue
            
    kg.close()
    return total_triplets

def step4_build_faiss_with_metadata(documents: list):
    """Build Vector Store with source tracking for RAGAS"""
    logger.info("STEP 4: Building FAISS Vector Store...")
    faiss_store = FAISSVectorStore(
        collection_name=settings.faiss.collection_name,
        embedding_model=settings.faiss.embedding_model,
        persistence_dir=str(settings.faiss.persistence_dir)
    )
    
    docs_ready = []
    for d in documents:
        docs_ready.append({
            "content": d.content if hasattr(d, 'content') else str(d),
            "metadata": d.metadata if hasattr(d, 'metadata') else {"source": "TNAU"}
        })
    
    faiss_store.add_documents(docs_ready)
    logger.info("FAISS Store Complete.")

def step5_optimized_nli_download():
    """Download NLI model with 8GB VRAM optimization (FP16)"""
    logger.info("STEP 5: Memory-Optimized NLI Download...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "cross-encoder/nli-deberta-v3-base" # Best for your project
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load in FP16 to save memory on RTX 3060
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    logger.info(f"NLI Model {model_name} cached successfully.")

def main():
    start_time = time.time()
    print("--- Agri-RAG Ingestion System Initialized ---")
    
    try:
        # 1. Load
        documents = step1_load_documents()
        
        # 2. Extract & Build KG (The Symbolic Part)
        # This handles Verbalization and Incremental Loading
        count = step2_extract_and_verbalize(documents)
        
        # 3. Vector DB (The Neural Retrieval Part)
        step4_build_faiss_with_metadata(documents)
        
        # 4. Logic Guard Prep
        step5_optimized_nli_download()
        
        duration = (time.time() - start_time) / 60
        logger.info(f"Ingestion Complete! {count} verbalized triplets ingested in {duration:.2f} mins.")
        logger.info(f"Knowledge Graph is live in Neo4j. Triplets exported for NLI: {OUTPUT_TRIPLETS}")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Critical Ingestion Error: {e}")

if __name__ == "__main__":
    main()