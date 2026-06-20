"""
Agri-RAG Phase 1: Advanced Ingestion Engine (Optimized)
Author: Sachidanand C G (USN: 1RV24SCS11)
Features: Table-aware extraction, Verbalized Triplets, Incremental Neo4j Merging.
Optimizations: Parallel API calls, Batch Neo4j inserts, Local cache.
"""

import signal
import sys
import time
import json
import logging
import re
import random
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
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
GROUND_TRUTH_FILE = Path("data/ragas_ground_truth.json")
CACHE_DIR = Path("data/cache/extraction_cache")
OUTPUT_TRIPLETS.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optimization configs
MAX_WORKERS = 8  # Parallel workers for cache/processing
MAX_CONCURRENT_API = 2  # Max concurrent OpenRouter calls (account-level rate limit)
BATCH_SIZE = 50  # Neo4j batch inserts
shutdown_requested = False  # Global flag for Ctrl+C handling

# Pool of free OpenRouter models for parallel extraction
# Each model has its own rate limit, so 8 threads across 8 models avoids rate limiting
FREE_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "minimax/minimax-m2.5:free",
]

MODEL_POOL = cycle(FREE_MODELS)
_model_lock = threading.Lock()
_api_semaphore = threading.Semaphore(MAX_CONCURRENT_API)

def get_next_model() -> str:
    """Thread-safe round-robin model selection"""
    with _model_lock:
        return next(MODEL_POOL)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        logger.warning("\nInterrupt received! Shutting down workers... Press Ctrl+C again to force quit.")
    else:
        logger.error("Force quit!")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def setup_neo4j_schema(kg: KnowledgeGraph):
    """Ensure unique constraints for incremental loading with specific labels"""
    logger.info("Configuring Neo4j Schema (Constraints & Indices)...")
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Crop) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pest) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chemical) REQUIRE ch.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.category)"
    ]
    for cmd in constraints:
        try:
            kg.query(cmd)
        except Exception as e:
            logger.debug(f"Constraint might already exist: {e}")

def get_label(entity_name: str, context: str, category: str = None) -> str:
    """Heuristic to map entity to specific Neo4j Labels"""
    if category:
        category_map = {
            'crop': 'Crop',
            'disease': 'Disease',
            'pest': 'Pest',
            'chemical': 'Chemical',
            'symptom': 'Symptom'
        }
        if category.lower() in category_map:
            return category_map[category.lower()]
    
    crops = ['rice', 'wheat', 'maize', 'sugarcane', 'cotton', 'sorghum', 'millet']
    diseases = ['blast', 'blight', 'rot', 'wilt', 'rust', 'spot', 'mosaic']
    pests = ['bollworm', 'stem borer', 'aphid', 'thrips', 'whitefly', 'caterpillar']
    chemicals = ['urea', 'npk', 'pesticide', 'fungicide', 'insecticide', 'dosage', 'fertilizer']
    
    name_low = entity_name.lower()
    if any(c in name_low for c in crops): return "Crop"
    if any(d in name_low for d in diseases): return "Disease"
    if any(p in name_low for p in pests): return "Pest"
    if any(ch in name_low for ch in chemicals): return "Chemical"
    if "symptom" in context.lower() or "look" in context.lower() or "appear" in context.lower(): return "Symptom"
    return "Entity"

def verbalize_triplet(subject: str, relation: str, obj: str, category: str = None) -> str:
    """Generate a natural language sentence from a triplet for NLI premise"""
    relation_text = relation.replace('_', ' ').lower()
    
    if category == 'Disease' or any(d in relation_text for d in ['disease', 'affects', 'causes']):
        return f"{subject} is affected by {obj}."
    elif category == 'Chemical' or any(c in relation_text for c in ['requires', 'needs', 'apply', 'dosage']):
        return f"{subject} requires {obj} for optimal growth."
    elif category == 'Pest' or any(p in relation_text for p in ['pest', 'controlled', 'prevented']):
        return f"{subject} can be controlled by {obj}."
    elif category == 'Symptom':
        return f"{subject} shows symptoms of {obj}."
    else:
        return f"{subject} {relation_text} {obj}."

def is_table_content(content: str) -> bool:
    """Detect if content contains tabular data"""
    table_indicators = [
        r'\|.*\|',
        r'\t.*\t',
        r'dosage|rate|amount|per hectare|kg/ha|l/ha|g/ha',
        r'crop.*disease.*control',
        r'stage.*fertilizer.*dose'
    ]
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in table_indicators)

def get_cache_key(content: str) -> str:
    """Generate cache key from content hash"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_from_cache(cache_key: str) -> list:
    """Load extracted triplets from cache if available"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed for {cache_key}: {e}")
    return None

def save_to_cache(cache_key: str, triplets: list):
    """Save extracted triplets to cache"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Cache save failed for {cache_key}: {e}")

def step1_load_documents(data_dir: str = "data") -> list:
    """Load documents with table-preservation settings"""
    logger.info("STEP 1: Loading Documents...")
    loader = DataLoader(chunk_size=600, chunk_overlap=100)
    documents = loader.load_directory(data_dir)
    logger.info(f"Loaded {len(documents)} document chunks (Table-Aware).")
    return documents

def extract_single_chunk(doc_idx: int, doc, model_name: str) -> dict:
    """Extract triplets from a single document chunk (for parallel execution)"""
    try:
        raw_content = doc.content if hasattr(doc, 'content') else str(doc)
        cache_key = get_cache_key(raw_content)
        
        # Check cache first
        cached = load_from_cache(cache_key)
        if cached is not None:
            logger.info(f"Doc {doc_idx}: Loaded from cache ({len(cached)} triplets)")
            return {
                'doc_idx': doc_idx,
                'triplets': cached,
                'is_table': is_table_content(raw_content),
                'source': doc.metadata.get('source', 'TNAU_Manual'),
                'raw_content': raw_content,
                'from_cache': True
            }
        
        is_table = is_table_content(raw_content)
        if is_table:
            logger.debug(f"Doc {doc_idx}: Detected table content")
        
        # Throttle concurrent API calls to avoid account-level rate limits
        with _api_semaphore:
            # Create a fresh extractor with a unique model from the pool
            extractor = TripletExtractor(
                provider="openrouter",
                model_name=model_name,
            temperature=0.1,
            max_tokens=2000,
            custom_prompt="""Extract agricultural triplets from the text below. Return ONLY valid JSON.

Format: {"triplets": [{"subject": "...", "relation": "...", "object": "...", "verbalized": "...", "category": "..."}]}

Rules:
- subject: main entity (crop, disease, pest, chemical)
- relation: action (requires, prevents, causes, controls, affects)
- object: target entity or value
- verbalized: full sentence describing the fact
- category: Crop, Disease, Pest, Chemical, or Symptom

For tables, extract each row as separate triplets.
For dosage info, capture crop, fertilizer/pesticide, dosage, application method.
For diseases, capture disease, symptoms, control measures, prevention.

Text:
"""
        )
        
        res = extractor.extract_triplets(raw_content)
        
        # Convert to serializable format
        triplet_dicts = []
        for t in res:
            verbalized = getattr(t, 'verbalized', None)
            if not verbalized:
                category = getattr(t, 'category', None)
                verbalized = verbalize_triplet(t.subject, t.relation, t.object, category)
            
            category = getattr(t, 'category', None)
            triplet_dicts.append({
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "verbalized": verbalized,
                "category": category or "General",
            })
        
        # Save to cache
        save_to_cache(cache_key, triplet_dicts)
        
        return {
            'doc_idx': doc_idx,
            'triplets': triplet_dicts,
            'is_table': is_table,
            'source': doc.metadata.get('source', 'TNAU_Manual'),
            'raw_content': raw_content,
            'from_cache': False
        }
        
    except Exception as e:
        logger.error(f"Error extracting doc {doc_idx}: {e}")
        return {
            'doc_idx': doc_idx,
            'triplets': [],
            'is_table': False,
            'source': 'unknown',
            'raw_content': '',
            'from_cache': False,
            'error': str(e)
        }

def batch_insert_to_neo4j(kg: KnowledgeGraph, batch: list):
    """Insert a batch of triplets to Neo4j using UNWIND for efficiency"""
    if not batch:
        return
    
    query = """
    UNWIND $batch AS item
    MERGE (s:Label {name: item.sub})
    WITH s, item
    CALL apoc.create.addLabels(s, [item.label]) YIELD node AS labeled_s
    MERGE (o:Entity {name: item.obj})
    MERGE (labeled_s)-[r:REL {type: item.rel}]->(o)
    SET r.verbalized = item.verb,
        r.source = item.src,
        r.category = item.cat,
        r.is_table_derived = item.is_table
    RETURN count(*) AS count
    """
    
    # Simpler approach without apoc - use dynamic labels via separate queries per label type
    # Group by label for efficient batch inserts
    by_label = {}
    for t in batch:
        label = t['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(t)
    
    total_inserted = 0
    for label, items in by_label.items():
        params = {
            "items": [{
                "sub": t['subject'],
                "obj": t['object'],
                "verb": t['verbalized'],
                "src": t['source'],
                "cat": t['category'],
                "rel": t['relation'].upper(),
                "is_table": t['is_table_derived']
            } for t in items]
        }
        
        cypher = f"""
        UNWIND $items AS item
        MERGE (s:{label} {{name: item.sub}})
        MERGE (o:Entity {{name: item.obj}})
        MERGE (s)-[r:{label}_REL]->(o)
        SET r.verbalized = item.verb,
            r.source = item.src,
            r.category = item.cat,
            r.is_table_derived = item.is_table,
            r.type = item.rel
        """
        
        try:
            kg.query(cypher, parameters=params)
            total_inserted += len(items)
        except Exception as e:
            logger.error(f"Batch insert failed for label {label}: {e}")
            # Fallback to individual inserts
            for t in items:
                try:
                    single_query = f"""
                    MERGE (s:{label} {{name: $sub}})
                    MERGE (o:Entity {{name: $obj}})
                    MERGE (s)-[r:{label}_REL]->(o)
                    SET r.verbalized = $verb, r.source = $src, r.category = $cat, r.is_table_derived = $is_table, r.type = $rel
                    """
                    kg.query(single_query, parameters={
                        "sub": t['subject'],
                        "obj": t['object'],
                        "verb": t['verbalized'],
                        "src": t['source'],
                        "cat": t['category'],
                        "is_table": t['is_table_derived'],
                        "rel": t['relation'].upper()
                    })
                    total_inserted += 1
                except Exception as e2:
                    logger.error(f"Single insert failed: {e2}")
    
    return total_inserted

def step2_extract_and_verbalize(documents: list) -> int:
    """Extract S-R-O and generate Verbalized Premises for NLI (Optimized)"""
    logger.info("STEP 2: Triplet Extraction & Verbalization (Parallel + Cached + Batch)...")
    
    kg = KnowledgeGraph(uri=settings.neo4j.uri, username=settings.neo4j.username, password=settings.neo4j.password)
    setup_neo4j_schema(kg)
    
    total_triplets = 0
    cache_hits = 0
    ground_truth_pairs = []
    all_triplets_for_file = []
    
    # Phase 1: Parallel extraction with caching
    logger.info(f"Phase 1: Extracting {len(documents)} chunks with {MAX_WORKERS} parallel workers...")
    extraction_start = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(extract_single_chunk, idx, doc, get_next_model()): idx
            for idx, doc in enumerate(documents, 1)
        }
        
        completed = 0
        for future in as_completed(futures):
            global shutdown_requested
            if shutdown_requested:
                logger.info("Shutdown requested - cancelling remaining tasks...")
                executor.shutdown(wait=False, cancel_futures=True)
                break
            
            result = future.result()
            completed += 1
            
            if result.get('from_cache'):
                cache_hits += 1
            
            triplets = result.get('triplets', [])
            raw_content = result.get('raw_content', '')
            source = result.get('source', 'TNAU_Manual')
            is_table = result.get('is_table', False)
            
            for t in triplets:
                label = get_label(t['subject'], raw_content, t.get('category'))
                t_data = {
                    **t,
                    "label": label,
                    "source": source,
                    "is_table_derived": is_table
                }
                all_triplets_for_file.append(t_data)
                
                if t.get('category') in ['Disease', 'Pest', 'Chemical']:
                    qa_pair = generate_ground_truth_qa(t_data)
                    if qa_pair:
                        ground_truth_pairs.append(qa_pair)
            
            total_triplets += len(triplets)
            
            if completed % 50 == 0:
                elapsed = time.time() - extraction_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {completed}/{len(documents)} chunks ({total_triplets} triplets, {cache_hits} cache hits, {rate:.1f} chunks/sec)")
    
    extraction_time = time.time() - extraction_start
    logger.info(f"Extraction complete: {total_triplets} triplets in {extraction_time:.1f}s ({len(documents)/extraction_time:.1f} chunks/sec)")
    
    if shutdown_requested:
        logger.warning("Shutdown requested - saving partial results...")
    
    # Phase 2: Batch Neo4j inserts
    logger.info(f"Phase 2: Batch inserting {total_triplets} triplets to Neo4j...")
    neo4j_start = time.time()
    
    for i in range(0, len(all_triplets_for_file), BATCH_SIZE):
        if shutdown_requested:
            break
        batch = all_triplets_for_file[i:i + BATCH_SIZE]
        inserted = batch_insert_to_neo4j(kg, batch)
        if inserted:
            logger.info(f"Batch {i//BATCH_SIZE + 1}: Inserted {inserted} triplets")
    
    neo4j_time = time.time() - neo4j_start
    logger.info(f"Neo4j batch inserts complete in {neo4j_time:.1f}s")
    
    # Phase 3: Save to file
    logger.info(f"Phase 3: Saving {len(all_triplets_for_file)} triplets to {OUTPUT_TRIPLETS}...")
    with open(OUTPUT_TRIPLETS, 'w', encoding='utf-8') as f:
        for t in all_triplets_for_file:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    kg.close()
    
    if ground_truth_pairs:
        save_ground_truth_dataset(ground_truth_pairs)
    
    logger.info(f"Cache stats: {cache_hits}/{len(documents)} chunks served from cache")
    
    return total_triplets

def generate_ground_truth_qa(triplet_data: dict) -> dict:
    """Generate Q&A pairs from triplets for RAGAS evaluation"""
    category = triplet_data.get('category', '')
    subject = triplet_data['subject']
    obj = triplet_data['object']
    
    qa_templates = {
        'Disease': [
            {"question": f"How to control {obj} in {subject}?", "answer": triplet_data['verbalized']},
            {"question": f"What causes {obj} in {subject}?", "answer": triplet_data['verbalized']}
        ],
        'Pest': [
            {"question": f"How to prevent {obj} in {subject}?", "answer": triplet_data['verbalized']},
            {"question": f"What controls {obj} affecting {subject}?", "answer": triplet_data['verbalized']}
        ],
        'Chemical': [
            {"question": f"What fertilizer does {subject} need?", "answer": triplet_data['verbalized']},
            {"question": f"How much {obj} should I apply to {subject}?", "answer": triplet_data['verbalized']}
        ]
    }
    
    if category in qa_templates:
        return random.choice(qa_templates[category])
    return None

def save_ground_truth_dataset(qa_pairs: list):
    """Save ground truth Q&A pairs for RAGAS evaluation"""
    if not qa_pairs:
        return
    
    seen_questions = set()
    unique_pairs = []
    for pair in qa_pairs:
        if pair['question'] not in seen_questions:
            seen_questions.add(pair['question'])
            unique_pairs.append(pair)
    
    if len(unique_pairs) > 20:
        unique_pairs = random.sample(unique_pairs, 20)
    
    output = {
        "questions": unique_pairs,
        "metadata": {
            "source": "TNAU_Manuals_Auto_Generated",
            "count": len(unique_pairs),
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(unique_pairs)} ground truth Q&A pairs to {GROUND_TRUTH_FILE}")

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
    
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    logger.info(f"NLI Model {model_name} cached successfully.")

def main():
    start_time = time.time()
    print("--- Agri-RAG Ingestion System Initialized (Optimized) ---")
    print(f"Optimizations: {MAX_WORKERS} parallel workers, {BATCH_SIZE} batch Neo4j inserts, local cache")
    
    try:
        documents = step1_load_documents()
        count = step2_extract_and_verbalize(documents)
        step4_build_faiss_with_metadata(documents)
        step5_optimized_nli_download()
        
        duration = (time.time() - start_time) / 60
        logger.info(f"Ingestion Complete! {count} verbalized triplets ingested in {duration:.2f} mins.")
        logger.info(f"Knowledge Graph is live in Neo4j. Triplets exported for NLI: {OUTPUT_TRIPLETS}")
        logger.info(f"Ground truth Q&A pairs saved to: {GROUND_TRUTH_FILE}")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Critical Ingestion Error: {e}")

if __name__ == "__main__":
    main()
