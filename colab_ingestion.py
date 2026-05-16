# ======================================================================================
# AGRI-RAG PHASE 1: TURBO INGESTION (COLAB + AURA + GROQ)
# Student: Sachidanand C G | USN: 1RV24SCS11 | RVCE
# ======================================================================================

# 1. INSTALL DEPENDENCIES
!pip install -q langchain langchain-groq neo4j sentence-transformers pypdf faiss-cpu

import asyncio
import json
import logging
import time
import os
from google.colab import files
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from neo4j import GraphDatabase

# --- 2. CONFIGURATION (ENTER YOUR KEYS HERE) ---
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"
AURA_URI = "neo4j+s://PASTE_YOUR_INSTANCE_ID.databases.neo4j.io"
AURA_USER = "neo4j"
AURA_PWD = "PASTE_YOUR_AURA_PASSWORD_HERE"

# Logic Settings
CHUNK_SIZE = 1200 
CHUNK_OVERLAP = 150
CONCURRENT_TASKS = 3 # Keep it low for Groq Free Tier to avoid Rate Limits

# Files for Phase 2 Evaluation
TRIPLETS_FILE = "triplets_verbalized.jsonl"
RAGAS_GT_FILE = "ragas_ground_truth.jsonl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgriRAG")

# --- 3. NEO4J SCHEMA SETUP ---
def setup_aura_schema():
    """Initializes constraints in Neo4j Aura to prevent duplicates"""
    try:
        driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PWD))
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.label)"
        ]
        with driver.session() as session:
            for c in constraints:
                session.run(c)
        driver.close()
        logger.info("✅ Neo4j Aura Schema initialized (Constraints & Indices set).")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j Aura: {e}")
        raise

# --- 4. ASYNC EXTRACTION (TRIPLETS + VERBALIZATION + RAGAS QA) ---
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.1)

async def extract_and_generate_gt(chunk_text, semaphore):
    """Processes one chunk to get triplets and one Q&A pair for RAGAS"""
    async with semaphore:
        prompt = f"""
        Act as an Indian Agriculture Expert. Analyze the text/table below:
        1. Extract (Subject, Relation, Object) triplets.
        2. Provide a 'verbalized' full sentence for each triplet.
        3. Classify Subject: Crop, Disease, Pest, Chemical, or Symptom.
        4. Generate ONE Question-Answer pair based strictly on this text for RAGAS evaluation.

        Text: {chunk_text}

        Return ONLY a JSON object:
        {{
          "triplets": [
            {{"subject": "Rice", "relation": "requires", "object": "Nitrogen", "verbalized": "Rice requires Nitrogen for optimal growth.", "category": "Crop"}}
          ],
          "ragas_qa": {{"question": "...", "answer": "...", "ground_truth": "..."}}
        }}
        """
        try:
            response = await llm.ainvoke(prompt)
            clean_json = response.content.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"⚠️ Extraction Error in chunk: {e}")
            return None

# --- 5. BATCH UPLOAD TO NEO4J ---
BATCH_QUERY = """
UNWIND $batch AS row
MERGE (s:Entity {name: row.subject})
ON CREATE SET s.label = row.category
MERGE (o:Entity {name: row.object})
WITH s, o, row
CALL apoc.create.relationship(s, row.relation, {verbalized: row.verbalized, source: 'TNAU_Manual_2020'}, o) 
YIELD rel
RETURN count(*)
"""

def upload_to_aura(triplets):
    if not triplets: return
    try:
        driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PWD))
        with driver.session() as session:
            session.run(BATCH_QUERY, batch=triplets)
        driver.close()
    except Exception as e:
        logger.error(f"⚠️ Neo4j Batch Upload Failed: {e}")

# --- 6. MAIN ORCHESTRATOR ---
async def run_ingestion(pdf_file_name):
    if not os.path.exists(pdf_file_name):
        logger.error(f"❌ File {pdf_file_name} not found! Upload it to Colab folder.")
        return

    setup_aura_schema()
    
    # Load and Chunk
    loader = PyPDFLoader(pdf_file_name)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(pages)
    
    logger.info(f"📄 Loaded {pdf_file_name}. Created {len(chunks)} chunks.")
    
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    start_time = time.time()

    # Clear old files
    for f in [TRIPLETS_FILE, RAGAS_GT_FILE]:
        if os.path.exists(f): os.remove(f)

    # Parallel Execution
    tasks = [extract_and_generate_gt(c.page_content, semaphore) for c in chunks]
    
    logger.info(f"🚀 Starting Extraction & Sync (Concurrent Tasks: {CONCURRENT_TASKS})...")
    
    # Process in batches of 30 chunks to show progress
    for i in range(0, len(tasks), 30):
        batch_results = await asyncio.gather(*tasks[i:i+30])
        
        valid_triplets = []
        for res in batch_results:
            if res:
                valid_triplets.extend(res.get('triplets', []))
                
                # Save Ground Truth for Phase 2 Evaluation
                with open(RAGAS_GT_FILE, 'a') as f:
                    f.write(json.dumps(res.get('ragas_qa')) + "\n")
                
                # Save Triplets for local backup
                with open(TRIPLETS_FILE, 'a') as f:
                    for t in res.get('triplets', []):
                        f.write(json.dumps(t) + "\n")

        upload_to_aura(valid_triplets)
        logger.info(f"✅ Progress: {min(i+30, len(chunks))}/{len(chunks)} chunks synced.")

    duration = (time.time() - start_time) / 60
    logger.info(f"🎉 SUCCESS! Ingested into Aura in {duration:.2f} mins.")
    
    # Trigger File Downloads for Phase 2
    print("\n--- DOWNLOAD EVALUATION FILES ---")
    files.download(RAGAS_GT_FILE)
    files.download(TRIPLETS_FILE)

# --- 7. START PROCESS ---
# Replace "tnau_rice.pdf" with your actual file name uploaded to Colab
PDF_NAME = "tnau_rice.pdf" 

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply() # Required to run async in Jupyter/Colab
    asyncio.run(run_ingestion(PDF_NAME))