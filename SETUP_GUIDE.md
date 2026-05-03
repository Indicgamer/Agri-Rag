# 🚀 AGRI-RAG COMPLETE STARTUP GUIDE
## All Problems Solved - Follow This Exactly

---

## ✅ WHAT WAS WRONG
1. ❌ Expired OpenAI API key (security risk + non-functional)
2. ❌ No external services running (Neo4j, Ollama)
3. ❌ Large PDF would cause huge API costs
4. ❌ No error handling if APIs fail
5. ❌ Entire app in memory for large files

## ✅ WHAT WE FIXED
1. ✅ Switched to free local Ollama LLM
2. ✅ Removed exposed API key from `.env`
3. ✅ Created optimized PDF streaming processor
4. ✅ Created safe `.env` template
5. ✅ Ready for graceful error handling

---

## 📋 PREREQUISITES

### Option A: Use Ollama (RECOMMENDED - FREE & LOCAL)
```bash
# 1. Install Ollama from https://ollama.ai
# 2. Download a model (in new terminal, keep it running):
ollama pull llama2
ollama serve

# 3. Verify it's running:
curl http://localhost:11434/api/tags
# Should show list of models
```

### Option B: Use Neo4j (OPTIONAL - Graph Database)
```bash
# Only if you want to test graph retrieval
# Install Docker first, then:
docker run --name agri-rag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Verify: http://localhost:7474/
```

---

## 🎯 QUICKEST WAY TO TEST (5 MINUTES)

### Terminal 1: Start Ollama
```bash
ollama serve
# Keep this running in background
```

### Terminal 2: Run Demo (No Setup Needed)
```bash
cd "d:\4th Sem Mtech\agri rag"
source venv/Scripts/activate  # or .venv/Scripts/activate
python local_demo.py "How can I prevent fungal blast in rice?"
```

**Expected Output**:
```
✓ Processing query...
✓ Retrieving relevant facts...
✓ Pruning with NLI...
✓ Generating response...
🤖 Response: [grounded agricultural advice]
```

---

## 🎯 STREAMLIT WEB APP (10 MINUTES)

### Terminal 1: Start Ollama
```bash
ollama serve
```

### Terminal 2: Start Streamlit
```bash
cd "d:\4th Sem Mtech\agri rag"
source venv/Scripts/activate
streamlit run ui/app.py
```

**Opens at**: http://localhost:8501

**Steps in UI**:
1. Click "🔧 Initialize System" button (sidebar)
2. Wait for all components to load
3. Enter question in "Ask Agricultural Questions" tab
4. Click "🔍 Get Advice"

---

## 📄 PROCESSING LARGE PDF (YOUR USE CASE)

### Method 1: Stream-Process with Batching (RECOMMENDED)
```python
# Script to efficiently process Agriculture-CPG-2020.pdf
# (Add to your project)

from src.utils.pdf_processor_optimized import OptimizedPDFProcessor
from src.models.triplet_extractor import TripletExtractor

processor = OptimizedPDFProcessor(
    chunk_size=512,
    chunk_overlap=50,
    batch_size=10,
    max_pages=5  # Test with first 5 pages
)

extractor = TripletExtractor()  # Uses Ollama

batch_count = 0
total_triplets = 0

for batch in processor.process_pdf_streaming("data/Agriculture-CPG-2020.pdf"):
    batch_count += 1
    print(f"Processing batch {batch_count}: {len(batch)} chunks")
    
    # Process each chunk
    for chunk in batch:
        triplets = extractor.extract_triplets(
            chunk.content,
            document_source=chunk.source,
            document_page=chunk.page
        )
        total_triplets += len(triplets)
        print(f"  Page {chunk.page}: {len(triplets)} triplets extracted")

print(f"\n✓ Total: {total_triplets} triplets from {batch_count} batches")
```

### Method 2: Use Sample First (SAFEST)
```bash
# Test with small sample before processing entire PDF
python -c "
from src.utils.data_loader import DataLoader

loader = DataLoader()
docs = loader.load_txt('data/sample_rice_blast.txt')
print(f'Sample loaded: {len(docs)} chunks')
for doc in docs[:3]:
    print(f'  - {doc.content[:60]}...')
"
```

---

## 🔍 TROUBLESHOOTING

### Problem: "Connection refused" on Ollama
```bash
# Make sure Ollama is running in another terminal:
ollama serve
# If still failing:
curl http://localhost:11434/api/tags  # Test connection
```

### Problem: "Neo4j connection failed"
```bash
# Neo4j is optional, you can skip it
# Just disable in ui/app.py or use local_demo.py instead
```

### Problem: "CUDA out of memory" or slow responses
```bash
# Ollama uses CPU by default
# If you have GPU, check Ollama docs for GPU setup
# Otherwise use smaller model:
ollama pull mistral  # Smaller & faster
```

### Problem: "PyPDF2 not found"
```bash
source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 COST COMPARISON

| Scenario | Method | Cost | Speed | Memory |
|----------|--------|------|-------|--------|
| **Test locally** | Ollama | $0 | Slow | Low |
| **Process large PDF** | Ollama Batch | $0 | Slow | Low |
| **Production** | OpenAI | $$$$ | Fast | Low |
| **Hybrid** | Ollama extraction + OpenAI response | $$ | Medium | Low |

---

## 🎓 NEXT STEPS

### Immediate (Done)
- ✅ Fixed `.env` configuration
- ✅ Created optimized PDF processor
- ✅ Removed exposed API key

### This Session
- [ ] Start Ollama service
- [ ] Test with `local_demo.py`
- [ ] Run Streamlit app

### Future
- [ ] Process full PDF with batching
- [ ] Set up Neo4j for graph retrieval
- [ ] Add to Git with proper `.gitignore`
- [ ] Implement retry logic for production

---

## 📚 KEY FILES

| File | Purpose |
|------|---------|
| `.env` | Configuration (UPDATED) |
| `.env.example.safe` | Template for safe setup |
| `src/utils/pdf_processor_optimized.py` | NEW: Efficient PDF streaming |
| `local_demo.py` | Quick test without services |
| `ui/app.py` | Streamlit web interface |
| `main.py` | RAG pipeline orchestrator |

---

## ⚠️ SECURITY REMINDERS

```bash
# NEVER commit these to Git:
echo ".env" >> .gitignore
echo "*.pdf" >> .gitignore  # Keep PDFs local
echo "data/chroma_db/*" >> .gitignore

# ALWAYS use placeholders in .env.example:
# ❌ OPENAI_API_KEY=sk-proj-abc...xyz
# ✅ OPENAI_API_KEY=your_key_here
```

---

## 🆘 STILL STUCK?

Run this diagnostic:
```bash
# Check Python
python --version  # Should be 3.11 or 3.12

# Check venv activation
which python  # Should show your venv path

# Check Ollama
curl http://localhost:11434/api/tags

# Check dependencies
pip list | grep -E "ollama|langchain|chromadb|neo4j"
```

Good luck! 🌾
