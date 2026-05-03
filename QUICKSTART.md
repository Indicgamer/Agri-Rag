# Agri-RAG Quickstart

## Fastest Check

Use the dependency-light demo first:

```powershell
python local_demo.py "How can I prevent fungal blast in rice?"
```

If Python is not available, install Python 3.12 and create a new environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Full Stack Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env`:

```powershell
Copy-Item .env.example .env
```

To use an API instead of local Ollama, edit `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

To use local Ollama, keep:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b
```

Start Ollama:

```powershell
ollama serve
ollama pull llama3
```

Start Neo4j:

```powershell
docker run --name agri-rag-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

Run Streamlit:

```powershell
streamlit run ui/app.py
```

## Useful Python Snippets

Load text/PDF documents:

```python
from src.utils.data_loader import DataLoader

loader = DataLoader(chunk_size=512, chunk_overlap=50)
documents = loader.load_directory("data")
```

Extract triplets:

```python
from src.models.triplet_extractor import TripletExtractor

extractor = TripletExtractor(model_name="llama3")
triplets = extractor.extract_triplets(
    "Proper drainage prevents fungal blast in rice.",
    document_source="demo.txt",
)
```

Run the app object:

```python
from main import get_app

app = get_app()
app.initialize_all()
result = app.query("How can I prevent fungal blast in rice?")
print(result.response)
```

## Notes

- Use Python 3.11 or 3.12, not Python 3.14.
- The full app expects Ollama and Neo4j to be running.
- `local_demo.py` is intentionally small and does not replace the full ML
  pipeline. It is a sanity check for the architecture.
