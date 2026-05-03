# Agri-RAG Build Path

## 1. Install a Stable Python

Use Python 3.11 or 3.12 for this project. The current workspace venv points to
Python 3.14, which is not a good target for the ML dependencies in
`requirements.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is unavailable, install Python 3.12 and reopen the terminal.

## 2. Verify the Local Demo First

The local demo uses only the standard library:

```powershell
python local_demo.py "How can I prevent fungal blast in rice?"
```

This validates the project flow before connecting Ollama, ChromaDB, Neo4j, and
the NLI model.

## 3. Bring Up External Services

Choose one LLM provider.

OpenAI API:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Ollama:

```powershell
ollama serve
ollama pull llama3.2:1b
```

Neo4j with Docker:

```powershell
docker run --name agri-rag-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

Then copy `.env.example` to `.env` and set the Neo4j password.

## 4. Run the App

```powershell
streamlit run ui/app.py
```
