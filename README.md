# Agri-RAG

NLI-gated retrieval augmented generation for reliable Indian agriculture
advisory. The project combines document ingestion, triplet extraction,
ChromaDB vector retrieval, Neo4j knowledge-graph retrieval, NLI pruning, and a
grounded response generator.

## Current Build Status

This repository now has two paths:

1. `local_demo.py` - a dependency-light standard-library demo of the full RAG
   flow. Use this first to verify the pipeline idea.
2. Full stack - Streamlit + ChromaDB + Neo4j + Ollama + DeBERTa, used for the
   actual Phase 3 implementation.

The existing `venv/` in this workspace points to an inaccessible Python 3.14
install. Use Python 3.11 or 3.12 for the ML stack.

## LLM Provider

The project supports two LLM modes:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b
```

or:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

The LLM is used in two places:

- Triplet extraction: document chunks -> subject/relation/object facts for Neo4j
- Response generation: pruned retrieved facts -> final grounded advisory answer

Neo4j stores the KG, but the LLM creates the candidate triplets that are loaded
into it.

For strict, non-demo execution, see `docs/REAL_MODE.md`.

## Quick Start

Run the local demo:

```powershell
python local_demo.py "How can I prevent fungal blast in rice?"
```

Create a clean environment for the full stack:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is unavailable, install Python 3.12 and reopen the terminal.

## External Services

Ollama:

```powershell
ollama serve
ollama pull llama3
```

Neo4j with Docker:

```powershell
docker run --name agri-rag-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

Environment:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` with the Neo4j password.

Run the Streamlit app:

```powershell
streamlit run ui/app.py
```

## Project Structure

```text
agri rag/
  configs/              Settings and YAML config
  data/                 Local source documents and generated stores
  docs/                 Build notes
  notebooks/            Experiments
  src/
    models/             Triplet extraction, response generation, pipeline
    pruning/            NLI pruning
    retrieval/          ChromaDB, Neo4j, hybrid retrieval
    utils/              Data loading
  ui/                   Streamlit app
  local_demo.py         Standard-library end-to-end demo
```

## Core Flow

```text
User question
  -> Hybrid retrieval from vector store and knowledge graph
  -> NLI pruning of retrieved facts
  -> Grounded response generation
  -> Citations, confidence, and warnings
```

## Research Scope

Target crops: rice, wheat, cotton, maize, and sugarcane.

Trusted source direction: TNAU crop guides, ICAR/KVK manuals, and CIBRC
pesticide safety information.

Primary research contribution: NLI-gated pruning to reduce context clutter and
surface contradictions before generation.
