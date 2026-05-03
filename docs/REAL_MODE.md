# Running Agri-RAG In Real Mode

Real mode means:

- LLM triplet extraction uses a configured LLM provider.
- LLM response generation uses a configured LLM provider.
- NLI pruning uses a HuggingFace transformer model.
- Rule-based and deterministic fallbacks are disabled.

## Recommended `.env`

Use OpenAI if your API key has quota:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

ALLOW_RULE_FALLBACK=false
ALLOW_RESPONSE_FALLBACK=false

NLI_PROVIDER=transformers
NLI_MODEL=cross-encoder/nli-deberta-v3-base
NLI_DEVICE=cpu

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
AGRI_RAG_NEO4J_DATABASE=neo4j
```

Use local Ollama if API quota is unavailable:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434

ALLOW_RULE_FALLBACK=false
ALLOW_RESPONSE_FALLBACK=false

NLI_PROVIDER=transformers
NLI_MODEL=cross-encoder/nli-deberta-v3-base
NLI_DEVICE=cpu
```

## Demo Mode

Only use this when you need the app to keep running without model/API access:

```env
NLI_PROVIDER=heuristic
ALLOW_RULE_FALLBACK=true
ALLOW_RESPONSE_FALLBACK=true
```

## Data

Place real source documents in `data/`:

- `.txt` files for cleaned advisory text
- `.pdf` files for guides/manuals

Then run:

```bash
streamlit run ui/app.py
```

In the app:

1. Initialize System
2. Data Management
3. Clear KG + Vector Store if you previously loaded demo data
4. Build Knowledge Base

If graph counts stay at zero in real mode, check the logs for LLM extraction
errors. In strict real mode, failed extraction will not silently create fake
triplets.
