# Colab Ingestion Workflow

Use Colab for the expensive document-to-triplet extraction step, then import the
exported triplets into your local Neo4j.

## 1. Upload Files To Colab

Create this folder in Colab and upload PDFs/TXT files:

```python
!mkdir -p /content/agri_data
```

Upload `notebooks/colab_ingestion.py` to Colab, or copy it into a cell/file.

## 2. Install Dependencies

```python
!pip install -q pypdf tqdm pandas requests
```

## 3. Set API Key

For Groq:

```python
import os
os.environ["PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = "your_key_here"
os.environ["MODEL_NAME"] = "llama-3.3-70b-versatile"
```

If that model is unavailable on your Groq account, set `MODEL_NAME` to another
Groq chat model you have access to.

For OpenAI-compatible usage:

```python
import os
os.environ["PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "your_key_here"
os.environ["MODEL_NAME"] = "gpt-4o-mini"
```

## 4. Run Extraction

Small test run first:

```python
!python colab_ingestion.py \
  --input_dir /content/agri_data \
  --output_dir /content/agri_exports \
  --max_chunks 20
```

Full run:

```python
!python colab_ingestion.py \
  --input_dir /content/agri_data \
  --output_dir /content/agri_exports
```

Outputs:

- `/content/agri_exports/chunks.jsonl`
- `/content/agri_exports/triplets.jsonl`
- `/content/agri_exports/triplets.csv`
- `/content/agri_exports/neo4j_import_triplets.cypher`
- `/content/agri_exports/errors.jsonl` if any chunks fail

The script resumes automatically if `triplets.jsonl` already exists.

## 5. Download Triplets

Download `/content/agri_exports/triplets.jsonl` to your project machine.

## 6. Import Locally

From your local project root:

```powershell
python scripts/import_triplets_to_neo4j.py --triplets path\to\triplets.jsonl
```

To replace the current graph:

```powershell
python scripts/import_triplets_to_neo4j.py --triplets path\to\triplets.jsonl --clear_first
```

Keep vector retrieval local by ingesting the same documents through your app, or
use your existing ChromaDB if it already has the PDF chunks.
