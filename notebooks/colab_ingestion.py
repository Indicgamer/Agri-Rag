"""
Colab-ready ingestion program for Agri-RAG.

Use this in Google Colab to do the expensive document -> triplet extraction step
with more RAM/compute or a stronger API model, then bring the exported files back
to the local app.

Colab quick start:
    !pip install -q pypdf tqdm pandas requests
    !python colab_ingestion.py --input_dir /content/agri_data --output_dir /content/agri_exports

Supported providers:
    PROVIDER=groq
    GROQ_API_KEY=...
    MODEL_NAME=llama-3.3-70b-versatile

    PROVIDER=openai
    OPENAI_API_KEY=...
    MODEL_NAME=gpt-4o-mini

    PROVIDER=openrouter
    OPENROUTER_API_KEY=...
    MODEL_NAME=tencent/hy3-preview
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from tqdm import tqdm

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - Colab install step handles this.
    PdfReader = None


RELATION_TYPES = {
    "AFFECTS",
    "CAUSES",
    "PREVENTS",
    "TREATS",
    "RECOMMENDS",
    "CONTRAINDICATED_FOR",
    "OCCURS_IN",
    "REQUIRES",
    "IMPROVES",
    "REDUCES",
    "RELATED_TO",
}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: Optional[int]
    text: str


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str
    confidence: float
    source_doc: str
    source_page: Optional[int]
    source_chunk_id: str
    source_text: str
    extractor_model: str


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"Page \d+|^\d+$", "", text, flags=re.MULTILINE)
    return "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip())


def chunk_text(
    text: str,
    source: str,
    page: Optional[int],
    chunk_words: int,
    overlap_words: int,
) -> List[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        digest = hashlib.sha1(f"{source}:{page}:{start}:{chunk[:80]}".encode("utf-8")).hexdigest()[:16]
        chunks.append(Chunk(chunk_id=digest, source=source, page=page, text=chunk))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def load_pdf(path: Path, chunk_words: int, overlap_words: int, max_pages: Optional[int]) -> List[Chunk]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Run: !pip install pypdf")

    reader = PdfReader(str(path))
    chunks: List[Chunk] = []
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    for page_idx, page in enumerate(pages, start=1):
        text = clean_text(page.extract_text() or "")
        chunks.extend(chunk_text(text, path.name, page_idx, chunk_words, overlap_words))
    return chunks


def load_txt(path: Path, chunk_words: int, overlap_words: int) -> List[Chunk]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return chunk_text(text, path.name, None, chunk_words, overlap_words)


def load_documents(input_dir: Path, chunk_words: int, overlap_words: int, max_pages: Optional[int]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            chunks.extend(load_pdf(path, chunk_words, overlap_words, max_pages))
        elif path.suffix.lower() in {".txt", ".md"}:
            chunks.extend(load_txt(path, chunk_words, overlap_words))
    return chunks


def build_prompt(text: str) -> str:
    relations = ", ".join(sorted(RELATION_TYPES))
    return f"""Extract agricultural knowledge triplets from the text.

Return ONLY valid JSON as an array. Each item must have:
subject, relation, object, confidence

Rules:
- Use only facts explicitly present in the text.
- relation must be one of: {relations}
- Keep entity phrases short but meaningful.
- Do not invent facts.
- If no clear agricultural triplets exist, return [].

Example:
Text: Proper drainage prevents fungal blast in rice fields.
Output: [{{"subject":"Proper drainage","relation":"PREVENTS","object":"fungal blast in rice fields","confidence":0.9}}]

Text:
{text}
"""


class ExtractorClient:
    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        elif self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            raise ValueError("Supported providers: groq, openai, openrouter")

        if not self.api_key:
            raise RuntimeError(f"Missing API key for provider={self.provider}")

    def extract(self, text: str, retries: int = 5) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_prompt(text)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        delay = 2.0
        for attempt in range(1, retries + 1):
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            if response.status_code in {429, 500, 502, 503, 504}:
                wait = extract_retry_delay(response.text) or delay
                wait = min(wait, 180)
                print(f"Provider retry {attempt}/{retries}: HTTP {response.status_code}. Waiting {wait:.1f}s")
                time.sleep(wait + random.uniform(0, 1.5))
                delay = min(delay * 2, 60)
                continue

            raise RuntimeError(f"Provider error HTTP {response.status_code}: {response.text[:500]}")

        raise RuntimeError(f"Provider failed after {retries} retries")


def extract_retry_delay(text: str) -> Optional[float]:
    match = re.search(r"try again in\s+([0-9.]+)\s*([a-zA-Z]*)", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value * 60 if unit.startswith("m") else value


def parse_triplets(raw: str, chunk: Chunk, model: str) -> List[Triplet]:
    data = parse_json_array(raw)
    triplets: List[Triplet] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        relation = normalize_relation(str(item.get("relation", "")).strip())
        obj = str(item.get("object", "")).strip()
        if not is_valid_triplet(subject, relation, obj):
            continue
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        triplets.append(
            Triplet(
                subject=subject[:120],
                relation=relation,
                object=obj[:160],
                confidence=max(0.0, min(confidence, 1.0)),
                source_doc=chunk.source,
                source_page=chunk.page,
                source_chunk_id=chunk.chunk_id,
                source_text=chunk.text[:700],
                extractor_model=model,
            )
        )
    return triplets


def parse_json_array(raw: str) -> List[Dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def normalize_relation(relation: str) -> str:
    rel = relation.upper().replace(" ", "_").replace("-", "_")
    return rel if rel in RELATION_TYPES else "RELATED_TO"


def is_valid_triplet(subject: str, relation: str, obj: str) -> bool:
    if len(subject) < 2 or len(obj) < 2:
        return False
    if relation not in RELATION_TYPES:
        return False
    if subject.lower() == obj.lower():
        return False
    return True


def append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_chunks(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(row.get("source_chunk_id"))
            except json.JSONDecodeError:
                continue
    return done


def write_chunks(path: Path, chunks: List[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def write_triplets_csv(jsonl_path: Path, csv_path: Path) -> None:
    fields = [
        "subject",
        "relation",
        "object",
        "confidence",
        "source_doc",
        "source_page",
        "source_chunk_id",
        "source_text",
        "extractor_model",
    ]
    with jsonl_path.open("r", encoding="utf-8") as src, csv_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for line in src:
            row = json.loads(line)
            writer.writerow({field: row.get(field, "") for field in fields})


def write_neo4j_import_cypher(path: Path, csv_name: str) -> None:
    path.write_text(
        f"""// Copy {csv_name} to Neo4j import directory, then run this in Neo4j Browser.
LOAD CSV WITH HEADERS FROM 'file:///{csv_name}' AS row
WITH row
WHERE row.subject IS NOT NULL AND row.subject <> ''
  AND row.object IS NOT NULL AND row.object <> ''
MERGE (s:Entity {{id: row.subject}})
SET s.name = row.subject, s.type = 'ENTITY'
MERGE (o:Entity {{id: row.object}})
SET o.name = row.object, o.type = 'ENTITY'
WITH s, o, row
CALL apoc.create.relationship(
  s,
  row.relation,
  {{
    confidence: toFloat(row.confidence),
    source_doc: row.source_doc,
    source_page: row.source_page,
    source_chunk_id: row.source_chunk_id,
    source_text: row.source_text,
    extractor_model: row.extractor_model
  }},
  o
) YIELD rel
RETURN count(rel) AS relationships_imported;
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Agri-RAG triplets in Colab.")
    parser.add_argument("--input_dir", default="/content/agri_data", help="Folder containing PDF/TXT files.")
    parser.add_argument("--output_dir", default="/content/agri_exports", help="Folder for exported chunks/triplets.")
    parser.add_argument("--provider", default=os.getenv("PROVIDER", "groq"), choices=["groq", "openai", "openrouter"])
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", ""))
    parser.add_argument("--chunk_words", type=int, default=int(os.getenv("CHUNK_WORDS", "350")))
    parser.add_argument("--overlap_words", type=int, default=int(os.getenv("OVERLAP_WORDS", "40")))
    parser.add_argument("--max_pages", type=int, default=int(os.getenv("MAX_PAGES", "0")), help="0 means all pages.")
    parser.add_argument("--max_chunks", type=int, default=int(os.getenv("MAX_CHUNKS", "0")), help="0 means all chunks.")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.1")))
    parser.add_argument("--max_tokens", type=int, default=int(os.getenv("MAX_TOKENS", "700")))
    parser.add_argument("--sleep", type=float, default=float(os.getenv("REQUEST_SLEEP", "0.3")))
    args = parser.parse_args()

    # Set default model based on provider if not specified
    if not args.model:
        if args.provider == "openrouter":
            args.model = "minimax/minimax-m2.5:free"
        elif args.provider == "groq":
            args.model = "llama-3.3-70b-versatile"
        elif args.provider == "openai":
            args.model = "gpt-4o-mini"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.jsonl"
    triplets_path = output_dir / "triplets.jsonl"
    csv_path = output_dir / "triplets.csv"
    cypher_path = output_dir / "neo4j_import_triplets.cypher"

    chunks = load_documents(
        input_dir,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        max_pages=args.max_pages or None,
    )
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    write_chunks(chunks_path, chunks)
    print(f"Loaded {len(chunks)} chunks. Saved chunks to {chunks_path}")

    client = ExtractorClient(args.provider, args.model, args.temperature, args.max_tokens)
    done_chunks = load_done_chunks(triplets_path)
    print(f"Resuming with {len(done_chunks)} chunks already completed.")

    total_triplets = 0
    for chunk in tqdm(chunks, desc="Extracting triplets"):
        if chunk.chunk_id in done_chunks:
            continue
        try:
            raw = client.extract(chunk.text)
            triplets = parse_triplets(raw, chunk, args.model)
            append_jsonl(triplets_path, [asdict(t) for t in triplets])
            total_triplets += len(triplets)
        except Exception as exc:
            error_path = output_dir / "errors.jsonl"
            append_jsonl(error_path, [{"chunk_id": chunk.chunk_id, "source": chunk.source, "page": chunk.page, "error": str(exc)}])
        time.sleep(args.sleep)

    if triplets_path.exists():
        write_triplets_csv(triplets_path, csv_path)
        write_neo4j_import_cypher(cypher_path, csv_path.name)

    print(f"New triplets extracted in this run: {total_triplets}")
    print(f"Triplets JSONL: {triplets_path}")
    print(f"Triplets CSV:   {csv_path}")
    print(f"Neo4j Cypher:   {cypher_path}")


if __name__ == "__main__":
    main()
