"""
Import Colab-extracted triplets into the local Agri-RAG Neo4j database.

Example:
    python scripts/import_triplets_to_neo4j.py --triplets notebooks/agri_exports/triplets.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.settings import settings
from src.retrieval.neo4j_retriever import KnowledgeGraph


def load_triplets(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "subject": row.get("subject"),
                    "relation": row.get("relation"),
                    "object": row.get("object"),
                    "metadata": {
                        "confidence": row.get("confidence"),
                        "source_doc": row.get("source_doc"),
                        "source_page": row.get("source_page"),
                        "source_chunk_id": row.get("source_chunk_id"),
                        "source_text": row.get("source_text"),
                        "extractor_model": row.get("extractor_model"),
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import triplets.jsonl into Neo4j.")
    parser.add_argument("--triplets", required=True, help="Path to triplets.jsonl exported from Colab.")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--clear_first", action="store_true", help="Clear existing graph before import.")
    args = parser.parse_args()

    triplets_path = Path(args.triplets)
    triplets = load_triplets(triplets_path)
    print(f"Loaded {len(triplets)} triplets from {triplets_path}")

    kg = KnowledgeGraph(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database,
    )

    if args.clear_first:
        print("Clearing existing graph...")
        kg.clear_graph()

    total_added = 0
    for start in range(0, len(triplets), args.batch_size):
        batch = triplets[start:start + args.batch_size]
        total_added += kg.add_triplets_batch(batch)
        print(f"Imported {min(start + args.batch_size, len(triplets))}/{len(triplets)}")

    print(f"Done. Added {total_added} triplets.")
    print(kg.get_statistics())


if __name__ == "__main__":
    main()
