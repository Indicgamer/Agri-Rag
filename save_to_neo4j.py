"""
Quick script to save triplets from file to Neo4j
Run this anytime to sync file -> Neo4j
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.neo4j_retriever import KnowledgeGraph
from configs.settings import settings

print("Loading triplets from file...")

# Load triplets from file
triplets = []
with open('data/exports/triplets.jsonl', 'r') as f:
    for line in f:
        try:
            triplets.append(json.loads(line.strip()))
        except:
            pass

print(f"Loaded {len(triplets)} triplets from file")

# Connect to Neo4j
print("Connecting to Neo4j...")
kg = KnowledgeGraph(
    uri=settings.neo4j.uri,
    username=settings.neo4j.username,
    password=settings.neo4j.password
)

# Load to Neo4j
print("Loading triplets to Neo4j...")
kg.add_triplets_batch(triplets)
print(f"Loaded {len(triplets)} triplets to Neo4j!")

# Show stats
stats =kg.get_statistics()
print("\nNeo4j Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\nDone! Triplets saved to Neo4j.")