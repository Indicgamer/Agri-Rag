"""
Dependency-light local demo for the Agri-RAG pipeline.

This file uses only the Python standard library so the project can be tested
before installing ML/database dependencies. It mirrors the intended flow:
query -> retrieval -> pruning -> grounded response.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence


STOPWORDS = {
    "a", "an", "and", "are", "as", "can", "for", "how", "i", "in", "is",
    "it", "my", "of", "on", "or", "the", "to", "what", "with",
}


@dataclass
class Fact:
    content: str
    source: str
    crop: str
    tags: List[str]


@dataclass
class RetrievedFact:
    fact: Fact
    score: float
    reason: str


@dataclass
class PrunedFact:
    fact: Fact
    label: str
    score: float
    keep: bool


SAMPLE_FACTS = [
    Fact(
        content="Fungal blast affects rice and is favored by humid weather and poor field drainage.",
        source="demo_agri_notes.txt",
        crop="rice",
        tags=["rice", "fungal", "blast", "drainage", "humidity"],
    ),
    Fact(
        content="Proper drainage and avoiding water stagnation help prevent fungal blast in rice fields.",
        source="demo_agri_notes.txt",
        crop="rice",
        tags=["rice", "fungal", "blast", "prevention", "drainage"],
    ),
    Fact(
        content="Resistant rice varieties improve disease management where fungal blast is common.",
        source="demo_agri_notes.txt",
        crop="rice",
        tags=["rice", "fungal", "blast", "resistant", "variety"],
    ),
    Fact(
        content="Balanced nitrogen use is recommended because excessive urea can increase disease susceptibility.",
        source="demo_agri_notes.txt",
        crop="rice",
        tags=["rice", "urea", "nitrogen", "balanced", "disease"],
    ),
    Fact(
        content="Cotton benefits from potassium application for yield and plant strength.",
        source="demo_agri_notes.txt",
        crop="cotton",
        tags=["cotton", "potassium", "yield"],
    ),
    Fact(
        content="Seed treatment and timely sowing reduce early pest pressure in maize.",
        source="demo_agri_notes.txt",
        crop="maize",
        tags=["maize", "seed", "pest", "sowing"],
    ),
]


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def cosine_overlap(query_terms: Sequence[str], fact_terms: Sequence[str]) -> float:
    query_counts = {term: query_terms.count(term) for term in set(query_terms)}
    fact_counts = {term: fact_terms.count(term) for term in set(fact_terms)}
    shared = set(query_counts) & set(fact_counts)
    numerator = sum(query_counts[t] * fact_counts[t] for t in shared)
    query_norm = math.sqrt(sum(v * v for v in query_counts.values()))
    fact_norm = math.sqrt(sum(v * v for v in fact_counts.values()))
    if query_norm == 0 or fact_norm == 0:
        return 0.0
    return numerator / (query_norm * fact_norm)


def retrieve(query: str, facts: Iterable[Fact], top_k: int = 4) -> List[RetrievedFact]:
    query_terms = tokenize(query)
    results: List[RetrievedFact] = []

    for fact in facts:
        fact_terms = tokenize(fact.content + " " + " ".join(fact.tags))
        lexical_score = cosine_overlap(query_terms, fact_terms)
        tag_hits = len(set(query_terms) & set(fact.tags))
        tag_score = min(tag_hits / 3, 1.0)
        score = (0.65 * lexical_score) + (0.35 * tag_score)
        if score > 0:
            results.append(
                RetrievedFact(
                    fact=fact,
                    score=round(score, 3),
                    reason=f"{tag_hits} tag hits plus lexical overlap",
                )
            )

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def prune(query: str, retrieved: Sequence[RetrievedFact]) -> List[PrunedFact]:
    query_terms = set(tokenize(query))
    pruned: List[PrunedFact] = []

    for item in retrieved:
        fact_terms = set(tokenize(item.fact.content + " " + " ".join(item.fact.tags)))
        overlap = query_terms & fact_terms
        score = len(overlap) / max(len(query_terms), 1)
        label = "entailment" if score >= 0.25 else "neutral"
        pruned.append(
            PrunedFact(
                fact=item.fact,
                label=label,
                score=round(score, 3),
                keep=label == "entailment",
            )
        )

    kept = [item for item in pruned if item.keep]
    return kept if kept else pruned[:1]


def generate_response(query: str, pruned: Sequence[PrunedFact]) -> Dict:
    facts = [item.fact.content for item in pruned if item.keep or item.label == "entailment"]
    if not facts and pruned:
        facts = [pruned[0].fact.content]

    answer_lines = [
        "Based on the local demo knowledge base:",
        *[f"{idx}. {fact}" for idx, fact in enumerate(facts, start=1)],
    ]

    if not facts:
        answer_lines.append("I do not have enough grounded information to answer this query.")
    elif "prevent" in query.lower() and "blast" in query.lower():
        answer_lines.append(
            "For rice fungal blast, prioritize drainage, avoid stagnant water, use resistant varieties, "
            "and keep nitrogen application balanced."
        )

    citations = [
        {"source": item.fact.source, "fact": item.fact.content, "label": item.label}
        for item in pruned
    ]
    confidence = min(0.95, 0.45 + (0.15 * len(facts)))

    return {
        "query": query,
        "response": "\n".join(answer_lines),
        "confidence": round(confidence, 2),
        "citations": citations,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run(query: str) -> Dict:
    retrieved = retrieve(query, SAMPLE_FACTS)
    pruned = prune(query, retrieved)
    result = generate_response(query, pruned)
    result["pipeline"] = {
        "retrieved": [
            {"score": item.score, "reason": item.reason, "fact": item.fact.content}
            for item in retrieved
        ],
        "pruned": [
            {
                "label": item.label,
                "score": item.score,
                "kept": item.keep,
                "fact": item.fact.content,
            }
            for item in pruned
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dependency-light Agri-RAG demo.")
    parser.add_argument(
        "query",
        nargs="?",
        default="How can I prevent fungal blast in rice?",
        help="Agricultural question to ask.",
    )
    args = parser.parse_args()

    result = run(args.query)
    print(f"Query: {result['query']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print("\nAnswer:")
    print(result["response"])
    print("\nCitations:")
    for idx, citation in enumerate(result["citations"], start=1):
        print(f"[{idx}] {citation['source']} - {citation['fact']}")


if __name__ == "__main__":
    main()
