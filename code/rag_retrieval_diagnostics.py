from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from qdrant_client import QdrantClient

from rag_answer import build_evidence
from rag_evaluation import DEFAULT_ANSWERABLE_CASES, normalize_source_url
from rag_retrieval.bm25 import search_bm25
from rag_retrieval.hybrid import fuse_rrf
from rag_retrieval.semantic import SemanticRetriever
from rag_settings import RetrievalSettings


def expected_rank(items: list[Any], expected_sources: list[str]) -> int | None:
    expected = {normalize_source_url(source) for source in expected_sources}
    for rank, item in enumerate(items, start=1):
        source = getattr(item, "source", None) or getattr(item, "url", "")
        if normalize_source_url(source) in expected:
            return rank
    return None


def summarize_retrieval(cases: list[dict[str, Any]]) -> dict[str, float]:
    count = len(cases)
    if not count:
        return {}
    summary: dict[str, float] = {}
    for field, k in (("bm25Rank", 20), ("semanticRank", 20), ("fusedRank", 10)):
        ranks = [case[field] for case in cases]
        summary[f"{field}HitAt{k}"] = round(
            sum(rank is not None and rank <= k for rank in ranks) / count, 4
        )
        summary[f"{field}MrrAt{k}"] = round(
            sum(1 / rank for rank in ranks if rank is not None and rank <= k) / count,
            4,
        )
    summary["finalEvidenceRecall"] = round(
        sum(case["evidenceContainsExpected"] for case in cases) / count, 4
    )
    summary["averageRetrievalMs"] = round(
        sum(case["retrievalMs"] for case in cases) / count, 2
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose each retrieval stage for fixed questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_ANSWERABLE_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = RetrievalSettings.from_env()
    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    ollama = httpx.Client(timeout=60.0)
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=10.0)
    semantic = SemanticRetriever(
        ollama_client=ollama,
        qdrant_client=qdrant,
        model=settings.embedding_model,
        collection_name=settings.collection,
        vector_size=settings.vector_size,
    )
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            started = perf_counter()
            bm25 = search_bm25(settings.bm25_path, case["question"], limit=settings.bm25_limit)
            vectors = semantic.search(case["question"], limit=settings.semantic_limit)
            fused = fuse_rrf(bm25, vectors, limit=len(bm25) + len(vectors) or 1)
            evidence = build_evidence(
                fused[: settings.evidence_limit],
                max_context_chars=settings.max_context_chars,
            )
            expected = list(case["expected_sources"])
            records.append({
                "id": case["id"],
                "bm25Rank": expected_rank(bm25, expected),
                "semanticRank": expected_rank(vectors, expected),
                "fusedRank": expected_rank(fused, expected),
                "evidenceContainsExpected": expected_rank(evidence, expected) is not None,
                "retrievalMs": round((perf_counter() - started) * 1000, 2),
            })
            print(f"[{len(records)}/{len(cases)}] {case['id']}", flush=True)
    finally:
        ollama.close()
        qdrant.close()
    output = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {
            "collection": settings.collection,
            "embeddingModel": settings.embedding_model,
            "bm25Limit": settings.bm25_limit,
            "semanticLimit": settings.semantic_limit,
            "evidenceLimit": settings.evidence_limit,
            "maxContextChars": settings.max_context_chars,
        },
        "metrics": summarize_retrieval(records),
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
