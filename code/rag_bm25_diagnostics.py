from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from rag_evaluation import normalize_source_url
from rag_retrieval.bm25 import BM25Result, search_bm25
from rag_retrieval.bm25_query import build_match_query, supported_strategies
from rag_settings import DEFAULT_BM25_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = PROJECT_ROOT / "data/evaluation/lexical_questions.json"
DEFAULT_REFERENCE_QUESTIONS = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/bm25-diagnostic-results.json"


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(float(ordered[index]), 2)


def summarize_strategy(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    count = len(records)
    if not count:
        raise ValueError("cannot summarize an empty diagnostic run")
    ranks = [record["expectedRank"] for record in records]
    latencies = [float(record["latencyMs"]) for record in records]
    return {
        "caseCount": count,
        "hitAt5": round(sum(rank is not None and rank <= 5 for rank in ranks) / count, 4),
        "hitAt10": round(sum(rank is not None and rank <= 10 for rank in ranks) / count, 4),
        "hitAt20": round(sum(rank is not None and rank <= 20 for rank in ranks) / count, 4),
        "mrrAt10": round(sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / count, 4),
        "emptyResultRate": round(sum(record["resultCount"] == 0 for record in records) / count, 4),
        "p50LatencyMs": percentile_nearest_rank(latencies, 0.50),
        "p95LatencyMs": percentile_nearest_rank(latencies, 0.95),
    }


def _search_experimental(
    database_path: Path,
    query: str,
    *,
    strategy: str,
    limit: int,
) -> list[BM25Result]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    match_query = build_match_query(query, strategy)
    if not match_query:
        return []
    title_weight = 12.0 if strategy == "title_boost" else 5.0
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT chunks.chunk_id, chunks.title, chunks.text, chunks.source,
                   -bm25(chunks_fts, ?, 1.0) AS score
            FROM chunks_fts
            JOIN chunks ON chunks.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC, chunks.rowid
            LIMIT ?
            """,
            (title_weight, match_query, limit),
        ).fetchall()
    return [BM25Result(row[0], row[1], row[2], row[3], float(row[4])) for row in rows]


def search_with_strategy(
    database_path: Path,
    query: str,
    *,
    strategy: str,
    limit: int,
) -> list[BM25Result]:
    if strategy == "literal_phrase":
        return search_bm25(database_path, query, limit=limit)
    if strategy not in supported_strategies():
        raise ValueError(f"unknown BM25 query strategy: {strategy}")
    return _search_experimental(database_path, query, strategy=strategy, limit=limit)


def _expected_rank(results: Sequence[BM25Result], sources: Sequence[str]) -> int | None:
    expected = {normalize_source_url(source) for source in sources}
    for rank, result in enumerate(results, start=1):
        if normalize_source_url(result.source) in expected:
            return rank
    return None


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not 30 <= len(cases) <= 40:
        raise ValueError("lexical dataset must contain 30 to 40 cases")
    required_categories = {
        "entity-alias", "edition-version", "prerelease-snapshot", "command-id-nbt",
        "exact-number", "decorated-long-query",
    }
    categories = {case.get("category") for case in cases if isinstance(case, dict)}
    if not required_categories <= categories:
        raise ValueError("lexical dataset does not cover every required category")
    for case in cases:
        if not isinstance(case, dict) or not all(
            isinstance(case.get(field), str) and case[field].strip()
            for field in ("id", "question", "category")
        ):
            raise ValueError("each lexical case needs id, question, category, and expectedSources")
        sources = case.get("expectedSources")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and source.strip() for source in sources
        ):
            raise ValueError("each lexical case needs non-empty expectedSources")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("lexical case ids must be unique")
    return cases


def load_reference_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("reference dataset must be a JSON array")
    return [
        {
            "id": case["id"],
            "question": case["question"],
            "category": case.get("category", "reference"),
            "expectedSources": case["expected_sources"],
        }
        for case in payload
    ]


def run_diagnostics(
    *, database_path: Path, cases: Sequence[dict[str, Any]], strategies: Sequence[str], limit: int
) -> dict[str, Any]:
    strategy_outputs: dict[str, Any] = {}
    for strategy in strategies:
        records = []
        for case in cases:
            started = perf_counter()
            results = search_with_strategy(
                database_path, case["question"], strategy=strategy, limit=limit
            )
            latency_ms = round((perf_counter() - started) * 1000, 2)
            records.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expectedRank": _expected_rank(results, case["expectedSources"]),
                "resultCount": len(results),
                "latencyMs": latency_ms,
                "results": [
                    {"rank": rank, "chunkId": item.chunk_id, "source": item.source, "title": item.title}
                    for rank, item in enumerate(results, start=1)
                ],
            })
        strategy_outputs[strategy] = {"metrics": summarize_strategy(records), "cases": records}
    return strategy_outputs


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    code_paths = (Path(__file__), PROJECT_ROOT / "code/rag_retrieval/bm25_query.py")
    return {
        "revision": revision.stdout.strip() or "unknown",
        "worktreeDirty": bool(status.stdout.strip()),
        "diagnosticCodeSha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_paths
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reproducible BM25 query strategies.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--reference-questions", type=Path, default=DEFAULT_REFERENCE_QUESTIONS)
    parser.add_argument("--database", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--strategies", nargs="+", choices=supported_strategies(), default=list(supported_strategies()))
    args = parser.parse_args()
    cases = load_cases(args.questions)
    reference_cases = load_reference_cases(args.reference_questions)
    raw_questions = args.questions.read_bytes()
    output = {
        "runAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code": _git_state(),
        "dataset": {"path": str(args.questions), "sha256": hashlib.sha256(raw_questions).hexdigest(), "caseCount": len(cases)},
        "database": {"path": str(args.database), "sizeBytes": args.database.stat().st_size, "modifiedAt": datetime.fromtimestamp(args.database.stat().st_mtime).astimezone().isoformat(timespec="seconds")},
        "config": {"limit": args.limit, "strategies": args.strategies},
        "strategies": run_diagnostics(database_path=args.database, cases=cases, strategies=args.strategies, limit=args.limit),
        "referenceComparison": {
            "path": str(args.reference_questions),
            "caseCount": len(reference_cases),
            "strategies": run_diagnostics(
                database_path=args.database,
                cases=reference_cases,
                strategies=args.strategies,
                limit=args.limit,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, value in output["strategies"].items():
        print(name, json.dumps(value["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
