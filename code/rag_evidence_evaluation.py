from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_evidence import (
    EvidenceCandidate,
    SelectionConfig,
    select_evidence,
)
from rag_retrieval.factory import build_default_retriever
from rag_settings import RetrievalSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data/evaluation/v2/questions.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "data/processed/chunks.jsonl"


def _candidate(record: Mapping[str, Any]) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=str(record["chunk_id"]),
        title=str(record["title"]),
        text=str(record["text"]),
        source=str(record["source"]),
        document_id=(
            str(record["document_id"])
            if record.get("document_id") is not None
            else None
        ),
        chunk_index=(
            int(record["chunk_index"])
            if record.get("chunk_index") is not None
            else None
        ),
    )


def _evidence_record(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "chunkId": item.chunk_id,
        "componentChunkIds": list(item.component_chunk_ids),
        "title": item.title,
        "url": item.url,
        "text": item.text,
    }


def evaluate_strategy(
    snapshot: Mapping[str, Any],
    *,
    strategy: str,
    max_context_chars: int,
    max_chunks_per_source: int = 2,
    redundancy_threshold: float = 0.8,
) -> dict[str, Any]:
    """Evaluate one selector against annotated chunks and sources without model calls."""
    output_cases: list[dict[str, Any]] = []
    recalled_relevant = 0
    relevant_total = 0
    relevant_evidence_items = 0
    selected_evidence_items = 0
    source_hits = 0
    source_cases = 0
    valid_ids = 0
    evidence_count = 0

    for case in snapshot.get("cases", []):
        candidates = [_candidate(item) for item in case.get("candidates", [])]
        evidence = select_evidence(
            candidates,
            SelectionConfig(
                strategy=strategy,  # type: ignore[arg-type]
                max_context_chars=max_context_chars,
                max_chunks_per_source=max_chunks_per_source,
                redundancy_threshold=redundancy_threshold,
            ),
        )
        relevant = set(case.get("relevant_chunk_ids", []))
        acceptable_sources = set(case.get("acceptable_sources", []))
        selected_chunks = {
            chunk_id
            for item in evidence
            for chunk_id in item.component_chunk_ids
        }
        candidate_ids = {item.chunk_id for item in candidates}
        recalled_relevant += len(relevant & selected_chunks)
        relevant_total += len(relevant)
        if relevant:
            relevant_evidence_items += sum(
                bool(relevant & set(item.component_chunk_ids)) for item in evidence
            )
            selected_evidence_items += len(evidence)
        if acceptable_sources:
            source_cases += 1
            source_hits += int(any(item.url in acceptable_sources for item in evidence))

        ids_are_valid = [item.id for item in evidence] == list(
            range(1, len(evidence) + 1)
        ) and all(
            item.component_chunk_ids
            and set(item.component_chunk_ids).issubset(candidate_ids)
            for item in evidence
        )
        evidence_count += len(evidence)
        valid_ids += len(evidence) if ids_are_valid else 0
        output_cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "relevantSelected": sorted(relevant & selected_chunks),
                "relevantTotal": len(relevant),
                "evidenceChars": sum(len(item.text) for item in evidence),
                "evidence": [_evidence_record(item) for item in evidence],
            }
        )

    return {
        "configuration": {
            "strategy": strategy,
            "maxContextChars": max_context_chars,
            "maxChunksPerSource": max_chunks_per_source,
            "redundancyThreshold": redundancy_threshold,
        },
        "metrics": {
            "relevantEvidenceRecall": round(
                recalled_relevant / relevant_total, 4
            )
            if relevant_total
            else None,
            "contextPrecision": round(
                relevant_evidence_items / selected_evidence_items, 4
            )
            if selected_evidence_items
            else None,
            "expectedSourceRecall": round(source_hits / source_cases, 4)
            if source_cases
            else None,
            "evidenceIdValidityRate": round(valid_ids / evidence_count, 4)
            if evidence_count
            else 1.0,
            "selectedEvidenceCount": evidence_count,
        },
        "cases": output_cases,
    }


def compare_strategies(
    snapshot: Mapping[str, Any],
    *,
    max_context_chars: int,
    source_caps: Sequence[int] = (1, 2, 3, 4),
    redundancy_thresholds: Sequence[float] = (0.6, 0.7, 0.8, 0.9),
) -> dict[str, Any]:
    """Run required strategy variants over exactly the same candidate snapshot."""
    runs = {
        "ranked_first": evaluate_strategy(
            snapshot,
            strategy="ranked_first",
            max_context_chars=max_context_chars,
        ),
        "adjacent_merge": evaluate_strategy(
            snapshot,
            strategy="adjacent_merge",
            max_context_chars=max_context_chars,
        ),
    }
    for cap in source_caps:
        runs[f"source_cap_{cap}"] = evaluate_strategy(
            snapshot,
            strategy="source_cap",
            max_context_chars=max_context_chars,
            max_chunks_per_source=cap,
        )
    for threshold in redundancy_thresholds:
        runs[f"redundancy_suppression_{threshold:g}"] = evaluate_strategy(
            snapshot,
            strategy="redundancy_suppression",
            max_context_chars=max_context_chars,
            redundancy_threshold=threshold,
        )
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "snapshot": snapshot.get("configuration", {}),
        "caseIds": [case["id"] for case in snapshot.get("cases", [])],
        "runs": runs,
    }


def _load_chunk_metadata(path: Path) -> dict[str, tuple[str | None, int | None]]:
    metadata: dict[str, tuple[str | None, int | None]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            chunk = json.loads(line)
            values = chunk.get("metadata", {})
            metadata[chunk["id"]] = (
                values.get("document_id"),
                values.get("chunk_index"),
            )
    return metadata


def capture_snapshot(
    dataset_path: Path,
    chunks_path: Path,
    settings: RetrievalSettings,
) -> dict[str, Any]:
    """Capture retrieval candidates once so all selector experiments are offline."""
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    metadata = _load_chunk_metadata(chunks_path)
    retriever, ollama, qdrant = build_default_retriever(settings)
    cases: list[dict[str, Any]] = []
    try:
        for case in dataset["cases"]:
            results = retriever.search(
                case["question"],
                bm25_limit=settings.bm25_limit,
                semantic_limit=settings.semantic_limit,
                limit=settings.evidence_limit,
            )
            candidates = []
            for result in results:
                document_id, chunk_index = metadata.get(result.chunk_id, (None, None))
                candidates.append(
                    {
                        "chunk_id": result.chunk_id,
                        "title": result.title,
                        "text": result.text,
                        "source": result.source,
                        "score": result.score,
                        "bm25_rank": result.bm25_rank,
                        "semantic_rank": result.semantic_rank,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                    }
                )
            cases.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "kind": case["kind"],
                    "relevant_chunk_ids": case["relevant_chunk_ids"],
                    "acceptable_sources": case["acceptable_sources"],
                    "candidates": candidates,
                }
            )
    finally:
        ollama.close()
        qdrant.close()
    return {
        "schema_version": 1,
        "configuration": {
            "dataset": str(dataset_path),
            "datasetSha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "bm25Limit": settings.bm25_limit,
            "semanticLimit": settings.semantic_limit,
            "candidateLimit": settings.evidence_limit,
            "collection": settings.collection,
            "embeddingModel": settings.embedding_model,
        },
        "cases": cases,
    }


def snapshot_from_answer_run(
    answer_run_path: Path,
    questions_path: Path,
    chunks_path: Path,
) -> dict[str, Any]:
    """Adapt a saved full-chain run to the common offline candidate format."""
    answer_bytes = answer_run_path.read_bytes()
    answer_run = json.loads(answer_bytes)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    expected_by_id = {
        case["id"]: case.get("expected_sources", []) for case in questions
    }
    metadata = _load_chunk_metadata(chunks_path)
    cases = []
    for case in answer_run["cases"]:
        candidates = []
        for item in case.get("evidence", []):
            document_id, chunk_index = metadata.get(item["chunkId"], (None, None))
            candidates.append(
                {
                    "chunk_id": item["chunkId"],
                    "title": item["title"],
                    "text": item["text"],
                    "source": item["url"],
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                }
            )
        cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "kind": "answerable" if case["id"] in expected_by_id else "unanswerable",
                "relevant_chunk_ids": [],
                "acceptable_sources": expected_by_id.get(case["id"], []),
                "candidates": candidates,
            }
        )
    return {
        "schema_version": 1,
        "configuration": {
            "answerRun": str(answer_run_path),
            "answerRunSha256": hashlib.sha256(answer_bytes).hexdigest(),
            "candidateLimit": max(
                (len(case["candidates"]) for case in cases), default=0
            ),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture and compare deterministic evidence assembly strategies."
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--answer-run", type=Path)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/retrieval_questions.json",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = sum((args.capture, args.snapshot is not None, args.answer_run is not None))
    if modes != 1:
        parser.error("choose exactly one of --capture, --snapshot, or --answer-run")

    if args.capture:
        snapshot = capture_snapshot(
            args.dataset,
            args.chunks,
            RetrievalSettings.from_env(),
        )
        if args.snapshot_output:
            args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot_output.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.snapshot is not None:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    else:
        snapshot = snapshot_from_answer_run(
            args.answer_run,
            args.questions,
            args.chunks,
        )
        if args.snapshot_output:
            args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot_output.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    comparison = compare_strategies(
        snapshot,
        max_context_chars=RetrievalSettings.from_env().max_context_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
