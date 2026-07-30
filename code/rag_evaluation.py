from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit, urlunsplit

import anyio
import httpx

from rag_answer import (
    AnswerEvidence,
    DeepSeekAnswerClient,
    DeepSeekSettings,
    build_evidence,
)
from rag_retrieval.hybrid import HybridResult


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
INSUFFICIENT_EVIDENCE_MESSAGE = "现有知识库没有足够资料支持可靠回答。"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANSWERABLE_CASES = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
DEFAULT_UNANSWERABLE_CASES = (
    PROJECT_ROOT / "data/evaluation/answer_quality/unanswerable_questions.json"
)


class EvaluationRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]: ...


class EvaluationAnswerClient(Protocol):
    def stream_answer(
        self,
        question: str,
        evidence: Sequence[AnswerEvidence],
    ) -> Any: ...


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_cases(
    answerable_path: Path,
    unanswerable_path: Path,
) -> list[dict[str, Any]]:
    """Load and label the fixed answerable and unanswerable datasets."""
    answerable = [
        {**case, "kind": "answerable"}
        for case in _load_json(answerable_path)
    ]
    unanswerable = [
        {**case, "kind": "unanswerable", "expected_sources": []}
        for case in _load_json(unanswerable_path)
    ]
    cases = answerable + unanswerable
    identifiers = [str(case["id"]) for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation case ids must be unique")
    return cases


def validate_resume_output(
    output: Mapping[str, Any],
    *,
    expected_configuration: Mapping[str, Any],
    expected_case_ids: Sequence[str],
) -> set[str]:
    """Reject resume files that would mix different runs or datasets."""
    if output.get("configuration") != expected_configuration:
        raise ValueError("resume output configuration does not match this run")
    completed_ids = [str(case["id"]) for case in output.get("cases", [])]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("resume output contains duplicate case ids")
    unknown_ids = set(completed_ids) - set(expected_case_ids)
    if unknown_ids:
        raise ValueError(
            f"resume output contains unknown case ids: {', '.join(sorted(unknown_ids))}"
        )
    return set(completed_ids)


def normalize_source_url(value: str) -> str:
    """Normalize equivalent encoded and Unicode source URLs for comparison."""
    parsed = urlsplit(value.strip())
    path = unquote(parsed.path).rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.query,
            "",
        )
    )


def _ordered_unique(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(values))


def analyze_answer(
    *,
    answer: str,
    evidence: Sequence[Mapping[str, Any]],
    expected_sources: Sequence[str],
) -> dict[str, Any]:
    """Calculate deterministic retrieval and citation checks for one answer."""
    citation_occurrences = [
        int(match.group(1)) for match in CITATION_PATTERN.finditer(answer)
    ]
    citation_ids = _ordered_unique(citation_occurrences)
    evidence_by_id = {
        int(item["id"]): normalize_source_url(str(item["url"]))
        for item in evidence
    }
    valid_occurrences = [
        citation_id
        for citation_id in citation_occurrences
        if citation_id in evidence_by_id
    ]
    invalid_ids = [
        citation_id
        for citation_id in citation_ids
        if citation_id not in evidence_by_id
    ]
    expected = {normalize_source_url(source) for source in expected_sources}
    retrieved_sources = set(evidence_by_id.values())
    cited_sources = {
        evidence_by_id[citation_id]
        for citation_id in citation_ids
        if citation_id in evidence_by_id
    }
    citation_count = len(citation_occurrences)

    return {
        "expectedSourceRetrieved": (
            bool(expected & retrieved_sources) if expected else None
        ),
        "expectedSourceCited": bool(expected & cited_sources) if expected else None,
        "hasCitation": citation_count > 0,
        "citationIds": citation_ids,
        "citationCount": citation_count,
        "validCitationCount": len(valid_occurrences),
        "invalidCitationIds": invalid_ids,
        "citationValidityRate": (
            round(len(valid_occurrences) / citation_count, 4)
            if citation_count
            else None
        ),
        "containsForbiddenInternalText": any(
            marker in answer
            for marker in (
                "DEEPSEEK_API_KEY",
                "SYSTEM_PROMPT",
                "内部检索分数",
                "Bearer ",
            )
        ),
    }


def _evidence_record(item: AnswerEvidence) -> dict[str, Any]:
    return {
        "id": item.id,
        "chunkId": item.chunk_id,
        "title": item.title,
        "url": item.url,
        "text": item.text,
        "excerpt": item.excerpt,
    }


async def run_answer_case(
    case: Mapping[str, Any],
    *,
    retriever: EvaluationRetriever,
    answer_client: EvaluationAnswerClient,
) -> dict[str, Any]:
    """Run the same retrieval and generation path used by POST /answers."""
    question = str(case["question"])
    retrieval_started = perf_counter()
    results = await anyio.to_thread.run_sync(
        partial(
            retriever.search,
            question,
            bm25_limit=20,
            semantic_limit=20,
            limit=8,
        )
    )
    retrieval_ms = round((perf_counter() - retrieval_started) * 1_000, 2)
    evidence = build_evidence(results)

    generation_started = perf_counter()
    if evidence:
        chunks = [
            chunk
            async for chunk in answer_client.stream_answer(question, evidence)
        ]
        answer_text = "".join(chunks)
        status = "answered"
    else:
        answer_text = INSUFFICIENT_EVIDENCE_MESSAGE
        status = "insufficientEvidence"
    generation_ms = round((perf_counter() - generation_started) * 1_000, 2)

    evidence_records = [_evidence_record(item) for item in evidence]
    expected_sources = list(case.get("expected_sources", []))
    return {
        "id": str(case["id"]),
        "kind": str(case["kind"]),
        "category": str(case["category"]),
        "question": question,
        "expectedSources": expected_sources,
        "evidence": evidence_records,
        "answer": {
            "status": status,
            "text": answer_text,
        },
        "timingMs": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": round(retrieval_ms + generation_ms, 2),
        },
        "objective": analyze_answer(
            answer=answer_text,
            evidence=evidence_records,
            expected_sources=expected_sources,
        ),
    }


def _average(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize_results(
    raw_run: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine deterministic metrics with explicit human review scores."""
    cases = list(raw_run["cases"])
    answerable = [case for case in cases if case["kind"] == "answerable"]
    unanswerable = [case for case in cases if case["kind"] == "unanswerable"]

    missing_reviews = [case["id"] for case in cases if case["id"] not in reviews]
    if missing_reviews:
        raise ValueError(f"missing reviews for: {', '.join(missing_reviews)}")

    correctness = [float(reviews[case["id"]]["correctness"]) for case in answerable]
    completeness = [
        float(reviews[case["id"]]["completeness"]) for case in answerable
    ]
    faithfulness = [
        float(reviews[case["id"]]["faithfulness"]) for case in answerable
    ]
    abstentions = [
        bool(reviews[case["id"]]["abstainedReliably"]) for case in unanswerable
    ]
    for score in (*correctness, *completeness, *faithfulness):
        if score < 0 or score > 2:
            raise ValueError("manual scores must be between 0 and 2")

    citation_count = sum(
        int(case["objective"]["citationCount"]) for case in cases
    )
    valid_citation_count = sum(
        int(case["objective"]["validCitationCount"]) for case in cases
    )
    invalid_citation_ids = {
        case["id"]: case["objective"]["invalidCitationIds"]
        for case in cases
        if case["objective"]["invalidCitationIds"]
    }
    expected_retrieved = sum(
        bool(case["objective"]["expectedSourceRetrieved"]) for case in answerable
    )
    expected_cited = sum(
        bool(case["objective"]["expectedSourceCited"]) for case in answerable
    )
    forbidden_text_cases = [
        case["id"]
        for case in cases
        if case["objective"].get("containsForbiddenInternalText", False)
    ]

    metrics = {
        "answerableCount": len(answerable),
        "unanswerableCount": len(unanswerable),
        "correctnessAverage": _average(correctness),
        "completenessAverage": _average(completeness),
        "faithfulnessAverage": _average(faithfulness),
        "expectedSourceRetrievalRate": (
            round(expected_retrieved / len(answerable), 4) if answerable else 0.0
        ),
        "expectedSourceCitationRate": (
            round(expected_cited / len(answerable), 4) if answerable else 0.0
        ),
        "citationValidityRate": (
            round(valid_citation_count / citation_count, 4)
            if citation_count
            else 0.0
        ),
        "abstentionRate": (
            round(sum(abstentions) / len(unanswerable), 4)
            if unanswerable
            else 0.0
        ),
        "invalidCitationIdsByCase": invalid_citation_ids,
        "forbiddenInternalTextCases": forbidden_text_cases,
    }
    meets_thresholds = (
        metrics["correctnessAverage"] >= 1.6
        and metrics["completenessAverage"] >= 1.5
        and metrics["faithfulnessAverage"] >= 1.8
        and metrics["citationValidityRate"] == 1.0
        and metrics["expectedSourceCitationRate"] >= 0.8
        and metrics["abstentionRate"] >= 0.8
        and not invalid_citation_ids
        and not forbidden_text_cases
    )
    return {
        "metrics": metrics,
        "meetsSuggestedThresholds": meets_thresholds,
        "reviews": dict(reviews),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


async def run_evaluation(
    *,
    answerable_path: Path,
    unanswerable_path: Path,
    output_path: Path,
) -> None:
    """Run all fixed cases, saving after every case so an interrupted run can resume."""
    from rag_api import _build_default_retriever

    cases = load_evaluation_cases(answerable_path, unanswerable_path)
    settings = DeepSeekSettings.from_env()
    configuration = {
        "model": settings.model,
        "baseUrl": settings.base_url,
        "bm25Limit": 20,
        "semanticLimit": 20,
        "evidenceLimit": 8,
    }
    if output_path.exists():
        output = _load_json(output_path)
        completed_ids = validate_resume_output(
            output,
            expected_configuration=configuration,
            expected_case_ids=[str(case["id"]) for case in cases],
        )
    else:
        output = {
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "configuration": configuration,
            "cases": [],
        }
        completed_ids = set()

    retriever, ollama, qdrant = _build_default_retriever()
    deepseek_http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)
    )
    answer_client = DeepSeekAnswerClient(
        settings=settings,
        http_client=deepseek_http,
    )
    try:
        for index, case in enumerate(cases, start=1):
            if case["id"] in completed_ids:
                print(f"[{index}/{len(cases)}] skip {case['id']} (already complete)")
                continue
            print(f"[{index}/{len(cases)}] run {case['id']}")
            result = await run_answer_case(
                case,
                retriever=retriever,
                answer_client=answer_client,
            )
            output["cases"].append(result)
            _write_json(output_path, output)
            print(
                f"[{index}/{len(cases)}] saved {case['id']} "
                f"({result['timingMs']['total']} ms)"
            )
    finally:
        await deepseek_http.aclose()
        ollama.close()
        qdrant.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize the MCwiki answer-quality evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run live answer generation")
    run_parser.add_argument(
        "--answerable",
        type=Path,
        default=DEFAULT_ANSWERABLE_CASES,
    )
    run_parser.add_argument(
        "--unanswerable",
        type=Path,
        default=DEFAULT_UNANSWERABLE_CASES,
    )
    run_parser.add_argument("--output", type=Path, required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="combine raw results with manual reviews",
    )
    summarize_parser.add_argument("--input", type=Path, required=True)
    summarize_parser.add_argument("--reviews", type=Path, required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        asyncio.run(
            run_evaluation(
                answerable_path=args.answerable,
                unanswerable_path=args.unanswerable,
                output_path=args.output,
            )
        )
        return

    raw_run = _load_json(args.input)
    reviews = _load_json(args.reviews)
    summary = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sourceRun": str(args.input),
        "configuration": raw_run.get("configuration", {}),
        **summarize_results(raw_run, reviews),
    }
    _write_json(args.output, summary)
    print(f"saved summary to {args.output}")


if __name__ == "__main__":
    main()
