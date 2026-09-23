from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data/evaluation/v2/questions.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "data/processed/chunks.jsonl"

QUOTAS = {
    "single_fact": (8, 10),
    "precise_value": (8, 10),
    "semantic_paraphrase": (6, 8),
    "multi_facet": (10, 12),
    "cross_chunk": (6, 8),
    "unanswerable": (6, 8),
    "edition_conflict": (4, 6),
}
KINDS = {"answerable", "unanswerable"}
EDITIONS = {"all", "java", "bedrock"}
ANNOTATION_STATUSES = {"confirmed", "candidate"}
UNANSWERABLE_REASONS = {
    "out_of_domain",
    "future_information",
    "private_realtime_state",
    "knowledge_base_gap",
}
REQUIRED_FIELDS = {
    "id",
    "question",
    "kind",
    "category",
    "facets",
    "expected_answer_points",
    "preferred_sources",
    "acceptable_sources",
    "relevant_chunk_ids",
    "edition",
    "version_scope",
    "annotation_status",
}


class DatasetValidationError(ValueError):
    """Raised when an evaluation-v2 dataset violates its contract."""


def _require_string(value: Any, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{case_id}: {field} must be a non-empty string")
    return value


def _require_string_list(value: Any, field: str, case_id: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DatasetValidationError(f"{case_id}: {field} must be a list of strings")
    return value


def _validate_source_url(value: str, case_id: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zh.minecraft.wiki"
        or not parsed.path.startswith("/w/")
    ):
        raise DatasetValidationError(f"{case_id}: invalid source URL: {value}")


def load_chunk_ids(path: Path) -> set[str]:
    """Read stable chunk identifiers without retaining the large corpus in memory."""
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as chunks:
        for line_number, line in enumerate(chunks, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identifiers.add(_require_string(record.get("id"), "id", f"line {line_number}"))
            except (json.JSONDecodeError, AttributeError) as error:
                raise DatasetValidationError(
                    f"invalid chunk JSON on line {line_number}"
                ) from error
    return identifiers


def _validate_answer_points(value: Any, case_id: str) -> None:
    if not isinstance(value, list) or not value:
        raise DatasetValidationError(f"{case_id}: answer point list must not be empty")
    point_ids: set[str] = set()
    for point in value:
        if not isinstance(point, Mapping):
            raise DatasetValidationError(f"{case_id}: answer point must be an object")
        point_id = _require_string(point.get("id"), "answer point id", case_id)
        _require_string(point.get("description"), "answer point description", case_id)
        if point_id in point_ids:
            raise DatasetValidationError(f"{case_id}: duplicate answer point id: {point_id}")
        point_ids.add(point_id)


def _validate_case(case: Any, chunk_ids: set[str]) -> None:
    if not isinstance(case, Mapping):
        raise DatasetValidationError("every case must be an object")
    case_id = _require_string(case.get("id"), "id", "case")
    missing = REQUIRED_FIELDS - case.keys()
    if missing:
        raise DatasetValidationError(
            f"{case_id}: missing fields: {', '.join(sorted(missing))}"
        )
    _require_string(case.get("question"), "question", case_id)
    kind = _require_string(case.get("kind"), "kind", case_id)
    category = _require_string(case.get("category"), "category", case_id)
    if kind not in KINDS:
        raise DatasetValidationError(f"{case_id}: unsupported kind: {kind}")
    if category not in QUOTAS:
        raise DatasetValidationError(f"{case_id}: unsupported category: {category}")
    if (kind == "unanswerable") != (category == "unanswerable"):
        raise DatasetValidationError(
            f"{case_id}: unanswerable kind and category must be used together"
        )
    if case.get("edition") not in EDITIONS:
        raise DatasetValidationError(f"{case_id}: unsupported edition")
    if case.get("annotation_status") not in ANNOTATION_STATUSES:
        raise DatasetValidationError(f"{case_id}: unsupported annotation_status")
    _require_string(case.get("version_scope"), "version_scope", case_id)
    _require_string_list(case.get("facets"), "facets", case_id)
    preferred = _require_string_list(
        case.get("preferred_sources"), "preferred_sources", case_id
    )
    acceptable = _require_string_list(
        case.get("acceptable_sources"), "acceptable_sources", case_id
    )
    relevant = _require_string_list(
        case.get("relevant_chunk_ids"), "relevant_chunk_ids", case_id
    )
    _require_string_list(case.get("tags", []), "tags", case_id)
    for source in [*preferred, *acceptable]:
        _validate_source_url(source, case_id)
    if not set(preferred).issubset(acceptable):
        raise DatasetValidationError(
            f"{case_id}: preferred_sources must be included in acceptable_sources"
        )
    unknown_chunks = set(relevant) - chunk_ids
    if unknown_chunks:
        raise DatasetValidationError(
            f"{case_id}: unknown chunk ids: {', '.join(sorted(unknown_chunks))}"
        )

    answer_points = case.get("expected_answer_points")
    if kind == "answerable":
        _validate_answer_points(answer_points, case_id)
        if not acceptable:
            raise DatasetValidationError(
                f"{case_id}: answerable case needs an acceptable source"
            )
        if not relevant:
            raise DatasetValidationError(
                f"{case_id}: answerable case needs a relevant chunk"
            )
        if "reason" in case:
            raise DatasetValidationError(f"{case_id}: answerable case must not have reason")
    else:
        if any((answer_points, preferred, acceptable, relevant)):
            raise DatasetValidationError(
                f"{case_id}: unanswerable case must not declare sources, chunks, or answer points"
            )
        if case.get("reason") not in UNANSWERABLE_REASONS:
            raise DatasetValidationError(f"{case_id}: invalid unanswerable reason")


def validate_dataset(
    dataset: Any,
    chunk_ids: set[str],
    *,
    enforce_quotas: bool = True,
) -> None:
    """Validate structure, semantics, evidence references, and dataset quotas."""
    if not isinstance(dataset, Mapping):
        raise DatasetValidationError("dataset must be an object")
    if dataset.get("schema_version") != 2:
        raise DatasetValidationError("schema_version must be 2")
    _require_string(dataset.get("dataset_id"), "dataset_id", "dataset")
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise DatasetValidationError("cases must be a list")

    identifiers: set[str] = set()
    for case in cases:
        _validate_case(case, chunk_ids)
        case_id = str(case["id"])
        if case_id in identifiers:
            raise DatasetValidationError(f"duplicate id: {case_id}")
        identifiers.add(case_id)

    if not enforce_quotas:
        return
    if not 40 <= len(cases) <= 60:
        raise DatasetValidationError("dataset must contain 40 to 60 cases")
    counts = Counter(str(case["category"]) for case in cases)
    for category, (minimum, maximum) in QUOTAS.items():
        count = counts[category]
        if not minimum <= count <= maximum:
            raise DatasetValidationError(
                f"category {category} has {count} cases; expected {minimum} to {maximum}"
            )


def build_statistics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(cases),
        "kind": dict(sorted(Counter(str(case["kind"]) for case in cases).items())),
        "primary_category": dict(
            sorted(Counter(str(case["category"]) for case in cases).items())
        ),
        "edition": dict(
            sorted(Counter(str(case["edition"]) for case in cases).items())
        ),
        "annotation_status": dict(
            sorted(
                Counter(str(case["annotation_status"]) for case in cases).items()
            )
        ),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and inspect evaluation v2")
    parser.add_argument("command", choices=("validate", "stats"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset = _load_json(args.dataset)
    validate_dataset(dataset, load_chunk_ids(args.chunks))
    if args.command == "validate":
        print(f"valid: {args.dataset} ({len(dataset['cases'])} cases)")
    else:
        print(json.dumps(build_statistics(dataset["cases"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
