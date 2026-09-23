import json
import tempfile
import unittest
from pathlib import Path

from rag_evaluation_schema import (
    DatasetValidationError,
    build_statistics,
    load_chunk_ids,
    validate_dataset,
)


def answerable_case(**overrides):
    case = {
        "id": "sample-answerable",
        "question": "示例问题？",
        "kind": "answerable",
        "category": "single_fact",
        "facets": ["示例事实"],
        "expected_answer_points": [
            {"id": "sample-point", "description": "说明示例事实"}
        ],
        "preferred_sources": ["https://zh.minecraft.wiki/w/示例"],
        "acceptable_sources": ["https://zh.minecraft.wiki/w/示例"],
        "relevant_chunk_ids": ["known-chunk"],
        "edition": "all",
        "version_scope": "current",
        "annotation_status": "candidate",
    }
    case.update(overrides)
    return case


def unanswerable_case(**overrides):
    case = {
        "id": "sample-unanswerable",
        "question": "示例无答案问题？",
        "kind": "unanswerable",
        "category": "unanswerable",
        "facets": [],
        "expected_answer_points": [],
        "preferred_sources": [],
        "acceptable_sources": [],
        "relevant_chunk_ids": [],
        "edition": "all",
        "version_scope": "current",
        "reason": "out_of_domain",
        "annotation_status": "confirmed",
    }
    case.update(overrides)
    return case


class ChunkLoadingTests(unittest.TestCase):
    def test_loads_ids_from_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            path.write_text(
                '{"id":"one"}\n{"id":"two"}\n', encoding="utf-8"
            )

            self.assertEqual(load_chunk_ids(path), {"one", "two"})


class DatasetValidationTests(unittest.TestCase):
    def test_accepts_well_formed_cases_and_equivalent_sources(self):
        case = answerable_case(
            preferred_sources=["https://zh.minecraft.wiki/w/Java版1.21"],
            acceptable_sources=[
                "https://zh.minecraft.wiki/w/Java版1.21",
                "https://zh.minecraft.wiki/w/1.21",
            ],
        )

        validate_dataset(
            {"schema_version": 2, "dataset_id": "test", "cases": [case]},
            {"known-chunk"},
            enforce_quotas=False,
        )

    def test_rejects_duplicate_ids(self):
        case = answerable_case()
        with self.assertRaisesRegex(DatasetValidationError, "duplicate id"):
            validate_dataset(
                {"schema_version": 2, "dataset_id": "test", "cases": [case, case]},
                {"known-chunk"},
                enforce_quotas=False,
            )

    def test_rejects_unknown_chunks(self):
        with self.assertRaisesRegex(DatasetValidationError, "unknown chunk"):
            validate_dataset(
                {
                    "schema_version": 2,
                    "dataset_id": "test",
                    "cases": [answerable_case(relevant_chunk_ids=["missing"])],
                },
                {"known-chunk"},
                enforce_quotas=False,
            )

    def test_rejects_non_wiki_or_insecure_urls(self):
        for url in ("http://zh.minecraft.wiki/w/示例", "https://example.com/wiki"):
            with self.subTest(url=url), self.assertRaisesRegex(
                DatasetValidationError, "invalid source URL"
            ):
                validate_dataset(
                    {
                        "schema_version": 2,
                        "dataset_id": "test",
                        "cases": [answerable_case(acceptable_sources=[url])],
                    },
                    {"known-chunk"},
                    enforce_quotas=False,
                )

    def test_rejects_empty_answer_points(self):
        with self.assertRaisesRegex(DatasetValidationError, "answer point"):
            validate_dataset(
                {
                    "schema_version": 2,
                    "dataset_id": "test",
                    "cases": [answerable_case(expected_answer_points=[])],
                },
                {"known-chunk"},
                enforce_quotas=False,
            )

    def test_rejects_structural_errors(self):
        with self.assertRaisesRegex(DatasetValidationError, "question"):
            validate_dataset(
                {
                    "schema_version": 2,
                    "dataset_id": "test",
                    "cases": [answerable_case(question=3)],
                },
                {"known-chunk"},
                enforce_quotas=False,
            )

    def test_enforces_unanswerable_constraints(self):
        with self.assertRaisesRegex(DatasetValidationError, "must not declare sources"):
            validate_dataset(
                {
                    "schema_version": 2,
                    "dataset_id": "test",
                    "cases": [
                        unanswerable_case(
                            acceptable_sources=["https://zh.minecraft.wiki/w/示例"]
                        )
                    ],
                },
                {"known-chunk"},
                enforce_quotas=False,
            )

    def test_rejects_kind_and_primary_category_mismatch(self):
        with self.assertRaisesRegex(DatasetValidationError, "category"):
            validate_dataset(
                {
                    "schema_version": 2,
                    "dataset_id": "test",
                    "cases": [answerable_case(category="unanswerable")],
                },
                {"known-chunk"},
                enforce_quotas=False,
            )

    def test_versioned_v2_dataset_passes_full_validation(self):
        project_root = Path(__file__).resolve().parents[2]
        dataset = json.loads(
            (project_root / "data/evaluation/v2/questions.json").read_text(
                encoding="utf-8"
            )
        )

        validate_dataset(
            dataset,
            load_chunk_ids(project_root / "data/processed/chunks.jsonl"),
        )

        java_case = next(
            case
            for case in dataset["cases"]
            if case["id"] == "java-1-21-equivalent-sources"
        )
        self.assertEqual(
            java_case["preferred_sources"],
            ["https://zh.minecraft.wiki/w/Java%E7%89%881.21"],
        )
        self.assertIn(
            "https://zh.minecraft.wiki/w/1.21",
            java_case["acceptable_sources"],
        )


class StatisticsTests(unittest.TestCase):
    def test_reports_primary_categories_without_double_counting(self):
        cases = [
            answerable_case(tags=["cross_chunk"]),
            unanswerable_case(),
        ]

        statistics = build_statistics(cases)

        self.assertEqual(statistics["total"], 2)
        self.assertEqual(statistics["kind"], {"answerable": 1, "unanswerable": 1})
        self.assertEqual(
            statistics["primary_category"],
            {"single_fact": 1, "unanswerable": 1},
        )


if __name__ == "__main__":
    unittest.main()
