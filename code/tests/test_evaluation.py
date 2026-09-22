import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rag_answer import DeepSeekSettings
from rag_evaluation import (
    analyze_answer,
    build_evaluation_configuration,
    load_evaluation_cases,
    normalize_source_url,
    run_answer_case,
    run_evaluation,
    summarize_results,
    validate_complete_output,
    validate_resume_output,
)
from rag_retrieval.hybrid import HybridResult


class FakeRetriever:
    def search(self, query, *, bm25_limit, semantic_limit, limit):
        return [
            HybridResult(
                chunk_id="chunk-redstone",
                title="红石中继器",
                text="红石中继器可以延迟信号。",
                source="https://zh.minecraft.wiki/w/红石中继器",
                score=0.032,
                bm25_rank=1,
                semantic_rank=1,
            )
        ]


class FakeAnswerClient:
    async def stream_answer(self, question, evidence):
        yield "红石中继器可以延迟信号。"
        yield "[1]"


class SourceNormalizationTests(unittest.TestCase):
    def test_normalizes_encoded_and_unicode_wiki_urls(self):
        encoded = (
            "https://zh.minecraft.wiki/w/"
            "%E7%BA%A2%E7%9F%B3%E4%B8%AD%E7%BB%A7%E5%99%A8/"
        )

        self.assertEqual(
            normalize_source_url(encoded),
            normalize_source_url("https://zh.minecraft.wiki/w/红石中继器"),
        )


class LoadEvaluationCasesTests(unittest.TestCase):
    def test_labels_answerable_and_unanswerable_case_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            answerable_path = root / "answerable.json"
            unanswerable_path = root / "unanswerable.json"
            answerable_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "known",
                            "question": "已知问题",
                            "category": "known",
                            "expected_sources": ["https://example.test/known"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            unanswerable_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "unknown",
                            "question": "未知问题",
                            "category": "unknown",
                            "reason": "知识库不包含。",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_evaluation_cases(answerable_path, unanswerable_path)

        self.assertEqual([case["kind"] for case in cases], ["answerable", "unanswerable"])
        self.assertEqual(cases[1]["expected_sources"], [])

    def test_rejects_resuming_with_different_configuration(self):
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_resume_output(
                {
                    "configuration": {"model": "old-model"},
                    "cases": [],
                },
                expected_configuration={"model": "current-model"},
                expected_case_ids=["known"],
            )

    def test_configuration_fingerprint_changes_with_case_content(self):
        settings = DeepSeekSettings(
            api_key="test-secret",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
        )
        original = build_evaluation_configuration(
            settings,
            [
                {
                    "id": "known",
                    "question": "原始问题",
                    "kind": "answerable",
                    "category": "known",
                    "expected_sources": ["https://example.test/known"],
                }
            ],
        )
        changed = build_evaluation_configuration(
            settings,
            [
                {
                    "id": "known",
                    "question": "修改后的问题",
                    "kind": "answerable",
                    "category": "known",
                    "expected_sources": ["https://example.test/known"],
                }
            ],
        )

        self.assertNotEqual(
            original["datasetFingerprint"],
            changed["datasetFingerprint"],
        )
        self.assertIn("systemPromptFingerprint", original)
        self.assertEqual(original["maxContextChars"], 12_000)
        self.assertEqual(original["temperature"], 0.2)
        self.assertEqual(original["thinkingType"], "disabled")
        self.assertNotIn("test-secret", json.dumps(original))


class FixedRegressionSuiteTests(unittest.TestCase):
    def test_default_regression_suite_matches_versioned_manifest(self):
        project_root = Path(__file__).resolve().parents[2]
        evaluation_root = project_root / "data" / "evaluation"
        manifest = json.loads(
            (evaluation_root / "regression_suite.json").read_text(encoding="utf-8")
        )

        cases = load_evaluation_cases(
            evaluation_root / "retrieval_questions.json",
            evaluation_root / "answer_quality" / "unanswerable_questions.json",
        )
        answerable_ids = [
            case["id"] for case in cases if case["kind"] == "answerable"
        ]
        unanswerable_ids = [
            case["id"] for case in cases if case["kind"] == "unanswerable"
        ]

        self.assertEqual(manifest["suiteId"], "mcwiki-rag-regression-v1")
        self.assertEqual(answerable_ids, manifest["answerableCaseIds"])
        self.assertEqual(unanswerable_ids, manifest["unanswerableCaseIds"])
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({case["id"] for case in cases}), 18)
        self.assertTrue(
            (project_root / manifest["referenceBaseline"]).is_file()
        )


class AnswerAnalysisTests(unittest.TestCase):
    def test_reports_expected_source_and_invalid_citation_ids(self):
        result = analyze_answer(
            answer="中继器可以延迟信号。[1] 另见不存在的证据。[9]",
            evidence=[
                {
                    "id": 1,
                    "url": "https://zh.minecraft.wiki/w/红石中继器",
                },
                {
                    "id": 2,
                    "url": "https://zh.minecraft.wiki/w/红石比较器",
                },
            ],
            expected_sources=[
                "https://zh.minecraft.wiki/w/"
                "%E7%BA%A2%E7%9F%B3%E4%B8%AD%E7%BB%A7%E5%99%A8"
            ],
        )

        self.assertTrue(result["expectedSourceRetrieved"])
        self.assertTrue(result["expectedSourceCited"])
        self.assertEqual(result["citationIds"], [1, 9])
        self.assertEqual(result["invalidCitationIds"], [9])
        self.assertEqual(result["citationValidityRate"], 0.5)


class RunAnswerCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_retrieval_generation_and_objective_checks(self):
        case = {
            "id": "redstone",
            "question": "中继器有什么作用？",
            "category": "gameplay",
            "kind": "answerable",
            "expected_sources": ["https://zh.minecraft.wiki/w/红石中继器"],
        }

        result = await run_answer_case(
            case,
            retriever=FakeRetriever(),
            answer_client=FakeAnswerClient(),
        )

        self.assertEqual(result["answer"]["text"], "红石中继器可以延迟信号。[1]")
        self.assertEqual(result["answer"]["status"], "answered")
        self.assertTrue(result["objective"]["expectedSourceCited"])
        self.assertEqual(result["evidence"][0]["chunkId"], "chunk-redstone")


class EvaluationRunFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_failed_case_does_not_stop_batch_and_can_be_retried(self):
        cases = [
            {"id": "first", "question": "first", "kind": "answerable", "category": "test"},
            {"id": "second", "question": "second", "kind": "answerable", "category": "test"},
        ]
        answer_settings = DeepSeekSettings(api_key="test-key")
        resources = (
            object(),
            SimpleNamespace(close=lambda: None),
            SimpleNamespace(close=lambda: None),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "run.json"
            with patch("rag_evaluation.load_evaluation_cases", return_value=cases), patch(
                "rag_evaluation.DeepSeekSettings.from_env", return_value=answer_settings
            ), patch("rag_evaluation.build_default_retriever", return_value=resources), patch(
                "rag_evaluation.run_answer_case",
                new=AsyncMock(side_effect=[RuntimeError("temporary"), {"id": "second", "timingMs": {"total": 1}}]),
            ):
                await run_evaluation(
                    answerable_path=Path("unused"),
                    unanswerable_path=Path("unused"),
                    output_path=output_path,
                )
            first_run = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([case["id"] for case in first_run["cases"]], ["first", "second"])
            self.assertEqual(first_run["cases"][0]["error"], "RuntimeError")
            with self.assertRaisesRegex(ValueError, "failed cases"):
                validate_complete_output(first_run, expected_case_ids=["first", "second"])

            with patch("rag_evaluation.load_evaluation_cases", return_value=cases), patch(
                "rag_evaluation.DeepSeekSettings.from_env", return_value=answer_settings
            ), patch("rag_evaluation.build_default_retriever", return_value=resources), patch(
                "rag_evaluation.run_answer_case", new=AsyncMock(return_value={"id": "first", "timingMs": {"total": 1}})
            ) as retry:
                await run_evaluation(
                    answerable_path=Path("unused"),
                    unanswerable_path=Path("unused"),
                    output_path=output_path,
                )
            self.assertEqual(retry.call_count, 1)
            second_run = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual({case["id"] for case in second_run["cases"]}, {"first", "second"})
            self.assertEqual(len(second_run["cases"]), 2)


class SummaryTests(unittest.TestCase):
    def test_combines_objective_metrics_with_manual_reviews(self):
        raw_run = {
            "cases": [
                {
                    "id": "answerable",
                    "kind": "answerable",
                    "objective": {
                        "expectedSourceRetrieved": True,
                        "expectedSourceCited": True,
                        "citationCount": 2,
                        "validCitationCount": 1,
                        "invalidCitationIds": [9],
                    },
                },
                {
                    "id": "unanswerable",
                    "kind": "unanswerable",
                    "objective": {
                        "citationCount": 0,
                        "validCitationCount": 0,
                        "invalidCitationIds": [],
                    },
                },
            ]
        }
        reviews = {
            "answerable": {
                "correctness": 2,
                "completeness": 2,
                "faithfulness": 2,
                "notes": "回答正确，但包含一个无效引用。",
            },
            "unanswerable": {
                "abstainedReliably": True,
                "notes": "明确说明资料不足。",
            },
        }

        summary = summarize_results(
            raw_run,
            reviews,
            expected_case_ids=["answerable", "unanswerable"],
        )

        self.assertEqual(summary["metrics"]["answerableCount"], 1)
        self.assertEqual(summary["metrics"]["correctnessAverage"], 2.0)
        self.assertEqual(summary["metrics"]["expectedSourceCitationRate"], 1.0)
        self.assertEqual(summary["metrics"]["citationValidityRate"], 0.5)
        self.assertEqual(summary["metrics"]["abstentionRate"], 1.0)
        self.assertFalse(summary["meetsSuggestedThresholds"])

    def test_rejects_incomplete_raw_run_before_calculating_thresholds(self):
        raw_run = {
            "cases": [
                {
                    "id": "answerable",
                    "kind": "answerable",
                    "objective": {
                        "expectedSourceRetrieved": True,
                        "expectedSourceCited": True,
                        "citationCount": 1,
                        "validCitationCount": 1,
                        "invalidCitationIds": [],
                    },
                },
                {
                    "id": "unanswerable",
                    "kind": "unanswerable",
                    "objective": {
                        "citationCount": 0,
                        "validCitationCount": 0,
                        "invalidCitationIds": [],
                    },
                },
            ]
        }
        reviews = {
            "answerable": {
                "correctness": 2,
                "completeness": 2,
                "faithfulness": 2,
            },
            "unanswerable": {"abstainedReliably": True},
        }

        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize_results(
                raw_run,
                reviews,
                expected_case_ids=["answerable", "unanswerable", "missing"],
            )

    def test_rejects_string_abstention_review_value(self):
        raw_run = {
            "cases": [
                {
                    "id": "unanswerable",
                    "kind": "unanswerable",
                    "objective": {
                        "citationCount": 0,
                        "validCitationCount": 0,
                        "invalidCitationIds": [],
                    },
                }
            ]
        }
        reviews = {
            "unanswerable": {
                "abstainedReliably": "false",
            }
        }

        with self.assertRaisesRegex(ValueError, "boolean"):
            summarize_results(
                raw_run,
                reviews,
                expected_case_ids=["unanswerable"],
            )


if __name__ == "__main__":
    unittest.main()
