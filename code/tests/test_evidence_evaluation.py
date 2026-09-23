import unittest

from rag_evidence_evaluation import compare_strategies, evaluate_strategy


SNAPSHOT = {
    "schema_version": 1,
    "cases": [
        {
            "id": "multi",
            "question": "两个要点是什么？",
            "kind": "answerable",
            "relevant_chunk_ids": ["relevant-a", "relevant-b"],
            "acceptable_sources": ["https://example.test/a"],
            "candidates": [
                {
                    "chunk_id": "relevant-a",
                    "title": "A",
                    "text": "要点甲",
                    "source": "https://example.test/a",
                    "document_id": "doc-a",
                    "chunk_index": 0,
                },
                {
                    "chunk_id": "noise-a",
                    "title": "A",
                    "text": "无关历史",
                    "source": "https://example.test/a",
                    "document_id": "doc-a",
                    "chunk_index": 9,
                },
                {
                    "chunk_id": "relevant-b",
                    "title": "B",
                    "text": "要点乙",
                    "source": "https://example.test/b",
                    "document_id": "doc-b",
                    "chunk_index": 0,
                },
            ],
        },
        {
            "id": "empty",
            "question": "无答案",
            "kind": "unanswerable",
            "relevant_chunk_ids": [],
            "acceptable_sources": [],
            "candidates": [],
        },
    ],
}


class EvidenceEvaluationTests(unittest.TestCase):
    def test_reports_recall_precision_source_recall_and_per_case_evidence(self):
        result = evaluate_strategy(
            SNAPSHOT,
            strategy="source_cap",
            max_context_chars=100,
            max_chunks_per_source=1,
        )

        self.assertEqual(result["metrics"]["relevantEvidenceRecall"], 1.0)
        self.assertEqual(result["metrics"]["contextPrecision"], 1.0)
        self.assertEqual(result["metrics"]["expectedSourceRecall"], 1.0)
        self.assertEqual(result["metrics"]["evidenceIdValidityRate"], 1.0)
        self.assertEqual(
            result["cases"][0]["evidence"][0]["componentChunkIds"],
            ["relevant-a"],
        )

    def test_compare_runs_every_required_strategy_on_the_same_cases(self):
        comparison = compare_strategies(
            SNAPSHOT,
            max_context_chars=100,
            source_caps=(1, 2),
            redundancy_thresholds=(0.8,),
        )

        self.assertEqual(
            set(comparison["runs"]),
            {
                "ranked_first",
                "source_cap_1",
                "source_cap_2",
                "adjacent_merge",
                "redundancy_suppression_0.8",
            },
        )
        self.assertEqual(comparison["caseIds"], ["multi", "empty"])


if __name__ == "__main__":
    unittest.main()
