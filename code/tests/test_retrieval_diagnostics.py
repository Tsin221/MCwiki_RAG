import unittest
from types import SimpleNamespace

from rag_retrieval_diagnostics import expected_rank, summarize_retrieval


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_normalizes_source_and_preserves_rank(self):
        rows = [
            SimpleNamespace(source="https://zh.minecraft.wiki/w/其它"),
            SimpleNamespace(source="https://zh.minecraft.wiki/w/红石中继器"),
        ]
        expected = ["https://zh.minecraft.wiki/w/%E7%BA%A2%E7%9F%B3%E4%B8%AD%E7%BB%A7%E5%99%A8"]
        self.assertEqual(expected_rank(rows, expected), 2)

    def test_stage_metrics_expose_recall_and_ordering_gap(self):
        metrics = summarize_retrieval([
            {"bm25Rank": 1, "semanticRank": None, "fusedRank": 3, "evidenceContainsExpected": True, "retrievalMs": 100},
            {"bm25Rank": None, "semanticRank": 2, "fusedRank": 12, "evidenceContainsExpected": False, "retrievalMs": 200},
        ])
        self.assertEqual(metrics["bm25RankHitAt20"], 0.5)
        self.assertEqual(metrics["fusedRankHitAt10"], 0.5)
        self.assertEqual(metrics["fusedRankMrrAt10"], 0.1667)
        self.assertEqual(metrics["finalEvidenceRecall"], 0.5)
