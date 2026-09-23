import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_bm25_diagnostics import (
    percentile_nearest_rank,
    search_with_strategy,
    summarize_strategy,
)


class BM25DiagnosticMetricTests(unittest.TestCase):
    def test_summarizes_required_metrics(self):
        records = [
            {"expectedRank": 1, "resultCount": 20, "latencyMs": 1.0},
            {"expectedRank": 7, "resultCount": 20, "latencyMs": 2.0},
            {"expectedRank": 15, "resultCount": 20, "latencyMs": 3.0},
            {"expectedRank": None, "resultCount": 0, "latencyMs": 100.0},
        ]

        self.assertEqual(
            summarize_strategy(records),
            {
                "caseCount": 4,
                "hitAt5": 0.25,
                "hitAt10": 0.5,
                "hitAt20": 0.75,
                "mrrAt10": 0.2857,
                "emptyResultRate": 0.25,
                "p50LatencyMs": 2.0,
                "p95LatencyMs": 100.0,
            },
        )

    def test_nearest_rank_percentile_handles_single_value(self):
        self.assertEqual(percentile_nearest_rank([4.5], 0.95), 4.5)

    def test_missing_database_is_not_created(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "missing.db"

            with self.assertRaises(FileNotFoundError):
                search_with_strategy(
                    database,
                    "红石中继器",
                    strategy="token_or",
                    limit=20,
                )

            self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
