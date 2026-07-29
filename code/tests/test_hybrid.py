import unittest

from rag_retrieval.bm25 import BM25Result
from rag_retrieval.hybrid import HybridRetriever, fuse_rrf
from rag_retrieval.semantic import SemanticResult


def bm25_result(chunk_id, score=1.0):
    return BM25Result(
        chunk_id=chunk_id,
        title=f"title-{chunk_id}",
        text=f"text-{chunk_id}",
        source=f"https://example.test/{chunk_id}",
        score=score,
    )


def semantic_result(chunk_id, score=0.8):
    return SemanticResult(
        chunk_id=chunk_id,
        title=f"title-{chunk_id}",
        text=f"text-{chunk_id}",
        source=f"https://example.test/{chunk_id}",
        score=score,
    )


class RRFFusionTests(unittest.TestCase):
    def test_shared_chunk_receives_both_rank_contributions(self):
        results = fuse_rrf(
            [bm25_result("shared"), bm25_result("bm25-only")],
            [semantic_result("semantic-only"), semantic_result("shared")],
            rrf_k=60,
            limit=3,
        )

        self.assertEqual(
            [result.chunk_id for result in results],
            ["shared", "semantic-only", "bm25-only"],
        )
        self.assertEqual(results[0].bm25_rank, 1)
        self.assertEqual(results[0].semantic_rank, 2)
        self.assertAlmostEqual(results[0].score, 1 / 61 + 1 / 62)
        self.assertIsNone(results[1].bm25_rank)
        self.assertEqual(results[1].semantic_rank, 1)

    def test_ties_are_deterministic(self):
        results = fuse_rrf(
            [bm25_result("b")],
            [semantic_result("a")],
            rrf_k=60,
            limit=2,
        )

        self.assertEqual([result.chunk_id for result in results], ["a", "b"])

    def test_duplicate_chunk_in_one_route_is_counted_once(self):
        results = fuse_rrf(
            [bm25_result("same"), bm25_result("same")],
            [],
            rrf_k=60,
            limit=5,
        )

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 1 / 61)

    def test_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            fuse_rrf([], [], limit=0)
        with self.assertRaisesRegex(ValueError, "rrf_k"):
            fuse_rrf([], [], rrf_k=-1)


class HybridRetrieverTests(unittest.TestCase):
    def test_calls_both_routes_and_returns_fused_results(self):
        calls = []

        def search_bm25_route(query, limit):
            calls.append(("bm25", query, limit))
            return [bm25_result("shared"), bm25_result("lexical")]

        def search_semantic_route(query, limit):
            calls.append(("semantic", query, limit))
            return [semantic_result("shared"), semantic_result("semantic")]

        retriever = HybridRetriever(
            search_bm25=search_bm25_route,
            search_semantic=search_semantic_route,
        )

        results = retriever.search(
            "红石中继器",
            bm25_limit=20,
            semantic_limit=15,
            limit=3,
        )

        self.assertEqual(
            calls,
            [
                ("bm25", "红石中继器", 20),
                ("semantic", "红石中继器", 15),
            ],
        )
        self.assertEqual(results[0].chunk_id, "shared")

    def test_blank_query_skips_both_routes(self):
        def fail_if_called(query, limit):
            raise AssertionError("retrieval route should not be called")

        retriever = HybridRetriever(
            search_bm25=fail_if_called,
            search_semantic=fail_if_called,
        )

        self.assertEqual(retriever.search(" \n\t"), [])


if __name__ == "__main__":
    unittest.main()
