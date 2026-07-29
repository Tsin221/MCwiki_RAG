import unittest

from fastapi.testclient import TestClient

from rag_api import create_app
from rag_retrieval.hybrid import HybridResult


class FakeHybridRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, bm25_limit, semantic_limit, limit):
        self.calls.append(
            {
                "query": query,
                "bm25_limit": bm25_limit,
                "semantic_limit": semantic_limit,
                "limit": limit,
            }
        )
        return [
            HybridResult(
                chunk_id="chunk-redstone",
                title="红石中继器",
                text="红石中继器可以延迟红石信号。",
                source="https://example.test/redstone-repeater",
                score=0.032,
                bm25_rank=1,
                semantic_rank=2,
            )
        ]


class SearchApiTests(unittest.TestCase):
    def setUp(self):
        self.retriever = FakeHybridRetriever()
        self.client = TestClient(create_app(retriever=self.retriever))

    def test_search_returns_camel_case_hybrid_results(self):
        response = self.client.post(
            "/search",
            json={
                "query": "  红石中继器  ",
                "limit": 5,
                "bm25Limit": 20,
                "semanticLimit": 15,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "query": "红石中继器",
                "results": [
                    {
                        "chunkId": "chunk-redstone",
                        "title": "红石中继器",
                        "text": "红石中继器可以延迟红石信号。",
                        "source": "https://example.test/redstone-repeater",
                        "score": 0.032,
                        "bm25Rank": 1,
                        "semanticRank": 2,
                    }
                ],
            },
        )
        self.assertEqual(
            self.retriever.calls,
            [
                {
                    "query": "红石中继器",
                    "bm25_limit": 20,
                    "semantic_limit": 15,
                    "limit": 5,
                }
            ],
        )

    def test_blank_query_is_rejected_at_the_api_boundary(self):
        response = self.client.post("/search", json={"query": " \n\t"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.retriever.calls, [])

    def test_limits_outside_the_public_range_are_rejected(self):
        too_small = self.client.post(
            "/search",
            json={"query": "红石", "limit": 0},
        )
        too_large = self.client.post(
            "/search",
            json={"query": "红石", "semanticLimit": 101},
        )

        self.assertEqual(too_small.status_code, 422)
        self.assertEqual(too_large.status_code, 422)
        self.assertEqual(self.retriever.calls, [])


if __name__ == "__main__":
    unittest.main()
