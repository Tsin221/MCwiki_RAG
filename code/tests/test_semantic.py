import unittest
from types import SimpleNamespace

import httpx

from rag_retrieval.semantic import SemanticRetriever


class FakeQdrant:
    def __init__(self, points=None):
        self.points = points or []
        self.queries = []

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(points=self.points)


def embedding_client(embeddings):
    def handler(request):
        return httpx.Response(200, json={"embeddings": embeddings})

    return httpx.Client(transport=httpx.MockTransport(handler))


class SemanticRetrieverTests(unittest.TestCase):
    def test_returns_qdrant_results_with_the_shared_retrieval_fields(self):
        ollama = embedding_client([[0.1, 0.2, 0.3]])
        qdrant = FakeQdrant(
            [
                SimpleNamespace(
                    score=0.91,
                    payload={
                        "chunk_id": "chunk-redstone",
                        "title": "红石中继器",
                        "text": "红石中继器可以延迟红石信号。",
                        "source": "https://example.test/redstone-repeater",
                        "metadata": {"document_id": "doc-redstone", "chunk_index": 3},
                    },
                )
            ]
        )
        retriever = SemanticRetriever(
            ollama_client=ollama,
            qdrant_client=qdrant,
            vector_size=3,
        )

        results = retriever.search("中继器有什么用？", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-redstone")
        self.assertEqual(results[0].title, "红石中继器")
        self.assertEqual(results[0].text, "红石中继器可以延迟红石信号。")
        self.assertEqual(
            results[0].source,
            "https://example.test/redstone-repeater",
        )
        self.assertEqual(results[0].score, 0.91)
        self.assertEqual(results[0].document_id, "doc-redstone")
        self.assertEqual(results[0].chunk_index, 3)
        self.assertEqual(
            qdrant.queries,
            [
                {
                    "collection_name": "mcwiki_chunks",
                    "query": [0.1, 0.2, 0.3],
                    "limit": 5,
                    "with_payload": True,
                    "with_vectors": False,
                }
            ],
        )
        ollama.close()

    def test_blank_query_returns_empty_without_calling_services(self):
        ollama = embedding_client([[0.1, 0.2, 0.3]])
        qdrant = FakeQdrant()
        retriever = SemanticRetriever(
            ollama_client=ollama,
            qdrant_client=qdrant,
            vector_size=3,
        )

        self.assertEqual(retriever.search(" \t\n"), [])
        self.assertEqual(qdrant.queries, [])
        ollama.close()

    def test_rejects_non_positive_limit(self):
        ollama = embedding_client([[0.1, 0.2, 0.3]])
        retriever = SemanticRetriever(
            ollama_client=ollama,
            qdrant_client=FakeQdrant(),
            vector_size=3,
        )

        with self.assertRaisesRegex(ValueError, "limit"):
            retriever.search("红石", limit=0)
        ollama.close()

    def test_rejects_wrong_embedding_dimension(self):
        ollama = embedding_client([[0.1, 0.2]])
        retriever = SemanticRetriever(
            ollama_client=ollama,
            qdrant_client=FakeQdrant(),
            vector_size=3,
        )

        with self.assertRaisesRegex(ValueError, "dimension"):
            retriever.search("红石")
        ollama.close()

    def test_rejects_non_object_ollama_response(self):
        ollama = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=[])
            )
        )
        retriever = SemanticRetriever(
            ollama_client=ollama,
            qdrant_client=FakeQdrant(),
            vector_size=3,
        )

        with self.assertRaisesRegex(ValueError, "object"):
            retriever.search("红石")
        ollama.close()

    def test_rejects_malformed_qdrant_payload(self):
        ollama = embedding_client([[0.1, 0.2, 0.3]])
        qdrant = FakeQdrant(
            [
                SimpleNamespace(
                    score=0.5,
                    payload={"chunk_id": "missing-fields"},
                )
            ]
        )
        retriever = SemanticRetriever(
            ollama_client=ollama,
            qdrant_client=qdrant,
            vector_size=3,
        )

        with self.assertRaisesRegex(ValueError, "payload"):
            retriever.search("红石")
        ollama.close()


if __name__ == "__main__":
    unittest.main()
