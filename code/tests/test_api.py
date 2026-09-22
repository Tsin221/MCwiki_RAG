import json
import tempfile
import unittest
from collections.abc import AsyncIterator
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_answer import AnswerConfigurationError, AnswerEvidence, ModelUnavailableError
from rag_api import create_app
from rag_retrieval.hybrid import HybridResult
from rag_settings import RetrievalSettings


class FakeHybridRetriever:
    def __init__(self, *, results=None, error=None):
        self.calls = []
        self.results = results
        self.error = error

    def search(self, query, *, bm25_limit, semantic_limit, limit):
        self.calls.append(
            {
                "query": query,
                "bm25_limit": bm25_limit,
                "semantic_limit": semantic_limit,
                "limit": limit,
            }
        )
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
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


class FakeAnswerClient:
    def __init__(self, chunks=None, error=None):
        self.chunks = ["中继器可以延迟", "并增强红石信号。[1]"] if chunks is None else chunks
        self.error = error
        self.calls = []

    async def stream_answer(
        self,
        question: str,
        evidence: list[AnswerEvidence],
    ) -> AsyncIterator[str]:
        self.calls.append({"question": question, "evidence": evidence})
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = lines[0].removeprefix("event: ")
        data = lines[1].removeprefix("data: ")
        events.append((event_type, json.loads(data)))
    return events


class SearchApiTests(unittest.TestCase):
    def setUp(self):
        self.retriever = FakeHybridRetriever()
        self.client = TestClient(create_app(retriever=self.retriever))

    def test_search_failure_uses_shared_error_contract(self):
        client = TestClient(create_app(retriever=FakeHybridRetriever(error=RuntimeError("private detail"))))
        response = client.post("/search", json={"query": "红石"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_UNAVAILABLE")
        self.assertNotIn("private detail", response.text)

    def test_missing_route_and_wrong_method_use_shared_error_contract(self):
        self.assertEqual(self.client.get("/missing").json()["error"]["code"], "NOT_FOUND")
        response = self.client.get("/answers")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.json()["error"]["code"], "METHOD_NOT_ALLOWED")

    def test_wrong_qdrant_collection_blocks_retrieval_and_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            app = create_app(retriever=self.retriever, answer_client=FakeAnswerClient())
            app.state.retrieval_settings = RetrievalSettings(bm25_path=bm25_path)
            app.state.qdrant = SimpleNamespace(get_collection=lambda name: (_ for _ in ()).throw(RuntimeError("missing collection")))
            client = TestClient(app)
            self.assertEqual(client.get("/health").json(), {"api": "ok"})
            self.assertEqual(client.get("/ready").status_code, 503)
            response = client.post("/search", json={"query": "红石"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_UNAVAILABLE")
            self.assertEqual(self.retriever.calls, [])

    def test_wrong_qdrant_vector_dimension_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            app = create_app(retriever=self.retriever, answer_client=FakeAnswerClient())
            app.state.retrieval_settings = RetrievalSettings(bm25_path=bm25_path)
            app.state.qdrant = SimpleNamespace(get_collection=lambda name: SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=768))),
                status="green",
            ))
            response = TestClient(app).get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["qdrant"], "mismatch")

    def test_missing_answer_key_keeps_search_available(self):
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            settings = RetrievalSettings(bm25_path=bm25_path)
            collection = SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1024))),
                status="green",
            )
            qdrant = SimpleNamespace(get_collection=lambda name: collection, close=lambda: None)
            with patch("rag_api.RetrievalSettings.from_env", return_value=settings), patch(
                "rag_api._build_default_retriever",
                return_value=(self.retriever, SimpleNamespace(close=lambda: None), qdrant),
            ), patch("rag_api.DeepSeekSettings.from_env", side_effect=AnswerConfigurationError("missing")):
                with TestClient(create_app()) as client:
                    self.assertEqual(client.post("/search", json={"query": "红石"}).status_code, 200)
                    self.assertEqual(client.get("/ready").json()["answer_model"], "unavailable")
                    response = client.post("/answers", json={"question": "红石是什么？"})
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["error"]["code"], "CONFIGURATION_ERROR")

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


class AnswerApiTests(unittest.TestCase):
    def setUp(self):
        self.retriever = FakeHybridRetriever()
        self.answer_client = FakeAnswerClient()
        self.client = TestClient(
            create_app(
                retriever=self.retriever,
                answer_client=self.answer_client,
                cors_origins=["http://localhost:5173"],
                cookie_secure=False,
            )
        )

    def test_answers_streams_contract_and_sets_anonymous_cookie(self):
        response = self.client.post(
            "/answers",
            json={"question": "  红石中继器有什么作用？  "},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        self.assertEqual(
            parse_sse(response.text),
            [
                ("meta", {"question": "红石中继器有什么作用？"}),
                (
                    "sources",
                    {
                        "items": [
                            {
                                "id": 1,
                                "chunkId": "chunk-redstone",
                                "title": "红石中继器",
                                "url": "https://example.test/redstone-repeater",
                                "excerpt": "红石中继器可以延迟红石信号。",
                            }
                        ]
                    },
                ),
                ("delta", {"text": "中继器可以延迟"}),
                ("delta", {"text": "并增强红石信号。[1]"}),
                ("done", {"status": "answered"}),
            ],
        )
        cookie = SimpleCookie()
        cookie.load(response.headers["set-cookie"])
        visitor = cookie["mcwiki_visitor_id"]
        self.assertTrue(visitor["httponly"])
        self.assertEqual(visitor["samesite"].lower(), "lax")
        self.assertEqual(visitor["path"], "/")
        self.assertEqual(visitor["max-age"], "15552000")
        self.assertFalse(visitor["secure"])

    def test_existing_visitor_cookie_is_reused_without_resetting_it(self):
        self.client.cookies.set(
            "mcwiki_visitor_id",
            "7f3f4473-dc2a-4b46-a2fb-f22887bd9184",
        )

        response = self.client.post("/answers", json={"question": "红石是什么？"})

        self.assertNotIn("set-cookie", response.headers)

    def test_empty_evidence_skips_model_and_returns_stable_status(self):
        retriever = FakeHybridRetriever(results=[])
        answer_client = FakeAnswerClient()
        client = TestClient(
            create_app(retriever=retriever, answer_client=answer_client)
        )

        response = client.post("/answers", json={"question": "不存在的问题"})

        self.assertEqual(
            parse_sse(response.text),
            [
                ("meta", {"question": "不存在的问题"}),
                ("sources", {"items": []}),
                (
                    "delta",
                    {"text": "现有知识库没有足够资料支持可靠回答。"},
                ),
                ("done", {"status": "insufficientEvidence"}),
            ],
        )
        self.assertEqual(answer_client.calls, [])

    def test_model_failure_after_stream_start_becomes_error_event(self):
        client = TestClient(
            create_app(
                retriever=FakeHybridRetriever(),
                answer_client=FakeAnswerClient(
                    chunks=["已生成的部分。"],
                    error=ModelUnavailableError("secret upstream body"),
                ),
            )
        )

        response = client.post("/answers", json={"question": "红石是什么？"})
        events = parse_sse(response.text)

        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["code"], "MODEL_UNAVAILABLE")
        self.assertNotIn("secret upstream body", response.text)

    def test_unexpected_stream_failure_becomes_internal_error_event(self):
        client = TestClient(create_app(
            retriever=FakeHybridRetriever(),
            answer_client=FakeAnswerClient(chunks=[], error=RuntimeError("private detail")),
        ))
        response = client.post("/answers", json={"question": "红石是什么？"})
        self.assertEqual(parse_sse(response.text)[-1][1]["code"], "INTERNAL_ERROR")
        self.assertNotIn("private detail", response.text)

    def test_secure_cookie_setting_is_applied(self):
        client = TestClient(create_app(
            retriever=FakeHybridRetriever(),
            answer_client=FakeAnswerClient(),
            cookie_secure=True,
        ))
        response = client.post("/answers", json={"question": "红石是什么？"})
        cookie = SimpleCookie()
        cookie.load(response.headers["set-cookie"])
        self.assertTrue(cookie["mcwiki_visitor_id"]["secure"])

    def test_retrieval_failure_before_stream_uses_stable_json_error(self):
        client = TestClient(
            create_app(
                retriever=FakeHybridRetriever(error=RuntimeError("local path")),
                answer_client=FakeAnswerClient(),
            )
        )

        response = client.post("/answers", json={"question": "红石是什么？"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "RETRIEVAL_UNAVAILABLE",
                    "message": "知识库检索暂时不可用，请稍后重试。",
                }
            },
        )
        self.assertNotIn("local path", response.text)

    def test_validation_errors_use_the_shared_error_shape(self):
        response = self.client.post("/answers", json={"question": " \n\t"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_cors_preflight_allows_the_configured_credentialed_origin(self):
        response = self.client.options(
            "/answers",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://localhost:5173",
        )
        self.assertEqual(
            response.headers["access-control-allow-credentials"],
            "true",
        )


if __name__ == "__main__":
    unittest.main()
