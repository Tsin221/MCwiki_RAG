import json
import tempfile
import unittest
from collections.abc import AsyncIterator
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_answer import (
    AnswerConfigurationError,
    AnswerEvidence,
    DeepSeekSettings,
    ModelUnavailableError,
)
from rag_api import create_app
from rag_correction import DeepSeekEvidenceAssessor, EvidenceAssessment
from rag_query import QueryPlan, build_query_plan
from rag_reranker import CrossEncoderReranker, NoopReranker
from rag_retrieval.hybrid import HybridResult
from rag_settings import (
    CORRECTIVE_ENABLED,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    RetrievalSettings,
    STEP_BACK_QUERY_STRATEGY,
    VERIFICATION_ENABLED,
)
from rag_verification import (
    VERIFICATION_PASSED,
    VERIFICATION_UNAVAILABLE,
    VERIFICATION_UNSUPPORTED,
    ClaimCheck,
    DeepSeekAnswerVerifier,
    ModelVerdict,
)


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


class FakeQueryPlanner:
    def __init__(self, candidates=(), *, error=None):
        self.calls = []
        self.candidates = list(candidates)
        self.error = error

    async def plan(self, question: str) -> QueryPlan:
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return build_query_plan(
            question,
            self.candidates,
            strategy=STEP_BACK_QUERY_STRATEGY,
        )


class RecordingRetriever:
    """Retriever stub returning a scripted ranking per query."""

    def __init__(self, rankings=None, error=None):
        self.calls = []
        self.rankings = rankings or {}
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
        return self.rankings.get(query, [])[:limit]


def stub_result(chunk_id, *, text=None, document_id=None, chunk_index=None):
    return HybridResult(
        chunk_id=chunk_id,
        title=f"标题 {chunk_id}",
        text=text or f"{chunk_id} 的正文",
        source=f"https://example.test/{chunk_id}",
        score=0.03,
        bm25_rank=1,
        semantic_rank=1,
        document_id=document_id,
        chunk_index=chunk_index,
    )


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
                                "componentChunkIds": ["chunk-redstone"],
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


class StepBackAnswerApiTests(unittest.TestCase):
    question = "红石中继器有什么作用？"
    abstract_question = "红石信号的传输机制是什么"

    def post_answer(self, client, question=None):
        return client.post(
            "/answers",
            json={"question": self.question if question is None else question},
        )

    def sources_of(self, response):
        return next(
            data for event, data in parse_sse(response.text) if event == "sources"
        )

    def test_answers_retrieves_once_with_the_original_question_by_default(self):
        retriever = RecordingRetriever({self.question: [stub_result("original")]})
        answer_client = FakeAnswerClient()

        response = self.post_answer(
            TestClient(create_app(retriever=retriever, answer_client=answer_client))
        )

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            [self.question],
        )
        self.assertEqual(
            [call["limit"] for call in retriever.calls],
            [RetrievalSettings().candidate_limit],
        )
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["original"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_step_back_plan_runs_both_queries_and_merges_the_sources(self):
        retriever = RecordingRetriever(
            {
                self.question: [stub_result("shared"), stub_result("only-original")],
                self.abstract_question: [
                    stub_result("only-step-back"),
                    stub_result("shared"),
                ],
            }
        )
        planner = FakeQueryPlanner([self.abstract_question])
        answer_client = FakeAnswerClient()
        client = TestClient(
            create_app(
                retriever=retriever,
                answer_client=answer_client,
                query_planner=planner,
            )
        )

        response = self.post_answer(client, f"  {self.question}  ")

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            [self.question, self.abstract_question],
        )
        self.assertEqual(planner.calls, [self.question])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["shared", "only-step-back", "only-original"],
        )
        self.assertEqual(
            [item.chunk_id for item in answer_client.calls[0]["evidence"]],
            ["shared", "only-step-back", "only-original"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_fused_neighbours_are_merged_into_one_evidence_item(self):
        retriever = RecordingRetriever(
            {
                self.question: [
                    stub_result(
                        "chunk-0",
                        text="红石中继器可以延迟信号",
                        document_id="doc",
                        chunk_index=0,
                    )
                ],
                self.abstract_question: [
                    stub_result(
                        "chunk-1",
                        text="延迟信号并增强信号强度",
                        document_id="doc",
                        chunk_index=1,
                    )
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            app = create_app(
                retriever=retriever,
                answer_client=FakeAnswerClient(),
                query_planner=FakeQueryPlanner([self.abstract_question]),
            )
            app.state.retrieval_settings = RetrievalSettings(bm25_path=bm25_path)
            app.state.qdrant = SimpleNamespace(
                get_collection=lambda name: SimpleNamespace(
                    config=SimpleNamespace(
                        params=SimpleNamespace(
                            vectors=SimpleNamespace(size=RetrievalSettings().vector_size)
                        )
                    ),
                    status="green",
                )
            )

            response = self.post_answer(TestClient(app))

        items = self.sources_of(response)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["chunkId"], "chunk-0")
        self.assertEqual(items[0]["componentChunkIds"], ["chunk-0", "chunk-1"])

    def test_planning_failure_answers_with_the_original_question_only(self):
        retriever = RecordingRetriever({self.question: [stub_result("original")]})
        planner = FakeQueryPlanner(error=RuntimeError("planner endpoint down"))

        response = self.post_answer(
            TestClient(
                create_app(
                    retriever=retriever,
                    answer_client=FakeAnswerClient(),
                    query_planner=planner,
                )
            )
        )

        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["original"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_step_back_retrieval_failure_keeps_the_stable_json_error(self):
        retriever = RecordingRetriever(error=RuntimeError("local path"))
        client = TestClient(
            create_app(
                retriever=retriever,
                answer_client=FakeAnswerClient(),
                query_planner=FakeQueryPlanner([self.abstract_question]),
            )
        )

        response = self.post_answer(client)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "RETRIEVAL_UNAVAILABLE")
        self.assertNotIn("local path", response.text)

    def test_search_stays_a_single_query_endpoint(self):
        retriever = RecordingRetriever({self.question: [stub_result("original")]})
        planner = FakeQueryPlanner([self.abstract_question])
        client = TestClient(
            create_app(
                retriever=retriever,
                answer_client=FakeAnswerClient(),
                query_planner=planner,
            )
        )

        response = client.post("/search", json={"query": self.question})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(planner.calls, [])
        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(
            [item["chunkId"] for item in response.json()["results"]],
            ["original"],
        )


class BrokenScorer:
    """A Cross-Encoder that fails at inference time."""

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        raise RuntimeError("model inference failed")


class RecordingReranker:
    """Order the pool by a scripted chunk order and record every request."""

    def __init__(self, order=()):
        self.calls = []
        self.order = list(order)

    def readiness(self):
        return "ok"

    def rerank(self, question, candidates, *, limit):
        self.calls.append(
            {"question": question, "candidates": list(candidates), "limit": limit}
        )
        positions = {chunk_id: index for index, chunk_id in enumerate(self.order)}
        ranked = sorted(
            candidates,
            key=lambda item: positions.get(item.chunk_id, len(positions)),
        )
        return ranked[:limit]


def missing_model(_model_name):
    raise OSError("weights are not available locally")


class RerankerAnswerApiTests(unittest.TestCase):
    question = "红石中继器有什么作用？"
    abstract_question = "红石信号的传输机制是什么"

    def post_answer(self, client, question=None):
        return client.post(
            "/answers",
            json={"question": self.question if question is None else question},
        )

    def sources_of(self, response):
        return next(
            data for event, data in parse_sse(response.text) if event == "sources"
        )

    def test_answers_recalls_a_candidate_pool_then_answers_in_reranked_order(self):
        pool = [stub_result("a"), stub_result("b"), stub_result("c")]
        retriever = RecordingRetriever({self.question: pool})
        reranker = RecordingReranker(["c"])

        response = self.post_answer(
            TestClient(
                create_app(
                    retriever=retriever,
                    answer_client=FakeAnswerClient(),
                    reranker=reranker,
                )
            )
        )

        self.assertEqual(
            [call["limit"] for call in retriever.calls],
            [RetrievalSettings().candidate_limit],
        )
        self.assertEqual(reranker.calls[0]["question"], self.question)
        self.assertEqual(
            [item.chunk_id for item in reranker.calls[0]["candidates"]],
            ["a", "b", "c"],
        )
        self.assertEqual(reranker.calls[0]["limit"], RetrievalSettings().evidence_limit)
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["c", "a", "b"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_a_broken_cross_encoder_still_answers_from_the_rrf_order(self):
        retriever = RecordingRetriever(
            {self.question: [stub_result("a"), stub_result("b")]}
        )

        with self.assertLogs("rag_reranker", level="WARNING"):
            response = self.post_answer(
                TestClient(
                    create_app(
                        retriever=retriever,
                        answer_client=FakeAnswerClient(),
                        reranker=CrossEncoderReranker(scorer=BrokenScorer()),
                    )
                )
            )

        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["a", "b"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_the_reranker_scores_the_user_question_not_the_step_back_question(self):
        retriever = RecordingRetriever(
            {
                self.question: [stub_result("original")],
                self.abstract_question: [stub_result("abstract")],
            }
        )
        reranker = RecordingReranker()

        self.post_answer(
            TestClient(
                create_app(
                    retriever=retriever,
                    answer_client=FakeAnswerClient(),
                    query_planner=FakeQueryPlanner([self.abstract_question]),
                    reranker=reranker,
                )
            )
        )

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            [self.question, self.abstract_question],
        )
        self.assertEqual(reranker.calls[0]["question"], self.question)
        self.assertEqual(
            [item.chunk_id for item in reranker.calls[0]["candidates"]],
            ["abstract", "original"],
        )

    def test_ready_reports_the_disabled_reranker(self):
        client = TestClient(
            create_app(
                retriever=RecordingRetriever(),
                answer_client=FakeAnswerClient(),
                reranker=NoopReranker(),
            )
        )

        body = client.get("/ready")

        self.assertEqual(body.status_code, 200)
        self.assertEqual(body.json()["reranker"], "disabled")

    def test_ready_reports_a_cross_encoder_that_could_not_load(self):
        with self.assertLogs("rag_reranker", level="WARNING"):
            reranker = CrossEncoderReranker(loader=missing_model)
        client = TestClient(
            create_app(
                retriever=RecordingRetriever(),
                answer_client=FakeAnswerClient(),
                reranker=reranker,
            )
        )

        body = client.get("/ready")

        self.assertEqual(body.status_code, 503)
        self.assertEqual(body.json()["reranker"], "unavailable")

    def test_the_default_startup_path_honours_an_injected_reranker(self):
        reranker = RecordingReranker()
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            settings = RetrievalSettings(bm25_path=bm25_path)
            collection = SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=settings.vector_size))
                ),
                status="green",
            )
            qdrant = SimpleNamespace(
                get_collection=lambda name: collection,
                close=lambda: None,
            )
            with patch("rag_api.RetrievalSettings.from_env", return_value=settings), patch(
                "rag_api._build_default_retriever",
                return_value=(
                    RecordingRetriever(),
                    SimpleNamespace(close=lambda: None),
                    qdrant,
                ),
            ), patch(
                "rag_api.DeepSeekSettings.from_env",
                side_effect=AnswerConfigurationError("missing"),
            ):
                with TestClient(create_app(reranker=reranker)) as client:
                    body = client.get("/ready")

        self.assertEqual(body.json()["reranker"], "ok")


class ScriptedAssessor:
    """Return the scripted assessments in order; the last one repeats."""

    def __init__(self, assessments=(), *, error=None):
        self.calls = []
        self.assessments = list(assessments)
        self.error = error

    async def assess(self, question, evidence):
        self.calls.append({"question": question, "evidence": list(evidence)})
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.assessments) - 1)
        return self.assessments[index]

    def seen_chunk_ids(self, index):
        return [item.chunk_id for item in self.calls[index]["evidence"]]


def sufficient():
    return EvidenceAssessment(sufficient=True, reason="证据完整")


def insufficient(rewritten_query=None):
    return EvidenceAssessment(
        sufficient=False,
        missing_aspects=("延迟机制",),
        rewritten_query=rewritten_query,
        reason="缺少延迟机制的证据",
    )


class CorrectiveAnswerApiTests(unittest.TestCase):
    question = "红石中继器有什么作用？"
    rewritten = "红石中继器 延迟信号 增强信号"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.bm25_path = Path(directory.name) / "bm25.db"
        self.bm25_path.touch()

    def settings(self, *, corrective=CORRECTIVE_ENABLED, max_retrieval_rounds=2):
        return RetrievalSettings(
            bm25_path=self.bm25_path,
            corrective=corrective,
            max_retrieval_rounds=max_retrieval_rounds,
            candidate_limit=4,
            evidence_limit=2,
        )

    def build_client(
        self,
        retriever,
        assessor=None,
        answer_client=None,
        *,
        corrective=CORRECTIVE_ENABLED,
        max_retrieval_rounds=2,
    ):
        app = create_app(
            retriever=retriever,
            answer_client=answer_client or FakeAnswerClient(),
            corrective_assessor=assessor,
        )
        app.state.retrieval_settings = self.settings(
            corrective=corrective,
            max_retrieval_rounds=max_retrieval_rounds,
        )
        app.state.qdrant = SimpleNamespace(
            get_collection=lambda name: SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=RetrievalSettings().vector_size)
                    )
                ),
                status="green",
            )
        )
        return TestClient(app)

    def post_answer(self, client, question=None):
        return client.post(
            "/answers",
            json={"question": self.question if question is None else question},
        )

    def sources_of(self, response):
        return next(
            data for event, data in parse_sse(response.text) if event == "sources"
        )

    def test_the_switch_off_keeps_the_single_round_baseline(self):
        retriever = RecordingRetriever(
            {self.question: [stub_result("a"), stub_result("b")]}
        )
        assessor = ScriptedAssessor([insufficient(self.rewritten)])

        response = self.post_answer(
            self.build_client(retriever, assessor, corrective="none")
        )

        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(assessor.calls, [])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["a", "b"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_a_sufficient_first_round_answers_from_that_round_only(self):
        retriever = RecordingRetriever(
            {self.question: [stub_result("a"), stub_result("b")]}
        )
        assessor = ScriptedAssessor([sufficient()])
        answer_client = FakeAnswerClient()

        response = self.post_answer(
            self.build_client(retriever, assessor, answer_client)
        )

        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(len(assessor.calls), 1)
        self.assertEqual(assessor.seen_chunk_ids(0), ["a", "b"])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["a", "b"],
        )
        self.assertEqual(
            [item.chunk_id for item in answer_client.calls[0]["evidence"]],
            ["a", "b"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_a_corrected_round_announces_the_merged_evidence_only(self):
        retriever = RecordingRetriever(
            {
                self.question: [stub_result("a"), stub_result("b")],
                self.rewritten: [stub_result("c")],
            }
        )
        assessor = ScriptedAssessor([insufficient(self.rewritten), sufficient()])
        answer_client = FakeAnswerClient()

        response = self.post_answer(
            self.build_client(retriever, assessor, answer_client)
        )

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            [self.question, self.rewritten],
        )
        self.assertEqual(len(assessor.calls), 2)
        self.assertEqual(assessor.seen_chunk_ids(1), ["a", "c"])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["a", "c"],
        )
        self.assertEqual(
            [item.chunk_id for item in answer_client.calls[0]["evidence"]],
            ["a", "c"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_a_second_insufficient_check_refuses_and_never_generates(self):
        retriever = RecordingRetriever(
            {
                self.question: [stub_result("a")],
                self.rewritten: [stub_result("b")],
            }
        )
        # The assessor keeps offering a rewrite; the round budget is what stops it.
        assessor = ScriptedAssessor([insufficient(self.rewritten)])
        answer_client = FakeAnswerClient()

        response = self.post_answer(
            self.build_client(retriever, assessor, answer_client)
        )

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            [self.question, self.rewritten],
        )
        self.assertEqual(len(assessor.calls), 2)
        self.assertEqual(self.sources_of(response), {"items": []})
        self.assertEqual(
            parse_sse(response.text)[-2:],
            [
                ("delta", {"text": INSUFFICIENT_EVIDENCE_MESSAGE}),
                ("done", {"status": "insufficientEvidence"}),
            ],
        )
        self.assertEqual(answer_client.calls, [])

    def test_an_empty_first_round_can_still_be_corrected(self):
        retriever = RecordingRetriever(
            {self.question: [], self.rewritten: [stub_result("c")]}
        )
        assessor = ScriptedAssessor([insufficient(self.rewritten), sufficient()])

        response = self.post_answer(self.build_client(retriever, assessor))

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            [self.question, self.rewritten],
        )
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["c"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_no_evidence_is_never_passed_to_the_generator(self):
        retriever = RecordingRetriever({})
        assessor = ScriptedAssessor([insufficient(self.rewritten)])
        answer_client = FakeAnswerClient()

        response = self.post_answer(
            self.build_client(retriever, assessor, answer_client)
        )

        self.assertEqual(self.sources_of(response), {"items": []})
        self.assertEqual(parse_sse(response.text)[-1][1], {"status": "insufficientEvidence"})
        self.assertEqual(answer_client.calls, [])

    def test_a_failing_assessor_keeps_the_single_round_answer(self):
        retriever = RecordingRetriever(
            {self.question: [stub_result("a"), stub_result("b")]}
        )
        assessor = ScriptedAssessor(error=RuntimeError("assessment endpoint down"))

        with self.assertLogs("rag_correction", level="WARNING") as logs:
            response = self.post_answer(self.build_client(retriever, assessor))

        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["a", "b"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))
        self.assertIn("without a sufficiency check", "\n".join(logs.output))
        self.assertNotIn("assessment endpoint down", response.text)

    def test_corrective_without_an_assessor_stays_single_round(self):
        retriever = RecordingRetriever(
            {self.question: [stub_result("a"), stub_result("b")]}
        )

        with self.assertLogs("rag_api", level="WARNING") as logs:
            response = self.post_answer(
                self.build_client(retriever, assessor=None)
            )

        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(
            [item["chunkId"] for item in self.sources_of(response)["items"]],
            ["a", "b"],
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))
        self.assertIn("without an assessor", "\n".join(logs.output))

    def test_a_single_round_budget_answers_without_correcting(self):
        retriever = RecordingRetriever(
            {
                self.question: [stub_result("a")],
                self.rewritten: [stub_result("b")],
            }
        )
        assessor = ScriptedAssessor([insufficient(self.rewritten)])

        response = self.post_answer(
            self.build_client(retriever, assessor, max_retrieval_rounds=1)
        )

        self.assertEqual([call["query"] for call in retriever.calls], [self.question])
        self.assertEqual(self.sources_of(response), {"items": []})
        self.assertEqual(parse_sse(response.text)[-1][1], {"status": "insufficientEvidence"})

    def test_the_default_startup_path_builds_the_assessor_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            settings = RetrievalSettings(
                bm25_path=bm25_path, corrective=CORRECTIVE_ENABLED
            )
            collection = SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=settings.vector_size)
                    )
                ),
                status="green",
            )
            qdrant = SimpleNamespace(
                get_collection=lambda name: collection,
                close=lambda: None,
            )
            with patch("rag_api.RetrievalSettings.from_env", return_value=settings), patch(
                "rag_api._build_default_retriever",
                return_value=(
                    RecordingRetriever(),
                    SimpleNamespace(close=lambda: None),
                    qdrant,
                ),
            ), patch(
                "rag_api.DeepSeekSettings.from_env",
                return_value=DeepSeekSettings(api_key="test-key"),
            ):
                app = create_app()
                with TestClient(app) as client:
                    self.assertEqual(client.get("/ready").status_code, 200)

        self.assertIsInstance(app.state.corrective_assessor, DeepSeekEvidenceAssessor)

    def test_the_default_startup_path_builds_no_assessor_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            settings = RetrievalSettings(bm25_path=bm25_path)
            collection = SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=settings.vector_size)
                    )
                ),
                status="green",
            )
            qdrant = SimpleNamespace(
                get_collection=lambda name: collection,
                close=lambda: None,
            )
            with patch("rag_api.RetrievalSettings.from_env", return_value=settings), patch(
                "rag_api._build_default_retriever",
                return_value=(
                    RecordingRetriever(),
                    SimpleNamespace(close=lambda: None),
                    qdrant,
                ),
            ), patch(
                "rag_api.DeepSeekSettings.from_env",
                return_value=DeepSeekSettings(api_key="test-key"),
            ):
                app = create_app()
                with TestClient(app):
                    pass

        self.assertIsNone(app.state.corrective_assessor)


class ScriptedAnswerClient:
    """Answer with the scripted drafts in order; the last one repeats."""

    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls = []

    def _next(self, question, evidence, feedback):
        self.calls.append(
            {
                "question": question,
                "evidence": list(evidence),
                "feedback": feedback,
            }
        )
        return self.drafts[min(len(self.calls) - 1, len(self.drafts) - 1)]

    async def collect_answer(self, question, evidence, *, feedback=""):
        return self._next(question, evidence, feedback)

    async def stream_answer(self, question, evidence):
        yield self._next(question, evidence, "")

    def seen_chunk_ids(self, index):
        return [item.chunk_id for item in self.calls[index]["evidence"]]


class ScriptedModelVerifier:
    """Return the scripted verdicts in order; the last one repeats."""

    def __init__(self, verdicts=(), *, error=None):
        self.calls = []
        self.verdicts = list(verdicts)
        self.error = error

    async def verify(self, question, answer, evidence):
        self.calls.append(
            {"question": question, "answer": answer, "evidence": list(evidence)}
        )
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.verdicts) - 1)
        return self.verdicts[index]

    def seen_answers(self):
        return [call["answer"] for call in self.calls]


def claim(text, *, citation_ids=(1,), supported=True, reason=""):
    return ClaimCheck(
        claim=text,
        citation_ids=tuple(citation_ids),
        supported=supported,
        reason=reason,
    )


def verdict(*claims, useful=True, issues=()):
    return ModelVerdict(claims=tuple(claims), useful=useful, issues=tuple(issues))


class VerifiedAnswerApiTests(unittest.TestCase):
    question = "红石中继器有什么作用？"
    draft = "红石中继器可以延迟红石信号。[1]"
    rewritten = "红石中继器可以延迟并增强红石信号。[1]"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.bm25_path = Path(directory.name) / "bm25.db"
        self.bm25_path.touch()

    def settings(self, *, verification=VERIFICATION_ENABLED, max_answer_attempts=2):
        return RetrievalSettings(
            bm25_path=self.bm25_path,
            answer_verification=verification,
            max_answer_attempts=max_answer_attempts,
        )

    def build_client(
        self,
        answer_client,
        verifier=None,
        *,
        retrieval=None,
        verification=VERIFICATION_ENABLED,
        max_answer_attempts=2,
    ):
        app = create_app(
            retriever=retrieval or RecordingRetriever({self.question: [stub_result("a")]}),
            answer_client=answer_client,
            answer_verifier=verifier,
        )
        app.state.retrieval_settings = self.settings(
            verification=verification,
            max_answer_attempts=max_answer_attempts,
        )
        app.state.qdrant = SimpleNamespace(
            get_collection=lambda name: SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=RetrievalSettings().vector_size)
                    )
                ),
                status="green",
            )
        )
        return TestClient(app)

    def post_answer(self, client):
        return client.post("/answers", json={"question": self.question})

    def text_of(self, response):
        return "".join(
            data["text"]
            for event, data in parse_sse(response.text)
            if event == "delta"
        )

    def test_the_switch_off_streams_the_baseline_answer(self):
        answer_client = ScriptedAnswerClient([self.draft])
        verifier = ScriptedModelVerifier([verdict(claim("延迟信号"))])

        response = self.post_answer(
            self.build_client(
                answer_client, verifier, verification="none"
            )
        )

        self.assertEqual(verifier.calls, [])
        self.assertEqual(self.text_of(response), self.draft)
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))

    def test_a_verified_draft_is_sent_with_a_verified_status(self):
        answer_client = ScriptedAnswerClient([self.draft])
        verifier = ScriptedModelVerifier([verdict(claim("红石中继器可以延迟红石信号"))])

        response = self.post_answer(self.build_client(answer_client, verifier))
        events = parse_sse(response.text)

        self.assertEqual(len(answer_client.calls), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(verifier.calls[0]["answer"], self.draft)
        self.assertEqual(events[1][0], "sources")
        self.assertEqual(self.text_of(response), self.draft)
        self.assertEqual(
            events[-1],
            ("done", {"status": "answered", "verification": VERIFICATION_PASSED}),
        )

    def test_a_rejected_draft_never_reaches_the_browser(self):
        answer_client = ScriptedAnswerClient([self.draft, self.rewritten])
        verifier = ScriptedModelVerifier(
            [
                verdict(
                    claim("红石中继器可以延迟红石信号", supported=False, reason="证据[1]只说延迟，未提增强"),
                    issues=("第一条声明与证据不符",),
                ),
                verdict(claim("红石中继器可以延迟并增强红石信号")),
            ]
        )

        response = self.post_answer(self.build_client(answer_client, verifier))

        self.assertEqual(len(answer_client.calls), 2)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(verifier.seen_answers(), [self.draft, self.rewritten])
        self.assertEqual(self.text_of(response), self.rewritten)
        self.assertNotIn(self.draft, response.text)
        # The rewrite is asked for with the verifier's own findings, from the same evidence.
        self.assertIn("第一条声明与证据不符", answer_client.calls[1]["feedback"])
        self.assertEqual(answer_client.seen_chunk_ids(0), ["a"])
        self.assertEqual(answer_client.seen_chunk_ids(1), ["a"])
        self.assertEqual(
            parse_sse(response.text)[-1],
            ("done", {"status": "answered", "verification": VERIFICATION_PASSED}),
        )

    def test_a_second_failure_becomes_an_explicit_insufficient_status(self):
        answer_client = ScriptedAnswerClient([self.draft, self.rewritten])
        verifier = ScriptedModelVerifier(
            [
                verdict(claim("红石中继器可以延迟红石信号", supported=False, reason="不受支持")),
                verdict(claim("红石中继器可以延迟并增强红石信号", supported=False, reason="仍不受支持")),
            ]
        )

        response = self.post_answer(self.build_client(answer_client, verifier))
        events = parse_sse(response.text)

        self.assertEqual(len(answer_client.calls), 2)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(
            events[-2:],
            [
                ("delta", {"text": INSUFFICIENT_EVIDENCE_MESSAGE}),
                (
                    "done",
                    {
                        "status": "insufficientEvidence",
                        "verification": VERIFICATION_UNSUPPORTED,
                    },
                ),
            ],
        )
        self.assertNotIn(self.draft, response.text)
        self.assertNotIn(self.rewritten, response.text)

    def test_an_invented_citation_is_stopped_before_the_verifier_is_called(self):
        answer_client = ScriptedAnswerClient(["红石中继器可以延迟红石信号。[9]"])
        verifier = ScriptedModelVerifier([verdict(claim("延迟信号"))])

        response = self.post_answer(self.build_client(answer_client, verifier))

        self.assertEqual(verifier.calls, [])
        self.assertEqual(len(answer_client.calls), 2)
        self.assertEqual(parse_sse(response.text)[-1][1]["verification"], VERIFICATION_UNSUPPORTED)
        self.assertEqual(self.text_of(response), INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertNotIn("[9]", response.text)

    def test_a_sub_question_left_unanswered_is_rewritten_once(self):
        answer_client = ScriptedAnswerClient([self.draft, self.rewritten])
        verifier = ScriptedModelVerifier(
            [
                verdict(
                    claim("红石中继器可以延迟红石信号"),
                    useful=False,
                    issues=("没有回答合成方式",),
                ),
                verdict(claim("红石中继器可以延迟并增强红石信号")),
            ]
        )

        response = self.post_answer(self.build_client(answer_client, verifier))

        self.assertEqual(self.text_of(response), self.rewritten)
        self.assertEqual(len(verifier.calls), 2)
        self.assertIn("没有回答合成方式", answer_client.calls[1]["feedback"])

    def test_a_single_attempt_budget_verifies_without_rewriting(self):
        answer_client = ScriptedAnswerClient([self.draft, self.rewritten])
        verifier = ScriptedModelVerifier(
            [verdict(claim("红石中继器可以延迟红石信号", supported=False))]
        )

        response = self.post_answer(
            self.build_client(answer_client, verifier, max_answer_attempts=1)
        )

        self.assertEqual(len(answer_client.calls), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(self.text_of(response), INSUFFICIENT_EVIDENCE_MESSAGE)

    def test_a_broken_verifier_answers_with_an_unavailable_status(self):
        answer_client = ScriptedAnswerClient([self.draft])
        verifier = ScriptedModelVerifier(error=RuntimeError("verification endpoint down"))

        with self.assertLogs("rag_verification", level="WARNING") as logs:
            response = self.post_answer(self.build_client(answer_client, verifier))

        self.assertEqual(self.text_of(response), self.draft)
        self.assertEqual(
            parse_sse(response.text)[-1],
            ("done", {"status": "answered", "verification": VERIFICATION_UNAVAILABLE}),
        )
        self.assertEqual(len(answer_client.calls), 1)
        self.assertIn("without a claim check", "\n".join(logs.output))
        self.assertNotIn("verification endpoint down", response.text)

    def test_verification_needs_an_answer_client_that_can_collect_a_draft(self):
        # FakeAnswerClient predates the verification layer and only streams.
        answer_client = FakeAnswerClient()
        verifier = ScriptedModelVerifier([verdict(claim("延迟信号"))])

        with self.assertLogs("rag_api", level="WARNING") as logs:
            response = self.post_answer(self.build_client(answer_client, verifier))

        self.assertEqual(verifier.calls, [])
        self.assertEqual(
            [data["text"] for event, data in parse_sse(response.text) if event == "delta"],
            answer_client.chunks,
        )
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))
        self.assertIn("collect a draft", "\n".join(logs.output))

    def test_verification_without_a_verifier_falls_back_to_the_draft(self):
        answer_client = ScriptedAnswerClient([self.draft])

        with self.assertLogs("rag_api", level="WARNING") as logs:
            response = self.post_answer(self.build_client(answer_client, verifier=None))

        self.assertEqual(self.text_of(response), self.draft)
        self.assertEqual(parse_sse(response.text)[-1], ("done", {"status": "answered"}))
        self.assertIn("without a verifier", "\n".join(logs.output))

    def test_the_default_startup_path_builds_the_verifier_when_enabled(self):
        app = self.startup_app(verification=VERIFICATION_ENABLED)

        self.assertIsInstance(app.state.answer_verifier, DeepSeekAnswerVerifier)

    def test_the_default_startup_path_builds_no_verifier_by_default(self):
        app = self.startup_app(verification="none")

        self.assertIsNone(app.state.answer_verifier)

    def startup_app(self, *, verification):
        with tempfile.TemporaryDirectory() as directory:
            bm25_path = Path(directory) / "bm25.db"
            bm25_path.touch()
            settings = RetrievalSettings(
                bm25_path=bm25_path, answer_verification=verification
            )
            collection = SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=settings.vector_size)
                    )
                ),
                status="green",
            )
            qdrant = SimpleNamespace(
                get_collection=lambda name: collection,
                close=lambda: None,
            )
            with patch("rag_api.RetrievalSettings.from_env", return_value=settings), patch(
                "rag_api._build_default_retriever",
                return_value=(
                    RecordingRetriever(),
                    SimpleNamespace(close=lambda: None),
                    qdrant,
                ),
            ), patch(
                "rag_api.DeepSeekSettings.from_env",
                return_value=DeepSeekSettings(api_key="test-key"),
            ):
                app = create_app()
                with TestClient(app):
                    pass
        return app


if __name__ == "__main__":
    unittest.main()
