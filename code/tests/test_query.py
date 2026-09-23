import json
import unittest
from types import SimpleNamespace

import httpx

from rag_query import (
    DeepSeekStepBackPlanner,
    OriginalQueryPlanner,
    QueryPlan,
    build_query_plan,
    build_query_planner,
)
from rag_settings import (
    MAX_RETRIEVAL_QUERIES,
    ORIGINAL_QUERY_STRATEGY,
    STEP_BACK_QUERY_STRATEGY,
)


CHAT_SETTINGS = SimpleNamespace(
    api_key="test-secret",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-pro",
)


def _chat_response(step_back_question: str) -> httpx.Response:
    content = json.dumps(
        {"step_back_question": step_back_question},
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


class QueryPlanTests(unittest.TestCase):
    def test_keeps_the_normalized_question_first_and_drops_duplicates(self):
        plan = build_query_plan(
            "  1.21.5 加入了什么？  ",
            ["版本更新包含哪些内容的背景原理是什么", "  1.21.5 加入了什么？ ", "  "],
            strategy=STEP_BACK_QUERY_STRATEGY,
        )

        self.assertEqual(plan.original, "1.21.5 加入了什么？")
        self.assertEqual(
            plan.retrieval_queries,
            ("1.21.5 加入了什么？", "版本更新包含哪些内容的背景原理是什么"),
        )
        self.assertEqual(plan.strategy, STEP_BACK_QUERY_STRATEGY)

    def test_never_exceeds_the_hard_query_limit(self):
        plan = build_query_plan(
            "红石中继器怎么用？",
            ["第一个候选", "第二个候选", "第三个候选"],
            strategy=STEP_BACK_QUERY_STRATEGY,
        )

        self.assertEqual(len(plan.retrieval_queries), MAX_RETRIEVAL_QUERIES)
        self.assertEqual(plan.retrieval_queries[0], "红石中继器怎么用？")

    def test_max_queries_of_one_keeps_only_the_original_question(self):
        plan = build_query_plan("红石中继器怎么用？", ["候选"], max_queries=1)

        self.assertEqual(plan.retrieval_queries, ("红石中继器怎么用？",))

    def test_blank_question_plans_no_queries(self):
        plan = build_query_plan("   ", ["候选"])

        self.assertEqual(plan.original, "")
        self.assertEqual(plan.retrieval_queries, ())

    def test_rejects_a_max_query_budget_above_the_hard_limit(self):
        with self.assertRaisesRegex(ValueError, "max_queries"):
            build_query_plan("红石中继器怎么用？", ["候选"], max_queries=3)

    def test_rejects_plans_that_break_the_public_contract(self):
        with self.assertRaisesRegex(ValueError, "original question"):
            QueryPlan(
                original="红石中继器怎么用？",
                retrieval_queries=("其它问题",),
                strategy=STEP_BACK_QUERY_STRATEGY,
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            QueryPlan(
                original="红石中继器怎么用？",
                retrieval_queries=("红石中继器怎么用？", "一", "二"),
                strategy=STEP_BACK_QUERY_STRATEGY,
            )
        with self.assertRaisesRegex(ValueError, "drop the original question"):
            QueryPlan(
                original="红石中继器怎么用？",
                retrieval_queries=(),
                strategy=STEP_BACK_QUERY_STRATEGY,
            )


class OriginalQueryPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_plans_one_retrieval_query_with_the_users_wording(self):
        plan = await OriginalQueryPlanner().plan("  1.21.5 加入了什么？  ")

        self.assertEqual(plan.original, "1.21.5 加入了什么？")
        self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))
        self.assertEqual(plan.strategy, ORIGINAL_QUERY_STRATEGY)

    async def test_blank_question_plans_no_queries(self):
        plan = await OriginalQueryPlanner().plan("   ")

        self.assertEqual(plan.retrieval_queries, ())


class StepBackPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def _plan_with(self, handler, question="1.21.5 加入了什么？", **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            planner = DeepSeekStepBackPlanner(
                settings=CHAT_SETTINGS,
                http_client=http_client,
                **kwargs,
            )
            return await planner.plan(question)

    async def test_exact_name_question_keeps_the_original_wording_first(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers["authorization"]
            captured["body"] = json.loads(request.content)
            return _chat_response("版本更新的背景原理与定义是什么")

        plan = await self._plan_with(handler)

        self.assertEqual(plan.strategy, STEP_BACK_QUERY_STRATEGY)
        self.assertEqual(plan.retrieval_queries[0], "1.21.5 加入了什么？")
        self.assertEqual(
            plan.retrieval_queries,
            ("1.21.5 加入了什么？", "版本更新的背景原理与定义是什么"),
        )
        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer test-secret")
        body = captured["body"]
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertFalse(body["stream"])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["temperature"], 0.0)
        self.assertIn("step_back_question", body["messages"][0]["content"])
        self.assertIn("1.21.5 加入了什么？", body["messages"][1]["content"])
        self.assertNotIn("test-secret", json.dumps(body))

    async def test_background_question_is_requested_for_a_mechanism_question(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response("红石信号的传输机制是什么")

        plan = await self._plan_with(
            handler,
            question="红石中继器如何延迟信号？",
        )

        self.assertEqual(
            plan.retrieval_queries,
            ("红石中继器如何延迟信号？", "红石信号的传输机制是什么"),
        )

    async def test_declined_abstraction_keeps_only_the_original_question(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response("   ")

        plan = await self._plan_with(handler)

        self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))

    async def test_invalid_model_output_falls_back_to_the_original_question(self):
        cases = {
            "invalid json": "not json at all",
            "json array": json.dumps(["问题"], ensure_ascii=False),
            "missing field": json.dumps({"question": "问题"}, ensure_ascii=False),
            "non string field": json.dumps({"step_back_question": 7}),
            "over long question": json.dumps(
                {"step_back_question": "背景" * 200},
                ensure_ascii=False,
            ),
        }

        for label, content in cases.items():
            with self.subTest(label=label):
                response = httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": content}}]},
                )

                async def handler(
                    request: httpx.Request, response: httpx.Response = response
                ) -> httpx.Response:
                    return response

                plan = await self._plan_with(handler)

                self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))
                self.assertEqual(plan.strategy, STEP_BACK_QUERY_STRATEGY)

    async def test_duplicate_abstraction_is_dropped(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response("  1.21.5 加入了什么？ ")

        plan = await self._plan_with(handler)

        self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))

    async def test_malformed_response_body_falls_back_to_the_original_question(self):
        cases = {
            "no choices": {"choices": []},
            "choices not a list": {"choices": "invalid"},
            "no message": {"choices": [{}]},
            "content not a string": {"choices": [{"message": {"content": 7}}]},
        }

        for label, payload in cases.items():
            with self.subTest(label=label):
                response = httpx.Response(200, json=payload)

                async def handler(
                    request: httpx.Request, response: httpx.Response = response
                ) -> httpx.Response:
                    return response

                plan = await self._plan_with(handler)

                self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))

    async def test_transport_failures_fall_back_to_the_original_question(self):
        async def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("upstream host details", request=request)

        async def http_error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="upstream unavailable")

        async def invalid_json_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"{not json",
            )

        async def unexpected_handler(request: httpx.Request) -> httpx.Response:
            raise RuntimeError("planner bug that must not break retrieval")

        for label, handler in {
            "timeout": timeout_handler,
            "http error": http_error_handler,
            "invalid body": invalid_json_handler,
            "unexpected error": unexpected_handler,
        }.items():
            with self.subTest(label=label):
                plan = await self._plan_with(handler)

                self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))
                self.assertEqual(plan.strategy, STEP_BACK_QUERY_STRATEGY)

    async def test_one_query_budget_skips_the_model_call(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("the model must not be called")

        plan = await self._plan_with(handler, max_queries=1)

        self.assertEqual(plan.retrieval_queries, ("1.21.5 加入了什么？",))

    async def test_blank_question_skips_the_model_call(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("the model must not be called")

        plan = await self._plan_with(handler, question="   ")

        self.assertEqual(plan.retrieval_queries, ())

    def test_rejects_invalid_configuration(self):
        unused_client = SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "timeout"):
            DeepSeekStepBackPlanner(
                settings=CHAT_SETTINGS,
                http_client=unused_client,
                timeout=0,
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            DeepSeekStepBackPlanner(
                settings=CHAT_SETTINGS,
                http_client=unused_client,
                temperature=3,
            )
        with self.assertRaisesRegex(ValueError, "max_queries"):
            DeepSeekStepBackPlanner(
                settings=CHAT_SETTINGS,
                http_client=unused_client,
                max_queries=MAX_RETRIEVAL_QUERIES + 1,
            )


class BuildQueryPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_strategy_never_calls_a_model(self):
        planner = build_query_planner(ORIGINAL_QUERY_STRATEGY)

        plan = await planner.plan("红石中继器怎么用？")

        self.assertIsInstance(planner, OriginalQueryPlanner)
        self.assertEqual(plan.retrieval_queries, ("红石中继器怎么用？",))
        self.assertEqual(plan.strategy, ORIGINAL_QUERY_STRATEGY)

    async def test_step_back_without_a_chat_client_degrades_to_the_original_question(self):
        planner = build_query_planner(STEP_BACK_QUERY_STRATEGY)

        plan = await planner.plan("红石中继器怎么用？")

        self.assertIsInstance(planner, OriginalQueryPlanner)
        self.assertEqual(plan.retrieval_queries, ("红石中继器怎么用？",))

    async def test_step_back_with_a_chat_client_uses_the_model(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response("红石信号的传输机制是什么")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            planner = build_query_planner(
                STEP_BACK_QUERY_STRATEGY,
                settings=CHAT_SETTINGS,
                http_client=http_client,
                timeout=5.0,
            )
            plan = await planner.plan("红石中继器怎么用？")

        self.assertIsInstance(planner, DeepSeekStepBackPlanner)
        self.assertEqual(plan.strategy, STEP_BACK_QUERY_STRATEGY)
        self.assertEqual(
            plan.retrieval_queries,
            ("红石中继器怎么用？", "红石信号的传输机制是什么"),
        )

    def test_rejects_an_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported query strategy"):
            build_query_planner("multi_query")


if __name__ == "__main__":
    unittest.main()
