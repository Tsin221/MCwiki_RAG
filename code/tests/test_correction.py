import json
import unittest

import httpx

from rag_answer import AnswerEvidence
from rag_correction import (
    CORRECTIVE_DISABLED,
    CORRECTIVE_ENABLED,
    CorrectiveCoordinator,
    DeepSeekEvidenceAssessor,
    EvidenceAssessment,
    EvidenceAssessmentError,
    build_evidence_assessor,
    parse_assessment,
)
from rag_retrieval.hybrid import HybridResult
from rag_settings import (
    CORRECTIVE_DISABLED as SETTINGS_CORRECTIVE_DISABLED,
    CORRECTIVE_ENABLED as SETTINGS_CORRECTIVE_ENABLED,
    MAX_RETRIEVAL_ROUNDS,
    RetrievalSettings,
)


QUESTION = "红石中继器有什么作用？"
REWRITTEN = "红石中继器 延迟信号 增强信号"


def candidate(chunk_id, *, text=None):
    return HybridResult(
        chunk_id=chunk_id,
        title=f"标题 {chunk_id}",
        text=text or f"{chunk_id} 的正文",
        source=f"https://example.test/{chunk_id}",
        score=0.03,
        bm25_rank=1,
        semantic_rank=1,
    )


def evidence_item(item_id, chunk_id, *, text=None):
    normalized = text or f"{chunk_id} 的正文"
    return AnswerEvidence(
        id=item_id,
        chunk_id=chunk_id,
        component_chunk_ids=(chunk_id,),
        title=f"标题 {chunk_id}",
        url=f"https://example.test/{chunk_id}",
        text=normalized,
        excerpt=normalized,
    )


def corrective_settings(**overrides):
    values = {
        "corrective": CORRECTIVE_ENABLED,
        "bm25_limit": 12,
        "semantic_limit": 14,
        "candidate_limit": 4,
        "evidence_limit": 2,
        "max_context_chars": 2_000,
    }
    values.update(overrides)
    return RetrievalSettings(**values)


class ScriptedRetrieval:
    """Recall a scripted pool per query and rerank by a scripted chunk order."""

    def __init__(self, pools=None, *, order=(), recall_error=None):
        self.recalls = []
        self.reranks = []
        self.pools = pools or {}
        self.order = list(order)
        self.recall_error = recall_error

    async def recall(self, question, *, bm25_limit, semantic_limit, limit):
        self.recalls.append(
            {
                "query": question,
                "bm25_limit": bm25_limit,
                "semantic_limit": semantic_limit,
                "limit": limit,
            }
        )
        if self.recall_error is not None:
            raise self.recall_error
        return list(self.pools.get(question, []))[:limit]

    async def rerank(self, question, candidates, *, limit):
        pool = list(candidates)
        self.reranks.append({"question": question, "candidates": pool, "limit": limit})
        positions = {chunk_id: index for index, chunk_id in enumerate(self.order)}
        ranked = sorted(
            enumerate(pool),
            key=lambda entry: (positions.get(entry[1].chunk_id, len(positions)), entry[0]),
        )
        return [item for _, item in ranked[:limit]]

    def recalled_queries(self):
        return [call["query"] for call in self.recalls]

    def reranked_ids(self, index):
        return [item.chunk_id for item in self.reranks[index]["candidates"]]


class ScriptedAssessor:
    """Return the scripted assessments in order; the last one repeats."""

    def __init__(self, assessments, *, error=None):
        self.calls = []
        self.assessments = list(assessments)
        self.error = error

    async def assess(self, question, evidence):
        self.calls.append({"question": question, "evidence": list(evidence)})
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.assessments) - 1)
        return self.assessments[index]


def insufficient(**overrides):
    values = {
        "sufficient": False,
        "missing_aspects": ("延迟机制",),
        "rewritten_query": REWRITTEN,
        "reason": "缺少延迟机制的证据",
    }
    values.update(overrides)
    return EvidenceAssessment(**values)


def sufficient():
    return EvidenceAssessment(sufficient=True, reason="证据完整")


class ParseAssessmentTests(unittest.TestCase):
    def test_reads_the_full_structure(self):
        assessment = parse_assessment(
            json.dumps(
                {
                    "sufficient": False,
                    "missing_aspects": ["延迟机制", " 增强信号 "],
                    "rewritten_query": " 红石中继器 延迟 ",
                    "reason": " 缺少机制 ",
                }
            )
        )

        self.assertFalse(assessment.sufficient)
        self.assertEqual(assessment.missing_aspects, ("延迟机制", "增强信号"))
        self.assertEqual(assessment.rewritten_query, "红石中继器 延迟")
        self.assertEqual(assessment.reason, "缺少机制")

    def test_optional_fields_default_to_an_empty_judgement(self):
        assessment = parse_assessment('{"sufficient": true}')

        self.assertTrue(assessment.sufficient)
        self.assertEqual(assessment.missing_aspects, ())
        self.assertIsNone(assessment.rewritten_query)
        self.assertEqual(assessment.reason, "")

    def test_a_blank_rewritten_query_means_there_is_nothing_to_rewrite(self):
        assessment = parse_assessment(
            '{"sufficient": false, "rewritten_query": "   ", "reason": "缺失"}'
        )

        self.assertIsNone(assessment.rewritten_query)

    def test_the_missing_aspect_list_is_capped(self):
        assessment = parse_assessment(
            json.dumps(
                {
                    "sufficient": False,
                    "missing_aspects": [f"方面{index}" for index in range(20)],
                }
            )
        )

        self.assertEqual(len(assessment.missing_aspects), 8)

    def test_rejects_every_malformed_reply(self):
        for payload in (
            "not json",
            "[1, 2]",
            '{"missing_aspects": []}',
            '{"sufficient": "yes"}',
            '{"sufficient": 1}',
            '{"sufficient": false, "missing_aspects": "延迟机制"}',
            '{"sufficient": false, "missing_aspects": [1]}',
            '{"sufficient": false, "rewritten_query": 42}',
            '{"sufficient": false, "rewritten_query": "%s"}' % ("长" * 300),
            '{"sufficient": false, "reason": ["不是字符串"]}',
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(EvidenceAssessmentError):
                    parse_assessment(payload)

    def test_the_dataclass_refuses_a_blank_or_oversized_rewrite(self):
        with self.assertRaisesRegex(ValueError, "rewritten_query"):
            EvidenceAssessment(sufficient=False, rewritten_query="  ")
        with self.assertRaisesRegex(ValueError, "rewritten_query"):
            EvidenceAssessment(sufficient=False, rewritten_query="长" * 300)


class FakeChatSettings:
    api_key = "test-key"
    base_url = "https://api.example.test"
    model = "deepseek-v4-pro"


def chat_response(*, content=None, status_code=200, text=None):
    request = httpx.Request("POST", "https://api.example.test/chat/completions")
    if content is not None:
        return httpx.Response(
            status_code,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )
    return httpx.Response(status_code, text=text or "", request=request)


class FakeChatHttpClient:
    """Return one prepared response or raise one prepared transport error."""

    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response
        self.error = error

    async def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return self.response


class DeepSeekEvidenceAssessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_a_bounded_structured_request_and_parses_the_reply(self):
        client = FakeChatHttpClient(
            chat_response(
                content='{"sufficient": false, "missing_aspects": ["延迟机制"], '
                '"rewritten_query": "红石中继器 延迟", "reason": "缺少机制"}'
            )
        )
        assessor = DeepSeekEvidenceAssessor(
            settings=FakeChatSettings(),
            http_client=client,
            timeout=7.5,
        )

        assessment = await assessor.assess(
            QUESTION,
            [evidence_item(1, "chunk-a", text="中继器会延迟信号")],
        )

        self.assertFalse(assessment.sufficient)
        self.assertEqual(assessment.rewritten_query, "红石中继器 延迟")
        call = client.calls[0]
        self.assertEqual(call["url"], "https://api.example.test/chat/completions")
        self.assertEqual(call["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(call["timeout"], 7.5)
        self.assertEqual(call["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(call["json"]["stream"], False)
        self.assertEqual(call["json"]["temperature"], 0.0)
        self.assertEqual(call["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(call["json"]["response_format"], {"type": "json_object"})
        messages = call["json"]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn(QUESTION, messages[1]["content"])
        self.assertIn("[1] 标题：标题 chunk-a", messages[1]["content"])
        self.assertIn("中继器会延迟信号", messages[1]["content"])

    async def test_an_empty_first_round_is_still_assessed(self):
        client = FakeChatHttpClient(chat_response(content='{"sufficient": false}'))
        assessor = DeepSeekEvidenceAssessor(
            settings=FakeChatSettings(), http_client=client
        )

        await assessor.assess(QUESTION, [])

        self.assertIn("没有检索到任何证据", client.calls[0]["json"]["messages"][1]["content"])

    async def test_unusable_replies_raise_a_predictable_error(self):
        responses = {
            "invalid json body": chat_response(text="<html>gateway</html>"),
            "invalid assessment json": chat_response(content="当然可以回答"),
            "missing content": httpx.Response(
                200,
                json={"choices": [{"message": {}}]},
                request=httpx.Request("POST", "https://api.example.test"),
            ),
        }

        for label, response in responses.items():
            with self.subTest(case=label):
                assessor = DeepSeekEvidenceAssessor(
                    settings=FakeChatSettings(),
                    http_client=FakeChatHttpClient(response),
                )
                with self.assertRaises(EvidenceAssessmentError):
                    await assessor.assess(QUESTION, [evidence_item(1, "chunk-a")])

    async def test_transport_failures_and_timeouts_raise_the_same_error(self):
        cases = {
            "timeout": FakeChatHttpClient(error=httpx.TimeoutException("too slow")),
            "connect error": FakeChatHttpClient(error=httpx.ConnectError("unreachable")),
            "error status": FakeChatHttpClient(
                chat_response(status_code=500, text="upstream")
            ),
        }

        for label, client in cases.items():
            with self.subTest(case=label):
                assessor = DeepSeekEvidenceAssessor(
                    settings=FakeChatSettings(), http_client=client
                )
                with self.assertRaises(EvidenceAssessmentError):
                    await assessor.assess(QUESTION, [evidence_item(1, "chunk-a")])

    async def test_rejects_invalid_construction_arguments(self):
        client = FakeChatHttpClient(chat_response(content="{}"))

        with self.assertRaisesRegex(ValueError, "timeout"):
            DeepSeekEvidenceAssessor(
                settings=FakeChatSettings(), http_client=client, timeout=0
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            DeepSeekEvidenceAssessor(
                settings=FakeChatSettings(), http_client=client, temperature=3
            )
        with self.assertRaisesRegex(ValueError, "thinking_type"):
            DeepSeekEvidenceAssessor(
                settings=FakeChatSettings(), http_client=client, thinking_type=" "
            )


class BuildEvidenceAssessorTests(unittest.TestCase):
    def test_the_strategy_names_match_the_settings_module(self):
        self.assertEqual(
            (CORRECTIVE_DISABLED, CORRECTIVE_ENABLED),
            (SETTINGS_CORRECTIVE_DISABLED, SETTINGS_CORRECTIVE_ENABLED),
        )

    def test_the_disabled_strategy_builds_no_assessor(self):
        self.assertIsNone(build_evidence_assessor(CORRECTIVE_DISABLED))

    def test_the_enabled_strategy_needs_a_chat_client(self):
        self.assertIsNone(
            build_evidence_assessor(CORRECTIVE_ENABLED, settings=FakeChatSettings())
        )

    def test_the_enabled_strategy_builds_the_model_assessor_with_a_client(self):
        assessor = build_evidence_assessor(
            CORRECTIVE_ENABLED,
            settings=FakeChatSettings(),
            http_client=FakeChatHttpClient(chat_response(content="{}")),
            timeout=9.0,
        )

        self.assertIsInstance(assessor, DeepSeekEvidenceAssessor)

    def test_missing_client_is_logged_and_degrades(self):
        with self.assertLogs("rag_correction", level="WARNING") as logs:
            assessor = build_evidence_assessor(
                CORRECTIVE_ENABLED, settings=FakeChatSettings()
            )

        self.assertIsNone(assessor)
        self.assertIn("single-round retrieval", "\n".join(logs.output))

    def test_rejects_an_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported corrective strategy"):
            build_evidence_assessor("rewrite_everything")


class CorrectiveCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def build(self, retriever, assessor, **overrides):
        return CorrectiveCoordinator(
            retriever=retriever,
            assessor=assessor,
            settings=corrective_settings(**overrides),
        )

    async def test_a_sufficient_first_round_retrieves_exactly_once(self):
        retriever = ScriptedRetrieval(
            {QUESTION: [candidate("a"), candidate("b")]}, order=["a", "b"]
        )
        assessor = ScriptedAssessor([sufficient()])

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION])
        self.assertEqual(len(retriever.reranks), 1)
        self.assertEqual(len(assessor.calls), 1)
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertTrue(result.sufficient)
        self.assertEqual([item.chunk_id for item in result.evidence], ["a", "b"])
        self.assertEqual(
            retriever.recalls[0],
            {"query": QUESTION, "bm25_limit": 12, "semantic_limit": 14, "limit": 4},
        )
        self.assertEqual(retriever.reranks[0]["limit"], 2)
        self.assertEqual(retriever.reranks[0]["question"], QUESTION)

    async def test_a_second_round_merges_reranks_and_rechecks(self):
        retriever = ScriptedRetrieval(
            {
                QUESTION: [candidate("a"), candidate("b")],
                REWRITTEN: [candidate("c"), candidate("b")],
            },
            order=["c"],
        )
        assessor = ScriptedAssessor([insufficient(), sufficient()])

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION, REWRITTEN])
        self.assertEqual(len(retriever.reranks), 2)
        self.assertEqual(len(assessor.calls), 2)
        self.assertEqual(result.retrieval_rounds, 2)
        self.assertTrue(result.sufficient)
        # Both rounds land in one deduplicated pool before the single final rerank,
        # and the chunk both rounds found keeps its fusion advantage.
        self.assertEqual(retriever.reranked_ids(1), ["b", "a", "c"])
        self.assertEqual(retriever.reranks[1]["question"], QUESTION)
        self.assertEqual(retriever.reranks[1]["limit"], 2)
        self.assertEqual([item.chunk_id for item in result.evidence], ["c", "b"])

    async def test_the_second_check_is_final_even_when_the_model_offers_more(self):
        retriever = ScriptedRetrieval(
            {
                QUESTION: [candidate("a")],
                REWRITTEN: [candidate("b")],
            }
        )
        assessor = ScriptedAssessor(
            [insufficient(), insufficient(rewritten_query="再试一次 检索")]
        )

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION, REWRITTEN])
        self.assertEqual(len(assessor.calls), 2)
        self.assertEqual(result.retrieval_rounds, 2)
        self.assertFalse(result.sufficient)
        self.assertEqual([item.chunk_id for item in result.evidence], ["a", "b"])

    async def test_an_insufficient_round_without_a_rewrite_stops_immediately(self):
        retriever = ScriptedRetrieval({QUESTION: [candidate("a")]})
        assessor = ScriptedAssessor(
            [insufficient(rewritten_query=None, missing_aspects=("版本号",))]
        )

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION])
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertFalse(result.sufficient)
        self.assertEqual([item.chunk_id for item in result.evidence], ["a"])

    async def test_a_rewrite_of_the_original_question_is_not_retrieved_again(self):
        retriever = ScriptedRetrieval({QUESTION: [candidate("a")]})
        assessor = ScriptedAssessor([insufficient(rewritten_query=f"  {QUESTION} ")])

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION])
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertFalse(result.sufficient)

    async def test_an_empty_first_round_may_be_rewritten_once(self):
        retriever = ScriptedRetrieval(
            {QUESTION: [], REWRITTEN: [candidate("c")]}, order=["c"]
        )
        assessor = ScriptedAssessor([insufficient(), sufficient()])

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION, REWRITTEN])
        self.assertEqual(result.retrieval_rounds, 2)
        self.assertTrue(result.sufficient)
        self.assertEqual([item.chunk_id for item in result.evidence], ["c"])

    async def test_no_evidence_is_never_reported_as_sufficient(self):
        retriever = ScriptedRetrieval({QUESTION: []})
        assessor = ScriptedAssessor([sufficient()])

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(result.evidence, ())
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertFalse(result.sufficient)

    async def test_a_failing_assessor_keeps_the_single_round_result(self):
        for error in (RuntimeError("assessment service down"), EvidenceAssessmentError("bad json")):
            with self.subTest(error=type(error).__name__):
                retriever = ScriptedRetrieval({QUESTION: [candidate("a")]})
                assessor = ScriptedAssessor([sufficient()], error=error)

                with self.assertLogs("rag_correction", level="WARNING") as logs:
                    result = await self.build(retriever, assessor).run(QUESTION)

                self.assertEqual(retriever.recalled_queries(), [QUESTION])
                self.assertEqual(result.retrieval_rounds, 1)
                self.assertTrue(result.sufficient)
                self.assertEqual([item.chunk_id for item in result.evidence], ["a"])
                self.assertIn("without a sufficiency check", "\n".join(logs.output))

    async def test_a_single_round_budget_never_corrects(self):
        retriever = ScriptedRetrieval(
            {QUESTION: [candidate("a")], REWRITTEN: [candidate("b")]}
        )
        assessor = ScriptedAssessor([insufficient()])

        result = await self.build(
            retriever, assessor, max_retrieval_rounds=1
        ).run(QUESTION)

        self.assertEqual(retriever.recalled_queries(), [QUESTION])
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertFalse(result.sufficient)

    async def test_the_merged_pool_respects_the_candidate_budget(self):
        retriever = ScriptedRetrieval(
            {
                QUESTION: [candidate("a"), candidate("b")],
                REWRITTEN: [candidate(f"c{index}") for index in range(4)],
            }
        )
        assessor = ScriptedAssessor([insufficient(), sufficient()])

        result = await self.build(retriever, assessor).run(QUESTION)

        self.assertEqual(len(retriever.reranked_ids(1)), 4)
        self.assertEqual(result.retrieval_rounds, 2)

    async def test_retrieval_failures_reach_the_caller(self):
        retriever = ScriptedRetrieval(recall_error=RuntimeError("qdrant unavailable"))

        with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
            await self.build(retriever, ScriptedAssessor([sufficient()])).run(QUESTION)

    def test_rejects_a_round_budget_outside_the_hard_limit(self):
        retriever = ScriptedRetrieval()
        assessor = ScriptedAssessor([sufficient()])

        for rounds in (0, MAX_RETRIEVAL_ROUNDS + 1):
            with self.subTest(rounds=rounds):
                with self.assertRaisesRegex(ValueError, "max_retrieval_rounds"):
                    self.build(retriever, assessor, max_retrieval_rounds=rounds)

    def test_rejects_being_built_while_corrective_retrieval_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "must be enabled"):
            CorrectiveCoordinator(
                retriever=ScriptedRetrieval(),
                assessor=ScriptedAssessor([sufficient()]),
                settings=RetrievalSettings(corrective=CORRECTIVE_DISABLED),
            )


class AssessorFallbackTests(unittest.IsolatedAsyncioTestCase):
    """The coordinator keeps the retrieved evidence whatever the assessor does."""

    def build(self, client):
        return CorrectiveCoordinator(
            retriever=ScriptedRetrieval({QUESTION: [candidate("a")]}),
            assessor=DeepSeekEvidenceAssessor(
                settings=FakeChatSettings(), http_client=client
            ),
            settings=corrective_settings(),
        )

    async def test_an_unusable_reply_keeps_the_retrieved_evidence(self):
        client = FakeChatHttpClient(chat_response(content="我觉得这些资料够用了"))

        with self.assertLogs("rag_correction", level="WARNING") as logs:
            result = await self.build(client).run(QUESTION)

        self.assertEqual(result.retrieval_rounds, 1)
        self.assertEqual([item.chunk_id for item in result.evidence], ["a"])
        self.assertIn("without a sufficiency check", "\n".join(logs.output))

    async def test_a_timeout_keeps_the_retrieved_evidence(self):
        client = FakeChatHttpClient(error=httpx.TimeoutException("too slow"))

        with self.assertLogs("rag_correction", level="WARNING"):
            result = await self.build(client).run(QUESTION)

        self.assertEqual(result.retrieval_rounds, 1)
        self.assertTrue(result.sufficient)


if __name__ == "__main__":
    unittest.main()
