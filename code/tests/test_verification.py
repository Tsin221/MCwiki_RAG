import json
import unittest

import httpx

from rag_evidence import AnswerEvidence
from rag_settings import (
    MAX_ANSWER_ATTEMPTS,
    VERIFICATION_DISABLED as SETTINGS_VERIFICATION_DISABLED,
    VERIFICATION_ENABLED as SETTINGS_VERIFICATION_ENABLED,
)
from rag_verification import (
    MAX_REPORTED_ISSUES,
    VERIFICATION_DISABLED,
    VERIFICATION_ENABLED,
    AnswerVerification,
    AnswerVerificationCoordinator,
    AnswerVerificationError,
    ClaimCheck,
    DeepSeekAnswerVerifier,
    ModelVerdict,
    build_answer_verifier,
    build_verification,
    citation_audit,
    parse_verdict,
    rewrite_feedback,
)


QUESTION = "红石中继器有什么作用？"
FIRST_DRAFT = "红石中继器可以延迟红石信号。[1]"
REWRITTEN_DRAFT = "红石中继器可以延迟并增强红石信号。[1]"
CITATION_ISSUE = "回答引用了不存在的证据编号 [9]。"


def evidence_item(item_id, *, text=None):
    normalized = text or f"证据 {item_id} 的正文"
    return AnswerEvidence(
        id=item_id,
        chunk_id=f"chunk-{item_id}",
        component_chunk_ids=(f"chunk-{item_id}",),
        title=f"标题 {item_id}",
        url=f"https://example.test/{item_id}",
        text=normalized,
        excerpt=normalized,
    )


def supported_check(claim="红石中继器可以延迟红石信号", *, citation_ids=(1,), reason="证据[1]直接说明"):
    return ClaimCheck(
        claim=claim,
        citation_ids=tuple(citation_ids),
        supported=True,
        reason=reason,
    )


def unsupported_check(claim="红石中继器可以增强红石信号", *, citation_ids=(1,), reason="证据[1]没有提供该结论"):
    return ClaimCheck(
        claim=claim,
        citation_ids=tuple(citation_ids),
        supported=False,
        reason=reason,
    )


def verdict(*claims, useful=True, issues=()):
    return ModelVerdict(claims=tuple(claims), useful=useful, issues=tuple(issues))


class CitationAuditTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [evidence_item(1), evidence_item(2)]

    def test_reads_numbered_markers_once_and_in_order(self):
        audit = citation_audit("先看[2]，再看[1]，最后仍是[2]。", self.evidence)

        self.assertEqual(audit.cited, (2, 1))
        self.assertEqual(audit.invalid, ())
        self.assertTrue(audit.is_usable)

    def test_flags_numbers_that_were_never_offered_as_evidence(self):
        audit = citation_audit("中继器可以延迟信号[1]，并且可以增强信号[9]。", self.evidence)

        self.assertEqual(audit.cited, (1, 9))
        self.assertEqual(audit.invalid, (9,))
        self.assertFalse(audit.is_usable)

    def test_an_answer_without_markers_is_not_usable(self):
        audit = citation_audit("红石中继器可以延迟红石信号。", self.evidence)

        self.assertEqual(audit.cited, ())
        self.assertEqual(audit.invalid, ())
        self.assertFalse(audit.is_usable)

    def test_brackets_that_are_not_evidence_numbers_are_ignored(self):
        audit = citation_audit("在 [1.16] 版本后加入，见 [ 1 ] 与 [备注]。", self.evidence)

        self.assertEqual(audit.cited, ())
        self.assertFalse(audit.is_usable)


class ParseVerdictTests(unittest.TestCase):
    def test_reads_the_full_structure(self):
        parsed = parse_verdict(
            json.dumps(
                {
                    "claims": [
                        {
                            "claim": " 红石中继器可以延迟红石信号 ",
                            "citations": [2, 1, 2],
                            "supported": True,
                            "reason": " 证据直接说明 ",
                        },
                        {
                            "claim": "可以增强红石信号",
                            "citations": [],
                            "supported": False,
                            "reason": "没有引用",
                        },
                    ],
                    "useful": False,
                    "issues": [" 缺少增强信号的证据 ", ""],
                }
            )
        )

        self.assertFalse(parsed.useful)
        self.assertEqual(parsed.issues, ("缺少增强信号的证据",))
        self.assertEqual(parsed.claims[0].claim, "红石中继器可以延迟红石信号")
        self.assertEqual(parsed.claims[0].citation_ids, (1, 2))
        self.assertTrue(parsed.claims[0].supported)
        self.assertEqual(parsed.claims[0].reason, "证据直接说明")
        self.assertEqual(parsed.claims[1].citation_ids, ())
        self.assertFalse(parsed.claims[1].supported)

    def test_optional_fields_default_to_an_empty_list(self):
        parsed = parse_verdict('{"claims": [], "useful": true}')

        self.assertEqual(parsed.claims, ())
        self.assertTrue(parsed.useful)
        self.assertEqual(parsed.issues, ())

    def test_the_claim_and_issue_lists_are_capped(self):
        parsed = parse_verdict(
            json.dumps(
                {
                    "claims": [
                        {"claim": f"声明 {index}", "citations": [1], "supported": True}
                        for index in range(40)
                    ],
                    "useful": True,
                    "issues": [f"问题 {index}" for index in range(30)],
                }
            )
        )

        self.assertEqual(len(parsed.claims), 24)
        self.assertEqual(len(parsed.issues), 8)

    def test_rejects_every_malformed_reply(self):
        for payload in (
            "not json",
            "[1, 2]",
            '{"useful": true}',
            '{"claims": [], }',
            '{"claims": "红石", "useful": true}',
            '{"claims": [1], "useful": true}',
            '{"claims": [{"claim": 1, "citations": [1], "supported": true}], "useful": true}',
            '{"claims": [{"claim": "  ", "citations": [1], "supported": true}], "useful": true}',
            '{"claims": [{"claim": "声明", "citations": "1", "supported": true}], "useful": true}',
            '{"claims": [{"claim": "声明", "citations": [0], "supported": true}], "useful": true}',
            '{"claims": [{"claim": "声明", "citations": [true], "supported": true}], "useful": true}',
            '{"claims": [{"claim": "声明", "citations": [1], "supported": "yes"}], "useful": true}',
            '{"claims": [{"claim": "声明", "citations": [1], "supported": true, "reason": 7}], "useful": true}',
            '{"claims": [{"claim": "%s", "citations": [1], "supported": true}], "useful": true}'
            % ("长" * 400),
            '{"claims": [{"claim": "声明", "citations": [1], "supported": true, "reason": "%s"}], "useful": true}'
            % ("长" * 400),
            '{"claims": [], "useful": "yes"}',
            '{"claims": [], "useful": 1}',
            '{"claims": [], "useful": true, "issues": "缺少证据"}',
            '{"claims": [], "useful": true, "issues": [7]}',
            '{"claims": [], "useful": true, "issues": ["%s"]}' % ("长" * 400),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(AnswerVerificationError):
                    parse_verdict(payload)

    def test_the_dataclass_refuses_bad_claims(self):
        with self.assertRaisesRegex(ValueError, "claim"):
            ClaimCheck(claim="   ", citation_ids=(1,), supported=True)
        with self.assertRaisesRegex(ValueError, "claim"):
            ClaimCheck(claim="长" * 400, citation_ids=(1,), supported=True)
        with self.assertRaisesRegex(ValueError, "reason"):
            ClaimCheck(claim="声明", citation_ids=(1,), supported=True, reason="长" * 400)
        with self.assertRaisesRegex(ValueError, "citation_ids"):
            ClaimCheck(claim="声明", citation_ids=(0,), supported=True)


class BuildVerificationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [evidence_item(1), evidence_item(2)]

    def build(self, answer, model_verdict):
        return build_verification(answer, self.evidence, model_verdict)

    def test_a_fully_supported_useful_answer_passes(self):
        verification = self.build(
            FIRST_DRAFT,
            verdict(supported_check(), useful=True),
        )

        self.assertTrue(verification.supported)
        self.assertTrue(verification.useful)
        self.assertTrue(verification.acceptable)
        self.assertEqual(verification.issues, ())

    def test_a_supported_answer_that_misses_a_sub_question_is_not_acceptable(self):
        verification = self.build(
            FIRST_DRAFT,
            verdict(supported_check(), useful=False, issues=("遗漏了合成方式",)),
        )

        self.assertTrue(verification.supported)
        self.assertFalse(verification.useful)
        self.assertFalse(verification.acceptable)
        self.assertEqual(verification.issues, ("遗漏了合成方式",))

    def test_an_answer_without_citations_never_passes(self):
        verification = self.build(
            "红石中继器可以延迟红石信号。",
            verdict(supported_check(citation_ids=())),
        )

        self.assertFalse(verification.supported)
        self.assertIn("回答没有标注任何证据编号。", verification.issues)

    def test_an_out_of_range_citation_never_passes(self):
        verification = self.build(
            "红石中继器可以延迟红石信号。[9]",
            verdict(supported_check(citation_ids=(1,))),
        )

        self.assertFalse(verification.supported)
        self.assertIn(CITATION_ISSUE, verification.issues)

    def test_a_claim_without_a_citation_never_passes(self):
        verification = self.build(
            FIRST_DRAFT,
            verdict(supported_check(citation_ids=())),
        )

        self.assertFalse(verification.supported)
        self.assertTrue(
            any("没有标注证据编号" in issue for issue in verification.issues),
            verification.issues,
        )

    def test_a_claim_that_points_at_an_unknown_number_never_passes(self):
        verification = self.build(
            "红石中继器可以延迟红石信号。[1]",
            verdict(supported_check(citation_ids=(1, 9))),
        )

        self.assertFalse(verification.supported)
        self.assertTrue(
            any("[9]" in issue for issue in verification.issues),
            verification.issues,
        )

    def test_an_unsupported_claim_is_reported_with_its_reason(self):
        verification = self.build(
            "红石中继器可以增强红石信号。[2]",
            verdict(unsupported_check(citation_ids=(2,), reason="证据[2]说的是延迟而不是增强")),
        )

        self.assertFalse(verification.supported)
        self.assertIn("证据[2]说的是延迟而不是增强", verification.issues[0])
        self.assertIn("红石中继器可以增强红石信号", verification.issues[0])

    def test_an_empty_claim_list_cannot_pass(self):
        verification = self.build(FIRST_DRAFT, verdict())

        self.assertFalse(verification.supported)
        self.assertIn("核验没有返回任何可判定的事实声明。", verification.issues)

    def test_deterministic_findings_come_before_the_model_issues(self):
        verification = self.build(
            "红石中继器可以延迟红石信号。[9]",
            verdict(unsupported_check(citation_ids=(9,)), issues=("模型发现的问题",)),
        )

        self.assertEqual(verification.issues[0], CITATION_ISSUE)
        self.assertEqual(verification.issues[-1], "模型发现的问题")

    def test_the_reported_issues_are_capped(self):
        claims = tuple(
            unsupported_check(claim=f"未被支持的声明 {index}", citation_ids=(1,))
            for index in range(20)
        )

        verification = self.build(FIRST_DRAFT, verdict(*claims))

        self.assertEqual(len(verification.issues), MAX_REPORTED_ISSUES)
        self.assertFalse(verification.supported)


class RewriteFeedbackTests(unittest.TestCase):
    def test_repeats_the_findings_and_the_bounded_rules(self):
        verification = AnswerVerification(
            supported=False,
            useful=True,
            issues=("回答引用了不存在的证据编号 [9]。",),
        )

        feedback = rewrite_feedback(verification)

        self.assertIn("上一版回答没有通过证据核验", feedback)
        self.assertIn("- 回答引用了不存在的证据编号 [9]。", feedback)
        self.assertIn("不得引用不存在的编号", feedback)
        self.assertIn("每个子问题都要回答", feedback)

    def test_a_failure_without_findings_still_asks_for_a_rewrite(self):
        feedback = rewrite_feedback(AnswerVerification(supported=False, useful=False))

        self.assertIn("没有通过核验", feedback)

    def test_the_feedback_stays_bounded(self):
        verification = AnswerVerification(
            supported=False,
            useful=True,
            issues=tuple(f"问题{index}：" + "长" * 400 for index in range(20)),
        )

        feedback = rewrite_feedback(verification)

        self.assertLessEqual(len(feedback), 900)


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


class DeepSeekAnswerVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_a_bounded_structured_request_and_parses_the_reply(self):
        client = FakeChatHttpClient(
            chat_response(
                content=json.dumps(
                    {
                        "claims": [
                            {
                                "claim": "红石中继器可以延迟红石信号",
                                "citations": [1],
                                "supported": True,
                                "reason": "证据直接说明",
                            }
                        ],
                        "useful": True,
                        "issues": [],
                    }
                )
            )
        )
        verifier = DeepSeekAnswerVerifier(
            settings=FakeChatSettings(),
            http_client=client,
            timeout=7.5,
        )

        parsed = await verifier.verify(
            QUESTION,
            FIRST_DRAFT,
            [evidence_item(1, text="中继器会延迟信号")],
        )

        self.assertTrue(parsed.useful)
        self.assertEqual(parsed.claims[0].citation_ids, (1,))
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
        self.assertIn("只做判断，不要重写回答", messages[0]["content"])
        self.assertIn(QUESTION, messages[1]["content"])
        self.assertIn(FIRST_DRAFT, messages[1]["content"])
        self.assertIn("[1] 标题：标题 1", messages[1]["content"])
        self.assertIn("中继器会延迟信号", messages[1]["content"])
        self.assertNotIn("test-key", json.dumps(call["json"]))

    async def test_an_empty_evidence_list_is_still_checked(self):
        client = FakeChatHttpClient(
            chat_response(content='{"claims": [], "useful": true}')
        )
        verifier = DeepSeekAnswerVerifier(
            settings=FakeChatSettings(), http_client=client
        )

        await verifier.verify(QUESTION, FIRST_DRAFT, [])

        self.assertIn("没有可用的证据", client.calls[0]["json"]["messages"][1]["content"])

    async def test_unusable_replies_raise_a_predictable_error(self):
        responses = {
            "invalid json body": chat_response(text="<html>gateway</html>"),
            "invalid verdict json": chat_response(content="这版回答没问题"),
            "missing content": httpx.Response(
                200,
                json={"choices": [{"message": {}}]},
                request=httpx.Request("POST", "https://api.example.test"),
            ),
        }

        for label, response in responses.items():
            with self.subTest(case=label):
                verifier = DeepSeekAnswerVerifier(
                    settings=FakeChatSettings(),
                    http_client=FakeChatHttpClient(response),
                )
                with self.assertRaises(AnswerVerificationError):
                    await verifier.verify(QUESTION, FIRST_DRAFT, [evidence_item(1)])

    async def test_transport_failures_and_timeouts_raise_the_same_error(self):
        cases = {
            "timeout": FakeChatHttpClient(error=httpx.TimeoutException("too slow")),
            "connect error": FakeChatHttpClient(error=httpx.ConnectError("unreachable")),
            "error status": FakeChatHttpClient(chat_response(status_code=500, text="upstream")),
        }

        for label, client in cases.items():
            with self.subTest(case=label):
                verifier = DeepSeekAnswerVerifier(
                    settings=FakeChatSettings(), http_client=client
                )
                with self.assertRaises(AnswerVerificationError):
                    await verifier.verify(QUESTION, FIRST_DRAFT, [evidence_item(1)])

    async def test_rejects_invalid_construction_arguments(self):
        client = FakeChatHttpClient(chat_response(content="{}"))

        with self.assertRaisesRegex(ValueError, "timeout"):
            DeepSeekAnswerVerifier(
                settings=FakeChatSettings(), http_client=client, timeout=0
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            DeepSeekAnswerVerifier(
                settings=FakeChatSettings(), http_client=client, temperature=3
            )
        with self.assertRaisesRegex(ValueError, "thinking_type"):
            DeepSeekAnswerVerifier(
                settings=FakeChatSettings(), http_client=client, thinking_type=" "
            )


class BuildAnswerVerifierTests(unittest.TestCase):
    def test_the_strategy_names_match_the_settings_module(self):
        self.assertEqual(
            (VERIFICATION_DISABLED, VERIFICATION_ENABLED),
            (SETTINGS_VERIFICATION_DISABLED, SETTINGS_VERIFICATION_ENABLED),
        )

    def test_the_disabled_strategy_builds_no_verifier(self):
        self.assertIsNone(build_answer_verifier(VERIFICATION_DISABLED))

    def test_the_enabled_strategy_needs_a_chat_client(self):
        self.assertIsNone(
            build_answer_verifier(VERIFICATION_ENABLED, settings=FakeChatSettings())
        )

    def test_the_enabled_strategy_builds_the_model_verifier_with_a_client(self):
        verifier = build_answer_verifier(
            VERIFICATION_ENABLED,
            settings=FakeChatSettings(),
            http_client=FakeChatHttpClient(chat_response(content="{}")),
            timeout=9.0,
        )

        self.assertIsInstance(verifier, DeepSeekAnswerVerifier)

    def test_missing_client_is_logged_and_degrades(self):
        with self.assertLogs("rag_verification", level="WARNING") as logs:
            verifier = build_answer_verifier(
                VERIFICATION_ENABLED, settings=FakeChatSettings()
            )

        self.assertIsNone(verifier)
        self.assertIn("without a claim check", "\n".join(logs.output))

    def test_rejects_an_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported verification strategy"):
            build_answer_verifier("verify_everything")


class ScriptedAnswerer:
    """Write the scripted drafts in order; the last one repeats."""

    def __init__(self, drafts):
        self.calls = []
        self.drafts = list(drafts)

    async def collect_answer(self, question, evidence, *, feedback=""):
        self.calls.append(
            {"question": question, "evidence": list(evidence), "feedback": feedback}
        )
        index = min(len(self.calls) - 1, len(self.drafts) - 1)
        return self.drafts[index]

    def seen_answer_ids(self, index):
        return [item.id for item in self.calls[index]["evidence"]]


class ScriptedVerifier:
    """Return the scripted verdicts in order, or raise on a scripted call."""

    def __init__(self, verdicts, *, error=None, error_at=None):
        self.calls = []
        self.verdicts = list(verdicts)
        self.error = error
        self.error_at = error_at

    async def verify(self, question, answer, evidence):
        self.calls.append(
            {"question": question, "answer": answer, "evidence": list(evidence)}
        )
        if self.error is not None and (
            self.error_at is None or len(self.calls) == self.error_at
        ):
            raise self.error
        index = min(len(self.calls) - 1, len(self.verdicts) - 1)
        return self.verdicts[index]

    def seen_answers(self):
        return [call["answer"] for call in self.calls]


class AnswerVerificationCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def build(self, answerer, verifier, **overrides):
        return AnswerVerificationCoordinator(
            answerer=answerer,
            verifier=verifier,
            **overrides,
        )

    def setUp(self):
        self.evidence = [evidence_item(1), evidence_item(2)]

    async def test_a_verified_first_draft_is_sent_as_it_is(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT])
        verifier = ScriptedVerifier([verdict(supported_check())])

        result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertEqual(result.answer, FIRST_DRAFT)
        self.assertEqual(result.answer_attempts, 1)
        self.assertFalse(result.withheld)
        self.assertTrue(result.verification.acceptable)
        self.assertEqual(len(answerer.calls), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(verifier.calls[0]["question"], QUESTION)
        self.assertEqual(answerer.seen_answer_ids(0), [1, 2])
        self.assertEqual(answerer.calls[0]["feedback"], "")

    async def test_a_failed_first_draft_is_rewritten_once_and_checked_again(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT, REWRITTEN_DRAFT])
        verifier = ScriptedVerifier(
            [
                verdict(unsupported_check(citation_ids=(9,)), issues=("引用编号不存在",)),
                verdict(supported_check()),
            ]
        )

        result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertEqual(result.answer, REWRITTEN_DRAFT)
        self.assertEqual(result.answer_attempts, 2)
        self.assertFalse(result.withheld)
        self.assertTrue(result.verification.acceptable)
        self.assertEqual(len(answerer.calls), 2)
        self.assertEqual(len(verifier.calls), 2)
        # The second check reads the rewritten draft, not the rejected one.
        self.assertEqual(verifier.seen_answers(), [FIRST_DRAFT, REWRITTEN_DRAFT])
        # The rewrite is asked for with the verifier's own findings and no new evidence.
        self.assertIn("引用编号不存在", answerer.calls[1]["feedback"])
        self.assertIn("不得引用不存在的编号", answerer.calls[1]["feedback"])
        self.assertEqual(answerer.calls[1]["evidence"], answerer.calls[0]["evidence"])

    async def test_a_second_failure_withholds_the_rewrite(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT, REWRITTEN_DRAFT])
        verifier = ScriptedVerifier([verdict(unsupported_check())])

        result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertEqual(result.answer, REWRITTEN_DRAFT)
        self.assertEqual(result.answer_attempts, 2)
        self.assertTrue(result.withheld)
        self.assertFalse(result.verification.acceptable)
        self.assertEqual(len(answerer.calls), 2)
        self.assertEqual(len(verifier.calls), 2)

    async def test_an_invented_citation_is_rejected_without_a_model_call(self):
        answerer = ScriptedAnswerer(["中继器可以延迟信号[9]。"])
        verifier = ScriptedVerifier([verdict(supported_check())])

        result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertEqual(verifier.calls, [])
        self.assertEqual(result.answer_attempts, 2)
        self.assertTrue(result.withheld)
        self.assertIn(CITATION_ISSUE, result.verification.issues)
        self.assertIn(CITATION_ISSUE, answerer.calls[1]["feedback"])

    async def test_a_citation_free_draft_is_rejected_without_a_model_call(self):
        answerer = ScriptedAnswerer(["中继器可以延迟信号。"])
        verifier = ScriptedVerifier([verdict(supported_check())])

        result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertEqual(verifier.calls, [])
        self.assertTrue(result.withheld)
        self.assertIn("回答没有标注任何证据编号。", result.verification.issues)

    async def test_a_broken_verifier_keeps_the_draft_without_claiming_it_was_checked(self):
        for error in (
            RuntimeError("verification service down"),
            AnswerVerificationError("bad json"),
        ):
            with self.subTest(error=type(error).__name__):
                answerer = ScriptedAnswerer([FIRST_DRAFT])
                verifier = ScriptedVerifier([], error=error)

                with self.assertLogs("rag_verification", level="WARNING") as logs:
                    result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

                self.assertEqual(result.answer, FIRST_DRAFT)
                self.assertEqual(result.answer_attempts, 1)
                self.assertFalse(result.withheld)
                self.assertIsNone(result.verification)
                self.assertEqual(len(answerer.calls), 1)
                self.assertIn("without a claim check", "\n".join(logs.output))

    async def test_a_verifier_that_breaks_on_the_second_check_keeps_the_rewrite(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT, REWRITTEN_DRAFT])
        verifier = ScriptedVerifier(
            [verdict(unsupported_check())],
            error=httpx.TimeoutException("too slow"),
            error_at=2,
        )

        with self.assertLogs("rag_verification", level="WARNING"):
            result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertEqual(result.answer, REWRITTEN_DRAFT)
        self.assertEqual(result.answer_attempts, 2)
        self.assertFalse(result.withheld)
        self.assertIsNone(result.verification)
        self.assertEqual(len(answerer.calls), 2)

    async def test_a_single_attempt_budget_never_rewrites(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT, REWRITTEN_DRAFT])
        verifier = ScriptedVerifier([verdict(unsupported_check())])

        result = await self.build(
            answerer, verifier, max_attempts=1
        ).run(QUESTION, self.evidence)

        self.assertEqual(result.answer, FIRST_DRAFT)
        self.assertEqual(result.answer_attempts, 1)
        self.assertTrue(result.withheld)
        self.assertEqual(len(answerer.calls), 1)
        self.assertEqual(len(verifier.calls), 1)

    async def test_a_sub_question_that_was_never_answered_is_rewritten(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT, REWRITTEN_DRAFT])
        verifier = ScriptedVerifier(
            [
                verdict(supported_check(), useful=False, issues=("遗漏了合成方式",)),
                verdict(supported_check()),
            ]
        )

        result = await self.build(answerer, verifier).run(QUESTION, self.evidence)

        self.assertFalse(result.withheld)
        self.assertEqual(result.answer, REWRITTEN_DRAFT)
        self.assertIn("遗漏了合成方式", answerer.calls[1]["feedback"])

    def test_rejects_a_budget_outside_the_hard_limit(self):
        answerer = ScriptedAnswerer([FIRST_DRAFT])
        verifier = ScriptedVerifier([verdict(supported_check())])

        for attempts in (0, MAX_ANSWER_ATTEMPTS + 1):
            with self.subTest(attempts=attempts):
                with self.assertRaisesRegex(ValueError, "max_attempts"):
                    self.build(answerer, verifier, max_attempts=attempts)


if __name__ == "__main__":
    unittest.main()
