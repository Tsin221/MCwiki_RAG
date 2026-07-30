import json
import unittest

import httpx

from rag_answer import (
    AnswerConfigurationError,
    AnswerEvidence,
    DeepSeekAnswerClient,
    DeepSeekSettings,
    ModelResponseError,
    ModelTimeoutError,
    build_evidence,
)
from rag_retrieval.hybrid import HybridResult


def hybrid_result(
    chunk_id: str,
    *,
    title: str = "红石中继器",
    text: str = "红石中继器可以延迟并增强红石信号。",
    source: str = "https://zh.minecraft.wiki/w/红石中继器",
) -> HybridResult:
    return HybridResult(
        chunk_id=chunk_id,
        title=title,
        text=text,
        source=source,
        score=0.032,
        bm25_rank=1,
        semantic_rank=2,
    )


class BuildEvidenceTests(unittest.TestCase):
    def test_assigns_stable_ids_and_removes_duplicate_text(self):
        results = [
            hybrid_result("first"),
            hybrid_result("duplicate"),
            hybrid_result(
                "second",
                title="比较器",
                text="红石比较器可以比较信号强度。",
                source="https://zh.minecraft.wiki/w/红石比较器",
            ),
        ]

        evidence = build_evidence(results, max_context_chars=1_000)

        self.assertEqual([item.id for item in evidence], [1, 2])
        self.assertEqual([item.chunk_id for item in evidence], ["first", "second"])
        self.assertEqual(evidence[0].excerpt, evidence[0].text)

    def test_truncates_context_from_lower_ranked_results(self):
        results = [
            hybrid_result("first", text="甲" * 80),
            hybrid_result("second", text="乙" * 80),
        ]

        evidence = build_evidence(results, max_context_chars=100)

        self.assertEqual(len(evidence), 1)
        self.assertLessEqual(len(evidence[0].text), 100)


class DeepSeekSettingsTests(unittest.TestCase):
    def test_loads_defaults_and_requires_api_key(self):
        settings = DeepSeekSettings.from_env({"DEEPSEEK_API_KEY": " local-key "})

        self.assertEqual(settings.api_key, "local-key")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")
        self.assertEqual(settings.model, "deepseek-v4-pro")

        with self.assertRaises(AnswerConfigurationError):
            DeepSeekSettings.from_env({})


class DeepSeekAnswerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_validated_content_and_builds_numbered_context(self):
        captured_request = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["authorization"] = request.headers["authorization"]
            captured_request["body"] = json.loads(request.content)
            body = "\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"中继器可以"}}]}',
                    'data: {"choices":[{"delta":{"content":"延迟信号。[1]"}}]}',
                    "data: [DONE]",
                    "",
                ]
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekAnswerClient(
                settings=DeepSeekSettings(
                    api_key="test-secret",
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-pro",
                ),
                http_client=http_client,
            )
            evidence = [
                AnswerEvidence(
                    id=1,
                    chunk_id="chunk-redstone",
                    title="红石中继器",
                    url="https://zh.minecraft.wiki/w/红石中继器",
                    text="红石中继器可以延迟红石信号。",
                    excerpt="红石中继器可以延迟红石信号。",
                )
            ]

            chunks = [
                chunk
                async for chunk in client.stream_answer("中继器有什么作用？", evidence)
            ]

        self.assertEqual(chunks, ["中继器可以", "延迟信号。[1]"])
        self.assertEqual(
            captured_request["url"],
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(captured_request["authorization"], "Bearer test-secret")
        self.assertEqual(captured_request["body"]["model"], "deepseek-v4-pro")
        self.assertTrue(captured_request["body"]["stream"])
        system_prompt = captured_request["body"]["messages"][0]["content"]
        self.assertIn(
            "问题包含多个子问题或“哪些”时，逐项回答",
            system_prompt,
        )
        self.assertIn(
            "数字、否定词和大小关系必须与证据保持一致",
            system_prompt,
        )
        self.assertIn("输出前逐项核对", system_prompt)
        self.assertIn(
            "[1] 标题：红石中继器",
            captured_request["body"]["messages"][1]["content"],
        )
        self.assertNotIn("test-secret", json.dumps(captured_request["body"]))

    async def test_rejects_malformed_stream_chunks(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"choices":"invalid"}\n\n',
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekAnswerClient(
                settings=DeepSeekSettings(
                    api_key="test-secret",
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-pro",
                ),
                http_client=http_client,
            )

            with self.assertRaises(ModelResponseError):
                _ = [
                    chunk
                    async for chunk in client.stream_answer(
                        "问题",
                        [
                            AnswerEvidence(
                                id=1,
                                chunk_id="chunk",
                                title="标题",
                                url="https://example.test/wiki",
                                text="证据",
                                excerpt="证据",
                            )
                        ],
                    )
                ]

    async def test_maps_transport_timeout_without_exposing_details(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("upstream host details", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekAnswerClient(
                settings=DeepSeekSettings(
                    api_key="test-secret",
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-pro",
                ),
                http_client=http_client,
            )

            with self.assertRaises(ModelTimeoutError):
                _ = [
                    chunk
                    async for chunk in client.stream_answer(
                        "问题",
                        [
                            AnswerEvidence(
                                id=1,
                                chunk_id="chunk",
                                title="标题",
                                url="https://example.test/wiki",
                                text="证据",
                                excerpt="证据",
                            )
                        ],
                    )
                ]


if __name__ == "__main__":
    unittest.main()
