import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from rag_reranker import (
    CROSS_ENCODER_RERANKER,
    DISABLED,
    NOOP_RERANKER,
    READY,
    UNAVAILABLE,
    CrossEncoderReranker,
    NoopReranker,
    RerankingRetriever,
    build_reranker,
)
from rag_retrieval.hybrid import HybridResult
from rag_settings import (
    CROSS_ENCODER_RERANKER as SETTINGS_CROSS_ENCODER_RERANKER,
    NOOP_RERANKER as SETTINGS_NOOP_RERANKER,
)


def candidate(
    chunk_id,
    *,
    text=None,
    score=0.03,
    bm25_rank=1,
    semantic_rank=1,
    document_id=None,
    chunk_index=None,
):
    return HybridResult(
        chunk_id=chunk_id,
        title=f"标题 {chunk_id}",
        text=text or f"{chunk_id} 的正文",
        source=f"https://example.test/{chunk_id}",
        score=score,
        bm25_rank=bm25_rank,
        semantic_rank=semantic_rank,
        document_id=document_id,
        chunk_index=chunk_index,
    )


class ScriptedScorer:
    """Return prepared scores for the pairs it is asked to score."""

    def __init__(self, scores=(), *, error=None):
        self.calls = []
        self.scores = list(scores)
        self.error = error
        self.options = None

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        self.calls.append([tuple(pair) for pair in pairs])
        self.options = {"batch_size": batch_size, "show_progress_bar": show_progress_bar}
        if self.error is not None:
            raise self.error
        return self.scores


def failing_loader(_model_name):
    raise OSError("model weights are not available locally")


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_reorders_the_pool_by_model_score_and_truncates_to_the_limit(self):
        scorer = ScriptedScorer([0.2, 0.9, 0.5, 0.1])
        reranker = CrossEncoderReranker(scorer=scorer)
        pool = [candidate(name) for name in ("first", "second", "third", "fourth")]

        reranked = reranker.rerank("红石中继器怎么用？", pool, limit=2)

        self.assertEqual([item.chunk_id for item in reranked], ["second", "third"])
        self.assertEqual([item.reranker_score for item in reranked], [0.9, 0.5])
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(len(scorer.calls[0]), 4)

    def test_keeps_retrieval_metadata_and_never_rewrites_the_rrf_score(self):
        reranker = CrossEncoderReranker(scorer=ScriptedScorer([0.4, 0.8]))
        pool = [
            candidate(
                "a",
                score=0.032,
                bm25_rank=3,
                semantic_rank=None,
                document_id="doc-a",
                chunk_index=7,
            ),
            candidate("b", score=0.016, bm25_rank=None, semantic_rank=2),
        ]

        reranked = reranker.rerank("红石中继器怎么用？", pool, limit=2)

        best = reranked[0]
        self.assertEqual(best.chunk_id, "b")
        self.assertEqual(best.score, 0.016)
        self.assertIsNone(best.bm25_rank)
        self.assertEqual(best.semantic_rank, 2)
        self.assertEqual(best.source, "https://example.test/b")
        self.assertEqual(best.title, "标题 b")
        self.assertEqual(reranked[1].score, 0.032)
        self.assertEqual(reranked[1].document_id, "doc-a")
        self.assertEqual(reranked[1].chunk_index, 7)
        self.assertEqual(reranked[1].bm25_rank, 3)

    def test_scores_the_original_question_against_title_and_text(self):
        scorer = ScriptedScorer([0.5, 0.5])
        reranker = CrossEncoderReranker(scorer=scorer, batch_size=4)

        reranker.rerank(
            "  红石中继器怎么用？  ",
            [candidate("a", text="中继器会延迟信号"), candidate("b", text="比较器输出强度")],
            limit=2,
        )

        self.assertEqual(
            scorer.calls[0],
            [
                ("红石中继器怎么用？", "标题 a\n中继器会延迟信号"),
                ("红石中继器怎么用？", "标题 b\n比较器输出强度"),
            ],
        )
        self.assertEqual(scorer.options, {"batch_size": 4, "show_progress_bar": False})

    def test_a_duplicate_chunk_is_scored_and_returned_once(self):
        scorer = ScriptedScorer([0.1, 0.9])
        reranker = CrossEncoderReranker(scorer=scorer)
        pool = [candidate("a"), candidate("b"), candidate("a", score=0.011)]

        reranked = reranker.rerank("红石中继器怎么用？", pool, limit=5)

        self.assertEqual([item.chunk_id for item in reranked], ["b", "a"])
        self.assertEqual(len(scorer.calls[0]), 2)
        self.assertEqual(reranked[1].score, 0.03)

    def test_equal_scores_keep_the_incoming_rrf_order(self):
        reranker = CrossEncoderReranker(scorer=ScriptedScorer([0.7, 0.7, 0.7]))

        reranked = reranker.rerank(
            "红石中继器怎么用？",
            [candidate("a"), candidate("b"), candidate("c")],
            limit=3,
        )

        self.assertEqual([item.chunk_id for item in reranked], ["a", "b", "c"])

    def test_inference_failure_returns_the_rrf_order_without_scores(self):
        scorer = ScriptedScorer(error=RuntimeError("CUDA out of memory"))
        reranker = CrossEncoderReranker(scorer=scorer)
        pool = [candidate("a"), candidate("b")]

        with self.assertLogs("rag_reranker", level="WARNING") as logs:
            reranked = reranker.rerank("红石中继器怎么用？", pool, limit=1)

        self.assertEqual([item.chunk_id for item in reranked], ["a"])
        self.assertIsNone(reranked[0].reranker_score)
        self.assertIn("keeping the RRF order", "\n".join(logs.output))
        self.assertEqual(reranker.readiness(), READY)

    def test_unusable_scores_fall_back_instead_of_misordering_the_pool(self):
        pool = [candidate("a"), candidate("b")]

        for scores in ([0.5], [0.5, float("nan")], [0.5, None]):
            with self.subTest(scores=scores):
                reranker = CrossEncoderReranker(scorer=ScriptedScorer(scores))
                with self.assertLogs("rag_reranker", level="WARNING"):
                    reranked = reranker.rerank("红石中继器怎么用？", pool, limit=2)
                self.assertEqual([item.chunk_id for item in reranked], ["a", "b"])
                self.assertIsNone(reranked[0].reranker_score)

    def test_a_model_that_cannot_load_degrades_to_the_rrf_order(self):
        with self.assertLogs("rag_reranker", level="WARNING") as logs:
            reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base", loader=failing_loader)

        self.assertEqual(reranker.readiness(), UNAVAILABLE)
        self.assertIn("could not be loaded", "\n".join(logs.output))
        self.assertIsNotNone(reranker.load_error)
        pool = [candidate("a"), candidate("b")]
        self.assertEqual(
            [item.chunk_id for item in reranker.rerank("红石", pool, limit=1)],
            ["a"],
        )

    def test_a_blank_question_or_an_empty_pool_skips_the_model(self):
        scorer = ScriptedScorer([0.9])
        reranker = CrossEncoderReranker(scorer=scorer)

        self.assertEqual(reranker.rerank("   ", [candidate("a")], limit=1)[0].chunk_id, "a")
        self.assertEqual(reranker.rerank("红石", [], limit=1), [])
        self.assertEqual(scorer.calls, [])

    def test_concurrent_requests_never_enter_the_model_at_the_same_time(self):
        overlap = []
        inside = threading.Event()

        class GuardedScorer:
            def predict(self, pairs, batch_size=32, show_progress_bar=None):
                if inside.is_set():
                    overlap.append(len(overlap))
                inside.set()
                time.sleep(0.05)
                inside.clear()
                return [0.5] * len(pairs)

        reranker = CrossEncoderReranker(scorer=GuardedScorer())
        pool = [candidate("a"), candidate("b")]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(reranker.rerank, "红石中继器怎么用？", pool, limit=2)
                for _ in range(2)
            ]
            results = [future.result() for future in futures]

        self.assertEqual(overlap, [])
        self.assertEqual(len(results), 2)

    def test_rejects_invalid_limits_and_construction_arguments(self):
        reranker = CrossEncoderReranker(scorer=ScriptedScorer([0.5]))

        with self.assertRaisesRegex(ValueError, "limit"):
            reranker.rerank("红石", [candidate("a")], limit=0)
        with self.assertRaisesRegex(ValueError, "model_name"):
            CrossEncoderReranker(model_name=" ")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            CrossEncoderReranker(batch_size=0, scorer=ScriptedScorer([0.5]))


class NoopRerankerTests(unittest.TestCase):
    def test_keeps_the_fused_order_and_truncates_to_the_limit(self):
        pool = [candidate("a"), candidate("b"), candidate("c")]

        reranked = NoopReranker().rerank("红石中继器怎么用？", pool, limit=2)

        self.assertEqual([item.chunk_id for item in reranked], ["a", "b"])
        self.assertEqual([item.score for item in reranked], [0.03, 0.03])
        self.assertIsNone(reranked[0].reranker_score)
        self.assertEqual(pool, [candidate("a"), candidate("b"), candidate("c")])

    def test_drops_a_repeated_chunk_and_reports_itself_as_disabled(self):
        reranker = NoopReranker()

        reranked = reranker.rerank("红石", [candidate("a"), candidate("a")], limit=5)

        self.assertEqual([item.chunk_id for item in reranked], ["a"])
        self.assertEqual(reranker.readiness(), DISABLED)

    def test_rejects_a_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            NoopReranker().rerank("红石", [candidate("a")], limit=-1)


class BuildRerankerTests(unittest.TestCase):
    def test_the_strategy_names_match_the_settings_module(self):
        self.assertEqual(
            (NOOP_RERANKER, CROSS_ENCODER_RERANKER),
            (SETTINGS_NOOP_RERANKER, SETTINGS_CROSS_ENCODER_RERANKER),
        )

    def test_the_default_strategy_is_the_disabled_reranker(self):
        reranker = build_reranker(NOOP_RERANKER)

        self.assertIsInstance(reranker, NoopReranker)
        self.assertEqual(reranker.readiness(), DISABLED)

    def test_the_disabled_reranker_needs_no_model_library(self):
        build_reranker(NOOP_RERANKER)

        self.assertNotIn("sentence_transformers", sys.modules)

    def test_rejects_an_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported reranker"):
            build_reranker("llm_reranker")


class ScriptedRecall:
    """Return a scripted candidate pool per query and record every call."""

    def __init__(self, pool=None, error=None):
        self.calls = []
        self.pool = pool or []
        self.error = error

    async def search(self, query, *, bm25_limit, semantic_limit, limit):
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
        return self.pool[:limit]


class RecordingReranker:
    """Record the pool it was given and order it by a scripted chunk order."""

    def __init__(self, order=None):
        self.calls = []
        self.order = list(order or [])

    def readiness(self):
        return READY

    def rerank(self, question, candidates, *, limit):
        self.calls.append({"question": question, "candidates": list(candidates), "limit": limit})
        positions = {chunk_id: index for index, chunk_id in enumerate(self.order)}
        ordered = sorted(
            candidates,
            key=lambda item: (positions.get(item.chunk_id, len(positions)), candidates.index(item)),
        )
        return ordered[:limit]


class RerankingRetrieverTests(unittest.IsolatedAsyncioTestCase):
    question = "红石中继器怎么用？"

    async def test_recalls_the_candidate_pool_then_reranks_down_to_the_limit(self):
        pool = [candidate("a"), candidate("b"), candidate("c"), candidate("d")]
        recall = ScriptedRecall(pool)
        reranker = RecordingReranker(["c", "a"])
        retriever = RerankingRetriever(retriever=recall, reranker=reranker)

        results = await retriever.search(
            self.question,
            bm25_limit=20,
            semantic_limit=20,
            candidate_limit=4,
            limit=2,
        )

        self.assertEqual([call["limit"] for call in recall.calls], [4])
        self.assertEqual(reranker.calls[0]["question"], self.question)
        self.assertEqual([item.chunk_id for item in reranker.calls[0]["candidates"]], ["a", "b", "c", "d"])
        self.assertEqual(reranker.calls[0]["limit"], 2)
        self.assertEqual([item.chunk_id for item in results], ["c", "a"])

    async def test_a_pool_smaller_than_the_keep_count_still_recalls_a_full_pool(self):
        recall = ScriptedRecall([candidate("a")])
        retriever = RerankingRetriever(retriever=recall, reranker=NoopReranker())

        await retriever.search(
            self.question,
            bm25_limit=20,
            semantic_limit=20,
            candidate_limit=3,
            limit=8,
        )

        self.assertEqual([call["limit"] for call in recall.calls], [8])

    async def test_the_disabled_reranker_returns_the_baseline_ranking(self):
        pool = [candidate("a"), candidate("b"), candidate("c")]
        baseline = [candidate("a"), candidate("b"), candidate("c")]
        recall = ScriptedRecall(pool)
        retriever = RerankingRetriever(retriever=recall, reranker=NoopReranker())

        results = await retriever.search(
            self.question,
            bm25_limit=20,
            semantic_limit=20,
            candidate_limit=20,
            limit=8,
        )

        self.assertEqual(results, baseline)

    async def test_a_second_round_uses_the_same_reranker(self):
        reranker = RecordingReranker(["b", "a"])
        retriever = RerankingRetriever(retriever=ScriptedRecall(), reranker=reranker)

        results = await retriever.rerank(
            self.question,
            [candidate("a"), candidate("b")],
            limit=1,
        )

        self.assertEqual([item.chunk_id for item in results], ["b"])

    async def test_rejects_invalid_limits_before_recalling(self):
        recall = ScriptedRecall()
        retriever = RerankingRetriever(retriever=recall, reranker=NoopReranker())

        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            await retriever.search(
                self.question,
                bm25_limit=20,
                semantic_limit=20,
                candidate_limit=0,
                limit=8,
            )
        with self.assertRaisesRegex(ValueError, "limit"):
            await retriever.search(
                self.question,
                bm25_limit=20,
                semantic_limit=20,
                candidate_limit=20,
                limit=0,
            )
        self.assertEqual(recall.calls, [])

    async def test_recall_failure_propagates_to_the_caller(self):
        retriever = RerankingRetriever(
            retriever=ScriptedRecall(error=RuntimeError("qdrant unavailable")),
            reranker=NoopReranker(),
        )

        with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
            await retriever.search(
                self.question,
                bm25_limit=20,
                semantic_limit=20,
                candidate_limit=20,
                limit=8,
            )


if __name__ == "__main__":
    unittest.main()
