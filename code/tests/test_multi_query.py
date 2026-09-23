import unittest

from rag_query import QueryPlan, build_query_plan
from rag_retrieval.hybrid import HybridResult
from rag_retrieval.multi_query import (
    DEFAULT_RRF_K,
    MultiQueryRetriever,
    fuse_rankings,
)
from rag_settings import ORIGINAL_QUERY_STRATEGY, STEP_BACK_QUERY_STRATEGY


def hybrid_result(
    chunk_id: str,
    *,
    score: float = 0.032,
    bm25_rank: int | None = 1,
    semantic_rank: int | None = 2,
    document_id: str | None = None,
    chunk_index: int | None = None,
) -> HybridResult:
    return HybridResult(
        chunk_id=chunk_id,
        title=f"标题 {chunk_id}",
        text=f"{chunk_id} 的正文",
        source=f"https://example.test/{chunk_id}",
        score=score,
        bm25_rank=bm25_rank,
        semantic_rank=semantic_rank,
        document_id=document_id,
        chunk_index=chunk_index,
    )


def rrf_score(rank: int, *, rrf_k: int = DEFAULT_RRF_K) -> float:
    return 1 / (rrf_k + rank)


class ScriptedRetriever:
    """Return a scripted ranking per query and record every call."""

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


class ScriptedPlanner:
    """Build a plan from the question under test, or fail on demand."""

    def __init__(self, candidates=(), *, strategy=ORIGINAL_QUERY_STRATEGY, error=None):
        self.calls = []
        self.candidates = list(candidates)
        self.strategy = strategy
        self.error = error

    async def plan(self, question: str) -> QueryPlan:
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return build_query_plan(
            question,
            self.candidates,
            strategy=self.strategy,
        )


class FuseRankingsTests(unittest.TestCase):
    def test_keeps_one_entry_per_chunk_and_rewards_two_route_hits(self):
        fused = fuse_rankings(
            [
                [hybrid_result("both", score=0.9), hybrid_result("original")],
                [hybrid_result("step-back"), hybrid_result("both", score=0.01)],
            ],
            limit=10,
        )

        self.assertEqual(
            [item.chunk_id for item in fused],
            ["both", "step-back", "original"],
        )
        self.assertAlmostEqual(fused[0].score, rrf_score(1) + rrf_score(2))
        self.assertAlmostEqual(fused[1].score, rrf_score(1))
        self.assertAlmostEqual(fused[2].score, rrf_score(2))

    def test_keeps_the_metadata_of_the_first_appearance(self):
        fused = fuse_rankings(
            [
                [hybrid_result("filler")],
                [
                    hybrid_result(
                        "shared",
                        bm25_rank=2,
                        semantic_rank=3,
                        document_id="doc-a",
                        chunk_index=4,
                    )
                ],
                [hybrid_result("shared", bm25_rank=7, semantic_rank=7)],
            ],
            limit=5,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].chunk_id, "shared")
        self.assertEqual(fused[0].bm25_rank, 2)
        self.assertEqual(fused[0].semantic_rank, 3)
        self.assertEqual(fused[0].document_id, "doc-a")
        self.assertEqual(fused[0].chunk_index, 4)

    def test_counts_a_duplicate_inside_one_ranking_once(self):
        fused = fuse_rankings(
            [
                [
                    hybrid_result("duplicate"),
                    hybrid_result("duplicate"),
                    hybrid_result("distinct"),
                ]
            ],
            limit=10,
        )

        self.assertEqual(
            [item.chunk_id for item in fused],
            ["duplicate", "distinct"],
        )
        self.assertAlmostEqual(fused[0].score, rrf_score(1))
        self.assertAlmostEqual(fused[1].score, rrf_score(2))

    def test_truncates_to_the_limit_with_a_deterministic_order(self):
        rankings = [[hybrid_result("a"), hybrid_result("b")], [hybrid_result("c")]]

        self.assertEqual(
            [item.chunk_id for item in fuse_rankings(rankings, limit=10)],
            ["a", "c", "b"],
        )
        self.assertEqual(
            [item.chunk_id for item in fuse_rankings(rankings, limit=2)],
            ["a", "c"],
        )

    def test_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            fuse_rankings([[hybrid_result("a")]], limit=0)
        with self.assertRaisesRegex(ValueError, "rrf_k"):
            fuse_rankings([[hybrid_result("a")]], limit=1, rrf_k=-1)


class MultiQueryRetrieverTests(unittest.IsolatedAsyncioTestCase):
    question = "红石中继器怎么用？"

    async def search(self, coordinator):
        return await coordinator.search(
            self.question,
            bm25_limit=20,
            semantic_limit=20,
            limit=8,
        )

    async def test_original_strategy_returns_the_baseline_ranking(self):
        baseline = [hybrid_result("first"), hybrid_result("second", score=0.011)]
        retriever = ScriptedRetriever({"红石中继器怎么用？": baseline})
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(),
        )

        results = await self.search(coordinator)

        self.assertEqual(results, baseline)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0]["query"], "红石中继器怎么用？")
        self.assertEqual(retriever.calls[0]["limit"], 8)

    async def test_step_back_runs_both_queries_in_plan_order(self):
        retriever = ScriptedRetriever(
            {
                "红石中继器怎么用？": [hybrid_result("original-hit")],
                "红石信号的传输机制是什么": [hybrid_result("step-back-hit")],
            }
        )
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(
                ["红石信号的传输机制是什么"],
                strategy=STEP_BACK_QUERY_STRATEGY,
            ),
        )

        results = await self.search(coordinator)

        self.assertEqual(
            [call["query"] for call in retriever.calls],
            ["红石中继器怎么用？", "红石信号的传输机制是什么"],
        )
        self.assertEqual(
            [call["bm25_limit"] for call in retriever.calls],
            [20, 20],
        )
        self.assertEqual([call["limit"] for call in retriever.calls], [8, 8])
        self.assertEqual(
            [item.chunk_id for item in results],
            ["original-hit", "step-back-hit"],
        )

    async def test_chunk_found_by_both_queries_is_kept_once_and_ranks_first(self):
        retriever = ScriptedRetriever(
            {
                "红石中继器怎么用？": [
                    hybrid_result("only-original", bm25_rank=1, semantic_rank=1),
                    hybrid_result("shared", bm25_rank=2, semantic_rank=3),
                ],
                "红石信号的传输机制是什么": [
                    hybrid_result("shared", bm25_rank=7, semantic_rank=8),
                    hybrid_result("only-step-back", bm25_rank=1, semantic_rank=1),
                ],
            }
        )
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(
                ["红石信号的传输机制是什么"],
                strategy=STEP_BACK_QUERY_STRATEGY,
            ),
        )

        results = await self.search(coordinator)

        self.assertEqual(
            [item.chunk_id for item in results],
            ["shared", "only-original", "only-step-back"],
        )
        self.assertAlmostEqual(results[0].score, rrf_score(1) + rrf_score(2))
        self.assertEqual(results[0].bm25_rank, 2)
        self.assertEqual(results[0].semantic_rank, 3)

    async def test_fused_pool_is_capped_at_the_requested_limit(self):
        retriever = ScriptedRetriever(
            {
                "红石中继器怎么用？": [
                    hybrid_result("a"),
                    hybrid_result("b"),
                    hybrid_result("c"),
                ],
                "红石信号的传输机制是什么": [hybrid_result("d")],
            }
        )
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(
                ["红石信号的传输机制是什么"],
                strategy=STEP_BACK_QUERY_STRATEGY,
            ),
        )

        results = await coordinator.search(
            "红石中继器怎么用？",
            bm25_limit=20,
            semantic_limit=20,
            limit=2,
        )

        self.assertEqual([item.chunk_id for item in results], ["a", "d"])
        self.assertEqual([call["limit"] for call in retriever.calls], [2, 2])

    async def test_planner_failure_retrieves_the_original_question_only(self):
        retriever = ScriptedRetriever({"红石中继器怎么用？": [hybrid_result("only")]})
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(error=RuntimeError("planner bug")),
        )

        results = await self.search(coordinator)

        self.assertEqual([item.chunk_id for item in results], ["only"])
        self.assertEqual(
            [call["query"] for call in retriever.calls],
            ["红石中继器怎么用？"],
        )

    async def test_blank_question_skips_retrieval(self):
        retriever = ScriptedRetriever()
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(["红石信号的传输机制是什么"]),
        )

        results = await coordinator.search(
            "   ",
            bm25_limit=20,
            semantic_limit=20,
            limit=8,
        )

        self.assertEqual(results, [])
        self.assertEqual(retriever.calls, [])

    async def test_retrieval_failure_propagates_to_the_caller(self):
        retriever = ScriptedRetriever(error=RuntimeError("qdrant unavailable"))
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(["红石信号的传输机制是什么"]),
        )

        with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
            await self.search(coordinator)

    async def test_rejects_invalid_limits(self):
        retriever = ScriptedRetriever()
        coordinator = MultiQueryRetriever(
            retriever=retriever,
            planner=ScriptedPlanner(),
        )

        with self.assertRaisesRegex(ValueError, "bm25_limit"):
            await coordinator.search(
                "红石中继器怎么用？",
                bm25_limit=0,
                semantic_limit=20,
                limit=8,
            )
        with self.assertRaisesRegex(ValueError, "limit"):
            await coordinator.search(
                "红石中继器怎么用？",
                bm25_limit=20,
                semantic_limit=20,
                limit=0,
            )
        self.assertEqual(retriever.calls, [])

    async def test_rejects_a_negative_rrf_k(self):
        with self.assertRaisesRegex(ValueError, "rrf_k"):
            MultiQueryRetriever(
                retriever=ScriptedRetriever(),
                planner=ScriptedPlanner(),
                rrf_k=-1,
            )

    async def test_exposed_plan_describes_the_strategy_that_ran(self):
        coordinator = MultiQueryRetriever(
            retriever=ScriptedRetriever(),
            planner=ScriptedPlanner(
                ["红石信号的传输机制是什么"],
                strategy=STEP_BACK_QUERY_STRATEGY,
            ),
        )

        plan = await coordinator.plan(self.question)

        self.assertEqual(plan.strategy, STEP_BACK_QUERY_STRATEGY)
        self.assertEqual(
            plan.retrieval_queries,
            ("红石中继器怎么用？", "红石信号的传输机制是什么"),
        )


if __name__ == "__main__":
    unittest.main()
