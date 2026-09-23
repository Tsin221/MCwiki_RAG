from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol

import anyio

from rag_query import (
    ORIGINAL_QUERY_STRATEGY,
    QueryPlan,
    QueryPlanner,
    build_query_plan,
)

from .hybrid import HybridResult


logger = logging.getLogger(__name__)
DEFAULT_RRF_K = 60


class HybridSearch(Protocol):
    """Anything that runs one hybrid retrieval pass for a single query."""

    def search(
        self,
        query: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]: ...


@dataclass(slots=True)
class _FusedCandidate:
    result: HybridResult
    score: float
    best_rank: int


def fuse_rankings(
    rankings: Sequence[Sequence[HybridResult]],
    *,
    limit: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridResult]:
    """Fuse per-query rankings by reciprocal rank, keeping one entry per chunk.

    Only ranks are combined, so scores from different queries never get added
    together. A chunk returned by several queries accumulates one contribution
    per ranking, and keeps the metadata of its first appearance, which is the
    earliest query in the given ranking order.
    """
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if rrf_k < 0:
        raise ValueError("rrf_k must be zero or greater")

    candidates: dict[str, _FusedCandidate] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for result in ranking:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            rank += 1
            contribution = 1 / (rrf_k + rank)
            existing = candidates.get(result.chunk_id)
            if existing is None:
                candidates[result.chunk_id] = _FusedCandidate(
                    result=result,
                    score=contribution,
                    best_rank=rank,
                )
            else:
                existing.score += contribution
                existing.best_rank = min(existing.best_rank, rank)

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.score,
            candidate.best_rank,
            candidate.result.chunk_id,
        ),
    )
    return [
        replace(candidate.result, score=candidate.score)
        for candidate in ordered[:limit]
    ]


class MultiQueryRetriever:
    """Plan the queries for a question, retrieve each, then fuse the rankings.

    A single-query plan returns the underlying ranking unchanged, so the
    ``original`` strategy returns exactly what the baseline returned.
    """

    __slots__ = ("_planner", "_retriever", "_rrf_k")

    def __init__(
        self,
        *,
        retriever: HybridSearch,
        planner: QueryPlanner,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be zero or greater")
        self._retriever = retriever
        self._planner = planner
        self._rrf_k = rrf_k

    async def plan(self, question: str) -> QueryPlan:
        """Return a query plan, falling back to the original question on failure."""
        try:
            return await self._planner.plan(question)
        except Exception as error:
            logger.warning(
                "Query planning failed (%s); retrieving the original question only",
                error,
            )
            return build_query_plan(question, strategy=ORIGINAL_QUERY_STRATEGY)

    async def search(
        self,
        question: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]:
        """Return up to ``limit`` candidates drawn from every planned query.

        Callers that need a larger pool for a later reranking stage simply pass
        a larger ``limit``; the fused ranking is truncated to it.
        """
        for name, value in (
            ("bm25_limit", bm25_limit),
            ("semantic_limit", semantic_limit),
            ("limit", limit),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        plan = await self.plan(question)
        if not plan.retrieval_queries:
            return []
        return await anyio.to_thread.run_sync(
            partial(
                self._retrieve,
                plan,
                bm25_limit=bm25_limit,
                semantic_limit=semantic_limit,
                limit=limit,
            )
        )

    def _retrieve(
        self,
        plan: QueryPlan,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]:
        rankings = [
            list(
                self._retriever.search(
                    query,
                    bm25_limit=bm25_limit,
                    semantic_limit=semantic_limit,
                    limit=limit,
                )
            )
            for query in plan.retrieval_queries
        ]
        if len(rankings) == 1:
            return rankings[0]
        return fuse_rankings(rankings, limit=limit, rrf_k=self._rrf_k)
