from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .bm25 import BM25Result
from .semantic import SemanticResult


RetrievalResult = BM25Result | SemanticResult
SearchRoute = Callable[[str, int], list[RetrievalResult]]


@dataclass(frozen=True, slots=True)
class HybridResult:
    chunk_id: str
    title: str
    text: str
    source: str
    score: float
    bm25_rank: int | None
    semantic_rank: int | None
    document_id: str | None = None
    chunk_index: int | None = None


@dataclass(slots=True)
class _Candidate:
    result: RetrievalResult
    score: float = 0.0
    bm25_rank: int | None = None
    semantic_rank: int | None = None


def fuse_rrf(
    bm25_results: Sequence[BM25Result],
    semantic_results: Sequence[SemanticResult],
    *,
    limit: int = 10,
    rrf_k: int = 60,
) -> list[HybridResult]:
    """Fuse ranked BM25 and semantic results using reciprocal rank fusion."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if rrf_k < 0:
        raise ValueError("rrf_k must be zero or greater")

    candidates: dict[str, _Candidate] = {}

    def add_route(
        results: Sequence[RetrievalResult],
        rank_field: str,
    ) -> None:
        seen: set[str] = set()
        rank = 0
        for result in results:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            rank += 1
            candidate = candidates.setdefault(
                result.chunk_id,
                _Candidate(result=result),
            )
            candidate.score += 1 / (rrf_k + rank)
            setattr(candidate, rank_field, rank)

    add_route(bm25_results, "bm25_rank")
    add_route(semantic_results, "semantic_rank")

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.score,
            min(
                rank
                for rank in (candidate.bm25_rank, candidate.semantic_rank)
                if rank is not None
            ),
            candidate.result.chunk_id,
        ),
    )
    return [
        HybridResult(
            chunk_id=candidate.result.chunk_id,
            title=candidate.result.title,
            text=candidate.result.text,
            source=candidate.result.source,
            score=candidate.score,
            bm25_rank=candidate.bm25_rank,
            semantic_rank=candidate.semantic_rank,
            document_id=candidate.result.document_id,
            chunk_index=candidate.result.chunk_index,
        )
        for candidate in ordered[:limit]
    ]


class HybridRetriever:
    """Run lexical and semantic retrieval, then fuse both rankings."""

    def __init__(
        self,
        *,
        search_bm25: SearchRoute,
        search_semantic: SearchRoute,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be zero or greater")
        self._search_bm25 = search_bm25
        self._search_semantic = search_semantic
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        bm25_limit: int = 20,
        semantic_limit: int = 20,
        limit: int = 10,
    ) -> list[HybridResult]:
        for name, value in (
            ("bm25_limit", bm25_limit),
            ("semantic_limit", semantic_limit),
            ("limit", limit),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        normalized_query = query.strip()
        if not normalized_query:
            return []

        bm25_results = self._search_bm25(normalized_query, bm25_limit)
        semantic_results = self._search_semantic(normalized_query, semantic_limit)
        return fuse_rrf(
            bm25_results,
            semantic_results,
            limit=limit,
            rrf_k=self._rrf_k,
        )
