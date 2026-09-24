from __future__ import annotations

import inspect
import logging
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from typing import Any, Protocol, cast

import anyio

from rag_retrieval.hybrid import HybridResult


logger = logging.getLogger(__name__)

NOOP_RERANKER = "none"
CROSS_ENCODER_RERANKER = "cross_encoder"
SUPPORTED_RERANKERS: tuple[str, ...] = (NOOP_RERANKER, CROSS_ENCODER_RERANKER)
DEFAULT_RERANKER = NOOP_RERANKER
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANK_BATCH_SIZE = 16
DEFAULT_RERANK_MAX_LENGTH = 512

READY = "ok"
DISABLED = "disabled"
UNAVAILABLE = "unavailable"


class Reranker(Protocol):
    """Reorder an already recalled candidate pool.

    A reranker can only change the order of the candidates it is given and can
    never add a chunk that the retrieval stage did not recall. It returns at most
    ``limit`` candidates, and every returned candidate keeps the ``chunk_id`` and
    retrieval metadata it arrived with.
    """

    def rerank(
        self,
        question: str,
        candidates: Sequence[HybridResult],
        *,
        limit: int,
    ) -> list[HybridResult]: ...

    def readiness(self) -> str:
        """Report ``ok``, ``disabled`` or ``unavailable`` for the readiness probe."""
        ...


class PairScorer(Protocol):
    """A Cross-Encoder that scores one ``(question, document)`` pair per input."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class CandidateRecall(Protocol):
    """The async recall stage: one question in, a fused candidate pool out."""

    async def search(
        self,
        query: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]: ...


ScorerLoader = Callable[[str], PairScorer]


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")


def _validate_limits(**limits: int) -> None:
    for name, value in limits.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")


def _unique_by_chunk_id(candidates: Sequence[HybridResult]) -> list[HybridResult]:
    """Keep the first occurrence of each chunk, with its own metadata."""
    seen: set[str] = set()
    unique: list[HybridResult] = []
    for candidate in candidates:
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)
        unique.append(candidate)
    return unique


class NoopReranker:
    """The disabled switch: the fused RRF order is already the final order."""

    __slots__ = ()

    def readiness(self) -> str:
        return DISABLED

    def rerank(
        self,
        question: str,
        candidates: Sequence[HybridResult],
        *,
        limit: int,
    ) -> list[HybridResult]:
        _validate_limit(limit)
        return _unique_by_chunk_id(candidates)[:limit]


def load_cross_encoder(model_name: str) -> PairScorer:
    """Load a real Cross-Encoder once; torch stays an optional dependency."""
    from sentence_transformers import CrossEncoder

    model: Any = CrossEncoder(model_name, max_length=DEFAULT_RERANK_MAX_LENGTH)
    return cast(PairScorer, model)


def _prediction_options(scorer: PairScorer, batch_size: int) -> dict[str, Any]:
    """Pass batch and progress options only when the installed scorer accepts them."""
    try:
        parameters = inspect.signature(scorer.predict).parameters
    except (TypeError, ValueError):
        return {}
    options: dict[str, Any] = {}
    if "batch_size" in parameters:
        options["batch_size"] = batch_size
    if "show_progress_bar" in parameters:
        options["show_progress_bar"] = False
    return options


def _pair(question: str, candidate: HybridResult) -> tuple[str, str]:
    return question, f"{candidate.title}\n{candidate.text}"


def _coerce_scores(scores: Sequence[float], expected: int) -> list[float] | None:
    try:
        values = [float(score) for score in scores]
    except (TypeError, ValueError):
        return None
    if len(values) != expected or not all(math.isfinite(value) for value in values):
        return None
    return values


def _reordered(
    candidates: Sequence[HybridResult],
    scores: Sequence[float],
    *,
    limit: int,
) -> list[HybridResult]:
    """Order by descending score; equal scores keep the incoming RRF order."""
    ranked = sorted(
        enumerate(candidates),
        key=lambda entry: (-scores[entry[0]], entry[0]),
    )
    return [
        replace(candidate, reranker_score=scores[index])
        for index, candidate in ranked[:limit]
    ]


class CrossEncoderReranker:
    """Rescore the candidate pool with a real Cross-Encoder.

    The model is loaded once, when the reranker is constructed, and every failure
    after that — a missing model, a missing library or a broken inference call —
    degrades to the incoming RRF order instead of failing the answer request.
    """

    __slots__ = ("_batch_size", "_load_error", "_lock", "_model_name", "_options", "_scorer")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        scorer: PairScorer | None = None,
        loader: ScorerLoader | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self._model_name = model_name.strip()
        self._batch_size = batch_size
        self._scorer: PairScorer | None = None
        self._options: dict[str, Any] = {}
        self._load_error: str | None = None
        self._lock = threading.Lock()

        try:
            loaded = scorer if scorer is not None else (loader or load_cross_encoder)(
                self._model_name
            )
        except Exception as error:
            self._load_error = f"{type(error).__name__}: {error}"
            logger.warning(
                "Cross-Encoder %s could not be loaded (%s); reranking falls back to RRF order",
                self._model_name,
                self._load_error,
            )
            return
        self._scorer = loaded
        self._options = _prediction_options(loaded, batch_size)
        logger.info("Cross-Encoder %s is loaded and ready", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def readiness(self) -> str:
        return READY if self._scorer is not None else UNAVAILABLE

    def rerank(
        self,
        question: str,
        candidates: Sequence[HybridResult],
        *,
        limit: int,
    ) -> list[HybridResult]:
        _validate_limit(limit)
        unique = _unique_by_chunk_id(candidates)
        normalized_question = question.strip()
        if not unique or self._scorer is None or not normalized_question:
            return unique[:limit]

        pairs = [_pair(normalized_question, candidate) for candidate in unique]
        try:
            # Every request shares one model instance, so inference is serialized
            # instead of letting concurrent calls interleave inside the model.
            with self._lock:
                scores = list(self._scorer.predict(pairs, **self._options))
        except Exception as error:
            logger.warning(
                "Cross-Encoder reranking failed (%s); keeping the RRF order",
                error,
            )
            return unique[:limit]

        values = _coerce_scores(scores, len(unique))
        if values is None:
            logger.warning(
                "Cross-Encoder returned unusable scores for %d candidates; "
                "keeping the RRF order",
                len(unique),
            )
            return unique[:limit]
        return _reordered(unique, values, limit=limit)


def build_reranker(
    strategy: str,
    *,
    model: str = DEFAULT_RERANKER_MODEL,
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
) -> Reranker:
    """Return the reranker for a configured strategy."""
    if strategy == NOOP_RERANKER:
        return NoopReranker()
    if strategy != CROSS_ENCODER_RERANKER:
        raise ValueError(f"unsupported reranker: {strategy}")
    return CrossEncoderReranker(model_name=model, batch_size=batch_size)


class RerankingRetriever:
    """The single candidate-recall-then-rerank entry of the answer pipeline.

    It recalls a candidate pool of ``candidate_limit`` candidates, lets the
    reranker put them in order, and only then truncates to ``limit``. Later
    stages — a corrective second round, for example — pass their own candidates
    through the same reranker instead of writing another ordering rule.
    """

    __slots__ = ("_reranker", "_retriever")

    def __init__(self, *, retriever: CandidateRecall, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    async def recall(
        self,
        question: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]:
        """Recall a candidate pool without reranking it.

        A later stage that merges several rounds of candidates recalls each round
        here and hands the merged pool to :meth:`rerank` once.
        """
        _validate_limits(bm25_limit=bm25_limit, semantic_limit=semantic_limit, limit=limit)
        return await self._retriever.search(
            question,
            bm25_limit=bm25_limit,
            semantic_limit=semantic_limit,
            limit=limit,
        )

    async def rerank(
        self,
        question: str,
        candidates: Sequence[HybridResult],
        *,
        limit: int,
    ) -> list[HybridResult]:
        """Score an externally recalled pool — the second round of a later stage."""
        return await anyio.to_thread.run_sync(
            partial(self._reranker.rerank, question, candidates, limit=limit)
        )

    async def search(
        self,
        question: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        candidate_limit: int,
        limit: int,
    ) -> list[HybridResult]:
        _validate_limits(
            bm25_limit=bm25_limit,
            semantic_limit=semantic_limit,
            candidate_limit=candidate_limit,
            limit=limit,
        )
        pool_limit = max(candidate_limit, limit)
        candidates = await self.recall(
            question,
            bm25_limit=bm25_limit,
            semantic_limit=semantic_limit,
            limit=pool_limit,
        )
        return await anyio.to_thread.run_sync(
            partial(self._reranker.rerank, question, candidates, limit=limit)
        )
