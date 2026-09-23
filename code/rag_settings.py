from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_COLLECTION = "mcwiki_chunks"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_QDRANT_URL = "http://127.0.0.1:6335"
DEFAULT_BM25_PATH = Path(__file__).resolve().parents[1] / "data/processed/bm25.db"
DEFAULT_BM25_LIMIT = 20
DEFAULT_SEMANTIC_LIMIT = 20
DEFAULT_EVIDENCE_LIMIT = 8
DEFAULT_MAX_CONTEXT_CHARS = 12_000
DEFAULT_EVIDENCE_STRATEGY = "adjacent_merge"
DEFAULT_DEEPSEEK_READ_TIMEOUT = 90.0
DEFAULT_COOKIE_MAX_AGE = 15_552_000
DEFAULT_ANSWER_TEMPERATURE = 0.2
DEFAULT_ANSWER_THINKING_TYPE = "disabled"
ORIGINAL_QUERY_STRATEGY = "original"
STEP_BACK_QUERY_STRATEGY = "step_back"
SUPPORTED_QUERY_STRATEGIES = (ORIGINAL_QUERY_STRATEGY, STEP_BACK_QUERY_STRATEGY)
DEFAULT_QUERY_STRATEGY = ORIGINAL_QUERY_STRATEGY
MIN_RETRIEVAL_QUERIES = 1
MAX_RETRIEVAL_QUERIES = 2
DEFAULT_QUERY_PLAN_TIMEOUT = 15.0
INSUFFICIENT_EVIDENCE_MESSAGE = "现有知识库没有足够资料支持可靠回答。"


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    value = int(values.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_int(
    values: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = int(values.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    value = float(values.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class RetrievalSettings:
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection: str = DEFAULT_COLLECTION
    vector_size: int = DEFAULT_VECTOR_SIZE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    bm25_path: Path = DEFAULT_BM25_PATH
    bm25_limit: int = DEFAULT_BM25_LIMIT
    semantic_limit: int = DEFAULT_SEMANTIC_LIMIT
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    evidence_strategy: str = DEFAULT_EVIDENCE_STRATEGY
    query_strategy: str = DEFAULT_QUERY_STRATEGY
    max_retrieval_queries: int = MAX_RETRIEVAL_QUERIES
    query_plan_timeout: float = DEFAULT_QUERY_PLAN_TIMEOUT

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RetrievalSettings:
        values = os.environ if environ is None else environ
        vector_size = _positive_int(values, "MCWIKI_VECTOR_SIZE", DEFAULT_VECTOR_SIZE)
        evidence_strategy = values.get(
            "MCWIKI_EVIDENCE_STRATEGY", DEFAULT_EVIDENCE_STRATEGY
        ).strip()
        if evidence_strategy not in {
            "ranked_first",
            "adjacent_merge",
        }:
            raise ValueError("MCWIKI_EVIDENCE_STRATEGY is invalid")
        query_strategy = values.get(
            "MCWIKI_QUERY_STRATEGY", DEFAULT_QUERY_STRATEGY
        ).strip()
        if query_strategy not in SUPPORTED_QUERY_STRATEGIES:
            raise ValueError("MCWIKI_QUERY_STRATEGY is invalid")
        return cls(
            qdrant_url=values.get("MCWIKI_QDRANT_URL", DEFAULT_QDRANT_URL).strip(),
            collection=values.get("MCWIKI_QDRANT_COLLECTION", DEFAULT_COLLECTION).strip(),
            vector_size=vector_size,
            embedding_model=values.get("MCWIKI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip(),
            bm25_path=Path(values.get("MCWIKI_BM25_DB_PATH", str(DEFAULT_BM25_PATH))).resolve(),
            bm25_limit=_positive_int(values, "MCWIKI_BM25_LIMIT", DEFAULT_BM25_LIMIT),
            semantic_limit=_positive_int(values, "MCWIKI_SEMANTIC_LIMIT", DEFAULT_SEMANTIC_LIMIT),
            evidence_limit=_positive_int(values, "MCWIKI_EVIDENCE_LIMIT", DEFAULT_EVIDENCE_LIMIT),
            max_context_chars=_positive_int(values, "MCWIKI_CONTEXT_BUDGET", DEFAULT_MAX_CONTEXT_CHARS),
            evidence_strategy=evidence_strategy,
            query_strategy=query_strategy,
            max_retrieval_queries=_bounded_int(
                values,
                "MCWIKI_MAX_RETRIEVAL_QUERIES",
                MAX_RETRIEVAL_QUERIES,
                minimum=MIN_RETRIEVAL_QUERIES,
                maximum=MAX_RETRIEVAL_QUERIES,
            ),
            query_plan_timeout=_positive_float(
                values, "MCWIKI_QUERY_PLAN_TIMEOUT", DEFAULT_QUERY_PLAN_TIMEOUT
            ),
        )


@dataclass(frozen=True)
class ServiceSettings:
    deepseek_read_timeout: float = DEFAULT_DEEPSEEK_READ_TIMEOUT
    cookie_max_age: int = DEFAULT_COOKIE_MAX_AGE
    answer_temperature: float = DEFAULT_ANSWER_TEMPERATURE
    answer_thinking_type: str = DEFAULT_ANSWER_THINKING_TYPE

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServiceSettings:
        values = os.environ if environ is None else environ
        timeout = float(values.get("MCWIKI_DEEPSEEK_READ_TIMEOUT", DEFAULT_DEEPSEEK_READ_TIMEOUT))
        temperature = float(values.get("MCWIKI_ANSWER_TEMPERATURE", DEFAULT_ANSWER_TEMPERATURE))
        thinking = values.get("MCWIKI_ANSWER_THINKING_TYPE", DEFAULT_ANSWER_THINKING_TYPE).strip()
        if timeout <= 0 or not 0 <= temperature <= 2 or not thinking:
            raise ValueError("invalid answer service settings")
        return cls(
            deepseek_read_timeout=timeout,
            cookie_max_age=_positive_int(values, "MCWIKI_COOKIE_MAX_AGE", DEFAULT_COOKIE_MAX_AGE),
            answer_temperature=temperature,
            answer_thinking_type=thinking,
        )
