from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client import QdrantClient

from rag_retrieval.bm25 import search_bm25
from rag_retrieval.hybrid import HybridResult, HybridRetriever
from rag_retrieval.semantic import SemanticRetriever


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=10, ge=1, le=100)
    bm25_limit: int = Field(default=20, ge=1, le=100, alias="bm25Limit")
    semantic_limit: int = Field(default=20, ge=1, le=100, alias="semanticLimit")

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class SearchResultResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(alias="chunkId")
    title: str
    text: str
    source: str
    score: float
    bm25_rank: int | None = Field(alias="bm25Rank")
    semantic_rank: int | None = Field(alias="semanticRank")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]


def _default_bm25_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed" / "bm25.db"


def _build_default_retriever() -> tuple[HybridRetriever, httpx.Client, QdrantClient]:
    ollama = httpx.Client(timeout=60.0)
    qdrant = QdrantClient(url="http://127.0.0.1:6333", timeout=60.0)
    semantic = SemanticRetriever(
        ollama_client=ollama,
        qdrant_client=qdrant,
    )
    bm25_path = _default_bm25_path()
    hybrid = HybridRetriever(
        search_bm25=lambda query, limit: search_bm25(
            bm25_path,
            query,
            limit=limit,
        ),
        search_semantic=lambda query, limit: semantic.search(
            query,
            limit=limit,
        ),
    )
    return hybrid, ollama, qdrant


def _default_lifespan() -> Any:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        retriever, ollama, qdrant = _build_default_retriever()
        app.state.retriever = retriever
        try:
            yield
        finally:
            ollama.close()
            qdrant.close()

    return lifespan


def create_app(*, retriever: HybridRetriever | None = None) -> FastAPI:
    app = FastAPI(
        title="MCwiki RAG Search API",
        lifespan=None if retriever is not None else _default_lifespan(),
    )
    if retriever is not None:
        app.state.retriever = retriever

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        results: list[HybridResult] = request.app.state.retriever.search(
            body.query,
            bm25_limit=body.bm25_limit,
            semantic_limit=body.semantic_limit,
            limit=body.limit,
        )
        return SearchResponse(
            query=body.query,
            results=[
                SearchResultResponse(
                    chunk_id=result.chunk_id,
                    title=result.title,
                    text=result.text,
                    source=result.source,
                    score=result.score,
                    bm25_rank=result.bm25_rank,
                    semantic_rank=result.semantic_rank,
                )
                for result in results
            ],
        )

    return app


app = create_app()
