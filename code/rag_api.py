from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client import QdrantClient

from rag_answer import (
    AnswerEvidence,
    DeepSeekAnswerClient,
    DeepSeekSettings,
    ModelTimeoutError,
    ModelUnavailableError,
    build_evidence,
)
from rag_retrieval.bm25 import search_bm25
from rag_retrieval.hybrid import HybridResult, HybridRetriever
from rag_retrieval.semantic import SemanticRetriever


VISITOR_COOKIE_NAME = "mcwiki_visitor_id"
VISITOR_COOKIE_MAX_AGE = 15_552_000
INSUFFICIENT_EVIDENCE_MESSAGE = "现有知识库没有足够资料支持可靠回答。"


class AnswerStreamer(Protocol):
    def stream_answer(
        self,
        question: str,
        evidence: Sequence[AnswerEvidence],
    ) -> AsyncIterator[str]: ...


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


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


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
        deepseek_http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)
        )
        app.state.retriever = retriever
        app.state.answer_client = DeepSeekAnswerClient(
            settings=DeepSeekSettings.from_env(),
            http_client=deepseek_http,
        )
        try:
            yield
        finally:
            await deepseek_http.aclose()
            ollama.close()
            qdrant.close()

    return lifespan


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


def _has_valid_visitor_cookie(request: Request) -> bool:
    value = request.cookies.get(VISITOR_COOKIE_NAME)
    if value is None:
        return False
    try:
        return UUID(value).version == 4
    except (ValueError, AttributeError):
        return False


def _public_source(item: AnswerEvidence) -> dict[str, Any]:
    return {
        "id": item.id,
        "chunkId": item.chunk_id,
        "title": item.title,
        "url": item.url,
        "excerpt": item.excerpt,
    }


def create_app(
    *,
    retriever: HybridRetriever | None = None,
    answer_client: AnswerStreamer | None = None,
    cors_origins: Sequence[str] | None = None,
    cookie_secure: bool | None = None,
) -> FastAPI:
    use_default_lifespan = retriever is None and answer_client is None
    app = FastAPI(
        title="MCwiki RAG API",
        lifespan=_default_lifespan() if use_default_lifespan else None,
    )
    if retriever is not None:
        app.state.retriever = retriever
    if answer_client is not None:
        app.state.answer_client = answer_client

    configured_origins = list(
        cors_origins
        if cors_origins is not None
        else [os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    secure_cookie = (
        cookie_secure
        if cookie_secure is not None
        else os.environ.get("MCWIKI_COOKIE_SECURE", "false").lower() == "true"
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            422,
            "VALIDATION_ERROR",
            "请求内容不符合要求。",
        )

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

    @app.post("/answers", response_model=None)
    async def create_answer(
        body: AnswerRequest,
        request: Request,
    ) -> Response:
        current_retriever = getattr(request.app.state, "retriever", None)
        current_answer_client = getattr(request.app.state, "answer_client", None)
        if current_retriever is None or current_answer_client is None:
            return _error_response(
                503,
                "CONFIGURATION_ERROR",
                "回答服务尚未正确配置。",
            )

        try:
            results = await anyio.to_thread.run_sync(
                partial(
                    current_retriever.search,
                    body.question,
                    bm25_limit=20,
                    semantic_limit=20,
                    limit=8,
                )
            )
        except Exception:
            return _error_response(
                503,
                "RETRIEVAL_UNAVAILABLE",
                "知识库检索暂时不可用，请稍后重试。",
            )

        evidence = build_evidence(results)

        async def event_stream() -> AsyncIterator[str]:
            yield _sse_event("meta", {"question": body.question})
            yield _sse_event(
                "sources",
                {"items": [_public_source(item) for item in evidence]},
            )
            if not evidence:
                yield _sse_event(
                    "delta",
                    {"text": INSUFFICIENT_EVIDENCE_MESSAGE},
                )
                yield _sse_event(
                    "done",
                    {"status": "insufficientEvidence"},
                )
                return

            try:
                async for text in current_answer_client.stream_answer(
                    body.question,
                    evidence,
                ):
                    yield _sse_event("delta", {"text": text})
                yield _sse_event("done", {"status": "answered"})
            except ModelTimeoutError:
                yield _sse_event(
                    "error",
                    {
                        "code": "REQUEST_TIMEOUT",
                        "message": "回答生成超时，请稍后重试。",
                    },
                )
            except ModelUnavailableError:
                yield _sse_event(
                    "error",
                    {
                        "code": "MODEL_UNAVAILABLE",
                        "message": "回答生成中断，请稍后重试。",
                    },
                )
            except Exception:
                yield _sse_event(
                    "error",
                    {
                        "code": "INTERNAL_ERROR",
                        "message": "回答生成中断，请稍后重试。",
                    },
                )

        response = StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
        if not _has_valid_visitor_cookie(request):
            response.set_cookie(
                key=VISITOR_COOKIE_NAME,
                value=str(uuid4()),
                max_age=VISITOR_COOKIE_MAX_AGE,
                path="/",
                secure=secure_cookie,
                httponly=True,
                samesite="lax",
            )
        return response

    return app


app = create_app()
