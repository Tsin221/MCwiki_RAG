from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol
from uuid import UUID, uuid4

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException

from rag_answer import (
    AnswerEvidence,
    AnswerConfigurationError,
    DeepSeekAnswerClient,
    DeepSeekSettings,
    ModelTimeoutError,
    ModelUnavailableError,
    build_evidence,
)
from rag_correction import (
    CorrectiveCoordinator,
    EvidenceAssessor,
    build_evidence_assessor,
)
from rag_query import OriginalQueryPlanner, QueryPlanner, build_query_planner
from rag_reranker import (
    DISABLED as RERANKER_DISABLED,
    READY as RERANKER_READY,
    NoopReranker,
    Reranker,
    RerankingRetriever,
    build_reranker,
)
from rag_retrieval.hybrid import HybridResult, HybridRetriever
from rag_retrieval.factory import build_default_retriever as _build_default_retriever
from rag_retrieval.multi_query import MultiQueryRetriever
from rag_settings import (
    CORRECTIVE_ENABLED,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    RetrievalSettings,
    ServiceSettings,
    VERIFICATION_ENABLED,
)
from rag_verification import (
    VERIFICATION_PASSED,
    VERIFICATION_UNAVAILABLE,
    VERIFICATION_UNSUPPORTED,
    AnswerVerification,
    AnswerVerificationCoordinator,
    AnswerVerifier,
    build_answer_verifier,
)


VISITOR_COOKIE_NAME = "mcwiki_visitor_id"
logger = logging.getLogger(__name__)


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


def _retrieval_checks(app: FastAPI) -> dict[str, str]:
    settings: RetrievalSettings | None = getattr(app.state, "retrieval_settings", None)
    if settings is None:
        available = "ok" if getattr(app.state, "retriever", None) else "unavailable"
        return {"bm25": available, "qdrant": available}
    bm25 = "ok" if settings.bm25_path.is_file() else "unavailable"
    try:
        info = app.state.qdrant.get_collection(settings.collection)
        vectors = info.config.params.vectors
        size = vectors.size if hasattr(vectors, "size") else None
        status = getattr(info.status, "value", info.status)
        qdrant = "ok" if size == settings.vector_size and status == "green" else "mismatch"
    except Exception:
        logger.exception("Qdrant collection check failed: %s at %s", settings.collection, settings.qdrant_url)
        qdrant = "unavailable"
    return {"bm25": bm25, "qdrant": qdrant}


def _reranker_check(app: FastAPI) -> str:
    """Report whether a configured Cross-Encoder is loaded and usable."""
    reranker: Reranker | None = getattr(app.state, "reranker", None)
    return RERANKER_DISABLED if reranker is None else reranker.readiness()


def _default_lifespan() -> Any:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = RetrievalSettings.from_env()
        retriever, ollama, qdrant = _build_default_retriever(settings)
        service_settings: ServiceSettings = app.state.service_settings
        deepseek_http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=service_settings.deepseek_read_timeout,
                write=30.0,
                pool=10.0,
            )
        )
        app.state.retriever = retriever
        app.state.qdrant = qdrant
        app.state.retrieval_settings = settings
        checks = _retrieval_checks(app)
        if any(value != "ok" for value in checks.values()):
            logger.error("Knowledge base is not ready at startup: %s", checks)
        answer_settings: DeepSeekSettings | None = None
        try:
            answer_settings = DeepSeekSettings.from_env()
            app.state.answer_client = DeepSeekAnswerClient(
                settings=answer_settings,
                http_client=deepseek_http,
                temperature=service_settings.answer_temperature,
                thinking_type=service_settings.answer_thinking_type,
            )
        except AnswerConfigurationError:
            logger.warning("DeepSeek answer service is unconfigured")
            app.state.answer_client = None
        planner = getattr(app.state, "query_planner", None) or build_query_planner(
            settings.query_strategy,
            settings=answer_settings,
            http_client=deepseek_http,
            timeout=settings.query_plan_timeout,
            max_queries=settings.max_retrieval_queries,
            thinking_type=settings.query_plan_thinking_type,
        )
        reranker = getattr(app.state, "reranker", None) or build_reranker(
            settings.reranker, model=settings.reranker_model
        )
        app.state.reranker = reranker
        app.state.query_retriever = RerankingRetriever(
            retriever=MultiQueryRetriever(
                retriever=retriever,
                planner=planner,
            ),
            reranker=reranker,
        )
        assessor = getattr(app.state, "corrective_assessor", None)
        if assessor is None:
            assessor = build_evidence_assessor(
                settings.corrective,
                settings=answer_settings,
                http_client=deepseek_http,
                timeout=settings.corrective_timeout,
            )
        app.state.corrective_assessor = assessor
        verifier = getattr(app.state, "answer_verifier", None)
        if verifier is None:
            verifier = build_answer_verifier(
                settings.answer_verification,
                settings=answer_settings,
                http_client=deepseek_http,
                timeout=settings.verification_timeout,
            )
        app.state.answer_verifier = verifier
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
        "componentChunkIds": list(item.component_chunk_ids),
        "title": item.title,
        "url": item.url,
        "excerpt": item.excerpt,
    }


async def _collect_evidence(
    question: str,
    *,
    query_retriever: RerankingRetriever,
    retrieval_settings: RetrievalSettings,
    assessor: EvidenceAssessor | None,
) -> tuple[list[AnswerEvidence], bool]:
    """Return the evidence the answer model may see, and whether it suffices.

    Corrective retrieval runs only when it is switched on and an assessor exists;
    its coordinator decides both rounds and the final evidence. Without it the
    single-round baseline runs unchanged, and evidence is sufficient exactly when
    something was retrieved.
    """
    if assessor is not None and retrieval_settings.corrective == CORRECTIVE_ENABLED:
        result = await CorrectiveCoordinator(
            retriever=query_retriever,
            assessor=assessor,
            settings=retrieval_settings,
        ).run(question)
        return list(result.evidence), result.sufficient

    results = await query_retriever.search(
        question,
        bm25_limit=retrieval_settings.bm25_limit,
        semantic_limit=retrieval_settings.semantic_limit,
        candidate_limit=retrieval_settings.candidate_limit,
        limit=retrieval_settings.evidence_limit,
    )
    evidence = build_evidence(
        results,
        max_context_chars=retrieval_settings.max_context_chars,
        strategy=retrieval_settings.evidence_strategy,
    )
    return evidence, bool(evidence)


def _answer_verifier(
    settings: RetrievalSettings,
    verifier: AnswerVerifier | None,
    answer_client: Any,
) -> AnswerVerifier | None:
    """Return the verifier only when it is switched on and its dependencies exist.

    Verification needs both a verifier and an answer client that can hand over a
    complete draft; without either one the request keeps the streaming baseline
    instead of failing.
    """
    if settings.answer_verification != VERIFICATION_ENABLED:
        return None
    if verifier is None:
        logger.warning(
            "Answer verification is enabled without a verifier; "
            "sending the draft without a claim check"
        )
        return None
    if not callable(getattr(answer_client, "collect_answer", None)):
        logger.warning(
            "Answer verification needs an answer client that can collect a draft; "
            "sending the draft without a claim check"
        )
        return None
    return verifier


def _verification_status(verification: AnswerVerification | None) -> str:
    """The coarse public state; never the model's reasoning or internal scores."""
    return VERIFICATION_UNAVAILABLE if verification is None else VERIFICATION_PASSED


def create_app(
    *,
    retriever: HybridRetriever | None = None,
    answer_client: AnswerStreamer | None = None,
    query_planner: QueryPlanner | None = None,
    reranker: Reranker | None = None,
    corrective_assessor: EvidenceAssessor | None = None,
    answer_verifier: AnswerVerifier | None = None,
    cors_origins: Sequence[str] | None = None,
    cookie_secure: bool | None = None,
) -> FastAPI:
    use_default_lifespan = retriever is None and answer_client is None
    app = FastAPI(
        title="MCwiki RAG API",
        lifespan=_default_lifespan() if use_default_lifespan else None,
    )
    app.state.service_settings = ServiceSettings.from_env()
    if query_planner is not None:
        app.state.query_planner = query_planner
    if reranker is not None:
        app.state.reranker = reranker
    if corrective_assessor is not None:
        app.state.corrective_assessor = corrective_assessor
    if answer_verifier is not None:
        app.state.answer_verifier = answer_verifier
    if retriever is not None:
        app.state.retriever = retriever
        if reranker is None:
            app.state.reranker = NoopReranker()
        app.state.query_retriever = RerankingRetriever(
            retriever=MultiQueryRetriever(
                retriever=retriever,
                planner=query_planner or OriginalQueryPlanner(),
            ),
            reranker=app.state.reranker,
        )
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

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, error: HTTPException) -> JSONResponse:
        if error.status_code == 404:
            return _error_response(404, "NOT_FOUND", "请求的接口不存在。")
        if error.status_code == 405:
            return _error_response(405, "METHOD_NOT_ALLOWED", "此接口不支持该请求方式。")
        return _error_response(error.status_code, "HTTP_ERROR", "请求失败。")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"api": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> Response:
        checks = _retrieval_checks(request.app)
        answer_model = "configured" if getattr(request.app.state, "answer_client", None) else "unavailable"
        reranker = _reranker_check(request.app)
        available = (
            all(value == "ok" for value in checks.values())
            and answer_model == "configured"
            and reranker in {RERANKER_READY, RERANKER_DISABLED}
        )
        return JSONResponse(
            status_code=200 if available else 503,
            content={
                "api": "ok",
                **checks,
                "answer_model": answer_model,
                "reranker": reranker,
                "ready": available,
            },
        )

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse | Response:
        if any(value != "ok" for value in _retrieval_checks(request.app).values()):
            return _error_response(503, "RETRIEVAL_UNAVAILABLE", "知识库检索暂时不可用，请稍后重试。")
        try:
            results: list[HybridResult] = request.app.state.retriever.search(
                body.query,
                bm25_limit=body.bm25_limit,
                semantic_limit=body.semantic_limit,
                limit=body.limit,
            )
        except Exception:
            logger.exception("Search retrieval failed")
            return _error_response(503, "RETRIEVAL_UNAVAILABLE", "知识库检索暂时不可用，请稍后重试。")
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
        current_query_retriever = getattr(request.app.state, "query_retriever", None)
        current_answer_client = getattr(request.app.state, "answer_client", None)
        current_assessor = getattr(request.app.state, "corrective_assessor", None)
        retrieval_settings: RetrievalSettings = (
            getattr(request.app.state, "retrieval_settings", None)
            or RetrievalSettings.from_env()
        )
        if current_answer_client is None:
            return _error_response(
                503,
                "CONFIGURATION_ERROR",
                "回答服务尚未正确配置。",
            )
        current_verifier = _answer_verifier(
            retrieval_settings,
            getattr(request.app.state, "answer_verifier", None),
            current_answer_client,
        )
        checks = await anyio.to_thread.run_sync(_retrieval_checks, request.app)
        if (
            current_retriever is None
            or current_query_retriever is None
            or any(value != "ok" for value in checks.values())
        ):
            return _error_response(503, "RETRIEVAL_UNAVAILABLE", "知识库检索暂时不可用，请稍后重试。")
        if retrieval_settings.corrective == CORRECTIVE_ENABLED and current_assessor is None:
            logger.warning(
                "Corrective retrieval is enabled without an assessor; "
                "answering from a single retrieval round"
            )

        try:
            evidence, sufficient = await _collect_evidence(
                body.question,
                query_retriever=current_query_retriever,
                retrieval_settings=retrieval_settings,
                assessor=current_assessor,
            )
        except Exception:
            logger.exception("Answer retrieval failed")
            return _error_response(
                503,
                "RETRIEVAL_UNAVAILABLE",
                "知识库检索暂时不可用，请稍后重试。",
            )

        async def event_stream() -> AsyncIterator[str]:
            # `sources` always describes the evidence handed to the answer model, so
            # a refused answer announces none of the evidence it did not send.
            published = evidence if sufficient else []
            yield _sse_event("meta", {"question": body.question})
            yield _sse_event(
                "sources",
                {"items": [_public_source(item) for item in published]},
            )
            if not published:
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
                if current_verifier is None:
                    # Without verification the answer streams straight through, exactly
                    # as it did before this layer existed.
                    async for text in current_answer_client.stream_answer(
                        body.question,
                        evidence,
                    ):
                        yield _sse_event("delta", {"text": text})
                    yield _sse_event("done", {"status": "answered"})
                    return
                result = await AnswerVerificationCoordinator(
                    answerer=current_answer_client,
                    verifier=current_verifier,
                    max_attempts=retrieval_settings.max_answer_attempts,
                ).run(body.question, evidence)
            except ModelTimeoutError:
                logger.exception("Answer generation timed out")
                yield _sse_event(
                    "error",
                    {
                        "code": "REQUEST_TIMEOUT",
                        "message": "回答生成超时，请稍后重试。",
                    },
                )
            except ModelUnavailableError:
                logger.exception("Answer model unavailable")
                yield _sse_event(
                    "error",
                    {
                        "code": "MODEL_UNAVAILABLE",
                        "message": "回答生成中断，请稍后重试。",
                    },
                )
            except Exception:
                logger.exception("Answer stream failed")
                yield _sse_event(
                    "error",
                    {
                        "code": "INTERNAL_ERROR",
                        "message": "回答生成中断，请稍后重试。",
                    },
                )
            else:
                # Only the verified text is ever sent: a draft that failed verification
                # is replaced by the refusal instead of reaching the browser.
                if result.withheld:
                    yield _sse_event("delta", {"text": INSUFFICIENT_EVIDENCE_MESSAGE})
                    yield _sse_event(
                        "done",
                        {
                            "status": "insufficientEvidence",
                            "verification": VERIFICATION_UNSUPPORTED,
                        },
                    )
                    return
                if result.answer:
                    yield _sse_event("delta", {"text": result.answer})
                yield _sse_event(
                    "done",
                    {
                        "status": "answered",
                        "verification": _verification_status(result.verification),
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
                max_age=request.app.state.service_settings.cookie_max_age,
                path="/",
                secure=secure_cookie,
                httponly=True,
                samesite="lax",
            )
        return response

    return app


app = create_app()
