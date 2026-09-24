from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from rag_answer import build_evidence
from rag_evidence import AnswerEvidence
from rag_query import ChatSettings
from rag_retrieval.hybrid import HybridResult
from rag_retrieval.multi_query import fuse_rankings
from rag_settings import (
    CORRECTIVE_DISABLED,
    CORRECTIVE_ENABLED,
    DEFAULT_CORRECTIVE_TIMEOUT,
    MAX_RETRIEVAL_ROUNDS,
    MIN_RETRIEVAL_ROUNDS,
    RetrievalSettings,
)


logger = logging.getLogger(__name__)

DEFAULT_ASSESSMENT_TEMPERATURE = 0.0
# The assessment runs before every generation on a corrective request, so thinking stays
# off: task 01 measured ~7s of extra latency per planning call with no better result.
DEFAULT_ASSESSMENT_THINKING_TYPE = "disabled"
MAX_MISSING_ASPECTS = 8
MAX_REWRITTEN_QUERY_CHARS = 200

ASSESSMENT_SYSTEM_PROMPT = """你是 Minecraft Wiki 检索质量评估器。用户会给出一个问题和本轮检索到的证据，你需要判断这些证据是否足以可靠回答该问题。

规则：
1. 只输出一个 JSON 对象，键名固定为 sufficient、missing_aspects、rewritten_query、reason，不要输出解释、前缀或 Markdown 代码块。
2. sufficient 为布尔值：仅当证据覆盖了问题的全部子问题且没有明显缺口时为 true。
3. missing_aspects 为字符串数组，逐条列出证据缺失的关键方面；sufficient 为 true 时输出空数组。
4. rewritten_query 为字符串或 null：sufficient 为 false 且存在更利于检索缺失方面的措辞时，写一个
   不超过 60 个字符的中文检索查询；sufficient 为 true 或无法改进时输出 null。
5. rewritten_query 只改变检索措辞，不得引入新的事实、版本号或原问题范围之外的假设，也不得只是重复原问题。
6. reason 用一句话说明判断依据，使用简体中文。
"""


class EvidenceAssessmentError(RuntimeError):
    """Raised when an assessment call cannot produce a usable structure."""


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """One judgement of whether the current evidence can answer the question."""

    sufficient: bool
    missing_aspects: tuple[str, ...] = ()
    rewritten_query: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.rewritten_query is None:
            return
        if not self.rewritten_query.strip():
            raise ValueError("rewritten_query must not be blank; use None instead")
        if len(self.rewritten_query) > MAX_REWRITTEN_QUERY_CHARS:
            raise ValueError(
                f"rewritten_query must not exceed {MAX_REWRITTEN_QUERY_CHARS} characters"
            )


@dataclass(frozen=True, slots=True)
class CorrectiveResult:
    """The evidence a bounded corrective retrieval run settled on."""

    evidence: tuple[AnswerEvidence, ...]
    retrieval_rounds: int
    sufficient: bool


class EvidenceAssessor(Protocol):
    """Judge whether the retrieved evidence is enough to answer the question."""

    async def assess(
        self,
        question: str,
        evidence: Sequence[AnswerEvidence],
    ) -> EvidenceAssessment: ...


class CorrectiveRetrieval(Protocol):
    """The recall-then-rerank entry a corrective round retrieves through."""

    async def recall(
        self,
        question: str,
        *,
        bm25_limit: int,
        semantic_limit: int,
        limit: int,
    ) -> list[HybridResult]: ...

    async def rerank(
        self,
        question: str,
        candidates: Sequence[HybridResult],
        *,
        limit: int,
    ) -> list[HybridResult]: ...


def _compact(text: str) -> str:
    return " ".join(text.split())


def _same_query(left: str, right: str) -> bool:
    return "".join(left.split()).casefold() == "".join(right.split()).casefold()


def parse_assessment(content: str) -> EvidenceAssessment:
    """Parse the assessor's JSON reply; every violation raises instead of guessing."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise EvidenceAssessmentError("assessment response is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise EvidenceAssessmentError("assessment response must be a JSON object")
    if "sufficient" not in payload:
        raise EvidenceAssessmentError("assessment response has no sufficient field")
    sufficient = payload["sufficient"]
    if not isinstance(sufficient, bool):
        raise EvidenceAssessmentError("sufficient must be a boolean")

    raw_aspects = payload.get("missing_aspects", [])
    if not isinstance(raw_aspects, list):
        raise EvidenceAssessmentError("missing_aspects must be an array")
    aspects: list[str] = []
    for aspect in raw_aspects:
        if not isinstance(aspect, str):
            raise EvidenceAssessmentError("missing_aspects entries must be strings")
        compact = _compact(aspect)
        if compact:
            aspects.append(compact)
        if len(aspects) >= MAX_MISSING_ASPECTS:
            break

    raw_query = payload.get("rewritten_query")
    if raw_query is not None and not isinstance(raw_query, str):
        raise EvidenceAssessmentError("rewritten_query must be a string or null")
    rewritten = _compact(raw_query) if isinstance(raw_query, str) else ""
    if len(rewritten) > MAX_REWRITTEN_QUERY_CHARS:
        raise EvidenceAssessmentError("rewritten_query is too long")

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise EvidenceAssessmentError("reason must be a string")

    return EvidenceAssessment(
        sufficient=sufficient,
        missing_aspects=tuple(aspects),
        rewritten_query=rewritten or None,
        reason=_compact(reason),
    )


def _completion_content(body: Any) -> str:
    """Read the assistant message out of a chat-completion response body."""
    if not isinstance(body, Mapping):
        raise EvidenceAssessmentError("assessment response must be a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise EvidenceAssessmentError("assessment response has no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        raise EvidenceAssessmentError("assessment response choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise EvidenceAssessmentError("assessment response content must be a string")
    return content


def _evidence_message(question: str, evidence: Sequence[AnswerEvidence]) -> str:
    sections = [
        "\n".join([f"[{item.id}] 标题：{item.title}", f"正文：{item.text}"])
        for item in evidence
    ]
    body = "\n\n".join(sections) if sections else "（本轮没有检索到任何证据。）"
    return f"问题：{question}\n\n本轮证据：\n\n{body}"


class DeepSeekEvidenceAssessor:
    """Ask DeepSeek whether the evidence suffices, and for one better query if not.

    The assessor raises :class:`EvidenceAssessmentError` for every unusable reply.
    Deciding what to do about it belongs to the coordinator, which keeps the
    single-round behaviour instead of failing the answer request.
    """

    __slots__ = (
        "_http_client",
        "_settings",
        "_temperature",
        "_thinking_type",
        "_timeout",
    )

    def __init__(
        self,
        *,
        settings: ChatSettings,
        http_client: httpx.AsyncClient,
        timeout: float = DEFAULT_CORRECTIVE_TIMEOUT,
        temperature: float = DEFAULT_ASSESSMENT_TEMPERATURE,
        thinking_type: str = DEFAULT_ASSESSMENT_THINKING_TYPE,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if not thinking_type.strip():
            raise ValueError("thinking_type must not be blank")
        self._settings = settings
        self._http_client = http_client
        self._timeout = timeout
        self._temperature = temperature
        self._thinking_type = thinking_type

    async def assess(
        self,
        question: str,
        evidence: Sequence[AnswerEvidence],
    ) -> EvidenceAssessment:
        body = await self._request(question, evidence)
        return parse_assessment(_completion_content(body))

    async def _request(
        self,
        question: str,
        evidence: Sequence[AnswerEvidence],
    ) -> Any:
        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": ASSESSMENT_SYSTEM_PROMPT},
                {"role": "user", "content": _evidence_message(question, evidence)},
            ],
            "stream": False,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "thinking": {"type": self._thinking_type},
        }
        headers = {
            "authorization": f"Bearer {self._settings.api_key}",
            "content-type": "application/json",
        }
        try:
            response = await self._http_client.post(
                f"{self._settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise EvidenceAssessmentError("assessment request timed out") from error
        except httpx.HTTPError as error:
            raise EvidenceAssessmentError("assessment request failed") from error
        try:
            return response.json()
        except ValueError as error:
            raise EvidenceAssessmentError(
                "assessment response is not valid JSON"
            ) from error


def build_evidence_assessor(
    strategy: str,
    *,
    settings: ChatSettings | None = None,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_CORRECTIVE_TIMEOUT,
) -> EvidenceAssessor | None:
    """Return the assessor for a configured strategy, or None when corrective is off."""
    if strategy == CORRECTIVE_DISABLED:
        return None
    if strategy != CORRECTIVE_ENABLED:
        raise ValueError(f"unsupported corrective strategy: {strategy}")
    if settings is None or http_client is None:
        logger.warning(
            "Corrective strategy %s has no chat client configured; "
            "keeping the single-round retrieval path",
            strategy,
        )
        return None
    return DeepSeekEvidenceAssessor(
        settings=settings,
        http_client=http_client,
        timeout=timeout,
    )


class CorrectiveCoordinator:
    """Retrieve, assess, and correct at most once before handing evidence on.

    The coordinator owns both the round limit and the evidence budget, so no
    model output can extend the loop: the first round is always retrieved with
    the user's question, and a second round happens only when the assessor
    judges the first round insufficient and offers a usable rewritten query.
    """

    __slots__ = ("_assessor", "_retriever", "_settings")

    def __init__(
        self,
        *,
        retriever: CorrectiveRetrieval,
        assessor: EvidenceAssessor,
        settings: RetrievalSettings,
    ) -> None:
        if not MIN_RETRIEVAL_ROUNDS <= settings.max_retrieval_rounds <= MAX_RETRIEVAL_ROUNDS:
            raise ValueError(
                "max_retrieval_rounds must be between "
                f"{MIN_RETRIEVAL_ROUNDS} and {MAX_RETRIEVAL_ROUNDS}"
            )
        if settings.corrective != CORRECTIVE_ENABLED:
            raise ValueError("corrective retrieval must be enabled")
        self._retriever = retriever
        self._assessor = assessor
        self._settings = settings

    async def run(self, question: str) -> CorrectiveResult:
        settings = self._settings
        pool = await self._retriever.recall(
            question,
            bm25_limit=settings.bm25_limit,
            semantic_limit=settings.semantic_limit,
            limit=settings.candidate_limit,
        )
        evidence = await self._collect(question, pool)
        rounds = 1
        assessment = await self._assess(question, evidence)

        while (
            assessment is not None
            and not assessment.sufficient
            and rounds < settings.max_retrieval_rounds
        ):
            rewritten = assessment.rewritten_query
            if rewritten is None or _same_query(rewritten, question):
                break
            second_round = await self._retriever.recall(
                rewritten,
                bm25_limit=settings.bm25_limit,
                semantic_limit=settings.semantic_limit,
                limit=settings.candidate_limit,
            )
            # One fused pool, one rerank, one evidence budget: the second round is
            # merged into the first instead of being judged on its own.
            pool = fuse_rankings([pool, second_round], limit=settings.candidate_limit)
            evidence = await self._collect(question, pool)
            rounds += 1
            assessment = await self._assess(question, evidence)

        return CorrectiveResult(
            evidence=tuple(evidence),
            retrieval_rounds=rounds,
            # No evidence can never be sufficient, whatever the assessor replied.
            sufficient=bool(evidence) and (assessment is None or assessment.sufficient),
        )

    async def _collect(
        self,
        question: str,
        pool: Sequence[HybridResult],
    ) -> list[AnswerEvidence]:
        ranked = await self._retriever.rerank(
            question,
            pool,
            limit=self._settings.evidence_limit,
        )
        return build_evidence(
            ranked,
            max_context_chars=self._settings.max_context_chars,
            strategy=self._settings.evidence_strategy,
        )

    async def _assess(
        self,
        question: str,
        evidence: Sequence[AnswerEvidence],
    ) -> EvidenceAssessment | None:
        """Assess once; a broken or slow assessor never blocks the answer."""
        try:
            return await self._assessor.assess(question, evidence)
        except Exception as error:
            logger.warning(
                "Evidence assessment failed (%s); answering from the retrieved "
                "evidence without a sufficiency check",
                error,
            )
            return None
