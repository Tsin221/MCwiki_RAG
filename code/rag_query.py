from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from rag_settings import (
    DEFAULT_QUERY_PLAN_TIMEOUT,
    DEFAULT_QUERY_STRATEGY,
    MAX_RETRIEVAL_QUERIES,
    ORIGINAL_QUERY_STRATEGY,
    STEP_BACK_QUERY_STRATEGY,
)


logger = logging.getLogger(__name__)

DEFAULT_STEP_BACK_TEMPERATURE = 0.0
MAX_STEP_BACK_QUESTION_CHARS = 120
_STEP_BACK_QUESTION_KEY = "step_back_question"
STEP_BACK_SYSTEM_PROMPT = """你是 Minecraft Wiki 检索助手。用户会给出一个具体问题，你需要把它抽象成一个更概括的背景问题，用于补充检索原理、定义或上位概念。

规则：
1. 只输出一个 JSON 对象，键名固定为 step_back_question，不要输出解释、前缀或 Markdown 代码块。
2. step_back_question 必须比原问题更抽象，关注背后的机制、原理、分类或定义。
3. 不要在抽象问题里重复原问题中的精确版本号、数字或物品 ID，这些内容由原问题负责召回。
4. 如果原问题本身已经是背景性问题，或抽象后不会带来新的检索方向，就把 step_back_question 设为空字符串。
5. 使用简体中文，只写一个问题，不超过 60 个字符。
"""


class QueryPlanningError(RuntimeError):
    """Raised when a step-back model call cannot produce a usable question."""


class ChatSettings(Protocol):
    """Connection settings for an OpenAI-compatible chat completion endpoint."""

    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """The queries one question is retrieved with, in execution order.

    ``strategy`` names the strategy that produced the plan; ``retrieval_queries``
    always starts with the normalized original question and never exceeds the
    hard query limit.
    """

    original: str
    retrieval_queries: tuple[str, ...]
    strategy: str

    def __post_init__(self) -> None:
        if len(self.retrieval_queries) > MAX_RETRIEVAL_QUERIES:
            raise ValueError(
                f"retrieval_queries must not exceed {MAX_RETRIEVAL_QUERIES} queries"
            )
        if self.retrieval_queries and self.retrieval_queries[0] != self.original:
            raise ValueError("retrieval_queries[0] must be the original question")
        if self.original and not self.retrieval_queries:
            raise ValueError("retrieval_queries must not drop the original question")


class QueryPlanner(Protocol):
    """Turn a question into the queries that should be retrieved."""

    async def plan(self, question: str) -> QueryPlan: ...


def _comparable_key(text: str) -> str:
    return "".join(text.split()).casefold()


def build_query_plan(
    question: str,
    candidates: Sequence[str] = (),
    *,
    strategy: str = DEFAULT_QUERY_STRATEGY,
    max_queries: int = MAX_RETRIEVAL_QUERIES,
) -> QueryPlan:
    """Keep the normalized question first and append usable, unique candidates.

    Empty and duplicate candidates are dropped instead of raising, because
    candidates come from model output and a poor abstraction must never block
    the original query.
    """
    original = question.strip()
    if not original:
        return QueryPlan(original="", retrieval_queries=(), strategy=strategy)
    if not 1 <= max_queries <= MAX_RETRIEVAL_QUERIES:
        raise ValueError(
            f"max_queries must be between 1 and {MAX_RETRIEVAL_QUERIES}"
        )

    queries = [original]
    seen = {_comparable_key(original)}
    for candidate in candidates:
        if len(queries) >= max_queries:
            break
        normalized = " ".join(candidate.split())
        if not normalized:
            continue
        key = _comparable_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        queries.append(normalized)
    return QueryPlan(
        original=original,
        retrieval_queries=tuple(queries),
        strategy=strategy,
    )


class OriginalQueryPlanner:
    """Retrieve with the user's own question only."""

    __slots__ = ()

    async def plan(self, question: str) -> QueryPlan:
        return build_query_plan(question, strategy=ORIGINAL_QUERY_STRATEGY)


def _message_content(body: Any) -> str:
    if not isinstance(body, Mapping):
        raise QueryPlanningError("step-back response must be a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise QueryPlanningError("step-back response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise QueryPlanningError("step-back response choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise QueryPlanningError("step-back response choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise QueryPlanningError("step-back response content must be a string")
    return content


def _parse_step_back(content: str) -> str | None:
    """Return the abstract question, or None when the model declined to add one."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise QueryPlanningError("step-back response is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise QueryPlanningError("step-back response must be a JSON object")
    if _STEP_BACK_QUESTION_KEY not in payload:
        raise QueryPlanningError(
            f"step-back response has no {_STEP_BACK_QUESTION_KEY} field"
        )
    value = payload[_STEP_BACK_QUESTION_KEY]
    if not isinstance(value, str):
        raise QueryPlanningError(f"{_STEP_BACK_QUESTION_KEY} must be a string")
    step_back = " ".join(value.split())
    if not step_back:
        return None
    if len(step_back) > MAX_STEP_BACK_QUESTION_CHARS:
        raise QueryPlanningError("step-back question is too long")
    return step_back


class DeepSeekStepBackPlanner:
    """Ask a chat model for one abstract background question, then fall back safely."""

    __slots__ = ("_http_client", "_max_queries", "_settings", "_temperature", "_timeout")

    def __init__(
        self,
        *,
        settings: ChatSettings,
        http_client: httpx.AsyncClient,
        timeout: float = DEFAULT_QUERY_PLAN_TIMEOUT,
        temperature: float = DEFAULT_STEP_BACK_TEMPERATURE,
        max_queries: int = MAX_RETRIEVAL_QUERIES,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if not 1 <= max_queries <= MAX_RETRIEVAL_QUERIES:
            raise ValueError(
                f"max_queries must be between 1 and {MAX_RETRIEVAL_QUERIES}"
            )
        self._settings = settings
        self._http_client = http_client
        self._timeout = timeout
        self._temperature = temperature
        self._max_queries = max_queries

    async def plan(self, question: str) -> QueryPlan:
        """Plan retrieval; every model-side failure keeps only the original question."""
        original = question.strip()
        if not original or self._max_queries < 2:
            return build_query_plan(
                original,
                strategy=STEP_BACK_QUERY_STRATEGY,
                max_queries=self._max_queries,
            )
        try:
            step_back = await self._request_step_back(original)
        except Exception as error:
            logger.warning(
                "Step-back planning failed (%s); retrieving the original question only",
                error,
            )
            step_back = None
        return build_query_plan(
            original,
            [step_back] if step_back else [],
            strategy=STEP_BACK_QUERY_STRATEGY,
            max_queries=self._max_queries,
        )

    async def _request_step_back(self, question: str) -> str | None:
        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": STEP_BACK_SYSTEM_PROMPT},
                {"role": "user", "content": f"原问题：{question}"},
            ],
            "stream": False,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
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
            raise QueryPlanningError("step-back request timed out") from error
        except httpx.HTTPError as error:
            raise QueryPlanningError("step-back request failed") from error
        try:
            body = response.json()
        except ValueError as error:
            raise QueryPlanningError("step-back response is not valid JSON") from error
        return _parse_step_back(_message_content(body))


def build_query_planner(
    strategy: str,
    *,
    settings: ChatSettings | None = None,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_QUERY_PLAN_TIMEOUT,
    max_queries: int = MAX_RETRIEVAL_QUERIES,
) -> QueryPlanner:
    """Return the planner for a configured strategy, degrading to original-only."""
    if strategy == ORIGINAL_QUERY_STRATEGY:
        return OriginalQueryPlanner()
    if strategy != STEP_BACK_QUERY_STRATEGY:
        raise ValueError(f"unsupported query strategy: {strategy}")
    if settings is None or http_client is None:
        logger.warning(
            "Query strategy %s has no chat client configured; "
            "retrieving the original question only",
            strategy,
        )
        return OriginalQueryPlanner()
    return DeepSeekStepBackPlanner(
        settings=settings,
        http_client=http_client,
        timeout=timeout,
        max_queries=max_queries,
    )
