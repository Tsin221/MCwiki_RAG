from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import QdrantClient

from rag_settings import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_SIZE


DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True, slots=True)
class SemanticResult:
    chunk_id: str
    title: str
    text: str
    source: str
    score: float


class SemanticRetriever:
    """Retrieve MCwiki chunks by embedding a query and searching Qdrant."""

    def __init__(
        self,
        *,
        ollama_client: httpx.Client,
        qdrant_client: QdrantClient,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = DEFAULT_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")
        self._ollama_client = ollama_client
        self._qdrant_client = qdrant_client
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._collection_name = collection_name
        self._vector_size = vector_size

    def search(self, query: str, *, limit: int = 10) -> list[SemanticResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        normalized_query = query.strip()
        if not normalized_query:
            return []

        vector = self._embed_query(normalized_query)
        response = self._qdrant_client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._result_from_point(point) for point in response.points]

    def _embed_query(self, query: str) -> list[float]:
        response = self._ollama_client.post(
            f"{self._ollama_url}/api/embed",
            json={"model": self._model, "input": [query]},
        )
        response.raise_for_status()
        response_data = response.json()
        if not isinstance(response_data, Mapping):
            raise ValueError("Ollama response must be a JSON object")
        embeddings = response_data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            actual_count = len(embeddings) if isinstance(embeddings, list) else "non-list"
            raise ValueError(f"Ollama returned {actual_count} embeddings; expected 1")
        vector = embeddings[0]
        if not isinstance(vector, list) or len(vector) != self._vector_size:
            actual_size = len(vector) if isinstance(vector, list) else "non-list"
            raise ValueError(
                f"query embedding has dimension {actual_size}; "
                f"expected {self._vector_size}"
            )
        return vector

    @staticmethod
    def _result_from_point(point: Any) -> SemanticResult:
        payload = point.payload
        if not isinstance(payload, Mapping):
            raise ValueError("Qdrant result payload must be an object")

        values: dict[str, str] = {}
        for field in ("chunk_id", "title", "text", "source"):
            value = payload.get(field)
            if not isinstance(value, str):
                raise ValueError(
                    f"Qdrant result payload field {field!r} must be a string"
                )
            values[field] = value
        if not values["chunk_id"] or not values["text"].strip():
            raise ValueError(
                "Qdrant result payload requires non-empty chunk_id and text"
            )

        try:
            score = float(point.score)
        except (TypeError, ValueError) as error:
            raise ValueError("Qdrant result score must be numeric") from error

        return SemanticResult(score=score, **values)
