from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models


DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_COLLECTION = "mcwiki_chunks"
DEFAULT_VECTOR_SIZE = 1024


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_path() -> Path:
    return project_root() / "data" / "processed" / "chunks.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(chunk, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            if not isinstance(chunk.get("id"), str) or not chunk["id"]:
                raise ValueError(f"line {line_number} has no non-empty string id")
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(f"line {line_number} has no non-empty string text")
            yield chunk


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mcwiki-rag:{chunk_id}"))


def build_payload(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["id"],
        "title": chunk.get("title", ""),
        "text": chunk["text"],
        "source": chunk.get("source", ""),
        "metadata": chunk.get("metadata", {}),
    }


def validate_embeddings(
    embeddings: Any,
    *,
    expected_count: int,
    vector_size: int,
) -> list[list[float]]:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        actual_count = len(embeddings) if isinstance(embeddings, list) else "non-list"
        raise ValueError(
            f"Ollama returned {actual_count} embeddings; expected {expected_count}"
        )
    for index, embedding in enumerate(embeddings):
        if not isinstance(embedding, list) or len(embedding) != vector_size:
            actual_size = len(embedding) if isinstance(embedding, list) else "non-list"
            raise ValueError(
                f"embedding {index} has dimension {actual_size}; expected {vector_size}"
            )
    return embeddings


def embed_texts(
    client: httpx.Client,
    *,
    ollama_url: str,
    model: str,
    texts: Sequence[str],
    vector_size: int,
) -> list[list[float]]:
    response = client.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": list(texts)},
    )
    response.raise_for_status()
    response_data = response.json()
    return validate_embeddings(
        response_data.get("embeddings"),
        expected_count=len(texts),
        vector_size=vector_size,
    )


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        return

    collection = client.get_collection(collection_name)
    vectors_config = collection.config.params.vectors
    if not isinstance(vectors_config, models.VectorParams):
        raise ValueError(
            f"collection {collection_name!r} uses named vectors; expected one dense vector"
        )
    if vectors_config.size != vector_size or vectors_config.distance != models.Distance.COSINE:
        raise ValueError(
            f"collection {collection_name!r} has vector config "
            f"{vectors_config.size}/{vectors_config.distance}; "
            f"expected {vector_size}/{models.Distance.COSINE}"
        )


def collect_existing_point_ids(
    client: QdrantClient,
    *,
    collection_name: str,
    page_size: int = 256,
) -> set[str]:
    point_ids: set[str] = set()
    offset: models.PointId | None = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.update(str(record.id) for record in records)
        if offset is None:
            return point_ids


def pending_chunks(
    chunks: Iterable[dict[str, Any]],
    existing_point_ids: set[str],
) -> Iterator[dict[str, Any]]:
    for chunk in chunks:
        if chunk_point_id(chunk["id"]) not in existing_point_ids:
            yield chunk


def batched(
    items: Iterable[dict[str, Any]],
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest(
    *,
    input_path: Path,
    ollama_url: str,
    qdrant_url: str,
    model: str,
    collection_name: str,
    vector_size: int,
    batch_size: int,
    timeout_seconds: float,
) -> int:
    total = sum(1 for _ in iter_jsonl(input_path))
    if total == 0:
        raise ValueError(f"no chunks found in {input_path}")

    qdrant = QdrantClient(url=qdrant_url, timeout=timeout_seconds)
    ensure_collection(
        qdrant,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    existing_point_ids = collect_existing_point_ids(
        qdrant,
        collection_name=collection_name,
    )
    remaining = sum(
        1 for _ in pending_chunks(iter_jsonl(input_path), existing_point_ids)
    )
    indexed = total - remaining
    print(
        f"Resuming with {indexed}/{total} chunks already present; "
        f"{remaining} remaining.",
        flush=True,
    )

    with httpx.Client(timeout=timeout_seconds) as ollama:
        chunks_to_index = pending_chunks(iter_jsonl(input_path), existing_point_ids)
        for batch in batched(chunks_to_index, batch_size):
            embeddings = embed_texts(
                ollama,
                ollama_url=ollama_url,
                model=model,
                texts=[chunk["text"] for chunk in batch],
                vector_size=vector_size,
            )
            points = [
                models.PointStruct(
                    id=chunk_point_id(chunk["id"]),
                    vector=embedding,
                    payload=build_payload(chunk),
                )
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ]
            qdrant.upsert(
                collection_name=collection_name,
                wait=True,
                points=points,
            )
            indexed += len(points)
            print(f"Indexed {indexed}/{total} chunks", flush=True)
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ollama embeddings and upsert all chunks into Qdrant."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexed = ingest(
        input_path=args.input,
        ollama_url=args.ollama_url,
        qdrant_url=args.qdrant_url,
        model=args.model,
        collection_name=args.collection,
        vector_size=args.vector_size,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        f"Completed: indexed {indexed} chunks into Qdrant collection "
        f"{args.collection!r}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
