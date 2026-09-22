from __future__ import annotations

import httpx
from qdrant_client import QdrantClient

from rag_retrieval.bm25 import search_bm25
from rag_retrieval.hybrid import HybridRetriever
from rag_retrieval.semantic import SemanticRetriever
from rag_settings import RetrievalSettings


def build_default_retriever(
    settings: RetrievalSettings | None = None,
) -> tuple[HybridRetriever, httpx.Client, QdrantClient]:
    settings = settings or RetrievalSettings.from_env()
    ollama = httpx.Client(timeout=60.0)
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=10.0)
    semantic = SemanticRetriever(
        ollama_client=ollama,
        qdrant_client=qdrant,
        model=settings.embedding_model,
        collection_name=settings.collection,
        vector_size=settings.vector_size,
    )
    hybrid = HybridRetriever(
        search_bm25=lambda query, limit: search_bm25(
            settings.bm25_path, query, limit=limit
        ),
        search_semantic=lambda query, limit: semantic.search(query, limit=limit),
    )
    return hybrid, ollama, qdrant
