from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_COLLECTION = "mcwiki_chunks"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_QDRANT_URL = "http://127.0.0.1:6335"
DEFAULT_BM25_PATH = Path(__file__).resolve().parents[1] / "data/processed/bm25.db"


@dataclass(frozen=True)
class RetrievalSettings:
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection: str = DEFAULT_COLLECTION
    vector_size: int = DEFAULT_VECTOR_SIZE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    bm25_path: Path = DEFAULT_BM25_PATH

    @classmethod
    def from_env(cls) -> RetrievalSettings:
        vector_size = int(os.environ.get("MCWIKI_VECTOR_SIZE", DEFAULT_VECTOR_SIZE))
        if vector_size <= 0:
            raise ValueError("MCWIKI_VECTOR_SIZE must be positive")
        return cls(
            qdrant_url=os.environ.get("MCWIKI_QDRANT_URL", DEFAULT_QDRANT_URL).strip(),
            collection=os.environ.get("MCWIKI_QDRANT_COLLECTION", DEFAULT_COLLECTION).strip(),
            vector_size=vector_size,
            embedding_model=os.environ.get("MCWIKI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip(),
            bm25_path=Path(os.environ.get("MCWIKI_BM25_DB_PATH", str(DEFAULT_BM25_PATH))).resolve(),
        )
