from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata TEXT NOT NULL
) STRICT;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title,
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, text)
    VALUES (new.rowid, new.title, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, old.text);
    INSERT INTO chunks_fts(rowid, title, text)
    VALUES (new.rowid, new.title, new.text);
END;
"""


@dataclass(frozen=True, slots=True)
class BM25Result:
    chunk_id: str
    title: str
    text: str
    source: str
    score: float


def _iter_chunks(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
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
            yield line_number, chunk


def _chunk_values(line_number: int, chunk: Mapping[str, Any]) -> tuple[str, ...]:
    chunk_id = chunk.get("id")
    text = chunk.get("text")
    title = chunk.get("title", "")
    source = chunk.get("source", "")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"line {line_number} has no non-empty string id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"line {line_number} has no non-empty string text")
    if not isinstance(title, str):
        raise ValueError(f"line {line_number} title must be a string")
    if not isinstance(source, str):
        raise ValueError(f"line {line_number} source must be a string")
    try:
        metadata = json.dumps(
            chunk.get("metadata", {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"line {line_number} metadata is not valid JSON") from error
    return chunk_id, title, text, source, metadata


def build_bm25_index(input_path: Path, database_path: Path) -> int:
    """Synchronize a trigram FTS5 index with a chunks JSONL file."""
    input_path = Path(input_path)
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "CREATE TEMP TABLE seen_chunks (chunk_id TEXT PRIMARY KEY) STRICT"
            )

            count = 0
            for line_number, chunk in _iter_chunks(input_path):
                values = _chunk_values(line_number, chunk)
                try:
                    connection.execute(
                        "INSERT INTO seen_chunks(chunk_id) VALUES (?)",
                        (values[0],),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"duplicate chunk id {values[0]!r} on line {line_number}"
                    ) from error
                connection.execute(
                    """
                    INSERT INTO chunks(chunk_id, title, text, source, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        title = excluded.title,
                        text = excluded.text,
                        source = excluded.source,
                        metadata = excluded.metadata
                    WHERE chunks.title <> excluded.title
                       OR chunks.text <> excluded.text
                       OR chunks.source <> excluded.source
                       OR chunks.metadata <> excluded.metadata
                    """,
                    values,
                )
                count += 1

            if count == 0:
                raise ValueError(f"no chunks found in {input_path}")

            connection.execute(
                """
                DELETE FROM chunks
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM seen_chunks
                    WHERE seen_chunks.chunk_id = chunks.chunk_id
                )
                """,
            )
            stored_count = connection.execute(
                "SELECT count(*) FROM chunks"
            ).fetchone()[0]
            if stored_count != count:
                raise RuntimeError(
                    f"BM25 index contains {stored_count} chunks; expected {count}"
                )

    return count


def _literal_fts_query(query: str) -> str:
    return f'"{query.replace('"', '""')}"'


def search_bm25(
    database_path: Path,
    query: str,
    *,
    limit: int = 10,
) -> list[BM25Result]:
    """Return BM25-ranked chunks; blank or unmatched queries return an empty list."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    normalized_query = query.strip()
    if not normalized_query:
        return []

    database_path = Path(database_path)
    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT
                chunks.chunk_id,
                chunks.title,
                chunks.text,
                chunks.source,
                -bm25(chunks_fts, 5.0, 1.0) AS score
            FROM chunks_fts
            JOIN chunks ON chunks.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC, chunks.rowid
            LIMIT ?
            """,
            (_literal_fts_query(normalized_query), limit),
        ).fetchall()

    return [
        BM25Result(
            chunk_id=row[0],
            title=row[1],
            text=row[2],
            source=row[3],
            score=float(row[4]),
        )
        for row in rows
    ]
