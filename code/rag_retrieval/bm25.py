from __future__ import annotations

import json
import re
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
_VERSION_QUERY_PATTERN = re.compile(
    r"(?:(?P<edition>java|bedrock|基岩|教育)\s*版?\s*)?"
    r"(?P<version>\d+(?:\.\d+)+(?:-(?:pre|rc)\d+)?)",
    re.IGNORECASE,
)
_VERSION_EDITION_NAMES = {
    "java": "Java",
    "bedrock": "基岩",
    "基岩": "基岩",
    "教育": "教育",
}
_VERSION_OVERVIEW_TERMS = (
    "版本",
    "更新",
    "加入",
    "新增",
    "内容",
    "改动",
    "变化",
    "特性",
    "发布",
)
_QUERY_DECORATION_PATTERN = re.compile(r"[\s?？!！,，。:：;；]+")


@dataclass(frozen=True, slots=True)
class BM25Result:
    chunk_id: str
    title: str
    text: str
    source: str
    score: float
    document_id: str | None = None
    chunk_index: int | None = None


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


def _search_phrase(query: str) -> str:
    match = _VERSION_QUERY_PATTERN.search(query)
    if match is None:
        return query

    query_without_product_name = re.sub(
        "minecraft",
        "",
        query,
        flags=re.IGNORECASE,
    )
    compact_query = _QUERY_DECORATION_PATTERN.sub(
        "",
        query_without_product_name,
    )
    compact_match = _QUERY_DECORATION_PATTERN.sub("", match.group(0))
    asks_about_version = compact_query.casefold() == compact_match.casefold()
    asks_about_changes = any(term in query for term in _VERSION_OVERVIEW_TERMS)
    if not asks_about_version and not asks_about_changes:
        return query

    edition = match.group("edition")
    version = match.group("version")
    if edition is None:
        return version
    canonical_edition = _VERSION_EDITION_NAMES[edition.casefold()]
    return f"{canonical_edition}版{version}"


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
        search_phrase = _search_phrase(normalized_query)
        rows = connection.execute(
            """
            WITH matches AS (
                SELECT
                    chunks.chunk_id,
                    chunks.title,
                    chunks.text,
                    chunks.source,
                    json_extract(chunks.metadata, '$.document_id') AS document_id,
                    json_extract(chunks.metadata, '$.chunk_index') AS chunk_index,
                    chunks.rowid,
                    -bm25(chunks_fts, 5.0, 1.0) AS lexical_score,
                    CASE
                        WHEN chunks.title = ? COLLATE NOCASE
                         AND chunks.rowid = (
                            SELECT min(title_match.rowid)
                            FROM chunks AS title_match
                            WHERE title_match.title = ? COLLATE NOCASE
                         )
                        THEN 1
                        ELSE 0
                    END AS exact_title_overview
                FROM chunks_fts
                JOIN chunks ON chunks.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
            )
            SELECT
                chunk_id,
                title,
                text,
                source,
                document_id,
                chunk_index,
                lexical_score
                    + exact_title_overview
                    * (max(lexical_score) OVER () + 1.0) AS score
            FROM matches
            ORDER BY score DESC, rowid
            LIMIT ?
            """,
            (
                search_phrase,
                search_phrase,
                _literal_fts_query(search_phrase),
                limit,
            ),
        ).fetchall()

    return [
        BM25Result(
            chunk_id=row[0],
            title=row[1],
            text=row[2],
            source=row[3],
            document_id=row[4],
            chunk_index=int(row[5]) if row[5] is not None else None,
            score=float(row[6]),
        )
        for row in rows
    ]
