from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


_INVISIBLE_CHARACTERS = str.maketrans("", "", "\u200b\u200c\u200d\ufeff")
_MULTIPLE_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_INLINE_WHITESPACE = re.compile(r"[ \t]+")
_BOUNDARIES = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ")


def clean_text(value: Any) -> str:
    """Normalize text while preserving paragraph boundaries."""
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))
    text = text.translate(_INVISIBLE_CHARACTERS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return _MULTIPLE_BLANK_LINES.sub("\n\n", text)


def split_text(text: str, *, max_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    """Split cleaned text near natural boundaries with a small overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars")

    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        hard_end = min(start + max_chars, len(cleaned))
        end = hard_end

        if hard_end < len(cleaned):
            search_start = max(start + 1, hard_end - max_chars // 3)
            for boundary in _BOUNDARIES:
                position = cleaned.rfind(boundary, search_start, hard_end)
                if position >= search_start:
                    end = position + len(boundary)
                    break

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap_chars)

    return chunks


def build_chunks(
    records: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> Iterator[dict[str, Any]]:
    """Convert source records into traceable retrieval chunks."""
    for record_index, record in enumerate(records):
        title = clean_text(record.get("title", ""))
        source = clean_text(record.get("source_url", ""))
        text = clean_text(record.get("text", ""))
        if not text:
            continue

        document_key = f"{title}\0{source}"
        document_id = hashlib.sha256(document_key.encode("utf-8")).hexdigest()[:16]

        for chunk_index, chunk_text in enumerate(
            split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        ):
            chunk_key = f"{document_id}\0{chunk_index}\0{chunk_text}"
            chunk_id = hashlib.sha256(chunk_key.encode("utf-8")).hexdigest()[:24]
            yield {
                "id": chunk_id,
                "title": title,
                "text": chunk_text,
                "source": source,
                "metadata": {
                    "document_id": document_id,
                    "record_index": record_index,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                },
            }


def load_records(input_path: Path) -> list[Mapping[str, Any]]:
    with input_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list):
        raise ValueError("input JSON must contain a list of records")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError("every input record must be a JSON object")
    return data


def write_jsonl(chunks: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk in chunks:
            json.dump(chunk, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
            count += 1

    return count
