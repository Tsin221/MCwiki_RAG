from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_chunks, load_records, write_jsonl


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_path() -> Path:
    return project_root() / "data" / "original_dataset.json"


def default_output_path() -> Path:
    return project_root() / "data" / "processed" / "chunks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and chunk RAG source data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Source JSON array.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Destination JSONL file.",
    )
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    chunks = build_chunks(
        records,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    chunk_count = write_jsonl(chunks, args.output)
    print(f"Wrote {chunk_count} chunks to {args.output}")


if __name__ == "__main__":
    main()
