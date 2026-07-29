from __future__ import annotations

import argparse
from pathlib import Path

from rag_retrieval.bm25 import build_bm25_index


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_path() -> Path:
    return project_root() / "data" / "processed" / "chunks.jsonl"


def default_output_path() -> Path:
    return project_root() / "data" / "processed" / "bm25.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SQLite FTS5 trigram BM25 index."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexed_count = build_bm25_index(args.input, args.output)
    print(f"Completed: indexed {indexed_count} chunks into {args.output}", flush=True)


if __name__ == "__main__":
    main()
