import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

from rag_ingest.bm25_index import default_input_path, default_output_path
from rag_retrieval.bm25 import build_bm25_index, search_bm25


class _BM25Fixture:
    def setUp(self):
        self.input_path = Path(__file__).parent / "_bm25_test_chunks.jsonl"
        self.database_path = Path(__file__).parent / "_bm25_test.db"
        self._remove_test_files()
        self.chunks = [
            {
                "id": "chunk-redstone",
                "title": "红石中继器",
                "text": "红石中继器可以延迟红石信号。",
                "source": "https://example.test/redstone-repeater",
                "metadata": {"chunk_index": 0, "tags": ["红石"]},
            },
            {
                "id": "chunk-java-121",
                "title": "Java版1.21",
                "text": "Minecraft Java版1.21加入了新的内容。",
                "source": "https://example.test/java-1.21",
                "metadata": {"chunk_index": 1},
            },
            {
                "id": "chunk-dirt",
                "title": "泥土",
                "text": "泥土是一种常见方块。",
                "source": "https://example.test/dirt",
                "metadata": {"chunk_index": 2},
            },
        ]
        self._write_chunks(self.chunks)

    def tearDown(self):
        self._remove_test_files()

    def _remove_test_files(self):
        for suffix in ("", "-shm", "-wal", "-journal"):
            self.database_path.with_name(
                f"{self.database_path.name}{suffix}"
            ).unlink(missing_ok=True)
        self.input_path.unlink(missing_ok=True)

    def _write_chunks(self, chunks):
        with self.input_path.open("w", encoding="utf-8", newline="\n") as output:
            for chunk in chunks:
                json.dump(chunk, output, ensure_ascii=False)
                output.write("\n")


class BM25IndexTests(_BM25Fixture, unittest.TestCase):
    def test_builds_trigram_index_and_preserves_all_fields(self):
        indexed_count = build_bm25_index(self.input_path, self.database_path)

        self.assertEqual(indexed_count, 3)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
                3,
            )
            self.assertEqual(
                connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0],
                3,
            )
            row = connection.execute(
                """
                SELECT chunk_id, title, text, source, metadata
                FROM chunks
                WHERE chunk_id = ?
                """,
                ("chunk-redstone",),
            ).fetchone()

        self.assertEqual(row[:4], tuple(self.chunks[0][key] for key in ("id", "title", "text", "source")))
        self.assertEqual(json.loads(row[4]), self.chunks[0]["metadata"])

    def test_repeated_build_updates_rows_and_removes_stale_chunks(self):
        build_bm25_index(self.input_path, self.database_path)
        updated_chunks = [
            {**self.chunks[0], "text": "红石中继器会增强并延迟信号。"},
            self.chunks[1],
        ]
        self._write_chunks(updated_chunks)

        self.assertEqual(build_bm25_index(self.input_path, self.database_path), 2)
        self.assertEqual(build_bm25_index(self.input_path, self.database_path), 2)

        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
            ).fetchall()
        self.assertEqual(
            rows,
            [
                ("chunk-java-121", self.chunks[1]["text"]),
                ("chunk-redstone", updated_chunks[0]["text"]),
            ],
        )


class BM25QueryTests(_BM25Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        build_bm25_index(self.input_path, self.database_path)

    def test_returns_required_fields_and_higher_is_better_score(self):
        results = search_bm25(self.database_path, "红石中继器", limit=5)

        self.assertEqual(results[0].chunk_id, "chunk-redstone")
        self.assertEqual(results[0].title, self.chunks[0]["title"])
        self.assertEqual(results[0].text, self.chunks[0]["text"])
        self.assertEqual(results[0].source, self.chunks[0]["source"])
        self.assertGreater(results[0].score, 0)

    def test_supports_minecraft_terms_and_version_numbers(self):
        minecraft_results = search_bm25(self.database_path, "Minecraft")
        version_results = search_bm25(self.database_path, "1.21")

        self.assertEqual(minecraft_results[0].chunk_id, "chunk-java-121")
        self.assertEqual(version_results[0].chunk_id, "chunk-java-121")

    def test_recalls_exact_version_title_from_natural_language_question(self):
        results = search_bm25(
            self.database_path,
            "Minecraft Java版1.21主要加入了哪些内容？",
        )

        self.assertEqual(results[0].chunk_id, "chunk-java-121")

    def test_empty_and_no_match_queries_return_empty_lists(self):
        self.assertEqual(search_bm25(self.database_path, ""), [])
        self.assertEqual(search_bm25(self.database_path, " \t\n"), [])
        self.assertEqual(search_bm25(self.database_path, "不存在的检索词xyz"), [])

    def test_treats_fts_syntax_as_literal_text(self):
        self.assertEqual(search_bm25(self.database_path, 'NEAR("'), [])

    def test_rejects_non_positive_limits(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            search_bm25(self.database_path, "红石", limit=0)


class BM25CliDefaultsTests(unittest.TestCase):
    def test_default_paths_use_processed_data_directory(self):
        project_root = Path(__file__).resolve().parents[2]

        self.assertEqual(
            default_input_path(),
            project_root / "data" / "processed" / "chunks.jsonl",
        )
        self.assertEqual(
            default_output_path(),
            project_root / "data" / "processed" / "bm25.db",
        )


if __name__ == "__main__":
    unittest.main()
