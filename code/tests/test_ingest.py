import unittest
from pathlib import Path

from rag_ingest.__main__ import default_input_path, default_output_path
from rag_ingest.pipeline import build_chunks, clean_text, split_text


class CleanTextTests(unittest.TestCase):
    def test_normalizes_whitespace_and_removes_invisible_characters(self):
        raw = "  第一行\u200b \r\n\r\n\r\n  第二行\t内容  "

        self.assertEqual(clean_text(raw), "第一行\n\n第二行 内容")


class SplitTextTests(unittest.TestCase):
    def test_splits_long_text_with_overlap_and_respects_max_size(self):
        text = "第一段内容。" * 20

        chunks = split_text(text, max_chars=30, overlap_chars=6)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 30 for chunk in chunks))
        self.assertEqual(chunks[0][-6:], chunks[1][:6])

    def test_rejects_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            split_text("内容", max_chars=10, overlap_chars=10)


class BuildChunksTests(unittest.TestCase):
    def test_builds_traceable_chunks_and_skips_empty_records(self):
        records = [
            {
                "title": "测试页面",
                "source_url": "https://example.test/page",
                "text": "第一段。\n\n第二段。",
            },
            {
                "title": "空页面",
                "source_url": "https://example.test/empty",
                "text": "   ",
            },
        ]

        chunks = list(build_chunks(records, max_chars=20, overlap_chars=4))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["title"], "测试页面")
        self.assertEqual(chunks[0]["source"], "https://example.test/page")
        self.assertEqual(chunks[0]["text"], "第一段。\n\n第二段。")
        self.assertEqual(chunks[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(chunks[0]["metadata"]["char_count"], 10)
        self.assertTrue(chunks[0]["id"])


class CliDefaultsTests(unittest.TestCase):
    def test_default_paths_use_the_project_root_data_directory(self):
        project_root = Path(__file__).resolve().parents[2]

        self.assertEqual(
            default_input_path(),
            project_root / "data" / "original_dataset.json",
        )
        self.assertEqual(
            default_output_path(),
            project_root / "data" / "processed" / "chunks.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
