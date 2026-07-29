import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_ingest.vector_ingest import (
    build_payload,
    chunk_point_id,
    collect_existing_point_ids,
    iter_jsonl,
    pending_chunks,
    validate_embeddings,
)


class JsonlReaderTests(unittest.TestCase):
    def test_reads_non_empty_json_objects(self):
        contents = "\n".join(
            [
                json.dumps({"id": "a", "text": "第一段"}),
                "",
                json.dumps({"id": "b", "text": "第二段"}),
            ]
        )

        with patch.object(Path, "open", return_value=io.StringIO(contents)):
            self.assertEqual(
                list(iter_jsonl(Path("chunks.jsonl"))),
                [{"id": "a", "text": "第一段"}, {"id": "b", "text": "第二段"}],
            )

    def test_reports_the_line_number_for_invalid_json(self):
        contents = '{"id":"a","text":"ok"}\nnot-json\n'
        with patch.object(Path, "open", return_value=io.StringIO(contents)):
            with self.assertRaisesRegex(ValueError, "line 2"):
                list(iter_jsonl(Path("chunks.jsonl")))


class PointConversionTests(unittest.TestCase):
    def test_point_id_is_stable_uuid_and_payload_keeps_chunk_id(self):
        chunk = {
            "id": "0123456789abcdef01234567",
            "title": "测试",
            "text": "正文",
            "source": "https://example.test",
            "metadata": {"chunk_index": 2},
        }

        self.assertEqual(chunk_point_id(chunk["id"]), chunk_point_id(chunk["id"]))
        self.assertEqual(len(chunk_point_id(chunk["id"])), 36)
        self.assertEqual(
            build_payload(chunk),
            {
                "chunk_id": chunk["id"],
                "title": "测试",
                "text": "正文",
                "source": "https://example.test",
                "metadata": {"chunk_index": 2},
            },
        )


class EmbeddingValidationTests(unittest.TestCase):
    def test_accepts_expected_batch_size_and_dimension(self):
        embeddings = [[0.0, 1.0], [1.0, 0.0]]

        self.assertIs(validate_embeddings(embeddings, expected_count=2, vector_size=2), embeddings)

    def test_rejects_wrong_vector_dimension(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_embeddings([[0.0]], expected_count=1, vector_size=2)


class ResumeTests(unittest.TestCase):
    def test_collects_point_ids_across_scroll_pages(self):
        class FakeQdrant:
            def scroll(self, **kwargs):
                if kwargs.get("offset") is None:
                    return [SimpleNamespace(id="point-a")], "next-page"
                return [SimpleNamespace(id="point-b")], None

        self.assertEqual(
            collect_existing_point_ids(FakeQdrant(), collection_name="chunks"),
            {"point-a", "point-b"},
        )

    def test_pending_chunks_skips_chunks_already_in_qdrant(self):
        chunks = [
            {"id": "existing", "text": "old"},
            {"id": "missing", "text": "new"},
        ]
        existing_ids = {chunk_point_id("existing")}

        self.assertEqual(
            list(pending_chunks(chunks, existing_ids)),
            [{"id": "missing", "text": "new"}],
        )


if __name__ == "__main__":
    unittest.main()
