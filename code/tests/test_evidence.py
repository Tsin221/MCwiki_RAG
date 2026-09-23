import unittest

from rag_evidence import (
    EvidenceCandidate,
    SelectionConfig,
    select_evidence,
)


def candidate(
    chunk_id: str,
    text: str,
    *,
    source: str = "https://example.test/page",
    document_id: str | None = None,
    chunk_index: int | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        title="测试页面",
        text=text,
        source=source,
        document_id=document_id,
        chunk_index=chunk_index,
    )


class EvidenceSelectionTests(unittest.TestCase):
    def test_ranked_first_keeps_stable_contiguous_ids_and_original_chunks(self):
        selected = select_evidence(
            [candidate("a", "甲"), candidate("b", "乙")],
            SelectionConfig(strategy="ranked_first", max_context_chars=10),
        )

        self.assertEqual([item.id for item in selected], [1, 2])
        self.assertEqual(
            [item.component_chunk_ids for item in selected],
            [("a",), ("b",)],
        )

    def test_empty_input_returns_no_evidence(self):
        self.assertEqual(
            select_evidence(
                [], SelectionConfig(strategy="ranked_first", max_context_chars=10)
            ),
            [],
        )

    def test_first_oversized_candidate_is_truncated_to_the_strict_budget(self):
        selected = select_evidence(
            [candidate("long", "甲乙丙丁戊")],
            SelectionConfig(strategy="ranked_first", max_context_chars=3),
        )

        self.assertEqual(selected[0].text, "甲乙丙")
        self.assertEqual(sum(len(item.text) for item in selected), 3)

    def test_source_cap_preserves_distinct_sections_up_to_the_configured_limit(self):
        selected = select_evidence(
            [
                candidate("a", "第一节"),
                candidate("b", "第二节"),
                candidate("c", "第三节"),
                candidate("d", "其他页", source="https://example.test/other"),
            ],
            SelectionConfig(
                strategy="source_cap",
                max_context_chars=100,
                max_chunks_per_source=2,
            ),
        )

        self.assertEqual(
            [item.component_chunk_ids for item in selected],
            [("a",), ("b",), ("d",)],
        )

    def test_adjacent_merge_removes_chunk_overlap_and_keeps_both_chunk_ids(self):
        selected = select_evidence(
            [
                candidate(
                    "second",
                    "重复结尾以及新事实",
                    document_id="doc",
                    chunk_index=2,
                ),
                candidate(
                    "first",
                    "前文内容重复结尾",
                    document_id="doc",
                    chunk_index=1,
                ),
            ],
            SelectionConfig(
                strategy="adjacent_merge",
                max_context_chars=100,
                min_merge_overlap_chars=4,
            ),
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].text, "前文内容重复结尾以及新事实")
        self.assertEqual(selected[0].component_chunk_ids, ("first", "second"))

    def test_redundancy_suppression_keeps_complementary_same_source_text(self):
        selected = select_evidence(
            [
                candidate("a", "苦力怕接近玩家后开始爆炸倒计时"),
                candidate("b", "苦力怕接近玩家后开始爆炸倒计时。"),
                candidate("c", "苦力怕可能掉落火药，也可能掉落唱片"),
            ],
            SelectionConfig(
                strategy="redundancy_suppression",
                max_context_chars=100,
                redundancy_threshold=0.8,
            ),
        )

        self.assertEqual(
            [item.component_chunk_ids for item in selected],
            [("a",), ("c",)],
        )

    def test_invalid_strategy_and_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strategy"):
            SelectionConfig(strategy="unknown")
        with self.assertRaisesRegex(ValueError, "max_context_chars"):
            SelectionConfig(strategy="ranked_first", max_context_chars=0)
        with self.assertRaisesRegex(ValueError, "max_chunks_per_source"):
            SelectionConfig(
                strategy="source_cap",
                max_chunks_per_source=0,
            )


if __name__ == "__main__":
    unittest.main()
