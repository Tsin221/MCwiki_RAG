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


class EvidenceExcerptTests(unittest.TestCase):
    """The card snippet: prose over table cells, whole sentences, marked omissions."""

    def excerpt(self, text, *, chunk_index=None, source="https://example.test/page"):
        selected = select_evidence(
            [candidate("a", text, source=source, chunk_index=chunk_index)],
            SelectionConfig(strategy="ranked_first", max_context_chars=9_000),
        )
        return selected[0]

    def test_a_short_prose_item_is_shown_as_it_is(self):
        text = "红石中继器可以延迟红石信号，并阻止信号倒流。"

        item = self.excerpt(text, chunk_index=0)

        self.assertEqual(item.excerpt, text)

    def test_a_chunk_that_starts_mid_article_says_so(self):
        text = "器可以延迟红石信号，并阻止信号倒流。"

        item = self.excerpt(text, chunk_index=4)

        self.assertEqual(item.excerpt, f"…{text}")

    def test_an_unknown_position_does_not_claim_to_be_a_fragment(self):
        text = "器可以延迟红石信号，并阻止信号倒流。"

        item = self.excerpt(text)

        self.assertEqual(item.excerpt, text)

    def test_a_flattened_table_shows_its_descriptive_cells_only(self):
        table = "\n".join(
            [
                "false",
                "true",
                "红石中继器处于锁存状态",
                "powered",
                "0x1 0x2",
                "方块接收到了红石信号",
            ]
        )

        item = self.excerpt(table, chunk_index=3)

        self.assertEqual(item.excerpt, "…红石中继器处于锁存状态 方块接收到了红石信号")

    def test_a_table_without_prose_falls_back_to_its_cells(self):
        table = "\n".join(["false", "true", "0x1", "powered", "north"])

        item = self.excerpt(table, chunk_index=2)

        self.assertEqual(item.excerpt, "…false true 0x1 powered north")

    def test_a_block_of_prose_keeps_its_short_lines(self):
        prose = "第一行短句\n第二行稍长一点的句子，说明机制。\n第三行也短"

        item = self.excerpt(prose, chunk_index=1)

        self.assertEqual(item.excerpt, f"…{' '.join(prose.splitlines())}")

    def test_a_long_snippet_ends_where_a_sentence_ends(self):
        text = "这是一个足够长的句子。" * 30

        item = self.excerpt(text, chunk_index=7)

        self.assertTrue(item.excerpt.startswith("…"))
        self.assertTrue(item.excerpt.endswith("。…"))
        self.assertLessEqual(len(item.excerpt), 222)

    def test_a_long_snippet_without_any_full_stop_is_still_cut(self):
        text = "没有句号的连续文本" * 40

        item = self.excerpt(text, chunk_index=2)

        self.assertTrue(item.excerpt.endswith("…"))
        # 219 characters of text plus the fragment and omission markers.
        self.assertEqual(len(item.excerpt), 221)

    def test_a_merged_group_is_a_fragment_only_when_it_starts_mid_article(self):
        for chunk_index, expected in ((0, False), (3, True)):
            with self.subTest(chunk_index=chunk_index):
                merged = select_evidence(
                    [
                        candidate(
                            "later",
                            "重复结尾以及新事实",
                            document_id="doc",
                            chunk_index=chunk_index + 1,
                        ),
                        candidate(
                            "earlier",
                            "前文内容重复结尾",
                            document_id="doc",
                            chunk_index=chunk_index,
                        ),
                    ],
                    SelectionConfig(
                        strategy="adjacent_merge",
                        max_context_chars=9_000,
                        min_merge_overlap_chars=4,
                    ),
                )[0]

                self.assertEqual(merged.component_chunk_ids, ("earlier", "later"))
                self.assertEqual(merged.excerpt.startswith("…"), expected)


if __name__ == "__main__":
    unittest.main()
