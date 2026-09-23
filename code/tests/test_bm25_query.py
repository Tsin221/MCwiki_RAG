import unittest

from rag_retrieval.bm25_query import (
    build_match_query,
    escape_fts_phrase,
    supported_strategies,
)


class BM25QueryStrategyTests(unittest.TestCase):
    def test_literal_phrase_escapes_fts_quotes_and_operators(self):
        self.assertEqual(escape_fts_phrase('NEAR("x") OR y'), '"NEAR(""x"") OR y"')

    def test_all_strategies_build_safe_non_empty_match_queries(self):
        malicious = '红石 OR title:"x" NEAR(" /give minecraft:stone'

        for strategy in supported_strategies():
            with self.subTest(strategy=strategy):
                match_query = build_match_query(malicious, strategy)
                self.assertTrue(match_query)
                self.assertNotEqual(match_query, malicious)
                self.assertEqual(match_query.count('"') % 2, 0)

    def test_entity_phrase_keeps_structured_identifiers(self):
        match_query = build_match_query(
            "请问 Java版1.20.5-rc2 的 /give 和 minecraft:stone 怎么用？",
            "entity_phrase",
        )

        self.assertIn('"Java版1.20.5-rc2"', match_query)
        self.assertIn('"/give"', match_query)
        self.assertIn('"minecraft:stone"', match_query)

    def test_token_or_uses_trigrams_without_fts_operators(self):
        match_query = build_match_query("能不能顺便告诉我红石中继器的延迟是多少？", "token_or")

        self.assertIn('"红石中"', match_query)
        self.assertIn(" OR ", match_query)

    def test_rejects_blank_query_and_unknown_strategy(self):
        self.assertEqual(build_match_query("  ", "literal_phrase"), "")
        with self.assertRaisesRegex(ValueError, "unknown BM25 query strategy"):
            build_match_query("红石", "unknown")


if __name__ == "__main__":
    unittest.main()
