import unittest

from rag_settings import RetrievalSettings, ServiceSettings


class RuntimeSettingsTests(unittest.TestCase):
    def test_retrieval_limits_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_BM25_LIMIT": "12",
            "MCWIKI_SEMANTIC_LIMIT": "14",
            "MCWIKI_EVIDENCE_LIMIT": "6",
            "MCWIKI_CONTEXT_BUDGET": "9000",
            "MCWIKI_EVIDENCE_STRATEGY": "ranked_first",
        })
        self.assertEqual((settings.bm25_limit, settings.semantic_limit, settings.evidence_limit), (12, 14, 6))
        self.assertEqual(settings.max_context_chars, 9000)
        self.assertEqual(settings.evidence_strategy, "ranked_first")

    def test_rejects_invalid_retrieval_and_service_limits(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_BM25_LIMIT"):
            RetrievalSettings.from_env({"MCWIKI_BM25_LIMIT": "0"})
        with self.assertRaisesRegex(ValueError, "invalid answer service"):
            ServiceSettings.from_env({"MCWIKI_ANSWER_TEMPERATURE": "3"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_EVIDENCE_STRATEGY"):
            RetrievalSettings.from_env({"MCWIKI_EVIDENCE_STRATEGY": "unknown"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_EVIDENCE_STRATEGY"):
            RetrievalSettings.from_env({"MCWIKI_EVIDENCE_STRATEGY": "source_cap"})

    def test_query_planning_defaults_keep_the_original_question_only(self):
        settings = RetrievalSettings.from_env({})

        self.assertEqual(settings.query_strategy, "original")
        self.assertEqual(settings.max_retrieval_queries, 2)
        self.assertEqual(settings.query_plan_timeout, 15.0)

    def test_query_planning_options_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_QUERY_STRATEGY": "step_back",
            "MCWIKI_MAX_RETRIEVAL_QUERIES": "1",
            "MCWIKI_QUERY_PLAN_TIMEOUT": "20",
        })

        self.assertEqual(settings.query_strategy, "step_back")
        self.assertEqual(settings.max_retrieval_queries, 1)
        self.assertEqual(settings.query_plan_timeout, 20.0)

    def test_rejects_invalid_query_planning_settings(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_QUERY_STRATEGY"):
            RetrievalSettings.from_env({"MCWIKI_QUERY_STRATEGY": "multi_query"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_RETRIEVAL_QUERIES"):
            RetrievalSettings.from_env({"MCWIKI_MAX_RETRIEVAL_QUERIES": "3"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_RETRIEVAL_QUERIES"):
            RetrievalSettings.from_env({"MCWIKI_MAX_RETRIEVAL_QUERIES": "0"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_QUERY_PLAN_TIMEOUT"):
            RetrievalSettings.from_env({"MCWIKI_QUERY_PLAN_TIMEOUT": "0"})

    def test_answer_service_options_can_be_overridden(self):
        settings = ServiceSettings.from_env({
            "MCWIKI_DEEPSEEK_READ_TIMEOUT": "45",
            "MCWIKI_COOKIE_MAX_AGE": "3600",
            "MCWIKI_ANSWER_TEMPERATURE": "0.4",
            "MCWIKI_ANSWER_THINKING_TYPE": "enabled",
        })
        self.assertEqual(settings.deepseek_read_timeout, 45)
        self.assertEqual(settings.cookie_max_age, 3600)
        self.assertEqual(settings.answer_temperature, 0.4)
        self.assertEqual(settings.answer_thinking_type, "enabled")
