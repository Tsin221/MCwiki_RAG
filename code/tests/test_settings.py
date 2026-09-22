import unittest

from rag_settings import RetrievalSettings, ServiceSettings


class RuntimeSettingsTests(unittest.TestCase):
    def test_retrieval_limits_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_BM25_LIMIT": "12",
            "MCWIKI_SEMANTIC_LIMIT": "14",
            "MCWIKI_EVIDENCE_LIMIT": "6",
            "MCWIKI_CONTEXT_BUDGET": "9000",
        })
        self.assertEqual((settings.bm25_limit, settings.semantic_limit, settings.evidence_limit), (12, 14, 6))
        self.assertEqual(settings.max_context_chars, 9000)

    def test_rejects_invalid_retrieval_and_service_limits(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_BM25_LIMIT"):
            RetrievalSettings.from_env({"MCWIKI_BM25_LIMIT": "0"})
        with self.assertRaisesRegex(ValueError, "invalid answer service"):
            ServiceSettings.from_env({"MCWIKI_ANSWER_TEMPERATURE": "3"})

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
