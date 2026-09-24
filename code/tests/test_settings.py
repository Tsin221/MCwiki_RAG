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
        self.assertEqual(settings.query_plan_thinking_type, "disabled")

    def test_query_planning_options_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_QUERY_STRATEGY": "step_back",
            "MCWIKI_MAX_RETRIEVAL_QUERIES": "1",
            "MCWIKI_QUERY_PLAN_TIMEOUT": "20",
            "MCWIKI_QUERY_PLAN_THINKING_TYPE": "enabled",
        })

        self.assertEqual(settings.query_strategy, "step_back")
        self.assertEqual(settings.max_retrieval_queries, 1)
        self.assertEqual(settings.query_plan_timeout, 20.0)
        self.assertEqual(settings.query_plan_thinking_type, "enabled")

    def test_rejects_invalid_query_planning_settings(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_QUERY_STRATEGY"):
            RetrievalSettings.from_env({"MCWIKI_QUERY_STRATEGY": "multi_query"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_RETRIEVAL_QUERIES"):
            RetrievalSettings.from_env({"MCWIKI_MAX_RETRIEVAL_QUERIES": "3"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_RETRIEVAL_QUERIES"):
            RetrievalSettings.from_env({"MCWIKI_MAX_RETRIEVAL_QUERIES": "0"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_QUERY_PLAN_TIMEOUT"):
            RetrievalSettings.from_env({"MCWIKI_QUERY_PLAN_TIMEOUT": "0"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_QUERY_PLAN_THINKING_TYPE"):
            RetrievalSettings.from_env({"MCWIKI_QUERY_PLAN_THINKING_TYPE": " "})

    def test_reranking_defaults_keep_the_fused_ranking(self):
        settings = RetrievalSettings.from_env({})

        self.assertEqual(settings.reranker, "none")
        self.assertEqual(settings.reranker_model, "BAAI/bge-reranker-base")
        self.assertEqual(settings.candidate_limit, 20)
        self.assertEqual(settings.evidence_limit, 8)

    def test_reranking_options_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_RERANKER": "cross_encoder",
            "MCWIKI_RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "MCWIKI_CANDIDATE_LIMIT": "30",
        })

        self.assertEqual(settings.reranker, "cross_encoder")
        self.assertEqual(settings.reranker_model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(settings.candidate_limit, 30)

    def test_rejects_invalid_reranking_settings(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_RERANKER"):
            RetrievalSettings.from_env({"MCWIKI_RERANKER": "llm_reranker"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_RERANKER_MODEL"):
            RetrievalSettings.from_env({"MCWIKI_RERANKER_MODEL": "  "})
        with self.assertRaisesRegex(ValueError, "MCWIKI_CANDIDATE_LIMIT"):
            RetrievalSettings.from_env({"MCWIKI_CANDIDATE_LIMIT": "0"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_CANDIDATE_LIMIT"):
            RetrievalSettings.from_env({
                "MCWIKI_RERANKER": "cross_encoder",
                "MCWIKI_CANDIDATE_LIMIT": "4",
            })

    def test_an_oversized_keep_count_is_only_rejected_when_reranking_is_enabled(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_CANDIDATE_LIMIT": "4",
            "MCWIKI_EVIDENCE_LIMIT": "8",
        })

        self.assertEqual((settings.candidate_limit, settings.evidence_limit), (4, 8))

    def test_corrective_retrieval_defaults_to_a_single_round(self):
        settings = RetrievalSettings.from_env({})

        self.assertEqual(settings.corrective, "none")
        self.assertEqual(settings.corrective_timeout, 15.0)
        self.assertEqual(settings.max_retrieval_rounds, 2)

    def test_corrective_retrieval_options_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_CORRECTIVE": "corrective",
            "MCWIKI_CORRECTIVE_TIMEOUT": "8",
            "MCWIKI_MAX_RETRIEVAL_ROUNDS": "1",
        })

        self.assertEqual(settings.corrective, "corrective")
        self.assertEqual(settings.corrective_timeout, 8.0)
        self.assertEqual(settings.max_retrieval_rounds, 1)

    def test_rejects_invalid_corrective_settings(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_CORRECTIVE "):
            RetrievalSettings.from_env({"MCWIKI_CORRECTIVE": "always"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_CORRECTIVE_TIMEOUT"):
            RetrievalSettings.from_env({"MCWIKI_CORRECTIVE_TIMEOUT": "0"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_RETRIEVAL_ROUNDS"):
            RetrievalSettings.from_env({"MCWIKI_MAX_RETRIEVAL_ROUNDS": "3"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_RETRIEVAL_ROUNDS"):
            RetrievalSettings.from_env({"MCWIKI_MAX_RETRIEVAL_ROUNDS": "0"})

    def test_answer_verification_defaults_to_sending_the_draft_unchecked(self):
        settings = RetrievalSettings.from_env({})

        self.assertEqual(settings.answer_verification, "none")
        self.assertEqual(settings.verification_timeout, 15.0)
        self.assertEqual(settings.max_answer_attempts, 2)

    def test_answer_verification_options_can_be_overridden(self):
        settings = RetrievalSettings.from_env({
            "MCWIKI_ANSWER_VERIFICATION": "verify",
            "MCWIKI_VERIFICATION_TIMEOUT": "8",
            "MCWIKI_MAX_ANSWER_ATTEMPTS": "1",
        })

        self.assertEqual(settings.answer_verification, "verify")
        self.assertEqual(settings.verification_timeout, 8.0)
        self.assertEqual(settings.max_answer_attempts, 1)

    def test_rejects_invalid_answer_verification_settings(self):
        with self.assertRaisesRegex(ValueError, "MCWIKI_ANSWER_VERIFICATION"):
            RetrievalSettings.from_env({"MCWIKI_ANSWER_VERIFICATION": "always"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_VERIFICATION_TIMEOUT"):
            RetrievalSettings.from_env({"MCWIKI_VERIFICATION_TIMEOUT": "0"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_ANSWER_ATTEMPTS"):
            RetrievalSettings.from_env({"MCWIKI_MAX_ANSWER_ATTEMPTS": "3"})
        with self.assertRaisesRegex(ValueError, "MCWIKI_MAX_ANSWER_ATTEMPTS"):
            RetrievalSettings.from_env({"MCWIKI_MAX_ANSWER_ATTEMPTS": "0"})

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
