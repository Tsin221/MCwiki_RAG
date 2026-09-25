# RAG V2 任务完成情况

更新时间：2026-09-24

本文件记录本目录五个任务的实现与验收状态，按 `README.md` 的「统一交接格式」组织。任务文档
中的验收复选框尚未勾选，本文件是当前的唯一完成情况记录。

核对方式：代码与 git 历史逐项比对，后端完整测试于 2026-09-24 在提交 `e1ef515`（01～03）与
`c6747fb`（04）上实际执行。任务文档中的「验收标准」逐条列在下面。01～04 均已合并到 `main`：
03、04 分别由分支 `codex/ragv2-corrective`、`codex/ragv2-answer-verification` 快进合并，
两个分支保留但不再更新。04 的分支另外带三个与任务本身无关的显示修复提交，见文末
「附：来源卡片可读性修复」。

## 总览

| 任务 | 状态 | 提交 | 默认开关 | 主要缺口 |
|---|---|---|---|---|
| 01 Step-back 双路检索 | 已实现、验收通过 | `2c3c498`、`dcacf1a` | `MCWIKI_QUERY_STRATEGY=original` | 无 18 题对照证据 |
| 02 Cross-Encoder 精排 | 已实现，验收缺一项 | `ae9f5bc` | `MCWIKI_RERANKER=none` | 真实模型中文烟雾测试与耗时记录缺失 |
| 03 Corrective RAG | 已实现、验收通过 | `c91a49d` | `MCWIKI_CORRECTIVE=none` | 无 18 题对照证据；未在真实 DeepSeek 上跑过 |
| 04 答案证据核验 | 已实现、验收通过 | `c6747fb` | `MCWIKI_ANSWER_VERIFICATION=none` | 无 18 题对照证据；真实模型只跑过 3 题烟雾测试 |
| 05 Adaptive 路由 | 未开始 | — | — | 依赖 04 |

01～04 都只完成了代码与单元测试，**都没有做过 18 题对照评测**，因此默认开关均为关闭状态。
按 `README.md` 共同规则第 9 条，关闭后行为必须与基线一致，这一点有测试保证。

---

## 任务 01：Step-back 双路检索

### 分支与提交

- `2c3c498` feat: add step-back query planning and multi-query fusion
- `dcacf1a` perf: disable model thinking for step-back planning by default

### 修改文件清单

新增（与任务文档「文件边界」的预计完全一致）：

- `code/rag_query.py`（284 行）：`QueryPlan`、`QueryPlanner` 协议、`OriginalQueryPlanner`、
  `DeepSeekStepBackPlanner`、`build_query_planner()`
- `code/rag_retrieval/multi_query.py`（186 行）：`fuse_rankings()`、`MultiQueryRetriever`
- `code/tests/test_query.py`（23 项用例）
- `code/tests/test_multi_query.py`（15 项用例）

修改：

- `code/rag_api.py`：`/answers` 接入多查询检索，装配 `MultiQueryRetriever`
- `code/rag_settings.py`：策略、查询数上限、规划超时、thinking 开关
- `code/tests/test_api.py`、`code/tests/test_settings.py`、`.env.example`

### 验证命令与结果

任务文档规定的针对性验证：

```powershell
uv run --with pytest python -m pytest -q tests/test_query.py tests/test_multi_query.py tests/test_api.py
```

本次核对的是完整后端测试，于 2026-09-24 实测通过：**185 passed、34 subtests passed、9.02s**。
本任务新增的用例为 `test_query.py` 23 项、`test_multi_query.py` 15 项，另有 `test_api.py` 与
`test_settings.py` 的增补用例。本次未单独重跑上面的针对性命令。

### 默认开关与启用方法

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MCWIKI_QUERY_STRATEGY` | `original` | 改 `step_back` 启用 |
| `MCWIKI_MAX_RETRIEVAL_QUERIES` | `2` | 硬边界 1～2，超出报错 |
| `MCWIKI_QUERY_PLAN_TIMEOUT` | `15` | 规划调用超时秒数 |
| `MCWIKI_QUERY_PLAN_THINKING_TYPE` | `disabled` | `dcacf1a` 实测开启 thinking 每次规划多约 7 秒且无收益，故默认关闭 |

### 失败回退

已实现且有对应测试，规划失败一律只检索原问题，不让 `/answers` 失败：模型返回非法 JSON
（`test_invalid_model_output_falls_back_to_the_original_question`）、响应体残缺
（`test_malformed_response_body_falls_back_to_the_original_question`）、传输异常
（`test_transport_failures_falls_back_to_the_original_question`）、模型拒绝回答
（`test_declined_abstraction_keeps_only_the_original_question`）、重复抽象问题
（`test_duplicate_abstraction_is_dropped`）、无 chat 客户端时降级
（`test_step_back_without_a_chat_client_degrades_to_the_original_question`）。

### 验收标准逐条

| 验收标准 | 结论 | 依据 |
|---|---|---|
| `original` 模式与当前基线一致 | 通过 | `test_original_strategy_returns_the_baseline_ranking`、`test_original_strategy_never_calls_a_model` |
| `step_back` 确实执行两路混合检索 | 通过 | `test_step_back_runs_both_queries_in_plan_order`、`test_step_back_with_a_chat_client_uses_the_model` |
| 相同 chunk 只保留一次并获得融合优势 | 通过 | `test_keeps_one_entry_per_chunk_and_rewards_two_route_hits`、`test_chunk_found_by_both_queries_is_kept_once_and_ranks_first`、`test_counts_a_duplicate_inside_one_ranking_once`、`test_keeps_the_metadata_of_the_first_appearance` |
| 非法输入与异常可靠退回原问题 | 通过 | 见上一节列举的 6 项用例 |
| 查询数硬限制为 2 | 通过 | `test_never_exceeds_the_hard_query_limit`、`test_rejects_a_max_query_budget_above_the_hard_limit`、`test_max_queries_of_one_keeps_only_the_original_question` |
| `/search` 契约不变 | 通过 | `test_search_stays_a_single_query_endpoint`、`test_search_returns_camel_case_hybrid_results` |
| 完整后端测试通过 | 通过 | 185 passed（2026-09-24 实测） |

### 未解决风险

- 没有任何 18 题对照结果，因此无法判断 `step_back` 相比 `original` 是提升还是退化。
- 查询数上限 2 同时硬编码在 `rag_settings.py` 与 `QueryPlan` 的校验中，改动需同时修改两处。

### 交付给任务 02 的接口

统一的「返回较大候选池」检索入口：`MultiQueryRetriever` 只产出 `HybridResult` 列表，
候选来自原查询还是 Step-back 查询对下游透明。任务 02 直接包装该入口，未感知查询规划。

---

## 任务 02：Cross-Encoder 候选精排

### 分支与提交

- `ae9f5bc` feat: add pluggable cross-encoder reranking

### 修改文件清单

新增（与任务文档的预计一致）：

- `code/rag_reranker.py`（330 行）：`Reranker` 协议、`NoopReranker`、`CrossEncoderReranker`、
  `build_reranker()`、`RerankingRetriever`
- `code/tests/test_reranker.py`（24 项用例）

修改：

- `code/rag_api.py`：`/answers` 接入精排，`_reranker_check()` 反映到 `/ready`
- `code/rag_retrieval/hybrid.py`：`HybridResult` 增加 `reranker_score`，与 RRF 分数分开保存
- `code/rag_settings.py`：精排开关、模型名、候选池大小，并强制候选池 ≥ 证据上限
- `code/pyproject.toml`、`code/uv.lock`：新增可选依赖 `reranker = ["sentence-transformers"]`
- `code/tests/test_api.py`、`code/tests/test_settings.py`、`.env.example`、`README.md`

任务文档要求「不要修改 chunk、索引内容、RRF 公式或证据相邻合并算法」，实际改动符合此边界。

### 验证命令与结果

任务文档规定的针对性验证：

```powershell
uv run --with pytest python -m pytest -q tests/test_reranker.py tests/test_hybrid.py tests/test_api.py tests/test_settings.py
```

本次核对的是完整后端测试，于 2026-09-24 实测通过：**185 passed、34 subtests passed、9.02s**。
本任务新增 `test_reranker.py` 24 项用例，另有 `test_api.py` 与 `test_settings.py` 的增补用例。
**未执行**真实模型烟雾测试（见下方验收缺口）。本次未单独重跑上面的针对性命令。

### 默认开关与启用方法

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MCWIKI_RERANKER` | `none` | 改 `cross_encoder` 启用 |
| `MCWIKI_RERANKER_MODEL` | `BAAI/bge-reranker-base` | 模型可替换 |
| `MCWIKI_CANDIDATE_LIMIT` | `20` | 精排候选池，必须 ≥ `MCWIKI_EVIDENCE_LIMIT`（8），否则启动即报错 |

启用前需先安装可选依赖：`uv sync --extra reranker`（拉取 sentence-transformers/torch，约 1.1 GB）。
模型在应用启动时加载一次并常驻，`/ready` 经 `_reranker_check()` 报告加载状态。

### 失败回退

已实现且有对应测试，任何失败都保持 RRF 顺序而不是报错：推理异常
（`test_inference_failure_returns_the_rrf_order_without_scores`）、分数不可用
（`test_unusable_scores_fall_back_instead_of_misordering_the_pool`）、模型无法加载
（`test_a_model_that_cannot_load_degrades_to_the_rrf_order`）、分数并列时保持原顺序
（`test_equal_scores_keep_the_incoming_rrf_order`）。未安装模型库时进程仍可启动
（`test_the_disabled_reranker_needs_no_model_library`）。

### 验收标准逐条

| 验收标准 | 结论 | 依据 |
|---|---|---|
| 假模型下证明精排改变顺序并严格截取 Top-K | 通过 | `test_reorders_the_pool_by_model_score_and_truncates_to_the_limit` |
| 重复 chunk 不会因精排重新出现 | 通过 | `test_a_duplicate_chunk_is_scored_and_returned_once` |
| `NoopReranker` 完整保持输入顺序 | 通过 | `test_keeps_the_fused_order_and_truncates_to_the_limit` |
| 异常时回退 RRF 顺序且接口仍可用 | 通过 | 见上一节列举的 4 项用例 |
| 配置为正且 `rerank_limit <= candidate_limit` | 通过 | `test_rejects_a_non_positive_limit`、`test_rejects_invalid_limits_and_construction_arguments`、`test_settings.py` 的候选池约束用例 |
| **本地启用真实模型完成一次中文精排，并在交接中记录模型和耗时** | **未满足** | 仓库内无任何精排运行记录，`data/evaluation/` 下无相关报告 |
| 完整后端测试通过 | 通过 | 185 passed（2026-09-24 实测） |

另外两项任务文档要求已由测试覆盖：精排分数不冒充 RRF 分数
（`test_keeps_retrieval_metadata_and_never_rewrites_the_rrf_score`）、精排使用原始用户问题而非
抽象问题（`test_the_reranker_scores_the_user_question_not_the_step_back_question`）、
第二轮复用同一精排器（`test_a_second_round_uses_the_same_reranker`）。

### 未解决风险

1. **验收第 6 项未完成**：既没有真实模型的中文精排运行记录，也没有耗时数据，因此无法判断
   开启精排的实际收益与单次延迟代价。这是本任务唯一的验收缺口。
2. 精排被 `threading.Lock` 全局串行化（`test_concurrent_requests_never_enter_the_model_at_the_same_time`
   固化了该行为），并发请求会在精排处排队，吞吐上限受限。
3. 精排只重排已召回候选，不能找回召回阶段遗漏的证据，`MCWIKI_CANDIDATE_LIMIT=20` 之外遗漏的
   期望来源仍无法补救。

### 交付给任务 03 的接口

统一的「候选召回并精排」入口：`RerankingRetriever`，其 `rerank()` 被显式暴露为公共方法，
供后续纠正检索把第二轮候选交给同一精排器，无需另写排序逻辑。任务 03 可直接复用。

---

## 任务 03：Corrective RAG 有界纠正检索

### 分支与提交

- `c91a49d` feat: add bounded corrective retrieval with one query rewrite
- `e1ef515` docs: record task 03 corrective retrieval status

分支 `codex/ragv2-corrective` 已快进合并到 `main`（`main` 上的这两个提交即为其内容）。

### 修改文件清单

新增（与任务文档「文件边界」的预计完全一致）：

- `code/rag_correction.py`（402 行）：`EvidenceAssessment`、`CorrectiveResult`、
  `EvidenceAssessor` 协议、`DeepSeekEvidenceAssessor`、`parse_assessment()`、
  `build_evidence_assessor()`、`CorrectiveCoordinator`
- `code/tests/test_correction.py`（32 项用例，含 20 项 subTest）

修改：

- `code/rag_api.py`：`/answers` 接入纠正检索，`sources` 改为在最终证据确定后发送，
  `create_app()` 增加 `corrective_assessor` 注入点，`_default_lifespan` 装配评估器
- `code/rag_reranker.py`：`RerankingRetriever` 拆出公共 `recall()`，`search()` 行为不变
- `code/rag_settings.py`：纠正开关、评估超时、最大轮数
- `code/tests/test_api.py`（新增 11 项用例）、`code/tests/test_settings.py`（新增 3 项用例）、
  `.env.example`、`README.md`

未改动 chunk、索引、RRF 公式、证据相邻合并与引用格式；未加入网页搜索、LangGraph、递归
Agent、第三轮检索。

### 验证命令与结果

任务文档规定的针对性验证：

```powershell
uv run --with pytest python -m pytest -q tests/test_correction.py tests/test_api.py
```

实测通过：**72 passed、20 subtests passed、5.66s**。

完整后端测试，于 2026-09-24 在提交 `c91a49d` 上实测通过：

```powershell
uv run --with pytest python -m pytest -q
```

**231 passed、54 subtests passed、9.42s**（新增 46 项：纠正 32、接口 11、配置 3）。

### 默认开关与启用方法

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MCWIKI_CORRECTIVE` | `none` | 改 `corrective` 启用 |
| `MCWIKI_CORRECTIVE_TIMEOUT` | `15` | 单次证据评估调用超时秒数 |
| `MCWIKI_MAX_RETRIEVAL_ROUNDS` | `2` | 硬边界 1～2，超出报错；1 表示只评估不纠正 |

评估调用复用回答模型的 DeepSeek 配置（`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` /
`DEEPSEEK_MODEL`），不需要额外配置；评估器 thinking 固定关闭（`DEFAULT_ASSESSMENT_THINKING_TYPE`），
理由与任务 01 相同：评估发生在生成之前，每请求都要付一次延迟。

### 纠正流程与控制流边界

```text
第一轮 recall(原问题, candidate_limit) → rerank → evidence → assess
  ├─ sufficient            → 结束（只检索一轮）
  ├─ 改写查询为空/等同原问题 → 结束（不重复检索）
  └─ 不充分且还有轮数配额    → 第二轮 recall(改写查询) → 与第一轮按 chunk_id
                             RRF 合并（上限 candidate_limit）→ rerank → evidence → assess
                             仍不充分 → sufficient=false，拒答
```

- 循环条件同时受 `max_retrieval_rounds` 与「改写查询可用」两个条件约束，模型输出无法扩大
  轮数：第二次评估即使再给出新改写，也因轮数用尽而结束；
- 第二轮候选与第一轮合并后只走一次精排与一次证据预算，精排使用**原始用户问题**评分；
- 评估器不可用（异常、超时、非法 JSON）时保留已检索到的证据并记录 `WARNING`，不阻塞回答；
- 空证据无论评估器如何回答都不会被判定为充分（`sufficient=bool(evidence) and …`），
  因此不存在「无证据生成答案」的路径。

### 失败回退

已实现且有对应测试：非法 JSON、字段类型错误、`rewritten_query` 超长
（`test_rejects_every_malformed_reply` 的 10 项 subTest）、传输异常与超时
（`test_transport_failures_and_timeouts_raise_the_same_error`）、评估器整体不可用
（`test_a_failing_assessor_keeps_the_single_round_result`、
`test_an_unusable_reply_keeps_the_retrieved_evidence`、`test_a_timeout_keeps_the_retrieved_evidence`）、
开关打开但没有评估器（`test_corrective_without_an_assessor_stays_single_round`）。
所有回退都保持单轮检索行为。

### 验收标准逐条

| 验收标准 | 结论 | 依据 |
|---|---|---|
| 充分证据路径只检索一轮，不产生多余改写 | 通过 | `test_a_sufficient_first_round_retrieves_exactly_once`、`test_a_sufficient_first_round_answers_from_that_round_only` |
| 不充分路径只允许一个改写查询和一次补充检索 | 通过 | `test_a_second_round_merges_reranks_and_rechecks`（恰好两次 recall）、`test_a_second_insufficient_check_refuses_and_never_generates`（评估器持续给出改写仍停在两轮） |
| 两轮结果按 `chunk_id` 去重，并经过统一精排与证据预算 | 通过 | `test_a_second_round_merges_reranks_and_rechecks`（合并池 `["b","a","c"]`，`b` 只出现一次）、`test_the_merged_pool_respects_the_candidate_budget` |
| 第二次检查仍不充分时不调用答案生成模型 | 通过 | `test_a_second_insufficient_check_refuses_and_never_generates`（`answer_client.calls == []`）、`test_no_evidence_is_never_passed_to_the_generator` |
| 评估器非法输出、超时和异常均有确定回退行为 | 通过 | 见上一节列举的用例 |
| 循环次数不能由模型输出扩大 | 通过 | 见上一节的用例；轮数上限同时硬编码在 `MAX_RETRIEVAL_ROUNDS` 与 `CorrectiveCoordinator` 校验中，`test_rejects_a_round_budget_outside_the_hard_limit` |
| SSE 的 `sources` 与最终实际交给生成模型的证据一致 | 通过 | `sources` 在证据确定后发送（`rag_api.py` 的 `event_stream`），`test_a_corrected_round_announces_the_merged_evidence_only` 断言来源、评估器输入与生成模型输入三者一致；拒答时来源为空 |
| 完整后端测试通过 | 通过 | 231 passed（2026-09-24 实测） |

另外两项设计取舍也已由测试固化：改写查询等同原问题时不再检索
（`test_a_rewrite_of_the_original_question_is_not_retrieved_again`）、空首轮允许一次改写
（`test_an_empty_first_round_is_still_assessed`、`test_an_empty_first_round_can_still_be_corrected`）。

### 未解决风险

1. **没有 18 题对照评测**，因此无法判断开启纠正检索后的净收益；每个请求多一次评估调用是
   确定的成本，收益不确定，默认保持关闭。
2. **从未调用真实 DeepSeek 跑过**：所有评估器测试都使用假 HTTP 客户端，评估提示词的实际
   输出质量、改写查询的可用性与真实超时分布均未验证。
3. 评估忽略 `missing_aspects`，只用 `rewritten_query` 一个字段驱动纠正；该字段目前仅供
   观察与后续任务使用，未被消费，也未计入回答提示词。
4. 轮数上限 2 同时存在于 `rag_settings.py` 与 `CorrectiveCoordinator` 校验中，改动需同步两处。

### 交付给任务 04 的接口

`CorrectiveCoordinator.run(question)` 返回 `CorrectiveResult(evidence, retrieval_rounds,
sufficient)`：证据已经过统一精排与证据预算，`sufficient` 是最终的充分性判断（评估器不可用时
回退为「有证据即充分」）。`/answers` 只把 `sufficient` 为真时的证据交给生成模型。任务 04 的
核验只能处理这份已定稿的证据，不得自行再次检索。

---

## 任务 04：Self-RAG 式答案证据核验

### 分支与提交

- `c6747fb` feat: add bounded answer verification with one rewrite
- `b973627` docs: record task 04 answer verification status
- `a503471` docs: record the task 04 docs commit and two interface tests

分支 `codex/ragv2-answer-verification` 已快进合并到 `main`（`main` 上的这三个提交即为其内容），
该分支保留但不再更新。分支上另有三项与任务 04 无关的来源卡片显示修复
（`016f5b2`、`2b44677`、`13b5584`），见文末「附：来源卡片可读性修复」。

### 修改文件清单

新增（与任务文档「文件边界」的预计一致）：

- `code/rag_verification.py`（596 行）：`ClaimCheck`、`CitationAudit`、`ModelVerdict`、
  `AnswerVerification`、`VerificationResult`、`AnswerVerifier` 与 `DraftAnswerer` 协议、
  `citation_audit()`、`build_verification()`、`rewrite_feedback()`、`parse_verdict()`、
  `DeepSeekAnswerVerifier`、`build_answer_verifier()`、`AnswerVerificationCoordinator`
- `code/tests/test_verification.py`（43 项用例，含 30 项 subTest）

修改：

- `code/rag_answer.py`：`DeepSeekAnswerClient.collect_answer()` 新增（整稿收集），
  `stream_answer()` 与 `_context_message()` 增加可选 `feedback`，为空时请求体与基线逐字节一致
- `code/rag_api.py`：`/answers` 接入核验流程，新增 `_answer_verifier()`（开关与依赖检查）、
  `_verification_status()`，`create_app()` 增加 `answer_verifier` 注入点，`_default_lifespan`
  装配核验器
- `code/rag_settings.py`：核验开关、核验超时、最大答案版本数
- `code/tests/test_api.py`（新增 14 项用例）、`code/tests/test_answer.py`（新增 1 项）、
  `code/tests/test_settings.py`（新增 3 项）、`.env.example`、`README.md`

未改动检索、重排、chunk、证据编号与相邻合并规则；未改动前端；未加入第三轮检索或自由循环。

### 验证命令与结果

任务文档规定的针对性验证：

```powershell
uv run --with pytest python -m pytest -q tests/test_verification.py tests/test_answer.py tests/test_api.py
```

实测通过：**105 passed、30 subtests passed、8.42s**。

完整后端测试，于 2026-09-24 在提交 `c6747fb` 上实测通过：

```powershell
uv run --with pytest python -m pytest -q
```

**292 passed、84 subtests passed、12.99s**（新增 61 项：核验 43、接口 14、回答 1、配置 3，
另有 30 项 subTest）。

### 默认开关与启用方法

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MCWIKI_ANSWER_VERIFICATION` | `none` | 改 `verify` 启用 |
| `MCWIKI_VERIFICATION_TIMEOUT` | `15` | 单次核验调用超时秒数 |
| `MCWIKI_MAX_ANSWER_ATTEMPTS` | `2` | 硬边界 1～2，超出报错；1 表示只核验不重写 |

核验调用复用回答模型的 DeepSeek 配置（`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` /
`DEEPSEEK_MODEL`），不需要额外配置；核验器 thinking 固定关闭
（`DEFAULT_VERIFICATION_THINKING_TYPE`），理由与任务 01/03 相同：核验发生在每次生成之后，
延迟由每个请求承担。

### 核验流程与控制流边界

```text
生成完整草稿（collect_answer，不向浏览器发送）
  → 确定性引用检查（解析 [n]）
      ├─ 没有引用 / 引用越界 → 直接判失败，不调用核验模型
      └─ 引用合法            → 调用核验模型，返回原子声明与逐条支持判断
  → 合并判定：supported（引用合法 ∧ 声明非空 ∧ 无不受支持声明）∧ useful（覆盖全部子问题）
      ├─ 通过 → 发送该草稿
      └─ 失败且还有版本配额 → 用同一份证据 + 核验发现重写一次 → 再次核验
                                ├─ 通过 → 发送重写稿
                                └─ 仍失败 → 拒答（insufficientEvidence）
```

- `supported` 由代码计算，不向模型索取：模型只能逐条给声明与支持判断，无法把整篇草稿
  判为「可用」；
- 确定性引用检查先于模型调用，越界引用的草稿不会消耗核验调用
  （`test_an_invented_citation_is_stopped_before_the_verifier_is_called`）；
- 生成与核验各最多两次（`AnswerVerificationCoordinator` 的 `max_attempts` 同时硬编码在
  `MAX_ANSWER_ATTEMPTS` 中），模型输出无法扩大；
- 重写只能复用最终证据：`collect_answer` 的第二版请求体以第一版逐字为前缀，只追加
  `rewrite_feedback()` 生成的有限指令（≤900 字符、≤6 条问题），核验器不可用时连重写都不会
  触发；
- 核验器故障（异常、超时、非法 JSON、缺键）不是失败而是「不可用」：保留当前草稿、记录
  `WARNING`、`done` 标记 `verification=unavailable`，不伪装为已验证；
- 空证据路径不受影响：`sufficient=false` 时仍在生成之前拒答，不进入核验。

### 失败回退

已实现且有对应测试：核验器抛异常与返回非法结构
（`test_a_broken_verifier_keeps_the_draft_without_claiming_it_was_checked`）、第二次核验超时
（`test_a_verifier_that_breaks_on_the_second_check_keeps_the_rewrite`）、模型回复缺
`claims`/`useful` 键、类型错误、编号非正整数、文本超长
（`test_rejects_every_malformed_reply` 的 20 项 subTest）、传输异常与错误状态
（`test_transport_failures_and_timeouts_raise_the_same_error`）、开关打开但没有核验器
（`test_verification_without_a_verifier_falls_back_to_the_draft`）、开关打开但回答客户端
不支持整稿收集（`test_verification_needs_an_answer_client_that_can_collect_a_draft`）。
所有回退都保持任务 03 的流式行为，并记录 `WARNING`。

### 验收标准逐条

| 验收标准 | 结论 | 依据 |
|---|---|---|
| 引用 `[n]` 必须指向实际证据编号，越界编号不会进入最终答案 | 通过 | `citation_audit()` 先于模型核验执行；`test_an_out_of_range_citation_never_passes`、`test_a_claim_that_points_at_an_unknown_number_never_passes`、接口层 `test_an_invented_citation_is_stopped_before_the_verifier_is_called`（断言 `"[9]" not in response.text`） |
| 每个模型识别出的关键事实都有引用并被对应证据支持 | 通过 | `build_verification()` 把「无引用」「引用不存在」「模型判为不受支持」的声明一律计入 unsupported；`test_a_claim_without_a_citation_never_passes`、`test_an_unsupported_claim_is_reported_with_its_reason`、接口层 `test_a_rejected_draft_never_reaches_the_browser` |
| 首稿失败时最多重写一次，重写后必定再次核验 | 通过 | `test_a_failed_first_draft_is_rewritten_once_and_checked_again`（两次生成、两次核验，且第二次核验读到的是重写稿）、`test_a_single_attempt_budget_never_rewrites`、接口层 `test_a_rejected_draft_never_reaches_the_browser` |
| 第二版仍失败时返回明确的资料不足状态 | 通过 | `test_a_second_failure_withholds_the_rewrite`；接口层 `test_a_second_failure_becomes_an_explicit_insufficient_status`（`delta` 为 `INSUFFICIENT_EVIDENCE_MESSAGE`，`done` 为 `{"status":"insufficientEvidence","verification":"unsupported"}`） |
| 浏览器不会收到后来被判失败的初稿内容 | 通过 | 开启核验时先 `collect_answer` 再发送，只有最终稿进入 `delta`；接口层断言被判失败的初稿文本不出现在整段响应中 |
| 核验关闭时回答行为与任务 03 基线一致 | 通过 | `test_the_switch_off_streams_the_baseline_answer`（注入了核验器也不调用、`done` 无 `verification` 字段）、`test_settings.py` 的默认值用例；03 的既有接口用例全部保持通过 |
| 核验器异常路径有测试并带明确状态 | 通过 | `test_a_broken_verifier_answers_with_an_unavailable_status`（`verification=unavailable`，且错误详情不进入响应）、`test_verification_without_a_verifier_falls_back_to_the_draft` |
| 完整后端测试通过 | 通过 | 290 passed、84 subtests passed（2026-09-24 实测） |

任务文档另有两项要求在测试中固化：核验对象是完整草稿（`collect_answer` 把整条已校验流拼成
一个字符串，`test_collects_the_whole_draft_and_appends_rewrite_feedback`）、子问题遗漏也算
失败并可触发一次重写（`test_a_supported_answer_that_misses_a_sub_question_is_not_acceptable`、
`test_a_sub_question_that_was_never_answered_is_rewritten`）。

与任务 03 的衔接同样有测试：两者同时开启时只检索一轮，核验器读到的证据与 `sources` 事件、
生成模型输入三者一致（`test_verification_follows_corrective_retrieval_on_the_same_evidence`），
空证据路径既不生成也不核验、且 `done` 不带 `verification` 字段
（`test_a_refused_answer_is_never_generated_or_verified`）。

### 真实模型烟雾测试（2026-09-24 实测）

任务 02、03 的已知缺口之一是「从未跑过真实模型」，本任务补上了这一类验证。方法：在本地真实
链路上（BM25 245 MB 索引 + Qdrant `mcwiki_chunks` 41,368 点 + Ollama `qwen3-embedding:0.6b`，
`DEEPSEEK_MODEL=deepseek-flash`），`MCWIKI_ANSWER_VERIFICATION=verify`，对
`data/evaluation/retrieval_questions.json` 的 3 道可答题各跑一次 `/answers`；随后用
`AnswerVerificationCoordinator` 直接复跑同一份证据，读取声明级判定与分阶段耗时。运行用的是
一次性临时脚本，跑完即删、未提交（共同规则第 7 条）。

| 题目 | 证据数 | 草稿字符 | 生成 | 核验 | 模型判定 | 结果 |
|---|---|---|---|---|---|---|
| `redstone-repeater-delay` | 7 | 508 | 1.60s | 2.41s | 8 条声明，0 条不受支持，useful=true | 一次通过，`verified` |
| `brewing-stand-fuel` | 5 | 318 | 1.07s | 2.02s | 9 条声明，0 条不受支持，useful=true | 一次通过，`verified` |
| `shulker-box-storage` | 6 | 213 | 1.10s | 1.52s | 6 条声明，0 条不受支持，useful=true | 一次通过，`verified` |

三点结论：

1. **核验器确实返回逐条声明**（6～9 条/篇），没有出现「正常回答被判为无声明」这一最担心的
   情形；3 道题全部一次通过，未触发重写。
2. **15 秒超时余量充足**：单次核验实测 1.52～2.41s，约为超时的 1/6～1/10；`unavailable`
   在本次运行中没有出现。
3. **`/answers` 端到端符合契约**：三题的 `done` 都是
   `{"status":"answered","verification":"verified"}`，每次都只有 1 个 `delta` 事件（整稿一次
   发送），`sources` 数量与证据一致；关闭核验的对照组为 326/195/83 个 `delta`，`done` 无
   `verification` 字段。端到端总耗时两种模式都在 0.76～5.39s 之间，首个请求含连接建立，
   样本量太小，不足以给出可信的延迟增量；上表的分阶段数据更可靠。

另用两个构造用例检验失败路径（均直接驱动协调器，不经过检索）：

- 把「请告诉我现在私人服务器里有哪些玩家在线」配上红石类证据：第一版是诚实的「证据不足」且
  **没有任何引用** → 确定性检查直接判失败（**未调用核验模型**）→ 重写后模型改写成带
  `[1]…[7]` 的「证据中没有相关内容，资料不足」→ 核验器判 4 条声明全部受支持，最终发送。
  观察：引用规则会把「无引用的诚实拒答」推向「带引用的诚实拒答」，结论仍正确，但读者会看到
  一段带编号引用、实际上并未回答问题的文字。
- 完全无证据：两版都是无引用的拒答 → 两次都被确定性检查拒绝 → `withheld=true`，`issues` 为
  「回答没有标注任何证据编号。」与「核验没有返回任何可判定的事实声明。」，核验模型一次都
  没有被调用。真实链路上无证据会在生成前就拒答，此用例是人工构造的边界验证。

样本仍是 3 道题加 2 个人工用例、单一模型（`deepseek-flash`）、单次运行，不能替代 18 题对照。

### 未解决风险

1. **没有 18 题对照评测**，也无法判断开启核验后的净收益；每个被核验的请求至少多一次核验
   调用（实测 1.5～2.4s），重写时再多一次生成与一次核验，成本确定而收益未测，默认保持关闭。
2. **真实模型只跑过一次 3 题烟雾测试**（见上一节），且用的是 `deepseek-flash`：核验提示词的
   实际判定质量只在 3 篇回答、共 23 条声明上被间接观察，没有人工核对过这些声明是否真的被
   引用证据支持；更换模型或改写提示词都需要重跑。原先担心的「真实模型习惯性返回空 claims」
   在这 3 题上没有出现，但样本不足以排除它。
3. `max_answer_attempts` 上限 2 同时存在于 `rag_settings.py` 与 `AnswerVerificationCoordinator`
   校验中（与任务 01/03 的同类常量一样），改动需同步两处。
4. 开启核验后首字节延迟增加：整稿生成完成后才开始发送 `delta`，前端在核验结束前只能看到
   `sources`；实测核验通过的回答以**单个** `delta` 事件下发，前端的逐字输出效果在核验模式下
   消失（基线为数百个 `delta`）。
5. 核验只覆盖回答文本与引用，不核验 `sources` 卡片本身；`sources` 仍按任务 03 的规则在生成
   之前发送，因此「拒答时浏览器已看到来源卡片」这一组合只可能出现在核验失败路径上。
6. 引用规则本身可能被绕过：实测中「无引用的诚实拒答」会被重写成「带引用的诚实拒答」，
   核验器把这类声明判为受支持。这不是错误答案，但它说明「有引用」并不等价于「回答了问题」，
   `useful` 的判定质量比引用检查更依赖模型。

### 交付给任务 05 的接口

`AnswerVerificationCoordinator(answerer=…, verifier=…, max_attempts=…).run(question, evidence)`
返回 `VerificationResult(answer, verification, answer_attempts)`：只接收「原始用户问题 + 最终
证据」，返回可发送的答案或一个应被拒答的结果（`withheld` 为真），不涉及检索、查询规划或
路由。`answer_verification=none` 时该流程整体不存在，`/answers` 回到任务 03 的流式基线。
任务 05 的 Adaptive Router 只负责选择取证路径，不得复制核验逻辑。

---

## 任务 05：未开始

无任何代码、测试或配置，`code/rag_router.py`、`code/rag_pipeline.py`、
`code/tests/test_router.py`、`code/tests/test_pipeline.py` 均不存在。依赖 04（已实现并合并到
`main`，提交 `c6747fb`）：可直接基于当前 `main`（`13b5584`）开工，按共同规则第 1 条另建分支。

任务 05 另需修改 `README.md` 的最终流程与配置说明，届时本文件与根目录 README 都需同步更新。

---

## 全局缺口

1. **01～04 都没有 18 题对照评测**，四者的默认开关因此都停留在关闭状态。开启任何一个
   之前，都应按根目录 README 的评测要求补齐对照并保留历史结果。
2. **02 的真实模型烟雾测试缺失**，03 也**从未调用真实 DeepSeek 评估器**；04 已于
   2026-09-24 补上一次 3 题真实烟雾测试（`deepseek-flash`，见任务 04 内一节，脚本为一次性
   临时脚本、未提交）。因此仓库内仍没有可复现的真实模型验证入口，02 与 03 的真实运行记录
   依旧为空。
3. 任务文档 `01-…md`～`04-…md` 的验收复选框仍是未勾选状态。本文件记录了逐条结论，
   但四份任务文档本身未更新，新读者仍会看到全空的复选框。
4. 评测集 v2 的 48 题中仅 7 题经人工确认，若后续任务要做定量对比，标注状态会是一个限制。
5. `docs/ARCHITECTURE.md` 记录的是 `main` 上的结构，03 合并后它有两处过期：一是缺少纠正检索
   这一层（第 2 节请求链路、第 3 节分层、第 4 节可插拔维度表、第 5 节降级表各需补一行），
   二是本任务改动使该文件引用的行号偏移（`rag_api.py` 的 `create_answer` 350→405、
   `_retrieval_checks` 106→112、`_reranker_check` 124→130、单轮检索调用 376→261
   （已移入新的 `_collect_evidence`，239 行起）、装配点 174→180、`/search` 检索 319→379、
   SSE 事件段 398-420→451 起；`rag_reranker.py` 的 `build_reranker` 258→264、
   `RerankingRetriever` 272→278）。该文件按自身约定在专门的文档提交里更新，本次未改动。
   04 使漂移进一步扩大：核验这一层、`rag_verification.py` 与 `rag_answer.py` 的
   `collect_answer` 均未进入该文件，`rag_api.py` 的行号再次整体偏移。

## 附：来源卡片可读性修复（不在任务 04 范围内）

跑完任务 04 的真实模型烟雾测试后，回头用真实数据看来源卡片，发现卡片文字在真实语料下难以
阅读，且**不是任务 04 引入的**（本任务未改动 `rag_evidence.py` 的 `_excerpt` 与 `rag_api.py`
的 `_public_source`；最后一次改动分别是很早的 `393e978` 与 `178cebc`）。三个原因叠加：

1. 切块从文章中间开始，摘录取的又是块开头，于是每张卡片都是半句开场；
2. 220 字的截断点落在任意位置，结尾也是半句；
3. Wiki 表格在语料里是「一行一个单元格」，摘要又把空白压成一行，于是表格块变成
   `false true powered 0x1` 这样的裸值串，看起来像重复 bug（实测 `方块状态` 一块 113 行、
   平均行长 6 字）。同页的重复单元格也确实存在，但那是语料与表格结构的产物，不是拼接错误。

在同一页面出现多张同名卡片的问题上，`docs/specs/qa-web-mvp.md` 第 17 节早已把
「同一 Wiki 页面多个 chunk 导致来源卡片重复」列为已知风险。

修复分两个提交：

- `016f5b2`（后端 `code/rag_evidence.py`）：摘要改为「正文优先 + 整句截断 + 片段标记」——
  丢掉表格里的裸值单元格、保留描述性单元格，在句末或分句处收尾，并在从文章中间开始的条目
  前加省略号，让省略看起来是有意的。位置未知（`chunk_index is None`）时不加标记。
  实测效果：`方块状态` 卡片从 `false true 红石中继器处于锁存状态 powered false true …`
  变为 `…红石中继器处于锁存状态 红石中继器没有被锁存，可以中继信号 方块接收到了红石信号 …`；
  `红石中继器` 卡片改为在句末收尾而不是断在词中间。新增 9 项用例（`test_evidence.py`）。
- `2b44677`（前端 `lib/sources.ts`、`components/chat/SourceCard.tsx`、`App.tsx`、`app.css`）：
  按 URL 归组，一个页面一张卡片，保留其首个证据的位置，组内每段仍带自己的 `[n]`；
  标题旁显示「N 段」，标题行显示「N 条证据 · M 个页面」。新增 3 项单元用例与 1 项界面用例。

影响面与边界：`excerpt` 只用于卡片，回答模型读的仍是完整 `text`，因此回答质量、检索、RRF、
相邻合并与引用编号规则都没有改变；`sources` 事件的字段与结构未变（归组在前端完成），
`docs/specs/qa-web-mvp.md` 的事件载荷与前端类型不需要修改；`rag_evidence.py` 新增常量与
`_AssembledCandidate` 的一个字段，属显示层内部结构。后端 301 项、前端 14 项测试通过。

**仍未修复**：拼接块内部的重复段落（例如 `酿造台` 那张卡片里「原主机版 加入了酿造台」出现
两次，`Tutorial:无延迟科技` 的 4 块合并后有一段约 60 字的正文重复）。它来自 `_merge_adjacent`
只在前后缀上找最长重叠，且页面自身也有重复内容；修它会改变**模型读到的证据文本**，属于任务
02/03 冻结的相邻合并算法，需单独立项。

## 如何更新本文件

任务状态变化时同步更新三处：本文件的总览表、对应任务的「验收标准逐条」表，以及
`README.md` 的执行顺序表。任务文档的验收复选框保持原样，作为任务下发时的原始要求。
