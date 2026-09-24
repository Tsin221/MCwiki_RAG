# RAG V2 任务完成情况

更新时间：2026-09-24

本文件记录本目录五个任务的实现与验收状态，按 `README.md` 的「统一交接格式」组织。任务文档
中的验收复选框尚未勾选，本文件是当前的唯一完成情况记录。

核对方式：代码与 git 历史逐项比对，后端测试于 2026-09-24 在提交 `be182b1` 上实际执行
（工作区另有一处文档链接修改，不影响测试）。任务文档中的「验收标准」逐条列在下面。

## 总览

| 任务 | 状态 | 提交 | 默认开关 | 主要缺口 |
|---|---|---|---|---|
| 01 Step-back 双路检索 | 已实现、验收通过 | `2c3c498`、`dcacf1a` | `MCWIKI_QUERY_STRATEGY=original` | 无 18 题对照证据 |
| 02 Cross-Encoder 精排 | 已实现，验收缺一项 | `ae9f5bc` | `MCWIKI_RERANKER=none` | 真实模型中文烟雾测试与耗时记录缺失 |
| 03 Corrective RAG | 未开始 | — | — | 依赖 02 |
| 04 答案证据核验 | 未开始 | — | — | 依赖 03 |
| 05 Adaptive 路由 | 未开始 | — | — | 依赖 04 |

01 与 02 都只完成了代码与单元测试，**都没有做过 18 题对照评测**，因此默认开关均为关闭状态。
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

## 任务 03～05：未开始

三个任务均无任何代码、测试或配置，下列文件不存在：

| 任务 | 预计新增文件 | 状态 |
|---|---|---|
| 03 Corrective RAG | `code/rag_correction.py`、`code/tests/test_correction.py` | 均不存在 |
| 04 答案证据核验 | `code/rag_verification.py`、`code/tests/test_verification.py` | 均不存在 |
| 05 Adaptive 路由 | `code/rag_router.py`、`code/rag_pipeline.py`、`code/tests/test_router.py`、`code/tests/test_pipeline.py` | 均不存在 |

依赖关系为串行：03 依赖 02 已合并（已满足），04 依赖 03，05 依赖 04。因此 04 与 05 当前**不满足**
开工前置条件，需按顺序推进。

任务 05 另需修改 `README.md` 的最终流程与配置说明，届时本文件与根目录 README 都需同步更新。

---

## 全局缺口

1. **01、02 都没有 18 题对照评测**，两者的默认开关因此都停留在关闭状态。开启任何一个之前，
   都应按根目录 README 的评测要求补齐对照并保留历史结果。
2. **02 的真实模型烟雾测试缺失**，这是五份任务文档中唯一一条「已在文档中被要求、但从未执行」
   的验收项。
3. 任务文档 `01-…md`、`02-…md` 的验收复选框仍是未勾选状态。本文件记录了逐条结论，但两份
   任务文档本身未更新，新读者仍会看到全空的复选框。
4. 评测集 v2 的 48 题中仅 7 题经人工确认，若后续任务要做定量对比，标注状态会是一个限制。

## 如何更新本文件

任务状态变化时同步更新三处：本文件的总览表、对应任务的「验收标准逐条」表，以及
`README.md` 的执行顺序表。任务文档的验收复选框保持原样，作为任务下发时的原始要求。
