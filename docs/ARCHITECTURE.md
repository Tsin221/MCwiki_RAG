# 系统架构与扩展指南

本文说明 MCwiki RAG 当前在 `main` 上的实现结构：各部分组成、在线请求链路、可插拔的扩展点、
必须保持的稳定契约，以及已知的架构权衡。

这类内容只在架构发生变化时更新，不随任务进度变化。当前任务进度见
[`tasks/ragV2/STATUS.md`](tasks/ragV2/STATUS.md)，接口与错误契约见
[`specs/qa-web-mvp.md`](specs/qa-web-mvp.md)，运行配置的权威清单见仓库根目录 `.env.example`。

## 1. 三个组成部分

| 部分 | 位置 | 职责 | 是否在在线链路中 |
|---|---|---|---|
| 在线服务 | `code/rag_api.py` 及其导入的模块 | FastAPI 应用，向外提供 `/search`、`/answers`、`/health`、`/ready` | 是 |
| 离线流水线 | `code/rag_ingest/` | 清洗、分块、建 BM25 索引、写 Qdrant 向量 | 否，需手工执行 |
| 评测与诊断工具 | `code/rag_evaluation.py`、`rag_evidence_evaluation.py`、`rag_retrieval_diagnostics.py`、`rag_bm25_diagnostics.py` | 离线批量评测与对照实验 | 否 |

在线服务的导入边界很窄：`rag_api.py` 只导入 `rag_answer`、`rag_query`、`rag_reranker`、
`rag_retrieval.*` 和 `rag_settings`，不导入任何评测或诊断模块。因此评测工具的改动不会影响
线上行为，线上也不需要评测依赖。

## 2. 在线请求链路

### `POST /answers`（完整链路）

`create_answer()`（`rag_api.py:350`）按顺序执行：

1. 从 `request.app.state` 取出 `retriever`、`query_retriever`、`answer_client`、`retrieval_settings`；
2. 未配置 `DEEPSEEK_API_KEY` 时直接返回 503 `CONFIGURATION_ERROR`；
3. `_retrieval_checks()`（`rag_api.py:106`）检查 BM25 文件存在与 Qdrant 集合状态，不通过返回 503 `RETRIEVAL_UNAVAILABLE`；
4. `query_retriever.search()`（`rag_api.py:376`）执行检索链；
5. `build_evidence()` → `select_evidence()`（`rag_evidence.py:213`）在字符预算内去重、编号；
6. 证据为空时只发固定拒答文案与 `done{status:"insufficientEvidence"}`，不调用生成模型；
7. 否则经 SSE 依次发送 `meta` → `sources` → 多个 `delta` → `done`（`rag_api.py:398-420`）。

### `POST /search`

走 `app.state.retriever`，即**裸 `HybridRetriever`**（`rag_api.py:319`），不含查询规划与精排，
且 limit 取自请求体而非配置。这是有意的接口边界：`tasks/ragV2/README.md` 的共同规则第 4 条
要求 `/search` 保持单次混合检索接口，并由 `test_search_stays_a_single_query_endpoint` 守住。

注意因此产生的差异：`/answers` 的 limit 来自 `RetrievalSettings`，`/search` 的来自请求体，
同一个问题经两个接口可能得到不同排序。把 `/search` 当作调试与检查接口，不要当作 `/answers`
的等价预览。

### 健康检查

`/health` 只表示 API 进程存活。`/ready` 经 `_retrieval_checks()` 与 `_reranker_check()`
（`rag_api.py:124`）分别报告 BM25、Qdrant、回答模型与精排状态。缺少 DeepSeek Key 时
`/search` 与 `/ready` 仍可用。

## 3. 检索链路的分层

三层装饰器，每层只依赖一个 `Protocol`，因此下游不需要知道上游做了什么：

```text
RerankingRetriever           候选池精排            rag_reranker.py:272
  └─ MultiQueryRetriever     查询规划 + 多查询融合  rag_retrieval/multi_query.py
       └─ HybridRetriever    BM25 + 语义 + RRF      rag_retrieval/hybrid.py
```

装配点在 `rag_api.py:174`。分层的好处是逐层可关：关闭精排（`NoopReranker`）和关闭查询规划
（`OriginalQueryPlanner`）后，链路退化为单次混合检索，与基线一致，这一点由测试保证。

各层内部的固定算法：

- RRF 融合 `fuse_rrf()`（`rag_retrieval/hybrid.py:43`），`k=60`，该常量写在代码里而不是配置中；
- 多查询融合 `fuse_rankings()`（`rag_retrieval/multi_query.py:45`），按每路名次再融合一次；
- 证据组装 `select_evidence()`（`rag_evidence.py:213`），去重后按字符预算编号。

## 4. 四个可插拔维度

| 维度 | 工厂 | 可选策略 | 环境变量 |
|---|---|---|---|
| 查询规划 | `build_query_planner()`（`rag_query.py:272`） | `original`、`step_back` | `MCWIKI_QUERY_STRATEGY` |
| 精排 | `build_reranker()`（`rag_reranker.py:258`） | `none`、`cross_encoder` | `MCWIKI_RERANKER` |
| 证据组装 | 按 `SelectionConfig.strategy` 分支 | 生产放行 `ranked_first`、`adjacent_merge` | `MCWIKI_EVIDENCE_STRATEGY` |
| 多查询融合 | 由查询计划条数隐含 | 单查询透传、多查询 RRF | `MCWIKI_MAX_RETRIEVAL_QUERIES` |

横向协议就是扩展接缝：`QueryPlanner`（`rag_query.py`）、`Reranker`（`rag_reranker.py`）、
`CandidateRecall`、`SearchRoute`、`AnswerStreamer`。新增一个策略的通用做法是：

1. 在对应模块实现协议，不修改调用方的装配逻辑；
2. 在 `rag_settings.py` 里加环境变量并做取值校验与默认值；
3. 在工厂函数里按配置分派，未知取值必须报错而不是静默降级；
4. 保留一个关闭态策略（如 `NoopReranker`）用于回归对比；
5. 任何失败都要退回关闭态行为，不让接口失败。

**注意：召回本体不是可插拔的。** `build_default_retriever()`（`rag_retrieval/factory.py:12`）
把「BM25 + 语义 + RRF 混合」写死，没有单路检索或替换融合算法的运行时开关。若要改动召回策略，
必须修改该工厂，且会同时影响 `/search` 与 `/answers`。

## 5. 降级设计

全链路遵循「增强能力失败不得导致服务失败」，这是本项目最一致的设计原则：

| 失败情形 | 行为 |
|---|---|
| 缺 `DEEPSEEK_API_KEY` | `/answers` 返回 503 `CONFIGURATION_ERROR`；`/search`、`/ready` 仍可用 |
| Step-back 规划失败、超时、JSON 非法 | 只检索原问题 |
| 精排模型未安装、加载失败、推理异常、分数非法 | 保持 RRF 顺序并记录日志，`/ready` 报 `reranker: unavailable` |
| Qdrant 或 BM25 不可用 | 503 `RETRIEVAL_UNAVAILABLE`，不返回部分结果 |

## 6. 数据产物与可重建性

| 产物 | 大小 | 生成命令 | 是否入库 |
|---|---|---|---|
| `data/original_dataset.json` | 约 52 MB，8,200 条 | 随仓库提供 | 是 |
| `data/processed/chunks.jsonl` | 约 68 MB，41,368 块 | `uv run python -m rag_ingest` | 否 |
| `data/processed/bm25.db` | 约 234 MB | `uv run python -m rag_ingest.bm25_index` | 否 |
| Qdrant 集合 `mcwiki_chunks` | — | `uv run python -m rag_ingest.vector_ingest` | 否 |

三者互相独立、可分别重建，向量入库支持断点续跑。

**切块是最上游的稳定契约。** `chunk_id = sha256(document_id + chunk_index + text)` 的前 24 位
（`rag_ingest/pipeline.py:70` 起），`document_id` 由标题与来源派生。因此任何改变清洗、分块长度、
重叠或标题前置逻辑的改动都会产生全新的 chunk ID，必须重新建 BM25 索引并重新入库向量，否则
索引与语料会静默失配。相邻证据合并也依赖 `document_id` 与 `chunk_index`，不能改动其语义。

## 7. 必须保持的稳定契约

以下内容被测试或前端消费固化，改动前需同步更新对应测试与文档：

- **SSE 事件契约**：事件名与顺序为 `meta`、`sources`、`delta`*、`done`，流中错误为 `error`。
  权威定义见 `specs/qa-web-mvp.md`，前端逐事件做运行时校验（`frontend/src/lib/api.ts`）。
- **证据编号**：正文中的 `[n]` 必须与 `sources` 数组的顺序和长度一致，`sources` 直接来自
  最终交给生成模型的证据。
- **RRF 分数语义**：`HybridResult.score` 是 RRF 融合分，精排分数单独放在 `reranker_score`，
  两者不得混用或互相冒充。
- **`/search` 语义**：保持单次混合检索，见第 2 节。
- **错误信封**：`{"error": {"code", "message"}}`，`code` 取值见 `specs/qa-web-mvp.md`。
- **引用合法性只在提示词层保证**：服务端不校验模型输出里的 `[n]` 是否越界，越界检查只存在于
  评测脚本的 `analyze_answer()`（`rag_evaluation.py`）。这是当前已知缺口，不是被有意设计的
  契约——若要做运行时校验，应作为新功能引入。

## 8. 配置机制

配置集中在 `code/rag_settings.py`：`RetrievalSettings`（检索、证据、查询规划、精排）与
`ServiceSettings`（DeepSeek 超时、温度、thinking 开关、Cookie 寿命），都是 frozen dataclass，
经 `from_env()` 读取 `MCWIKI_*` 环境变量并做范围校验，非法值在启动时即报错。

变量清单与逐项说明以仓库根目录 `.env.example` 为准，本文不再重复列举，避免两处内容各自过期。

需要知道的两条约束：`MCWIKI_CANDIDATE_LIMIT` 必须不小于 `MCWIKI_EVIDENCE_LIMIT`，否则启动
失败；`MCWIKI_MAX_RETRIEVAL_QUERIES` 的硬边界是 1～2，校验同时存在于 `rag_settings.py` 与
`QueryPlan` 不变式中，改动需同步两处。`.env.example` 中的默认值就是固定 18 题基线所依赖的
配置，改动默认值等同于改变基线。

## 9. 前端分层

前端是单页应用，问答与视觉严格分层：

- **React DOM 层**负责输入、流式回答、引用、来源卡片、错误状态与全部可交互控件；
- **Three.js 场景层**（`frontend/src/components/scene/`）只做装饰性背景，不承载文字或交互。

`useAnswerStream` 持有问答状态机（`idle`/`retrieving`/`generating`/`complete`/`insufficient`/
`error`/`interrupted`），场景层读取该状态做氛围变化。SSE 由 `frontend/src/lib/api.ts` 手写解析，
原因是需要 POST 加凭证 Cookie，不能用浏览器的 `EventSource`；解析器逐事件做运行时类型校验，
并在流结束前未收到 `done` 或 `error` 时判为连接中断。

场景层有三级降级：`lazy` + `Suspense` 异步加载，`ErrorBoundary` 捕获异常，`prefers-reduced-motion`
时直接渲染静态回退。任何 Three.js 问题都不阻塞问答主链路。

前端不配置 dev proxy，直连后端并依赖带 Cookie 的 CORS，因此浏览器 Origin 必须与 `.env` 中的
`FRONTEND_ORIGIN` 完全一致，且必须通过 `localhost` 访问。

## 10. 已知的架构权衡

这些是当前实现中刻意保留或已知的代价，不是待修的缺陷清单：

- **精排全局串行化**：`CrossEncoderReranker` 用一把 `threading.Lock` 包住模型推理，并发请求
  在精排处排队，吞吐上限受限。单机单模型场景下这是简单且可预测的选择。
- **每个请求都做健康检查**：`/search` 与 `/answers` 都会执行 `_retrieval_checks()`，其中包含
  一次 `get_collection()` 网络往返，换来的是失败时能返回准确的状态码而不是部分结果。
- **`/search` 与 `/answers` 不共享链路**：见第 2 节，后者是完整链路，前者是有意简化的调试接口。
- **引用越界不在服务端拦截**：依赖提示词约束，评测脚本事后检查。见第 7 节。
- **证据策略能力大于配置入口**：`rag_evidence.py` 实现四种策略，配置只放行两种；另外两种
  与 `bm25_query.py` 的四种查询策略只供离线诊断使用。
- **限额由 settings 而非请求决定**：`/answers` 不接受调用方调整 limit，保证线上行为与基线一致。

## 11. 相关文档

| 文档 | 内容 |
|---|---|
| [`../README.md`](../README.md) | 项目入口：当前状态、已知缺口、快速开始、接口与评测指标 |
| [`../项目启动说明.txt`](../项目启动说明.txt) | 逐步骤启动、依赖服务与常见问题排查 |
| [`specs/qa-web-mvp.md`](specs/qa-web-mvp.md) | 接口、SSE、错误码与 Cookie 契约 |
| [`tasks/ragV2/`](tasks/ragV2/README.md) | 当前任务包与完成情况 |
| [`../data/evaluation/answer_quality/README.md`](../data/evaluation/answer_quality/README.md) | 评测方法、评分规则与回归门槛 |
| [`过期文档/README.md`](过期文档/README.md) | 历史文档归档说明 |
