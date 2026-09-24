# RAG 助手 MVP 执行计划

目标：尽快跑通一条可评测的最小 RAG 链路，不提前建设复杂功能。

当前阶段：数据清洗、索引底座、在线混合检索和问答 Web MVP 的实现、联调与首份
完整链路质量基线均已完成。针对性质量修复和同一 18 题对比基线也已完成，暂不接入
Reranker。下一阶段继续深化基础 RAG 工程，并将同一 18 题固定为所有相关变更的回归
门槛；当前不做公开部署前的限流、请求预算、监控、HTTPS Cookie 和滥用防护。

## 已完成

- 8,200 条原始资料已清洗为 41,368 个文档块；
- Ollama 已部署 `qwen3-embedding:0.6b`；
- Qdrant 容器已运行并配置持久卷；
- Python 代码已迁入 `code/`；
- Python 3.12 和后端依赖已由 uv 锁定；
- 清洗分块、向量写入和 BM25 检索相关测试均通过；
- 已创建 Qdrant 集合 `mcwiki_chunks`（1024 维、Cosine）；
- 41,368 个文档块已全部生成向量并写入 Qdrant；
- 向量写入命令已支持幂等 upsert 和断点续跑；
- Qdrant 验收结果为 41,368 个点、状态 `green`、优化队列为空；
- SQLite FTS5 + trigram BM25 索引已完成，内容表与 FTS 表均为 41,368 条；
- BM25 构建命令支持重复同步，查询模块可返回 chunk ID、标题、正文、来源和分数；
- 查询文本的 Ollama embedding 和 Qdrant Top-K 语义召回已实现；
- 语义召回返回 chunk ID、标题、正文、来源和相似度，并校验外部响应；
- 真实服务查询已验收，可返回中文 Minecraft Wiki 的相关结果；
- BM25 + Embedding 双路召回与 RRF 融合已实现；
- FastAPI `POST /search` 已实现并通过真实服务请求验收；
- 已准备 15 个带正确来源的评测问题；
- 首次基线：Hit@10 为 14/15（93.33%），MRR@10 为 0.7911；
- DeepSeek `deepseek-v4-pro` 证据约束流式回答已实现；
- FastAPI `POST /answers`、匿名访客 Cookie 和统一错误处理已实现；
- React 流式问答、引用来源、安全 Markdown 和 Three.js 背景已实现；
- 三个真实问题已完成桌面端与 360px 移动端浏览器联调；
- 后端自动化测试 61 项、前端自动化测试 8 项全部通过；
- TypeScript 类型检查和前端生产构建通过。
- 15 个可回答问题与 3 个无答案问题的完整链路基线已生成；
- 平均正确性 1.80/2、完整性 1.80/2、证据忠实度 1.93/2；
- 引用编号有效率 100%，无答案可靠拒答率 100%。
- Java 版 1.21 版本查询规范化和标题优先召回已实现；
- 三个既定质量回归目标已修复，同一 18 题新基线的正确性、完整性和证据忠实度均为
  2.00/2；
- 修复后期望来源召回率和引用率均为 15/15（100%）。
- 固定回归集已确认为 15 个可回答问题和 3 个无答案问题，共 18 个唯一案例；
- 评测运行记录配置、数据集和系统提示词指纹，拒绝混合配置、重复案例和不完整结果。

## 已完成阶段：混合检索 MVP

以下目标已经全部完成：

1. 查询文本的 Ollama embedding 和 Qdrant Top-K 召回；
2. BM25 + Embedding 双路召回与 RRF 融合；
3. FastAPI `/search` 接口；
4. 10～20 个带正确来源的评测问题和首次检索基线。

评测集中保留了“Java版1.21主要加入了哪些内容？”这一项 Top-10 未命中问题，作为
后续检索质量优化的回归目标，不通过修改评测集掩盖失败。

Reranker 和未来 Agentic RAG 属于后续阶段。当前唯一的期望来源召回失败没有进入
双路 Top-20 候选，Reranker 无候选可排，因此暂不接入。

未来 Agentic RAG 方案已记录在 `docs/ideas/agentic-rag.md`，当前不实施，也不在
本阶段引入 LangGraph。

## 已完成阶段：问答 Web MVP 质量评测

问答 Web MVP 规格见 `docs/specs/qa-web-mvp.md`，实现任务 1～7 已完成。评测方法、
原始结果、人工复核和基线报告见 `data/evaluation/answer_quality/`。Three.js 前端
视觉方案及完成记录见 `docs/过期文档/frontend-threejs/`。

已实现：

1. 使用 `deepseek-v4-pro` 基于本地检索证据生成回答；
2. 提供基于 SSE 的流式 `POST /answers`；
3. 使用安全属性明确的匿名 Cookie 区分浏览器访客；
4. 实现带轻量 Three.js 动态背景的 React 流式问答界面；
5. 展示回答引用和 Wiki 来源；
6. 完成前后端联调和真实浏览器冒烟测试。

评测结论：

1. 首份基线达到建议门槛；
2. Java 版 1.21 内容页现已进入最终证据第 1 名并可正常回答；
3. 灾厄巡逻队回答不再反向解释 Java 版光照条件；
4. 苦力怕回答已覆盖特定条件下的音乐唱片和生物头颅；
5. 同一 18 题修复后基线全部通过，当前仍无须接入 Reranker。

匿名 Cookie 不等同于登录。第一版不引入 LangGraph、联网搜索、账号系统、会话
持久化和管理后台。

## 运行

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run --with pytest python -m pytest
uv run python -m rag_ingest
uv run python -m rag_ingest.bm25_index
uv run python -m rag_ingest.vector_ingest
uv run uvicorn rag_api:app --env-file ..\.env --host 127.0.0.1 --port 8000
```

```powershell
cd E:\Work\MCwiki_RAG\frontend
npm run test -- --run
npm run typecheck
npm run build
npm run dev
```

BM25 构建命令默认读取 `data/processed/chunks.jsonl` 并同步
`data/processed/bm25.db`，可安全重复执行，不会产生重复记录。

向量写入命令可重复执行。它会先读取 `mcwiki_chunks` 中已有的点 ID，只处理缺失
文档块，因此适合在长时间任务中断后继续运行。

下一步继续深化基础 RAG 工程：维护固定 18 题回归门槛，依据具体失败案例选择改进项。
限流、请求预算、监控、HTTPS Cookie 和滥用防护留到明确需要公开部署时再实施。
最新对比报告见 `data/evaluation/answer_quality/REPORT-2026-07-30-quality-fix.md`。
