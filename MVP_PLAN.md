# RAG 助手 MVP 执行计划

目标：尽快跑通一条可评测的最小 RAG 链路，不提前建设复杂功能。

当前阶段：数据清洗、索引底座和在线混合检索 MVP 已经完成并验收；下一阶段计划
实现 DeepSeek 回答生成和简单 Web 问答界面。

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
- 自动化测试共 35 项，全部通过。

## 已完成阶段：混合检索 MVP

以下目标已经全部完成：

1. 查询文本的 Ollama embedding 和 Qdrant Top-K 召回；
2. BM25 + Embedding 双路召回与 RRF 融合；
3. FastAPI `/search` 接口；
4. 10～20 个带正确来源的评测问题和首次检索基线。

评测集中保留了“Java版1.21主要加入了哪些内容？”这一项 Top-10 未命中问题，作为
后续检索质量优化的回归目标，不通过修改评测集掩盖失败。

Reranker、回答生成 LLM、前端和未来 Agentic RAG 属于后续阶段。

未来 Agentic RAG 方案已记录在 `docs/ideas/agentic-rag.md`，当前不实施，也不在
本阶段引入 LangGraph。

## 下一阶段：问答 Web MVP

下一阶段规格见 `docs/specs/qa-web-mvp.md`，当前已经确认，尚未开始实现。Three.js
前端视觉方案见 `docs/frontend-threejs/`。

目标包括：

1. 使用 `deepseek-v4-pro` 基于本地检索证据生成回答；
2. 提供基于 SSE 的流式 `POST /answers`；
3. 使用安全属性明确的匿名 Cookie 区分浏览器访客；
4. 实现带轻量 Three.js 动态背景的 React 流式问答界面；
5. 展示回答引用和 Wiki 来源；
6. 建立回答准确性、引用准确性和无答案处理基线。

匿名 Cookie 不等同于登录。第一版不引入 LangGraph、联网搜索、账号系统、会话
持久化和管理后台。

## 运行

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m rag_ingest
uv run python -m rag_ingest.bm25_index
uv run python -m rag_ingest.vector_ingest
uv run uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

BM25 构建命令默认读取 `data/processed/chunks.jsonl` 并同步
`data/processed/bm25.db`，可安全重复执行，不会产生重复记录。

向量写入命令可重复执行。它会先读取 `mcwiki_chunks` 中已有的点 ID，只处理缺失
文档块，因此适合在长时间任务中断后继续运行。

下一步先按问答 Web MVP 规格接入回答生成和前端，再根据完整链路评测决定是否需要
Reranker。
