# RAG 助手 MVP 执行计划

目标：尽快跑通一条可评测的最小 RAG 链路，不提前建设复杂功能。

## 已完成

- 8,200 条原始资料已清洗为 41,368 个文档块；
- Ollama 已部署 `qwen3-embedding:0.6b`；
- Qdrant 容器已运行并配置持久卷；
- Python 代码已迁入 `code/`；
- Python 3.12 和后端依赖已由 uv 锁定；
- 清洗分块与向量写入测试共 12 项通过；
- 已创建 Qdrant 集合 `mcwiki_chunks`（1024 维、Cosine）；
- 41,368 个文档块已全部生成向量并写入 Qdrant；
- 向量写入命令已支持幂等 upsert 和断点续跑；
- Qdrant 验收结果为 41,368 个点、状态 `green`、优化队列为空；
- SQLite FTS5 + trigram BM25 索引已完成，内容表与 FTS 表均为 41,368 条；
- BM25 构建命令支持重复同步，查询模块可返回 chunk ID、标题、正文、来源和分数；
- 自动化测试共 20 项，全部通过。

## 当前步骤

1. 实现查询文本的 Ollama embedding 和 Qdrant Top-K 召回；
2. 实现 BM25 + Embedding 双路召回与 RRF 融合；
3. 提供 FastAPI `/search` 接口；
4. 准备 10～20 个评测问题验证召回质量。

## 运行

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m rag_ingest
uv run python -m rag_ingest.bm25_index
uv run python -m rag_ingest.vector_ingest
```

BM25 构建命令默认读取 `data/processed/chunks.jsonl` 并同步
`data/processed/bm25.db`，可安全重复执行，不会产生重复记录。

向量写入命令可重复执行。它会先读取 `mcwiki_chunks` 中已有的点 ID，只处理缺失
文档块，因此适合在长时间任务中断后继续运行。

检索链路稳定后，再接入 Reranker、回答生成 LLM 和前端。
