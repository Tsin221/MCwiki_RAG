# MCwiki RAG 下一智能体交接说明

更新时间：2026-07-29

## 1. 项目目标

构建一个 Minecraft Wiki RAG 助手。当前首期目标是先跑通可评测的混合检索链路：

```text
SQLite FTS5 BM25 + Qdrant Embedding
                  ↓
              RRF 融合
                  ↓
            FastAPI /search
```

检索链路稳定后，再接入 Reranker、回答生成 LLM 和前端。当前不需要 MySQL、
Redis 或 Elasticsearch。

## 2. 已完成并验收

- 原始数据：`data/original_dataset.json`，8,200 条；
- 清洗分块：`data/processed/chunks.jsonl`，41,368 个文档块；
- Embedding：Ollama `qwen3-embedding:0.6b`；
- 向量维度：1024；
- Qdrant 地址：`http://127.0.0.1:6333`；
- Qdrant Dashboard：`http://127.0.0.1:6333/dashboard`；
- Qdrant collection：`mcwiki_chunks`；
- 距离函数：Cosine；
- Qdrant `points_count`：41,368；
- Collection 状态：`green`；
- Optimizer 状态：`ok`；
- 写入队列：空；
- BM25 数据库：`data/processed/bm25.db`；
- BM25 实现：SQLite FTS5 + trigram；
- BM25 内容表与 FTS 表：均为 41,368 条；
- BM25 查询模块：`code/rag_retrieval/bm25.py`；
- 语义查询模块：`code/rag_retrieval/semantic.py`；
- 查询 embedding 和 Qdrant Top-K 语义召回已完成；
- RRF 融合模块：`code/rag_retrieval/hybrid.py`；
- FastAPI 应用：`code/rag_api.py`；
- `POST /search` 已通过真实服务验收；
- 15 题评测集和首次基线已生成；
- 自动化测试：35 项通过。

向量写入已完成。再次运行写入命令时，程序会确认 41,368 条均已存在、剩余 0 条。

BM25 索引也已完成。重复执行构建命令不会产生重复记录，并会同步更新变化记录、
删除源文件中已不存在的记录。

## 3. 已完成的实现

### 3.1 数据清洗与分块

实现文件：`code/rag_ingest/pipeline.py`

- 统一 Unicode、换行和空白格式；
- 移除不可见字符；
- 优先在自然边界处分块；
- 支持重叠文本；
- 生成稳定、可追踪的文档与文档块 ID；
- 保存来源及分块 metadata。

### 3.2 向量写入

入口：

```powershell
cd E:\Work\MCwiki_RAG\code
uv run python -m rag_ingest.vector_ingest
```

实现文件：`code/rag_ingest/vector_ingest.py`

关键行为：

- 使用 Ollama `/api/embed` 批量生成 embedding；
- 每个 chunk ID 映射为稳定的 Qdrant UUID；
- 原始 chunk ID、标题、正文、来源和 metadata 保留在 payload；
- 使用 Qdrant upsert，重复执行不会产生重复点；
- 启动时通过 Qdrant scroll 分页读取已有点 ID；
- 中断后再次运行只处理缺失 chunks，支持断点续跑；
- 校验 Ollama 返回的向量数量和 1024 维度；
- 已有 collection 的向量维度或距离函数不匹配时会拒绝继续。

相关测试：`code/tests/test_vector_ingest.py`

### 3.3 BM25 索引与查询

入口：

```powershell
cd E:\Work\MCwiki_RAG\code
uv run python -m rag_ingest.bm25_index
```

实现文件：

- `code/rag_ingest/bm25_index.py`
- `code/rag_retrieval/bm25.py`

关键行为：

- 使用 SQLite FTS5 外部内容表和 `trigram` tokenizer；
- 保存 chunk ID、标题、正文、来源和 metadata；
- 重复构建会更新变化内容并清理失效记录；
- 查询结果返回 chunk ID、标题、正文、来源和 BM25 分数；
- 查询文本按字面文本处理，不开放 FTS5 查询语法；
- 空白查询和无命中查询返回空列表。

相关测试：`code/tests/test_bm25.py`

### 3.4 查询向量化与语义召回

实现文件：`code/rag_retrieval/semantic.py`

关键行为：

- `SemanticRetriever` 长期持有 Ollama HTTP 客户端和 Qdrant 客户端；
- 使用 `qwen3-embedding:0.6b` 为查询生成 1024 维向量；
- 查询 `mcwiki_chunks` 并返回统一的 `SemanticResult`；
- 结果包含 chunk ID、标题、正文、来源和相似度；
- 空白查询直接返回空列表；
- 校验 limit、Ollama embedding 数量与维度、Qdrant payload 和分数；
- 已使用真实 Ollama 与 Qdrant 完成查询验收。

相关测试：`code/tests/test_semantic.py`

## 4. 重要说明与边界

Qdrant Dashboard 中的 `indexed_vectors_count` 可能小于 `points_count`。低于
`indexing_threshold` 的尾部小 segment 会通过全量扫描参与检索，因此这不表示
向量缺失。完整性应以 `points_count == 41,368` 以及断点复检剩余 0 条为准。

不要删除或重建 `mcwiki_chunks`，除非用户明确要求重新生成全部向量。现有向量生成
耗时较长，但已经完整持久化到 Qdrant 数据卷 `mcwiki_qdrant_storage`。

当前工作区已经是 Git 仓库。修改前应检查工作区状态，并保留用户已有改动。

FastAPI、Uvicorn 等依赖已经安装，但当前还没有 FastAPI 应用或 `/search` 接口。
React 前端、Reranker 和回答生成 LLM 也尚未实现。

## 5. 当前阶段结论

本地混合检索 MVP 已完成，包括查询 embedding、Qdrant Top-K、BM25、RRF、
`POST /search` 和首次评测基线。

评测文件：

- `data/evaluation/retrieval_questions.json`
- `data/evaluation/baseline-2026-07-29.json`

首次基线：

- Hit@10：14/15（93.33%）；
- MRR@10：0.7911；
- 已知未命中：`java-1-21-content`。

不要删除失败问题或修改正确来源来提高表面指标。下一步应先分析版本问题未命中的
原因，或者在确定云端模型供应商后进入 Reranker 与回答生成阶段。

## 6. 后续顺序

1. 分析并优化 Java版 1.21 问题的检索未命中；
2. 根据评测结果决定是否接入 Reranker；
3. 选择云端回答生成 LLM；
4. 生成带可追踪来源的回答；
5. 实现前端；
6. 本地 RAG 稳定后，再评估 Agentic RAG。

未来受限联网 Agentic RAG 方案见 `docs/ideas/agentic-rag.md`，当前不实施，也不在
本阶段引入 LangGraph。

## 7. 常用命令

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m rag_ingest
uv run python -m rag_ingest.bm25_index
uv run python -m rag_ingest.vector_ingest
uv run uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

相关状态与设计文档：

- `PROJECT_STATUS.md`
- `MVP_PLAN.md`
- `RAG_ASSISTANT_GUIDE.md`
