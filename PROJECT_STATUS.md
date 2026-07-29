# RAG 助手当前状态

更新时间：2026-07-29

## 已准备

- 代码目录：`code/`；
- Python 版本与依赖管理：uv；
- Python 版本：3.12；
- 后端基础依赖：FastAPI、Uvicorn、HTTPX、Qdrant Client；
- 原始资料：`data/original_dataset.json`，共 8,200 条；
- 清洗分块：已生成 41,368 个文档块；
- Embedding：Ollama + `qwen3-embedding:0.6b`；
- 模型目录：`E:\Ollama\model`；
- 向量维度：1024；
- 向量数据库：Qdrant 1.18.2；
- Qdrant 集合：`mcwiki_chunks`，1024 维、Cosine 距离；
- 向量写入：41,368 个文档块已全部生成向量并写入；
- 向量写入命令支持断点续跑，重复执行会跳过 Qdrant 中已有的点；
- Qdrant 地址：`http://127.0.0.1:6333`；
- Qdrant Dashboard：`http://127.0.0.1:6333/dashboard`；
- Qdrant 容器：`mcwiki-qdrant`；
- Qdrant 数据卷：`mcwiki_qdrant_storage`；
- BM25 数据库：`data/processed/bm25.db`；
- BM25 索引：SQLite FTS5 + trigram，已索引全部 41,368 个文档块；
- BM25 构建命令支持重复执行，会更新变化记录并清理源文件中已不存在的记录；
- BM25 查询模块：`code/rag_retrieval/bm25.py`。

## 向量库验收

- `points_count`：41,368，与文档块数量一致；
- Collection 状态：`green`；
- Optimizer 状态：`ok`；
- 写入队列：空；
- 断点续跑复检：41,368 条已存在，剩余 0 条；
- 自动化测试：20 项通过。

Dashboard 中的 `indexed_vectors_count` 可能小于 `points_count`。低于
`indexing_threshold` 的尾部小 segment 会使用全量扫描，这不表示向量缺失；
是否完整应以 `points_count` 和断点续跑复检结果为准。

## BM25 索引验收

- `chunks` 内容表记录数：41,368；
- `chunks_fts` FTS5 索引记录数：41,368；
- 重复 chunk ID：0；
- 有效 metadata JSON：41,368；
- Tokenizer：`trigram`；
- SQLite `integrity_check`：`ok`；
- FTS5 外部内容一致性检查：通过；
- 重复执行全量构建后记录数仍为 41,368；
- 中文关键词、Minecraft 专有名词和版本号查询均能返回结果；
- 空白查询和无结果查询均返回空列表。

## 还需完成

1. 实现查询文本的 Ollama embedding 和 Qdrant Top-K 召回；
2. 实现 BM25 与 Embedding 双路召回；
3. 实现 RRF 排名融合；
4. 提供 `/search` 检索接口；
5. 接入 Reranker；
6. 选择回答生成 LLM；
7. 准备 10～20 个评测问题及正确来源。

## 暂时不需要

- MySQL；
- Redis；
- Elasticsearch。

第一版先完成：

```text
SQLite BM25 + Qdrant Embedding
              ↓
          RRF 融合
              ↓
        FastAPI /search
```

检索链路稳定后，再增加 Reranker、LLM 回答和前端。

## 常用命令

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m rag_ingest.bm25_index
uv run python -m rag_ingest.vector_ingest
```

`rag_ingest.bm25_index` 默认读取 `data/processed/chunks.jsonl`，同步构建
`data/processed/bm25.db`。重复执行不会产生重复记录。

`rag_ingest.vector_ingest` 使用稳定的 Qdrant 点 ID 和 upsert 写入；中断后可直接
重复运行，命令会读取已有点 ID，只为缺失文档块生成向量。
