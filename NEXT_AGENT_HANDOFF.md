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
- 自动化测试：12 项通过。

向量写入已完成。再次运行写入命令时，程序会确认 41,368 条均已存在、剩余 0 条。

## 3. 已实现的向量写入行为

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

## 4. 重要说明

Qdrant Dashboard 中的 `indexed_vectors_count` 可能小于 `points_count`。低于
`indexing_threshold` 的尾部小 segment 会通过全量扫描参与检索，因此这不表示
向量缺失。完整性应以 `points_count == 41,368` 以及断点复检剩余 0 条为准。

不要删除或重建 `mcwiki_chunks`，除非用户明确要求重新生成全部向量。现有向量生成
耗时较长，但已经完整持久化到 Qdrant 数据卷 `mcwiki_qdrant_storage`。

当前工作区不是 Git 仓库，不要假设可以提交或回滚 Git commit。

## 5. 下一项任务

创建 SQLite FTS5 BM25 索引。这是实现双路召回前唯一缺失的检索数据底座。

建议输出：

- 数据库：`data/processed/bm25.db`；
- 输入：`data/processed/chunks.jsonl`；
- 索引内容至少包括：chunk ID、title、text、source、metadata；
- 中文检索使用当前方案约定的 SQLite FTS5 + trigram；
- 提供可重复执行的索引构建命令；
- 提供 BM25 查询函数或模块，为后续 `/search` 复用。

## 6. 下一项任务验收标准

1. SQLite 数据库成功创建；
2. 索引文档数与 chunks 数量一致，均为 41,368；
3. 重复构建不会产生重复记录；
4. 能用中文关键词、Minecraft 专有名词和版本号执行查询；
5. 查询结果返回 chunk ID、title、text、source 和 BM25 分数；
6. 空查询与无结果查询有明确行为；
7. 新增逻辑有自动化测试，且现有 12 项测试继续通过；
8. 完成后更新 `PROJECT_STATUS.md`、`MVP_PLAN.md` 和
   `RAG_ASSISTANT_GUIDE.md`。

## 7. 后续顺序

完成 BM25 后按以下顺序继续：

1. 实现查询文本的 Ollama embedding；
2. 实现 Qdrant Top-K 语义召回；
3. 合并 BM25 与 Embedding 结果；
4. 实现 RRF 排名融合；
5. 提供 FastAPI `/search`；
6. 准备 10～20 个带正确来源的评测问题；
7. 检索质量稳定后再选择 Reranker 和回答生成 LLM。

## 8. 常用命令

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m rag_ingest
uv run python -m rag_ingest.vector_ingest
```

相关状态与设计文档：

- `PROJECT_STATUS.md`
- `MVP_PLAN.md`
- `RAG_ASSISTANT_GUIDE.md`
