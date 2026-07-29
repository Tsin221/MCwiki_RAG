# RAG 助手当前状态

更新时间：2026-07-29

## 阶段结论

数据清洗、向量入库、BM25、查询语义召回、RRF 融合和 FastAPI `/search` 均已完成
并验收。在线混合检索 MVP 已完成。

当前可以提供本地知识库的混合检索结果，但还不能提供完整 RAG 问答：Reranker、
回答生成 LLM 和前端均尚未接入。

下一阶段计划使用 DeepSeek `deepseek-v4-pro` 生成流式回答，并实现带匿名访客
Cookie 和轻量 Three.js 动态背景的 React 问答界面。规格见
`docs/specs/qa-web-mvp.md`，视觉方案见 `docs/frontend-threejs/`，当前尚未实施。

未来计划构建可信度优先、受限联网的 Agentic RAG；预期方案见
`docs/ideas/agentic-rag.md`。该方案当前暂缓实施，本阶段不引入 LangGraph。

## 已完成：数据与索引底座

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
- 语义查询模块：`code/rag_retrieval/semantic.py`；
- 查询文本使用 Ollama `qwen3-embedding:0.6b` 生成 1024 维向量；
- Qdrant Top-K 查询返回 chunk ID、标题、正文、来源和相似度；
- 语义查询模块会校验 limit、embedding 维度和 Qdrant payload；
- 真实服务查询已验收，可返回对应的中文 Minecraft Wiki 结果。
- RRF 融合模块：`code/rag_retrieval/hybrid.py`；
- FastAPI 应用：`code/rag_api.py`；
- 检索接口：`POST /search`；
- 评测集：`data/evaluation/retrieval_questions.json`，共 15 题；
- 首次基线：`data/evaluation/baseline-2026-07-29.json`；
- Hit@10：14/15（93.33%）；
- MRR@10：0.7911；
- 已知未命中：Java版 1.21 内容问题。

## 向量库验收

- `points_count`：41,368，与文档块数量一致；
- Collection 状态：`green`；
- Optimizer 状态：`ok`；
- 写入队列：空；
- 断点续跑复检：41,368 条已存在，剩余 0 条；
- 自动化测试：35 项通过。

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

## 混合检索 MVP 完成情况

1. 查询向量化与 Qdrant Top-K：已完成；
2. BM25 与 Embedding 双路召回：已完成；
3. RRF 排名融合：已完成；
4. `/search` 检索接口：已完成；
5. 10～20 个评测问题及正确来源：已完成；
6. 首次真实服务基线：已完成。

该阶段已完成。

## 后续阶段

1. 使用 `deepseek-v4-pro` 生成带可追踪来源的回答；
2. 提供基于 SSE 的流式 `POST /answers`；
3. 使用匿名 Cookie 区分浏览器访客；
4. 实现带轻量 Three.js 动态背景的 React 流式问答界面；
5. 补充完整链路评测与无答案处理；
6. 根据完整链路评测决定是否接入 Reranker。

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

检索链路已完成。下一步按 `docs/specs/qa-web-mvp.md` 增加 DeepSeek 回答和前端。

## 常用命令

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m rag_ingest.bm25_index
uv run python -m rag_ingest.vector_ingest
uv run uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

`rag_ingest.bm25_index` 默认读取 `data/processed/chunks.jsonl`，同步构建
`data/processed/bm25.db`。重复执行不会产生重复记录。

`rag_ingest.vector_ingest` 使用稳定的 Qdrant 点 ID 和 upsert 写入；中断后可直接
重复运行，命令会读取已有点 ID，只为缺失文档块生成向量。
