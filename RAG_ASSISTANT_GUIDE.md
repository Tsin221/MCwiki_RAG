# RAG 助手开发指引

## 1. 项目目标

本项目计划构建一个基于知识库的 RAG（Retrieval-Augmented Generation，检索增强生成）助手。

助手接收用户问题后，先从知识库中检索相关内容，再由大语言模型结合检索结果生成回答。这样可以减少模型凭空作答，并让回答尽可能基于已有资料。

当前文档用于说明首期技术方案。后续引入新的模型、检索方式或工程组件时，再持续更新。

## 2. 目标技术流程

```text
数据清洗与分块
      ↓
BM25 + Embedding 双路召回
      ↓
RRF 排名融合
      ↓
Reranker 精排
      ↓
LLM 生成回答
```

### 2.1 当前实现状态

- **已完成**：数据清洗与分块；
- **已完成**：Ollama Embedding 批量生成与 Qdrant 向量入库；
- **已完成**：SQLite FTS5 + trigram BM25 索引与查询模块；
- **已完成**：查询向量化与 Qdrant Top-K 语义召回；
- **已完成**：BM25 + Embedding 双路召回与 RRF 融合；
- **已完成**：FastAPI `POST /search`；
- **已完成**：15 题检索评测集与首次真实服务基线；
- **后续阶段**：Reranker、回答生成 LLM、引用展示和前端。

因此，本文中的完整流程是目标架构，不表示所有环节都已经实现。当前已经完成离线
知识库与索引底座，但尚未形成可对外提供问答的完整 RAG 助手。

项目未来计划扩展为可信度优先、受限联网的 Agentic RAG，但当前不实施。未来方案、
安全边界和实施前置条件见 `docs/ideas/agentic-rag.md`。

### 2.2 当前工程选型

- 后端：FastAPI；
- 前端规划：React + TypeScript + Vite，尚未开始实现；
- Python 版本与依赖：uv，代码统一放在 `code/`；
- BM25：SQLite FTS5 + trigram，适配中文关键词检索；
- Embedding：Ollama + `qwen3-embedding:0.6b`，输出 1024 维向量；
- 向量数据库：Qdrant，集合为 `mcwiki_chunks`，使用 1024 维向量和 Cosine 相似度；
- 融合方式：RRF；
- Reranker 和回答生成 LLM：待检索链路验证后确定。

当前阶段不引入 MySQL、Redis 或 Elasticsearch，以减少首期部署和维护成本。

## 3. 各环节说明

### 3.1 数据清洗与分块

将原始文档转换为适合检索的知识片段。

主要工作包括：

- 去除无关内容、重复内容和异常格式；
- 保留标题、章节、来源等必要信息；
- 按段落、标题或合适的长度进行分块；
- 为每个分块保存来源、位置、更新时间等元数据；
- 避免分块过大导致检索不准确，也避免过小导致语义不完整。

输出结果是结构统一、可追踪来源的文档块。

### 3.2 BM25 + Embedding 双路召回

使用两种检索方式并行查找候选文档块：

- **BM25 召回**：擅长匹配关键词、专有名词、版本号和精确表达；
- **Embedding 召回**：擅长理解语义，可以召回表达不同但含义相近的内容。

双路召回可以兼顾关键词匹配和语义理解，降低单一检索方式带来的遗漏。

BM25 索引已由 `data/processed/chunks.jsonl` 全量构建到
`data/processed/bm25.db`。SQLite 普通内容表保留 chunk ID、标题、正文、来源和
metadata；FTS5 外部内容表使用 `trigram` tokenizer 索引标题与正文。普通表和 FTS
表均已验收为 41,368 条。

BM25 索引构建命令：

```powershell
cd E:\Work\MCwiki_RAG\code
uv run python -m rag_ingest.bm25_index
```

该命令可重复执行：相同 chunk ID 会按源数据更新，不会产生重复记录，源文件中已
删除的 chunk 也会从数据库同步删除。可通过 `--input` 和 `--output` 覆盖默认路径。

可复用查询入口为：

```python
from pathlib import Path

from rag_retrieval.bm25 import search_bm25

results = search_bm25(
    Path("../data/processed/bm25.db"),
    "红石中继器",
    limit=10,
)
```

每个结果是 `BM25Result`，包含 `chunk_id`、`title`、`text`、`source` 和
`score`。`score` 已将 SQLite 原生 `bm25()` 的排序方向取反，因此数值越大越相关。
查询文本会被当作字面文本而不是 FTS5 查询语法；空白查询和无命中查询均返回
空列表，`limit <= 0` 会抛出 `ValueError`。

当前 41,368 个文档块已全部使用 `qwen3-embedding:0.6b` 生成向量并写入
`mcwiki_chunks`。每个文档块 ID 会映射为稳定的 Qdrant UUID，原始 chunk ID、
标题、正文、来源和 metadata 保留在 payload 中。写入采用 upsert，因此重复执行
不会生成重复点。

向量写入命令：

```powershell
cd E:\Work\MCwiki_RAG\code
uv run python -m rag_ingest.vector_ingest
```

命令启动时会通过 Qdrant scroll 分页读取已有点 ID，只为缺失文档块调用 Ollama，
因此支持中断后断点续跑。完整性以 Qdrant `points_count` 与源文档块数量一致为准；
小于 `indexing_threshold` 的尾部 segment 可能不会计入 `indexed_vectors_count`，
但仍可通过全量扫描参与检索。

查询语义召回入口为 `rag_retrieval.semantic.SemanticRetriever`。它使用相同的
Ollama 模型生成查询向量，再调用 Qdrant Top-K 查询；结果包含 chunk ID、标题、
正文、来源和相似度。空白查询返回空列表，`limit <= 0` 会抛出 `ValueError`，
Ollama embedding 维度和 Qdrant payload 也会在边界进行校验。

双路融合入口为 `rag_retrieval.hybrid.HybridRetriever`。它分别执行 BM25 和语义
召回，按 chunk ID 去重，再使用 RRF 合并两路名次。融合结果保留 BM25 排名、语义
排名和 RRF 分数，不直接混合数值范围不同的原始分数。

FastAPI 应用入口为 `rag_api:app`，对外提供 `POST /search`。请求支持 `query`、
`limit`、`bm25Limit` 和 `semanticLimit`；响应返回统一的 camelCase 结果字段。
空白问题和超出范围的 limit 会在 API 边界被拒绝。

### 3.3 RRF 排名融合

使用 RRF（Reciprocal Rank Fusion）合并 BM25 和 Embedding 的召回结果。

RRF 主要根据文档块在不同结果列表中的排名计算融合分数，不强依赖两种检索分数处于同一数值范围。融合后得到统一的候选结果列表。

### 3.4 Reranker 精排

将融合后的候选文档块交给 Reranker，根据“用户问题与文档块是否真正相关”重新排序。

精排阶段用于：

- 将最相关的内容排到前面；
- 过滤只有关键词命中但语义不相关的内容；
- 控制最终发送给 LLM 的上下文数量和长度。

### 3.5 LLM 生成回答

将用户问题和精排后的知识片段一起发送给 LLM，由 LLM 生成最终回答。

回答应遵循以下原则：

- 优先依据检索到的知识内容作答；
- 不确定或资料不足时明确说明；
- 尽量附带文档来源或引用片段；
- 不使用检索结果无法支持的内容补全事实；
- 当不同资料存在冲突时，提示用户并展示相关来源。

## 4. 一次问答的处理过程

1. 用户提交问题；
2. 系统对问题进行必要的标准化处理；
3. BM25 和 Embedding 分别召回候选文档块；
4. RRF 合并两路结果；
5. Reranker 对候选结果进行精排；
6. 系统选择排名靠前的文档块组成上下文；
7. LLM 根据问题和上下文生成回答；
8. 系统返回回答，并尽可能展示引用来源。

## 5. 分阶段实现范围

### 5.1 当前阶段：混合检索 MVP

当前阶段先跑通并评测检索链路：

- 支持一种或少量固定格式的数据源；
- 数据清洗、分块和元数据保存（已完成）；
- BM25 索引与查询（已完成）；
- 文档向量生成与 Qdrant 入库（已完成）；
- 查询文本向量化与 Qdrant Top-K 召回（已完成）；
- BM25 与 Embedding 双路召回（已完成）；
- RRF 排名融合（已完成）；
- FastAPI `/search`（已完成）；
- 10～20 个带正确来源的基础评测问题（已完成，共 15 题）。

上述检索、接口和评测工作已经完成，混合检索 MVP 已标记为完成。当前 15 题首次
基线的 Hit@10 为 14/15（93.33%），MRR@10 为 0.7911。Java版 1.21 内容问题未在
Top-10 命中，已保留为后续回归目标。

### 5.2 后续阶段：完整 RAG 问答

- 根据检索评测结果选择并接入 Reranker；
- 接入一个 LLM 生成回答；
- 在回答中保留可追踪的来源信息；
- 实现前端与完整问答交互；
- 评测回答准确性、引用准确性和无答案处理。

首期暂不追求复杂的 Agent、多轮任务编排或大量数据源接入。

## 6. 基础效果评估

建议至少关注以下指标：

- **召回率**：正确资料是否进入候选结果；
- **排序质量**：最相关资料是否位于前列；
- **回答准确性**：回答是否得到检索内容支持；
- **引用准确性**：引用是否与回答内容对应；
- **无答案处理**：资料不足时是否能避免编造；
- **响应时间**：一次完整问答的耗时是否可接受。

每次更换分块策略、Embedding 模型、Reranker 或 LLM 时，应使用同一组测试问题进行对比。

## 7. 后续可更新方向

后续可根据实际效果逐步评估：

- 查询改写、意图识别或多查询召回；
- 混合分块、父子分块或上下文扩展；
- 新的稀疏检索、向量检索或融合算法；
- 更适合业务数据的 Embedding 和 Reranker；
- 多轮对话与历史问题处理；
- 权限过滤、数据隔离和敏感信息保护；
- 缓存、监控、评测集和反馈闭环；
- 多模态资料或更多数据源。

这些方向不是当前固定方案，应在有明确问题和评测结果后再决定是否引入。

## 8. 文档更新约定

当技术方案发生变化时，请同步更新：

1. “目标技术流程”和“当前实现状态”；
2. 对应环节的职责和输入输出；
3. 首期实现范围或后续方向；
4. 评测结果及变更原因。

建议记录“为什么调整”以及调整前后的效果，避免只记录最终采用了什么技术。
