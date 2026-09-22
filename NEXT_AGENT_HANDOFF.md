# MCwiki RAG 下一智能体交接说明

更新时间：2026-07-30

## 1. 项目目标

构建一个 Minecraft Wiki RAG 助手。当前本地问答 Web MVP 已经形成完整链路：

```text
SQLite FTS5 BM25 + Qdrant Embedding
                  ↓
              RRF 融合
                  ↓
         证据去重与上下文限制
                  ↓
       DeepSeek 流式证据约束回答
                  ↓
          FastAPI /answers SSE
                  ↓
       React 问答、引用与来源展示
```

实现任务 1～7 已完成，首份完整链路质量基线已经生成并达到建议门槛。当前目标是
修复版本查询召回、证据反向解读和多子问题完整性。这三个目标现已完成，并使用同一
18 题生成了修复后对比基线。当前决定仍不接入 Reranker，也不需要 MySQL、Redis、
Elasticsearch、LangGraph 或联网检索。下一阶段继续深化基础 RAG 工程；限流、请求
预算、监控、HTTPS Cookie、滥用防护和公开部署均不属于当前任务。现有同一 18 题已
固定为所有后续 RAG 变更的回归门槛。

## 2. 已完成并验收

- 原始数据：`data/original_dataset.json`，8,200 条；
- 清洗分块：`data/processed/chunks.jsonl`，41,368 个文档块；
- Embedding：Ollama `qwen3-embedding:0.6b`；
- 向量维度：1024；
- 本机 Qdrant 地址：`http://127.0.0.1:6335`（`.env` 中的 `MCWIKI_QDRANT_URL`）；
- Qdrant Dashboard：`http://127.0.0.1:6335/dashboard`；
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
- DeepSeek 回答客户端：`code/rag_answer.py`；
- 流式回答接口：`POST /answers`；
- SSE 事件：`meta`、`sources`、`delta`、`done`，流中错误使用 `error`；
- 匿名访客 Cookie、显式 CORS Origin 和统一错误处理已完成；
- React + TypeScript + Vite 问答前端位于 `frontend/`；
- 安全 Markdown、来源卡片和 Three.js 异步背景已完成；
- 三个真实问题已完成桌面端与 360px 移动端浏览器联调；
- 后端自动化测试：61 项通过；
- 前端自动化测试：8 项通过；
- TypeScript 类型检查和前端生产构建通过。

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

### 3.5 混合检索与回答生成

实现文件：

- `code/rag_retrieval/hybrid.py`
- `code/rag_answer.py`

关键行为：

- BM25 与语义结果按 chunk ID 去重并使用 RRF 融合；
- 回答上下文限制来源数量和总字符数；
- DeepSeek `deepseek-v4-pro` 只由后端调用；
- 模型被要求只依据编号证据回答，证据不足时明确说明；
- 流式响应按 OpenAI 兼容 SSE 格式解析，并校验异常响应。

相关测试：`code/tests/test_hybrid.py`、`code/tests/test_answer.py`

### 3.6 API 与前端

实现文件：

- `code/rag_api.py`
- `frontend/src/`

关键行为：

- `POST /search` 保持可用；
- `POST /answers` 返回 `meta`、`sources`、`delta`、`done` 或 `error` 事件；
- 匿名访客 Cookie 使用 `HttpOnly`、`SameSite=Lax` 和 180 天有效期；
- 前端使用 `fetch` + `ReadableStream` 解析 POST SSE；
- 模型 Markdown 作为不可信内容处理，不渲染 HTML、模型链接或模型图片；
- Three.js 场景异步加载，失败时回退到静态背景，不阻塞问答主链路。

相关测试：`code/tests/test_api.py`、`frontend/src/*.test.ts*`

## 4. 重要说明与边界

Qdrant Dashboard 中的 `indexed_vectors_count` 可能小于 `points_count`。低于
`indexing_threshold` 的尾部小 segment 会通过全量扫描参与检索，因此这不表示
向量缺失。完整性应以 `points_count == 41,368` 以及断点复检剩余 0 条为准。

不要删除或重建 `mcwiki_chunks`，除非用户明确要求重新生成全部向量。现有向量生成
耗时较长，但已经完整持久化到 Qdrant 数据卷 `mcwiki_qdrant_storage`。

当前工作区已经是 Git 仓库。修改前应检查工作区状态，并保留用户已有改动。

FastAPI `/search`、`/answers`、DeepSeek 回答服务和 React 前端均已实现。Reranker
仍未接入，这是有意保留的评测后决策点，不是遗漏实现。

根目录 `.env` 含本地密钥配置并被 Git 忽略。不要把密钥内容写入文档、日志、前端
代码或版本控制。带 Cookie 的 CORS 要求浏览器 Origin 与 `FRONTEND_ORIGIN` 完全
一致；默认使用 `http://localhost:5173`。

## 5. 当前阶段结论

本地混合检索 MVP 和问答 Web MVP 的实现任务 1～7 均已完成，包括查询 embedding、
Qdrant Top-K、BM25、RRF、`POST /search`、DeepSeek 流式回答、`POST /answers`
、React/Three.js 前端和首份完整链路质量评测。

评测文件：

- `data/evaluation/regression_suite.json`
- `data/evaluation/retrieval_questions.json`
- `data/evaluation/baseline-2026-07-29.json`
- `data/evaluation/answer_quality/REPORT-2026-07-30.md`
- `data/evaluation/answer_quality/baseline-2026-07-30.json`
- `data/evaluation/answer_quality/REPORT-2026-07-30-quality-fix.md`
- `data/evaluation/answer_quality/baseline-2026-07-30-quality-fix.json`

首次基线：

- Hit@10：14/15（93.33%）；
- MRR@10：0.7911；
- 已知未命中：`java-1-21-content`。

修复后的完整回答基线：

- 平均正确性：2.00/2；
- 平均完整性：2.00/2；
- 平均证据忠实度：2.00/2；
- 期望来源召回率和引用率：15/15（100%）；
- 引用编号有效率：100%；
- 无答案可靠拒答率：3/3（100%）。

不要删除失败问题或修改正确来源来提高表面指标。Java 版 1.21 内容页现在通过版本
查询规范化与标题优先召回进入 BM25 第 1 名和最终证据第 1 名。当前没有已知的候选
排序瓶颈，因此仍暂不接入 Reranker。

## 6. 后续顺序

1. 保留同一 18 题作为分块、Embedding、检索、融合、上下文、提示词和回答模型变更的
   固定回归，不删除失败问题，不覆盖历史基线；
2. 每次运行记录配置、数据集和系统提示词指纹，拒绝混合不同配置或不完整的评测；
3. 根据固定回归暴露的问题，再选择查询处理、分块与上下文、检索与融合等基础 RAG
   工程改进；
4. 只有候选已召回但最终排序仍不足时，再评估 Reranker；
5. 基础 RAG 稳定后，再进入证据驱动的 Agentic RAG。

未来证据驱动的 Agentic RAG 方案见 `docs/ideas/agentic-rag.md`，当前不实施，也不在
本阶段引入 LangGraph。限流、请求预算、监控、HTTPS Cookie 和滥用防护留到项目明确
需要公开部署时再评估。

## 7. 常用命令

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

相关状态与设计文档：

- `PROJECT_STATUS.md`
- `MVP_PLAN.md`
- `RAG_ASSISTANT_GUIDE.md`
