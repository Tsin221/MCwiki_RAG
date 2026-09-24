# MCwiki RAG

基于中文 Minecraft Wiki 的检索增强问答（RAG）助手。本地知识库经 BM25 与向量双路召回、
RRF 融合后组成证据上下文，由 DeepSeek 依据证据流式生成带编号引用的中文回答。

> 本项目为非官方项目，与 Mojang Studios、Minecraft Wiki 均无关联。
> 仓库中的 Wiki 数据与 Minecraft 相关素材遵循其各自的原始许可，详见 [NOTICE.md](NOTICE.md)。

## 技术链路

```text
SQLite FTS5 BM25  +  Qdrant 向量召回
                  ↓
             RRF 排名融合
                  ↓
        Cross-Encoder 精排（可选）
                  ↓
          证据去重与上下文预算
                  ↓
     DeepSeek 流式证据约束回答（SSE）
                  ↓
      React + 安全 Markdown + 来源卡片
```

## 技术栈

| 部分 | 选型 |
| --- | --- |
| 后端 | Python 3.12 + FastAPI + uv |
| 关键词检索 | SQLite FTS5 + trigram，适配中文关键词与版本号 |
| 语义检索 | Ollama `qwen3-embedding:0.6b`，1024 维 |
| 向量数据库 | Qdrant，集合 `mcwiki_chunks`，Cosine 距离 |
| 融合方式 | RRF（Reciprocal Rank Fusion） |
| 候选精排 | 可选 Cross-Encoder（默认 `BAAI/bge-reranker-base`），关闭时保持 RRF 顺序 |
| 回答生成 | DeepSeek `deepseek-v4-pro`，HTTPX 流式调用 |
| 前端 | React + TypeScript + Vite + React Three Fiber |

## 目录结构

```text
code/                 FastAPI 后端、检索与入库模块、测试
  rag_ingest/         清洗分块、BM25 索引、向量入库
  rag_retrieval/      BM25、语义召回、混合检索与 RRF 融合
  rag_api.py          POST /search 与 POST /answers
frontend/             React 问答界面
data/                 original_dataset.json（随仓库提供）与评测数据
docs/                 方案、任务与交接文档
```

## 快速开始

### 前置依赖

- Python 3.12 与 [uv](https://docs.astral.sh/uv/)
- Node.js（运行前端）
- [Ollama](https://ollama.com/)，并已拉取 embedding 模型：`ollama pull qwen3-embedding:0.6b`
- Qdrant 实例（默认映射到 `127.0.0.1:6335`）
- 一个 DeepSeek API Key，仅由后端使用

启动 Qdrant，容器名与端口需与 `.env` 中的 `MCWIKI_QDRANT_URL` 一致：

```powershell
docker run -d --name mcwiki-qdrant --restart unless-stopped -p 127.0.0.1:6335:6333 -p 127.0.0.1:6336:6334 -v mcwiki_qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

### 1. 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env` 填入 `DEEPSEEK_API_KEY`。真实密钥只放在根目录 `.env`（该文件已被 `.gitignore`
忽略），不要放进 `frontend/`，也不要提交。

### 2. 构建本地索引

从 `code/` 目录依次执行，产物写入 `data/processed/`（已被 Git 忽略）：

```powershell
cd code
uv sync

# 清洗与分块：data/original_dataset.json → data/processed/chunks.jsonl
uv run python -m rag_ingest

# BM25 索引：chunks.jsonl → data/processed/bm25.db
uv run python -m rag_ingest.bm25_index

# 向量入库：chunks.jsonl → Qdrant 集合 mcwiki_chunks
uv run python -m rag_ingest.vector_ingest
```

本仓库提供的原始数据为 8,200 条，分块后得到 41,368 个文档块。`chunks.jsonl` 与 `bm25.db`
是可再生产物，约 68 MB 与 245 MB，因此未纳入版本管理。向量入库支持断点续跑：启动时会先
分页读取 Qdrant 中已有点，只为缺失的文档块调用 Ollama。

### 3. 启动服务

后端：

```powershell
cd code
uv run uvicorn rag_api:app --env-file ..\.env --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

浏览器打开 <http://localhost:5173>。

> 必须使用 `localhost`。后端使用带 Cookie 的 CORS，浏览器 Origin 必须与 `.env` 中的
> `FRONTEND_ORIGIN` 完全一致。仅检查端口监听无法确认连到的是本项目实例。

### 4. 可选：启用 Cross-Encoder 精排

默认关闭（`MCWIKI_RERANKER=none`），`/answers` 直接使用 RRF 顺序。开启后链路变为
「召回 `MCWIKI_CANDIDATE_LIMIT` 个候选 → 用真实 Cross-Encoder 对「问题、`title + text`」
配对打分 → 保留 `MCWIKI_EVIDENCE_LIMIT` 个 → 证据组装」。Cross-Encoder 只能调整已经召回
候选的顺序，不能找回召回阶段遗漏的证据。

```powershell
cd code
uv sync --extra reranker          # 安装 sentence-transformers/torch（可选依赖）
# 在 .env 中设置 MCWIKI_RERANKER=cross_encoder 后启动服务
```

模型在应用启动时加载一次并常驻；模型未安装、加载失败或推理异常时自动回退到 RRF 顺序并
记录日志，`/ready` 会报告 `reranker: unavailable`。无法访问 `huggingface.co` 的网络可先
设置 `HF_ENDPOINT=https://hf-mirror.com` 下载模型。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/search` | 混合检索，返回两路召回经 RRF 融合后的候选证据 |
| POST | `/answers` | 证据约束的流式回答，SSE 事件为 `meta`、`sources`、`delta`、`done`，流中错误为 `error` |
| GET | `/health` | 仅表示 API 进程存活 |
| GET | `/ready` | 检查 BM25、目标 Qdrant collection、回答服务与精排配置 |

交互式文档位于 <http://127.0.0.1:8000/docs>。缺少 DeepSeek Key 时 `/search` 仍可用，
`/answers` 返回 `CONFIGURATION_ERROR`。

## 测试

```powershell
# 后端
cd code
uv run --with pytest python -m pytest

# 前端
cd frontend
npm run test -- --run
npm run typecheck
npm run build
```

## 评测与回归

`data/evaluation/regression_suite.json`（`mcwiki-rag-regression-v1`）固定了 18 道题
（15 道可回答、3 道无答案）。任何涉及分块、Embedding、检索、融合、上下文、提示词或回答
模型的改动，都要用同一套题与既有基线对比，并保留历史结果。

| 指标 | 结果 |
| --- | --- |
| 检索 Hit@10（15 题） | 14/15（93.33%） |
| 检索 MRR@10 | 0.7911 |
| 期望来源进入证据（18 题回归） | 15/15 |
| 严格引用期望来源 | 14/15 |
| 无答案可靠拒答 | 3/3 |

各次评测的报告与失败归因见 `data/evaluation/` 下的 README 与 REPORT 文件。

## 文档

| 文档 | 内容 |
| --- | --- |
| `项目启动说明.txt` | 逐步骤的本地启动与常见问题排查 |
| `PROJECT_STATUS.md` | 当前状态、验收结果与后续工作 |
| `RAG_ASSISTANT_GUIDE.md` | 技术方案、各环节职责与评测约定 |
| `MVP_PLAN.md` | 首期范围与完成标准 |
| `NEXT_AGENT_HANDOFF.md` | 阶段交接信息 |
| `docs/ideas/agentic-rag.md` | 暂缓实施的证据驱动 Agentic RAG 方案 |

## 许可与数据来源

- 源代码以 MIT 许可发布，见 [LICENSE](LICENSE)。
- `data/original_dataset.json` 来自中文 Minecraft Wiki（<https://zh.minecraft.wiki/>），
  内容遵循 CC BY-NC-SA 3.0，**不适用 MIT 许可**。详见 [NOTICE.md](NOTICE.md)。
- Minecraft 是 Mojang Studios 的商标，本项目与其无关联。

## 当前不做

限流、请求预算、监控、HTTPS Cookie 与滥用防护留到明确需要公开部署时再评估。Cross-Encoder
精排已接入但默认关闭（`MCWIKI_RERANKER=none`），是否默认开启由固定 18 题回归决定；不引入
LangGraph、多 Agent，也不使用 MySQL、Redis 或 Elasticsearch。
