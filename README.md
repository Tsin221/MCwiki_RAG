# MCwiki RAG

基于中文 Minecraft Wiki 的检索增强问答（RAG）助手。本地知识库经 BM25 与向量双路召回、
RRF 融合后组成证据上下文，由 DeepSeek 依据证据流式生成带编号引用的中文回答。

> 本项目为非官方项目，与 Mojang Studios、Minecraft Wiki 均无关联。
> 仓库中的 Wiki 数据与 Minecraft 相关素材遵循其各自的原始许可，详见 [NOTICE.md](NOTICE.md)。

## 当前状态

问答 Web MVP（任务 1～7）已完成并形成质量基线，处于「完善基础 RAG 能力 + 固定 18 题回归门槛」
阶段。已实现：

- 8,200 条原始资料清洗分块为 41,368 个文档块，建立 SQLite FTS5 BM25 索引与 Qdrant 向量索引；
- BM25 + 语义双路召回、RRF 融合、相邻证据块合并与上下文预算控制；
- 基于 SSE 的流式证据约束回答，带编号引用与来源卡片；
- 查询规划可插拔：`original` 与 `step_back`（抽象问题 + 多查询 RRF 融合）；
- 候选精排可插拔：`none` 与 `cross_encoder`（Cross-Encoder 重排候选池）。

任务 03（纠正式检索）、04（答案核验）、05（自适应路由）尚未实现，规划见
[`docs/tasks/ragV2/`](docs/tasks/ragV2/README.md)。

### 已知缺口

- Cross-Encoder 精排已接入但**尚无 18 题 A/B 对比证据**，因此默认关闭；开启前应先按
  [评测与回归](#评测与回归)完成一次对照。
- `step_back` 查询规划同样缺少评测报告，默认保持 `original`。
- 评测集 v2 共 48 题，其中仅 7 题经人工确认答案要点，其余为候选标注。
- `POST /search` 走的是不含查询规划与精排的混合检索，与 `/answers` 不是同一条链路，
  两者对同一问题可能给出不同排序。
- 无鉴权、限流、监控与滥用防护，暂不具备公开部署条件。

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
docs/                 规格、规划与归档
  specs/              问答 Web MVP 的接口与错误契约
  tasks/ragV2/        当前任务包（03～05 待执行）
  ideas/              暂缓实施的 Agentic RAG 方案
  过期文档/           已结束或被现状取代的文档，仅作历史记录
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
| POST | `/search` | 混合检索，返回两路召回经 RRF 融合后的候选证据（不含查询规划与精排） |
| POST | `/answers` | 证据约束的流式回答，SSE 事件为 `meta`、`sources`、`delta`、`done`，流中错误为 `error` |
| GET | `/health` | 仅表示 API 进程存活 |
| GET | `/ready` | 检查 BM25、目标 Qdrant collection、回答服务与精排配置 |

交互式文档位于 <http://127.0.0.1:8000/docs>。缺少 DeepSeek Key 时 `/search` 仍可用，
`/answers` 返回 `CONFIGURATION_ERROR`。

## 测试

```powershell
# 后端：18 个测试文件、185 项用例，全部离线，不访问网络与真实数据库
cd code
uv run --with pytest python -m pytest

# 前端：3 个测试文件、10 项用例
cd frontend
npm run test -- --run
npm run typecheck
npm run build
```

## 评测与回归

`data/evaluation/regression_suite.json`（`mcwiki-rag-regression-v1`）固定了 18 道题
（15 道可回答、3 道无答案）。任何涉及分块、Embedding、检索、融合、上下文、提示词或回答
模型的改动，都要用同一套题与既有基线对比，并保留历史结果。

检索基线（`data/evaluation/baseline-2026-07-29.json`，15 题）：

| 指标 | 结果 |
| --- | --- |
| 检索 Hit@10 | 14/15（93.33%） |
| 检索 MRR@10 | 0.7911 |

回答质量基线（`data/evaluation/answer_quality/baseline-2026-09-23-evidence-assembly.json`，18 题）：

| 指标 | 结果 |
| --- | --- |
| 正确性 / 完整性 / 忠实度 | 2.00 / 2.00 / 2.00（满分 2） |
| 期望来源进入证据 | 15/15 |
| 严格引用期望来源 | 14/15 |
| 引用编号有效 | 100% |
| 无答案可靠拒答 | 3/3 |

最新结论见 [`data/evaluation/v2/REPORT-2026-09-23-evidence-assembly.md`](data/evaluation/v2/REPORT-2026-09-23-evidence-assembly.md)，
BM25 查询策略对照见 [`data/evaluation/REPORT-2026-09-23-bm25-diagnostic.md`](data/evaluation/REPORT-2026-09-23-bm25-diagnostic.md)。
评测方法、评分规则与门槛说明见 [`data/evaluation/answer_quality/README.md`](data/evaluation/answer_quality/README.md)，
评测集 v2 的样本结构与标注口径见 [`data/evaluation/v2/README.md`](data/evaluation/v2/README.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [`项目启动说明.txt`](项目启动说明.txt) | 逐步骤的本地启动、依赖服务与常见问题排查 |
| [`docs/specs/qa-web-mvp.md`](docs/specs/qa-web-mvp.md) | 问答 Web MVP 规格：UX、SSE 与错误契约、提示词约束、完成标准 |
| [`docs/tasks/ragV2/`](docs/tasks/ragV2/README.md) | 当前任务包：01 查询规划、02 精排已实现，03～05 待执行 |
| [`docs/ideas/agentic-rag.md`](docs/ideas/agentic-rag.md) | 暂缓实施的证据驱动 Agentic RAG 方案 |
| [`docs/过期文档/`](docs/过期文档/README.md) | 历史文档归档（MVP 计划、项目状态、技术指南、阶段交接与评审记录） |

## 许可与数据来源

- 源代码以 MIT 许可发布，见 [LICENSE](LICENSE)。
- `data/original_dataset.json` 来自中文 Minecraft Wiki（<https://zh.minecraft.wiki/>），
  内容遵循 CC BY-NC-SA 3.0，**不适用 MIT 许可**。详见 [NOTICE.md](NOTICE.md)。
- Minecraft 是 Mojang Studios 的商标，本项目与其无关联。

## 当前不做

引入 LangGraph、多 Agent，使用 MySQL、Redis 或 Elasticsearch，以及 GraphRAG。限流、请求预算、
监控、HTTPS Cookie 与滥用防护留到明确需要公开部署时再评估；开放互联网检索仅作为
`docs/ideas/agentic-rag.md` 中的前瞻方案，当前不实施。
