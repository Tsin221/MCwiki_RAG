# RAG 助手当前状态

更新时间：2026-07-30

## 阶段结论

本地知识库、混合检索、DeepSeek 流式回答、FastAPI 接口和 React 问答界面均已实现并完成联调。
当前系统可以在浏览器中提交 Minecraft 问题，经过 BM25 + Embedding 双路召回与 RRF
融合后，由 DeepSeek `deepseek-v4-pro` 依据证据生成中文回答，并展示编号引用和可点击来源。

问答 Web MVP 的实现任务 1～7 已完成。15 个可回答问题和 3 个无答案问题的首份
完整链路质量基线已经生成，整体达到建议门槛。版本号查询召回、证据反向解读和
多子问题完整性三个已定位问题也已完成定向修复并通过同一 18 题回归。当前仍不接入
Reranker。下一阶段继续深化基础 RAG 工程，不把公开部署前的限流、请求预算、监控、
HTTPS Cookie 和滥用防护作为当前任务。

现有 15 个可回答问题和 3 个无答案问题已固定为回归集。后续任何分块、Embedding、
检索、融合、上下文、提示词或回答模型变更，都必须使用同一 18 题与当前修复后基线
对比。未来计划构建证据驱动的 Agentic RAG；预期方案见
`docs/ideas/agentic-rag.md`。该方案当前暂缓实施，先继续夯实基础 RAG，本阶段不引入
LangGraph。

## 已完成能力

### 数据与检索

- 原始资料：`data/original_dataset.json`，共 8,200 条；
- 清洗分块：41,368 个文档块；
- BM25：SQLite FTS5 + trigram，索引位于 `data/processed/bm25.db`；
- Embedding：Ollama + `qwen3-embedding:0.6b`，向量维度 1024；
- 向量数据库：Qdrant，集合 `mcwiki_chunks`，Cosine 距离；
- 混合检索：BM25 与 Embedding 双路召回，使用 RRF 融合；
- 检索接口：`POST /search`；
- 15 题检索基线：Hit@10 为 14/15（93.33%），MRR@10 为 0.7911。
- 修复后的 18 题完整链路中，15 个可回答问题的期望来源均进入最终证据并被引用。

### 回答服务与 API

- DeepSeek 客户端与证据上下文构建：`code/rag_answer.py`；
- 模型：`deepseek-v4-pro`，仅由后端调用；
- 流式问答接口：`POST /answers`；
- SSE 事件：`meta`、`sources`、`delta`、`done`，流中错误使用 `error`；
- 统一处理配置错误、检索错误、模型错误和超时；
- 匿名访客 Cookie：`HttpOnly`、`SameSite=Lax`、180 天有效期；
- CORS 使用显式前端 Origin 并允许 Cookie，不使用通配符；
- API Key 只从根目录 `.env` 读取，不进入前端、日志或 Git。

### 前端

- React 19 + TypeScript + Vite；
- 使用 `fetch` + `ReadableStream` 解析 POST SSE；
- 支持问题提交、增量回答、来源展示、错误状态和重试；
- 回答使用受限 Markdown 渲染：丢弃 HTML，只允许文本、列表、强调、代码和表格等安全标签；
- Three.js / React Three Fiber 动态背景异步加载；
- 支持移动端质量降级、`prefers-reduced-motion`、页面隐藏暂停和静态背景回退；
- 匿名 Cookie 请求使用 `credentials: 'include'`。

## 验收结果

- Qdrant `points_count`：41,368，Collection 状态 `green`；
- 后端自动化测试：61 项通过；
- 前端自动化测试：8 项通过；
- TypeScript 类型检查：通过；
- 前端生产构建：通过；
- 真实 `/answers` SSE：HTTP 200，事件顺序与增量输出正确；
- 真实浏览器问题：
  - “钻石矿石在哪里生成？”
  - “红石中继器有什么作用？”
  - “如何找到下界要塞？”
- 三个问题均完成本地检索、DeepSeek 回答、Markdown 展示和 8 条来源展示；
- 桌面端与 360px 移动端视觉检查通过；
- 浏览器控制台：0 error、0 warning；
- 浏览器网络请求：`POST http://localhost:8000/answers` 均为 200。
- 完整链路质量评测：15 个可回答问题 + 3 个无答案问题；
- 修复后平均正确性、完整性、证据忠实度：均为 2.00/2；
- 修复后期望来源召回率与引用率：均为 15/15（100%），引用编号有效率：100%；
- 无答案可靠拒答率：3/3（100%）；
- 最新评测报告：`data/evaluation/answer_quality/REPORT-2026-07-30-quality-fix.md`。
- 固定回归清单：`data/evaluation/regression_suite.json`（`mcwiki-rag-regression-v1`）。

当前保留两项非阻断提示：

- FastAPI TestClient 依赖链提示未来迁移到 `httpx2`；
- Three.js 异步场景包超过 Vite 默认 500 kB 提示，但已从首屏主包拆分。

## 本地启动

前置条件：

- Ollama 已运行并安装 `qwen3-embedding:0.6b`；
- Qdrant 已运行，且 `mcwiki_chunks` 已完成入库；
- 根目录 `.env` 已填写 `DEEPSEEK_API_KEY`。

首次配置：

```powershell
cd E:\Work\MCwiki_RAG
Copy-Item .env.example .env
# 编辑 .env，只在本地填写 DEEPSEEK_API_KEY
```

启动后端：

```powershell
cd E:\Work\MCwiki_RAG\code
uv sync
uv run uvicorn rag_api:app --env-file ..\.env --host 127.0.0.1 --port 8000
```

启动前端：

```powershell
cd E:\Work\MCwiki_RAG\frontend
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`。不要改用 `http://127.0.0.1:5173`，除非同时把
`.env` 中的 `FRONTEND_ORIGIN` 改为完全一致的地址；带 Cookie 的 CORS 必须精确匹配 Origin。

## 验证命令

后端：

```powershell
cd E:\Work\MCwiki_RAG\code
uv run --with pytest python -m pytest
```

前端：

```powershell
cd E:\Work\MCwiki_RAG\frontend
npm run test -- --run
npm run typecheck
npm run build
```

## 当前链路

```text
SQLite BM25 + Qdrant Embedding
              ↓
          RRF 融合
              ↓
       证据去重与上下文限制
              ↓
     DeepSeek 流式证据约束回答
              ↓
       FastAPI POST /answers
              ↓
 React + 安全 Markdown + 来源卡片
```

## 后续工作

1. 将同一 18 题作为基础 RAG 的固定回归门槛，不删除失败问题，也不覆盖历史基线；
2. 每次改变分块、Embedding、检索、融合、上下文、提示词或回答模型时，生成可对比的
   新基线，并记录配置与数据集指纹；
3. 根据固定回归暴露的具体问题，再从查询处理、分块与上下文、检索与融合等方向选择
   下一项基础 RAG 工程改进；
4. 只有出现“候选已召回但最终排序仍不足”的证据时，再评估 Reranker；
5. 基础 RAG 稳定后，再进入证据驱动的 Agentic RAG。

## 当前不做

- MySQL、Redis、Elasticsearch；
- 登录、注册和服务端会话历史；
- 任意互联网搜索；
- LangGraph、Agentic 检索循环和多 Agent；
- 限流、请求预算、监控、HTTPS Cookie 和滥用防护；
- 公开生产环境部署与完整计费系统。
