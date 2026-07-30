# RAG 助手当前状态

更新时间：2026-07-30

## 阶段结论

本地知识库、混合检索、DeepSeek 流式回答、FastAPI 接口和 React 问答界面均已实现并完成联调。
当前系统可以在浏览器中提交 Minecraft 问题，经过 BM25 + Embedding 双路召回与 RRF
融合后，由 DeepSeek `deepseek-v4-pro` 依据证据生成中文回答，并展示编号引用和可点击来源。

问答 Web MVP 的实现任务 1～6 已完成。尚未完成的是基于现有 15 题评测集的完整回答质量与
引用准确性基线；是否接入 Reranker 应在该基线完成后决定。

未来计划构建可信度优先、受限联网的 Agentic RAG；预期方案见
`docs/ideas/agentic-rag.md`。该方案当前暂缓实施，本阶段不引入 LangGraph。

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
- 后端自动化测试：48 项通过；
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

1. 用现有 15 题生成首份回答准确性、引用准确性和无答案行为基线；
2. 根据完整链路评测决定是否接入 Reranker；
3. 公开部署前增加限流、预算、监控、HTTPS Cookie 和滥用防护；
4. 评估来源按页面进一步合并，减少同一 Wiki 页面多 chunk 带来的重复卡片。

## 当前不做

- MySQL、Redis、Elasticsearch；
- 登录、注册和服务端会话历史；
- 任意互联网搜索；
- LangGraph、Agentic 检索循环和多 Agent；
- 公开生产环境部署与完整计费系统。
