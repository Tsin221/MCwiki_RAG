# MCwiki RAG V2 核心技术任务包

更新时间：2026-09-23

## 目标

在现有 MCwiki RAG 上补齐查询增强、精排、检索纠正、答案核验和自适应路由。任务只改在线
RAG 链路，继续使用现有 chunk、SQLite FTS5 BM25、Qdrant 向量库、RRF 和相邻证据合并。

本任务包不安排重新切块、重建语料、GraphRAG、开放互联网搜索、大型评测平台或自由循环的
多 Agent 系统。

## 当前基线

```text
问题
  → BM25 Top-20 + 向量 Top-20
  → RRF(k=60)
  → Top-8
  → adjacent_merge（最多 12000 字符）
  → DeepSeek 依据证据生成并标注引用
```

主要入口：

- `code/rag_api.py`：`POST /search`、`POST /answers`；
- `code/rag_retrieval/hybrid.py`：BM25 与向量召回、RRF；
- `code/rag_evidence.py`：证据选择与相邻块合并；
- `code/rag_answer.py`：答案提示词与 DeepSeek 流式调用；
- `code/rag_settings.py`：运行配置。

## 技术来源与项目决策

以下名称和核心机制直接来自 `E:\学习笔记\RAG`：

- Step-back、Cross-Encoder：`RAG基础检索技术提纯.md`；
- Corrective RAG、Self-RAG、Adaptive RAG：`高级RAG技术提纯.md`；
- 有界循环、停止条件和失败降级：`Agentic RAG技术提纯.md`。

任务里的 Python 接口、候选数量、失败回退、功能开关和实施顺序是针对 MCwiki 当前代码的工程
决策，不是参考文档规定的标准参数。执行者不得把教学示例中的阈值直接复制为生产阈值。

## 执行顺序

| 顺序 | 文件 | 核心产出 | 依赖 |
|---|---|---|---|
| 1 | `01-step-back-retrieval.md` | 原问题与抽象问题双路检索 | 当前 `main` |
| 2 | `02-cross-encoder-reranker.md` | 可插拔真实 Cross-Encoder 精排 | 任务 01 |
| 3 | `03-corrective-rag.md` | 证据不足时一次纠正检索 | 任务 02 |
| 4 | `04-answer-verification.md` | 生成后证据支持核验与一次重写 | 任务 03 |
| 5 | `05-adaptive-routing.md` | 按问题选择固定检索管道 | 任务 04 |

五份任务可以交给五个执行者，但不能从同一个旧提交同时开发。后一任务必须基于前一任务已合并
的 `main`，或明确基于前一任务的提交创建分支。

## 共同规则

1. 每个执行者使用独立分支，分支名使用任务文档中的建议名称。
2. 不修改 `data/processed/chunks.jsonl`，不改变 chunk 长度、重叠量、稳定 ID 或入库流程。
3. 不删除 BM25、向量召回、RRF、相邻块合并和现有引用格式。
4. `/search` 保持单次混合检索接口；RAG V2 编排主要接入 `/answers`。
5. 所有模型判断都必须返回受约束的结构化结果；解析失败时走文档规定的回退路径。
6. 所有循环都有硬上限。禁止让模型自行无限检索、无限改写或无限重生成。
7. 不提交模型权重、缓存、`.env`、API Key、数据库副本和临时运行日志。
8. 测试重点是本任务的控制流、边界和回退。固定 18 题只作为最终烟雾回归，不扩建评测系统。
9. 新功能应能通过配置关闭；关闭后行为与当前基线一致。

## 统一验证

在 `code` 目录运行：

```powershell
uv run --with pytest python -m pytest -q
```

每个任务还必须运行各自文档列出的针对性测试。需要下载模型或调用 DeepSeek 的验证与单元测试
分开，单元测试必须使用假实现，不能依赖网络。

## 统一交接格式

执行者完成后提供：

- 分支名和提交哈希；
- 修改文件清单；
- 实际执行的验证命令及结果；
- 默认开关状态和启用方法；
- 失败回退是否经过测试；
- 尚未解决的风险；
- 下一任务可依赖的公共接口。

