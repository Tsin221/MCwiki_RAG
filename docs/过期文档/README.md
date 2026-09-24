# 过期文档说明

本目录保存已结束或已被现状取代的文档。它们仅作为历史记录保留，**不再反映当前状态**，
不要据此判断项目进度或实现的接口行为。

当前状态以根目录 [`README.md`](../../README.md) 为准；启动与排障见
[`项目启动说明.txt`](../../项目启动说明.txt)。

## 目录内容

### 根目录移入

| 文档 | 为什么归入过期 |
|---|---|
| `PROJECT_STATUS.md` | 停在 2026-09-22 的状态快照。其中"当前仍不接入 Reranker"在 Cross-Encoder 合入后已成事实错误，测试数量与最新评测报告也已落后。状态职责已由根目录 README 的「当前状态」一节承接 |
| `MVP_PLAN.md` | 首期 MVP 的范围与完成标准，任务已全部完成；正文仍指向 7 月的对比报告为"最新"，并同样写着"暂不接入 Reranker" |
| `RAG_ASSISTANT_GUIDE.md` | 首期技术方案。正文把精排描述为链路内的固定环节，而实际实现是可选的旁路开关，会误导读者 |
| `NEXT_AGENT_HANDOFF.md` | 停留在 2026-07-30 的阶段交接。其交接职能已被 `docs/tasks/ragV2/README.md` 取代，且同样有过期的精排表述 |

### `docs/` 下移入

| 目录 | 原位置 | 为什么归入过期 |
|---|---|---|
| `rag-next/` | `docs/tasks/rag-next/` | 该任务包自身的 README 已声明收尾：三项基础建设任务全部合入 `main`，原任务 04～06 取消。文件名保留 2026-09-23 的状态，与后续提交不再一致 |
| `frontend-threejs/` | `docs/frontend-threejs/` | 前端已实施并验收完成，属于已完成任务的设计记录。其中 `README.md` 第 1.1 节声称浮空方块岛、粒子和雾已实现，与同目录 `implementation-plan.md` 的更正说明和 `frontend/src/components/scene/` 的实际实现直接矛盾，该错误未修正 |
| `计划文档/` | `docs/计划文档/` | 2026-09 的评审记录与整改方案。其 README 已说明其中的描述反映评审当时的状态，相关整改均已在后续提交中完成 |

## 仍然有效的文档

- 根目录 `README.md`：项目入口，含当前状态、已知缺口、快速开始、接口与评测指标；
- 根目录 `项目启动说明.txt`：逐步骤启动与常见问题排查；
- `docs/specs/qa-web-mvp.md`：问答 Web MVP 的接口与错误契约，仍是有效规格；
- `docs/ideas/agentic-rag.md`：暂缓实施的未来方案，属于前瞻而非过期；
- `docs/tasks/ragV2/`：当前任务包，任务 01、02 已实现，03～05 待执行。

## 已知的失效引用

本目录内的文档引用的路径是移动前的位置（例如 `docs/frontend-threejs/implementation-plan.md`），
为保持历史记录原样未作改写。

跨目录引用已随之更新：

- `README.md` 不再索引移入本目录的四份根目录文档；
- `MVP_PLAN.md` 中的 Three.js 前端方案路径现指向 `docs/过期文档/frontend-threejs/`；
- `计划文档/README.md` 的导航句原指向根目录的 `PROJECT_STATUS.md` 与 `MVP_PLAN.md`，
  这两份现已归档，该句改为指向根目录 `README.md`。
