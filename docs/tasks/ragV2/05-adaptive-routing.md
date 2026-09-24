# 任务 05：Adaptive RAG 路由与统一编排

建议分支：`codex/ragv2-adaptive-routing`

依赖：任务 04 已合并。

## 任务目标

根据问题是否需要知识库事实以及取证复杂度，在执行前选择固定、有界的处理管道。把前四项技术
组合成清晰状态机，使简单问题不承担所有模型调用，复杂问题又能使用 Step-back 或子问题拆分。

## 参考资料中的技术

`E:\学习笔记\RAG\高级RAG技术提纯.md` 第 7 节将问题路由到 `direct`、`single`、`multi`
预设管道，并强调 Adaptive RAG 是先路由再执行，不是根据每轮工具结果自由循环。

`E:\学习笔记\RAG\Agentic RAG技术提纯.md` 要求为再检索设置步数、停止、超时和费用边界。
本任务采用这些控制原则，但不实现自由 Agent。

## MCwiki 工程决策

定义四种路由：

| 路由 | 适用问题 | 执行管道 |
|---|---|---|
| `fixed_reply` | 问候、能力说明 | 服务端固定回复，不调用知识库和生成模型 |
| `single` | 单一事实或精确实体问题 | 原问题检索 → 精排 → Corrective → 核验回答 |
| `step_back` | 需要背景概念才能回答 | 原问题 + 一个抽象问题 → 融合 → 后续统一流程 |
| `decompose` | 明确包含多个独立子问题、对比或组合条件 | 最多 3 个子问题分别检索 → 合并精排 → 后续统一流程 |

附加约束：

- 不提供“模型无证据直接回答 Minecraft 事实”的路径；
- `fixed_reply` 只允许有限白名单意图，分类不确定时回退 `single`；
- 子问题拆分最多 3 个，必须保留原始问题用于最终生成和答案核验；
- 各子问题结果按稳定 chunk ID 合并，统一精排，不能各自直接生成局部答案；
- Corrective 最多补检索一轮，答案最多重写一次，沿用任务 03、04 的上限；
- 第一版使用普通 Python 状态机和显式数据结构，不引入 LangGraph；
- 路由器异常、非法标签或超时统一回退 `single`。

## 实现范围

1. 新建路由协议、结构化决策和 DeepSeek Router；
2. 实现上述四个固定执行路径；
3. 新建统一 `RagPipeline`，把查询计划、召回、精排、纠正、证据组装、生成和核验串起来；
4. 将 `/answers` 中的编排逻辑移入管道，API 层只负责校验、错误映射、SSE 和 Cookie；
5. 在 `done` 事件中提供稳定的 `status`，可附加 `route`、检索轮数和是否重写等非敏感字段；
6. 增加总预算保护：最多 3 个初始检索查询、1 个纠正查询、2 次答案生成、2 次答案核验；
7. 为每条路由、非法路由、部分子问题无证据和预算耗尽建立测试。

建议状态：

```python
@dataclass(slots=True)
class RagState:
    question: str
    route: str
    retrieval_queries: list[str]
    candidates: list[HybridResult]
    evidence: list[AnswerEvidence]
    retrieval_rounds: int
    answer_attempts: int
    verification_attempts: int
    final_status: str
```

状态中只保存控制流程需要的数据，不保存或对外暴露模型隐藏推理。

## 文件边界

预计新增：

- `code/rag_router.py`；
- `code/rag_pipeline.py`；
- `code/tests/test_router.py`；
- `code/tests/test_pipeline.py`。

预计修改：

- `code/rag_api.py`；
- `code/rag_settings.py`；
- `code/tests/test_api.py`；
- `README.md` 中的最终流程和配置说明。

前四个任务的核心模块只允许为统一接口做小幅调整，不得在本任务中重写其算法。

## 验收标准

- [ ] 四种路由均能通过测试稳定进入对应固定管道；
- [ ] Minecraft 事实问题不会绕过检索直接让模型回答；
- [ ] `decompose` 最多产生 3 个子问题，最终只生成一份综合答案；
- [ ] 路由失败确定性回退 `single`；
- [ ] 所有模型与检索调用都受总预算约束；
- [ ] `/answers` 的 SSE 事件和前端消费方式保持兼容；
- [ ] 关闭 Adaptive Router 时可回到任务 04 的单路行为；
- [ ] 完整后端测试通过；
- [ ] 用问候、单事实、需要背景、多子问题、无答案各一个真实样例完成端到端烟雾验证。

## 针对性验证

```powershell
uv run --with pytest python -m pytest -q tests/test_router.py tests/test_pipeline.py tests/test_api.py
uv run --with pytest python -m pytest -q
```

真实模型烟雾验证只需上述五类问题，不要求扩建或重新标注评测集。

## 最终流程

```text
问题
  → Adaptive Router
  → fixed_reply / single / step_back / decompose
  → BM25 + 向量 + RRF 多候选召回
  → Cross-Encoder 精排
  → Corrective 充分性检查（必要时一次改写再检索）
  → adjacent_merge 证据组装
  → 生成草稿
  → 引用检查 + Self-RAG 式支持度/完整性核验
  → 必要时一次重写并复核
  → SSE 返回最终答案与实际来源
```

