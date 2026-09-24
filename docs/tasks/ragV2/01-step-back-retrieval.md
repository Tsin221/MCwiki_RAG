# 任务 01：Step-back 双路检索

建议分支：`codex/ragv2-step-back`

依赖：当前 `main`。

## 任务目标

实现参考资料中的 Step-back 查询扩展：始终保留用户原问题，并允许模型生成一个更抽象的背景
问题；分别运行现有混合检索，再按稳定 chunk ID 合并结果。该任务建立查询计划和多查询融合接口，
供后续纠正检索与自适应路由复用。

## 参考资料中的技术

`E:\学习笔记\RAG\RAG基础检索技术提纯.md` 第 5.3 节说明：具体问题与更抽象的背景问题
分别检索，再合并证据。参考脚本只展示了抽象问题，没有完成双路检索；本任务负责把链路真正接通。

## MCwiki 工程决策

- 本任务实现 Step-back，不把未在参考资料中展开的 Multi-Query 当作既定方案；
- 查询计划最多包含两个查询：原问题和一个抽象问题；
- 原问题必须排在第一位，精确名称、版本、数字和物品 ID 不得被改写后丢失；
- 每个查询继续走现有 BM25 + 向量 + RRF；
- 多查询结果按每路名次再次做 RRF，并按 `chunk_id` 去重，不直接相加不同查询的原始分数；
- 通过 `MCWIKI_QUERY_STRATEGY=original|step_back` 控制，默认 `original`，由任务 05 决定自动路由；
- Step-back 模型调用失败、超时、返回空文本或无效结构时，只使用原问题，不让 `/answers` 失败。

## 实现范围

1. 新建查询计划数据结构和规划器协议；
2. 提供原问题规划器与 DeepSeek Step-back 规划器；
3. 输出受约束 JSON，只接受一个非空抽象问题；
4. 新建多查询检索协调器，复用 `HybridRetriever.search`；
5. 把协调器接入 `/answers`，保留 `/search` 当前语义；
6. 将策略、最多查询数和规划超时纳入集中配置。

建议公共契约：

```python
@dataclass(frozen=True, slots=True)
class QueryPlan:
    original: str
    retrieval_queries: tuple[str, ...]
    strategy: str

class QueryPlanner(Protocol):
    async def plan(self, question: str) -> QueryPlan: ...
```

`retrieval_queries[0]` 必须等于规范化后的 `original`，总数不得超过 2。

## 文件边界

预计新增：

- `code/rag_query.py`；
- `code/rag_retrieval/multi_query.py`；
- `code/tests/test_query.py`；
- `code/tests/test_multi_query.py`。

预计修改：

- `code/rag_api.py`；
- `code/rag_settings.py`；
- `code/tests/test_api.py`。

不要修改 chunk 生成、BM25 建库、Qdrant 入库、证据合并和答案提示词。

## 验收标准

- [ ] `original` 模式的检索调用、排序和最终证据与当前基线一致；
- [ ] `step_back` 模式确实执行原问题和抽象问题两次混合检索；
- [ ] 相同 chunk 在两路出现时只保留一次，并能因两路命中获得融合优势；
- [ ] 模型返回非法 JSON、空问题、重复问题、超时或异常时可靠退回原问题；
- [ ] 查询数硬限制为 2，模型输出不能突破限制；
- [ ] `/search` 的请求和响应契约保持不变；
- [ ] 完整后端测试通过。

## 针对性验证

```powershell
uv run --with pytest python -m pytest -q tests/test_query.py tests/test_multi_query.py tests/test_api.py
```

至少覆盖三个行为样例：精确名称问题保留原词、需要上位概念的问题产生抽象查询、规划器故障时
只执行原查询。

## 交付给任务 02 的接口

交付一个返回较大候选池的统一检索入口。后续 Reranker 不应关心候选来自原查询还是 Step-back
查询，只读取统一的 `HybridResult` 列表。

