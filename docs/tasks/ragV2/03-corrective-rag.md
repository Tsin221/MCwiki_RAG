# 任务 03：Corrective RAG 有界纠正检索

建议分支：`codex/ragv2-corrective`

依赖：任务 02 已合并。

## 任务目标

在答案生成前判断首轮证据是否足以回答问题。证据不足时，生成一次更适合检索的查询并再检索
一次；合并、精排和重新检查后仍不足则明确拒答，不把低质量片段直接交给生成模型。

## 参考资料中的技术

`E:\学习笔记\RAG\高级RAG技术提纯.md` 第 6 节介绍 Corrective RAG：先评估首轮检索质量，
质量差时改写问题并补充检索。参考示例会转向占位“网页搜索”，但文档也指出来源权威性和兜底
幻觉风险。

## MCwiki 工程决策

- 纠正范围只限现有 MCwiki 知识库，不接开放互联网；
- 不把 RRF 分数当相关概率，不照搬教学示例的 `0.45` 阈值；
- 使用结构化证据评估结果：`sufficient`、`missing_aspects`、`rewritten_query`、`reason`；
- 最多两轮检索：原查询一次、改写查询一次；
- 第二轮候选与第一轮候选按稳定 chunk ID 合并，再走任务 02 的同一精排入口；
- 合并后再检查一次充分性；仍不足时返回现有固定“资料不足”消息；
- 评估器不可用时保留当前单轮 RAG 行为并记录日志，避免模型评估服务故障拖垮问答服务；
- 空首轮结果允许请求一次查询改写，但不得生成无证据答案。

## 实现范围

1. 新建证据充分性评估协议、结构化结果和 DeepSeek 实现；
2. 新建最多两轮的 Corrective 协调器；
3. 复用任务 01 查询检索、任务 02 精排和现有证据组装；
4. 将最终证据确定后再发送 SSE `sources` 事件；
5. 为关闭、仅首轮、纠正成功、纠正后仍不足四种路径建立测试；
6. 增加开关、评估超时和最大轮数配置，其中最大轮数代码层面不得超过 2。

建议数据结构：

```python
@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    sufficient: bool
    missing_aspects: tuple[str, ...]
    rewritten_query: str | None
    reason: str

@dataclass(frozen=True, slots=True)
class CorrectiveResult:
    evidence: tuple[AnswerEvidence, ...]
    retrieval_rounds: int
    sufficient: bool
```

## 文件边界

预计新增：

- `code/rag_correction.py`；
- `code/tests/test_correction.py`。

预计修改：

- `code/rag_api.py`；
- `code/rag_settings.py`；
- `code/tests/test_api.py`；
- 必要时小幅扩展任务 01、02 提供的公共入口。

不要加入网页搜索、LangGraph、递归 Agent、第三轮检索或 chunk 改造。

## 验收标准

- [ ] 充分证据路径只检索一轮，不产生多余改写；
- [ ] 不充分路径只允许一个改写查询和一次补充检索；
- [ ] 两轮结果按 `chunk_id` 去重，并经过统一精排与证据预算；
- [ ] 第二次检查仍不充分时不调用答案生成模型；
- [ ] 评估器非法输出、超时和异常均有确定回退行为；
- [ ] 循环次数不能由模型输出扩大；
- [ ] SSE 的 `sources` 与最终实际交给生成模型的证据一致；
- [ ] 完整后端测试通过。

## 针对性验证

```powershell
uv run --with pytest python -m pytest -q tests/test_correction.py tests/test_api.py
```

测试必须断言模型调用次数和检索调用次数，不能只检查最终字符串。

## 交付给任务 04 的接口

交付“最终证据 + 是否充分”的确定结果。答案核验只处理已经通过充分性检查的证据，不能自行
再次检索。

