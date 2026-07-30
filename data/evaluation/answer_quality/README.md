# 回答质量评测

本目录记录问答 Web MVP 的完整链路质量基线。评测固定使用现有
`../retrieval_questions.json` 中的 15 个可回答问题，并补充
`unanswerable_questions.json` 中的资料不足或域外问题。

## 评测目标

回答质量不能只用“模型觉得回答不错”来判断。本评测把结果拆成两层：

1. **自动检查**：期望 Wiki 页面是否进入证据、回答是否包含引用、引用编号是否存在、
   回答是否引用了期望页面；
2. **人工复核**：回答事实是否正确、是否覆盖问题要点、每个关键表述是否由对应证据
   支持，以及资料不足时是否明确拒答。

自动检查用于发现确定性的结构问题，人工复核用于判断语义和事实，二者不能互相替代。

## 固定配置

- 回答模型：环境变量 `DEEPSEEK_MODEL`，当前预期为 `deepseek-v4-pro`；
- BM25 候选数：20；
- Embedding 候选数：20；
- RRF 最终证据数：8；
- 回答 temperature：沿用 `rag_answer.py` 的 0.2；
- 可回答问题：`../retrieval_questions.json`；
- 无答案问题：`unanswerable_questions.json`。

每次改变分块、Embedding、融合参数、上下文预算、提示词或回答模型，都应生成新的
基线文件，不能覆盖旧结果。

## 人工评分规则

### 可回答问题

每题从三个维度评分，每项 0～2 分：

| 维度 | 2 分 | 1 分 | 0 分 |
|---|---|---|---|
| 正确性 | 核心事实正确，无实质错误 | 核心方向正确，但有轻微错误或含混 | 核心事实错误或无法回答 |
| 完整性 | 覆盖问题中的主要子问题 | 只覆盖部分要点 | 基本未覆盖 |
| 证据忠实度 | 关键表述均能由引用证据直接支持 | 大体有证据，但存在弱支持表述 | 存在无依据关键表述或错误归引 |

单题满分 6 分。复核者必须写简短说明，不能只给数字。

### 无答案问题

记录一个布尔结果：

- `true`：明确说明知识库无法可靠回答，没有用无关证据编造事实；
- `false`：给出无依据的确定性答案，或把无关 Minecraft 资料当作支持证据。

## MVP 建议门槛

- 可回答问题平均正确性不低于 1.6/2；
- 可回答问题平均完整性不低于 1.5/2；
- 可回答问题平均证据忠实度不低于 1.8/2；
- 引用编号有效率为 100%；
- 至少 80% 的可回答问题引用期望 Wiki 页面；
- 无答案问题可靠拒答率不低于 80%；
- 不允许出现引用不存在编号或泄露系统提示词、密钥、内部检索分数。

门槛用于决定当前链路是否足以继续优化，而不是为了删除失败问题。任何失败样例都应
保留在报告中。

## 运行

在 `code/` 目录执行：

```powershell
uv run --env-file ../.env python -m rag_evaluation run `
  --output ../data/evaluation/answer_quality/raw-2026-07-30.json
```

完成人工复核后生成汇总：

```powershell
uv run python -m rag_evaluation summarize `
  --input ../data/evaluation/answer_quality/raw-2026-07-30.json `
  --reviews ../data/evaluation/answer_quality/reviews-2026-07-30.json `
  --output ../data/evaluation/answer_quality/baseline-2026-07-30.json
```

原始生成结果、人工评分和最终汇总分开保存，以便审计评分是否与原回答一致。
