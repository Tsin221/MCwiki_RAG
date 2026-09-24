# 任务 01：建立评测集 v2 与标注规范

## 任务目标

建立一套比现有 18 题更可诊断的评测数据结构和首批题集。v2 必须区分“允许的权威来源”、
“首选正文来源”、“答案必答要点”和“相关证据块”，从而支持 Recall、Precision、排名、
要点覆盖和引用等分层指标。

建议分支：`codex/evaluation-v2`

## 背景

现有题集每道可回答题主要只有一个 `expected_sources`。Java 1.21 问题引用了更直接的
`Java版1.21` 正文页，却因没有引用题集指定的 `1.21` 消歧义页而失败。这说明现有标签无法
表达多个有效来源。现有 18 题也不足以稳定判断 BM25、跨块问题和多子问题完整性。

## 范围

1. 定义并文档化 v2 样本结构；
2. 建立 40～60 个新增问题，不能复制现有 18 题换说法充数；
3. 编写数据校验器和单元测试；
4. 生成题集统计报告；
5. 保持 v1 文件及现有评测命令兼容。

## 建议数据结构

每条样本至少包含：

```json
{
  "id": "stable-unique-id",
  "question": "用户问题",
  "kind": "answerable",
  "category": "multi_facet",
  "facets": ["爆炸条件", "普通掉落", "唱片条件", "头颅条件"],
  "expected_answer_points": [
    {"id": "music-disc-condition", "description": "说明唱片掉落的击杀条件"}
  ],
  "preferred_sources": ["https://zh.minecraft.wiki/w/..."],
  "acceptable_sources": ["https://zh.minecraft.wiki/w/..."],
  "relevant_chunk_ids": ["..."],
  "edition": "all",
  "version_scope": "current"
}
```

无答案题应包含 `reason`，并明确属于域外、未来、实时私有状态还是知识库缺失。字段命名可在
实现前微调，但必须能表达上述语义。

## 题型配额

- 单一事实：8～10；
- 精确版本、数字、日期或命令：8～10；
- 同义改写：6～8；
- 多子问题：10～12；
- 跨块整合：6～8；
- 无答案或域外：6～8；
- 冲突、版本或版本差异：4～6。

同一道题可以属于多个标签，但统计报告必须给出主类别，避免重复计数导致总数不清。

## 实施步骤

1. 抽样查看 `data/processed/chunks.jsonl`，选择确实有证据的问题；
2. 为答案要点和有效来源做人工标注；
3. 用稳定块 ID 标注相关证据，允许一个问题对应多个块；
4. 编写校验器，检查 ID 唯一、URL 域名、字段类型、块 ID 存在性、配额和无答案约束；
5. 新增只读统计命令，输出题数和类别分布；
6. 为 Java 1.21 建立 v2 标签示例：正文页为 preferred，消歧义页可作为 acceptable；
7. 不修改 `data/evaluation/regression_suite.json` 的 v1 定义。

## 建议文件边界

允许新增或修改：

- `data/evaluation/v2/` 下的新数据与说明；
- `code/rag_evaluation_schema.py` 或等价的新模块；
- `code/tests/test_evaluation_schema.py`；
- 本任务专属报告。

不要修改：

- `code/rag_api.py`、`code/rag_answer.py`、检索实现；
- 现有 18 题和历史结果文件；
- 生产默认配置。

## 验收标准

- [ ] 新增 40～60 个问题并符合题型配额；
- [ ] 每个可回答问题至少有一个 answer point 和一个 acceptable source；
- [ ] 标注的所有 chunk ID 都存在于当前 `chunks.jsonl`；
- [ ] Java 1.21 等价来源能被数据结构正确表达；
- [ ] 校验器能拒绝重复 ID、未知块、非法 URL、空要点和结构错误；
- [ ] v1 的 18 题测试保持通过；
- [ ] 不需要真实 DeepSeek 调用即可完成本任务。

## 验证

```powershell
cd E:\Work\MCwiki_RAG\code
uv run --with pytest python -m pytest -q
```

另运行新题集校验与统计命令，并把输出写入交接说明。

## 交接重点

说明哪些标签经过人工阅读原文确认，哪些只是候选；列出仍有争议的等价来源和证据块，不能把
未人工确认的自动生成标签描述为金标准。
