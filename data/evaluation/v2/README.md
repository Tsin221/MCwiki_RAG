# MCwiki RAG 评测集 v2

`questions.json` 是独立于固定 v1 回归集的诊断题集。它不替换
`data/evaluation/retrieval_questions.json` 或 `regression_suite.json`。

## 样本结构

每条样本都包含以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 稳定且全局唯一的题目 ID |
| `question` | 面向用户的中文问题 |
| `kind` | `answerable` 或 `unanswerable` |
| `category` | 唯一主类别，统计时只计一次 |
| `tags` | 可选辅助标签，不参与主类别配额 |
| `facets` | 问题要覆盖的子主题 |
| `expected_answer_points` | 可独立判定的必答要点，每项具有稳定 ID 和描述 |
| `preferred_sources` | 最直接、最适合作为正文引用的来源 |
| `acceptable_sources` | 所有可接受的权威来源，必须包含首选来源 |
| `relevant_chunk_ids` | 当前 `chunks.jsonl` 中支持答案的稳定块 ID |
| `edition` | `all`、`java` 或 `bedrock` |
| `version_scope` | `current`、具体版本、`future` 或 `custom` 等范围说明 |
| `reason` | 仅无答案题使用的拒答原因枚举 |
| `annotation_status` | `confirmed` 或 `candidate` |

无答案题不得声明答案要点、来源或证据块。`reason` 只能是：

- `out_of_domain`：不属于 Minecraft Wiki 知识问答；
- `future_information`：尚未发生或未公布的未来信息；
- `private_realtime_state`：私人账户、存档或实时服务器状态；
- `knowledge_base_gap`：问题可能有答案，但当前知识库没有所需私有或自定义资料。

## 来源与标注口径

- 来源必须使用 `https://zh.minecraft.wiki/w/...`。
- `preferred_sources` 表示回答优先引用的直接正文页；`acceptable_sources` 表示不会导致
  引用失败的等价权威页。
- `java-1-21-equivalent-sources` 以 `Java版1.21` 为首选正文，同时接受 `1.21`
  消歧义页，演示等价来源的表达方式。
- `relevant_chunk_ids` 表示已在当前语料中定位到的候选证据，不代表块内每句话都足以独立
  覆盖全部答案要点。跨块题必须列出多个候选块。
- `confirmed` 只用于已有回归报告明确人工复核的 Java 1.21 等价来源，以及按定义即可
  确认的无答案边界题。其余 41 条通过了块内容抽样和结构校验，但尚未逐条对照在线原文，
  因而标为 `candidate`，不能当作最终金标准。

## 校验与统计

在 `code` 目录运行：

```powershell
uv run python -m rag_evaluation_schema validate
uv run python -m rag_evaluation_schema stats
```

校验会检查结构、ID 唯一性、来源域名、答案要点、无答案约束、块 ID 存在性、40～60 题
总量以及各主类别配额。两个命令均为只读操作，不会调用 DeepSeek。
