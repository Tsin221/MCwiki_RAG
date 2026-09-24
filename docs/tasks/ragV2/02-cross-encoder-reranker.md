# 任务 02：Cross-Encoder 候选精排

建议分支：`codex/ragv2-reranker`

依赖：任务 01 已合并。

## 任务目标

在混合召回与证据组装之间加入可插拔的 Cross-Encoder。先保留更大的 RRF 候选池，再由真实
“问题、候选文本”配对模型重新排序，最后把 Top-K 交给现有 `adjacent_merge`。

## 参考资料中的技术

`E:\学习笔记\RAG\RAG基础检索技术提纯.md` 第 5.2 节使用
`BAAI/bge-reranker-base` 对小候选集精排，并明确指出：Cross-Encoder 只能调整已召回候选的
顺序，不能找回召回阶段遗漏的证据。

## MCwiki 工程决策

- 定义 `NoopReranker` 和 `CrossEncoderReranker`，调用方只依赖协议；
- 默认模型为 `BAAI/bge-reranker-base`，模型名必须可配置；
- 默认候选池为 20，精排后保留 8；两个数量分别配置，不能复用一个含义不清的 `limit`；
- 精排输入使用原始用户问题与候选 `title + text`；
- 结果继续携带原 `chunk_id`、来源、文档位置和两路召回名次；
- 精排分数单独保存，不能冒充 RRF 分数或概率；
- 模型未安装、加载失败或推理异常时回退到原 RRF 顺序，并记录日志；
- 模型权重在应用启动时加载一次，禁止每个请求重新加载。

## 实现范围

1. 新建 Reranker 协议与结果排序实现；
2. 接入可实际运行的 Cross-Encoder 依赖和模型配置；
3. 扩展候选结构以区分 `rrf_score` 与 `reranker_score`；
4. 修改回答链路为“召回候选池 → 精排 → Top-K → 证据组装”；
5. 保留关闭开关，关闭时输出与任务 01 基线一致；
6. readiness 能反映配置为 Cross-Encoder 时模型是否加载成功。

建议公共契约：

```python
class Reranker(Protocol):
    def rerank(
        self,
        question: str,
        candidates: Sequence[HybridResult],
        *,
        limit: int,
    ) -> list[HybridResult]: ...
```

若需要新增包装类型，必须保持 `chunk_id` 和原始检索元数据可追踪。

## 文件边界

预计新增：

- `code/rag_reranker.py`；
- `code/tests/test_reranker.py`。

预计修改：

- `code/pyproject.toml` 与 `code/uv.lock`；
- `code/rag_retrieval/hybrid.py`；
- `code/rag_api.py`；
- `code/rag_settings.py`；
- 对应现有测试。

不要修改 chunk、索引内容、RRF 公式或证据相邻合并算法。

## 验收标准

- [ ] 使用假模型时，可证明精排会改变候选顺序并严格截取 Top-K；
- [ ] 重复 chunk 不会因精排重新出现；
- [ ] `NoopReranker` 完整保持输入顺序；
- [ ] Cross-Encoder 异常时自动回退 RRF 顺序，回答接口仍可用；
- [ ] 配置数字必须为正，且 `rerank_limit <= candidate_limit`；
- [ ] 本地启用真实模型时能完成一次中文问题精排，并在交接中记录模型和耗时；
- [ ] 完整后端测试通过。

## 针对性验证

```powershell
uv run --with pytest python -m pytest -q tests/test_reranker.py tests/test_hybrid.py tests/test_api.py tests/test_settings.py
```

真实模型烟雾测试单独执行，不得让自动化测试下载模型。

## 交付给任务 03 的接口

交付统一的“候选召回并精排”入口。纠正检索获得第二轮候选后，也必须通过同一个 Reranker，
不能另写一套排序逻辑。

