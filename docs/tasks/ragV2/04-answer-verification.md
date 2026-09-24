# 任务 04：Self-RAG 式答案证据核验

建议分支：`codex/ragv2-answer-verification`

依赖：任务 03 已合并。

## 任务目标

在把答案发送给用户之前，检查答案中的关键事实是否受到所引证据支持、引用编号是否合法、回答
是否覆盖用户问题。第一次失败时允许基于同一份证据重写一次；重写后必须再次核验，仍失败则
返回资料不足，而不是发送未经支持的版本。

## 参考资料中的技术

`E:\学习笔记\RAG\高级RAG技术提纯.md` 第 5 节把判断拆为相关性 `IsRel`、证据支持
`IsSup` 和答案效用 `IsUse`，并指出参考脚本重写后没有复核是缺陷。

`E:\学习笔记\RAG\RAG评估技术提纯.md` 建议把答案拆成单一事实声明，再逐条判断能否由实际
上下文推出。本任务采用这一检查粒度，但不建设新的评测平台。

## MCwiki 工程决策

- 核验对象是完整草稿，因此不能把未经核验的 token 直接流给浏览器；
- 先在服务端收集草稿，核验通过后再通过现有 SSE `delta` 事件发送最终文本；
- 先做确定性引用检查，再做模型支持度与完整性检查；
- 核验模型返回原子声明、对应引用、是否支持及问题列表；
- 最多生成两版答案：初稿一次、失败后重写一次；第二版必须再次核验；
- 重写只能使用同一份最终证据，不能在本任务内触发第三轮检索；
- 核验器故障时的默认行为通过配置明确。首版建议保留基线答案并记录 `verificationUnavailable`，
  不伪装为“已验证”；
- 不向普通用户暴露模型思维过程、内部评分或提示词。

## 实现范围

1. 新建确定性引用解析与合法性检查；
2. 新建答案核验协议、结构化结果和 DeepSeek 实现；
3. 为答案客户端增加可收集完整草稿的调用方式；
4. 实现“生成 → 核验 → 可选重写 → 再核验”的有界流程；
5. 保持 SSE 事件名称兼容，并在 `done` 中增加可选的验证状态；
6. 建立无引用、越界引用、事实不受支持、遗漏子问题、重写成功和重写仍失败测试。

建议数据结构：

```python
@dataclass(frozen=True, slots=True)
class ClaimCheck:
    claim: str
    citation_ids: tuple[int, ...]
    supported: bool
    reason: str

@dataclass(frozen=True, slots=True)
class AnswerVerification:
    supported: bool
    useful: bool
    claims: tuple[ClaimCheck, ...]
    issues: tuple[str, ...]
```

## 文件边界

预计新增：

- `code/rag_verification.py`；
- `code/tests/test_verification.py`。

预计修改：

- `code/rag_answer.py`；
- `code/rag_api.py`；
- `code/rag_settings.py`；
- `code/tests/test_answer.py`；
- `code/tests/test_api.py`。

不要改变检索、重排、chunk 或证据编号规则。

## 验收标准

- [ ] 引用 `[n]` 必须指向实际证据编号，越界编号不会进入最终答案；
- [ ] 每个模型识别出的关键事实都有引用并被对应证据支持；
- [ ] 首稿失败时最多重写一次，重写后必定再次核验；
- [ ] 第二版仍失败时返回明确的资料不足状态；
- [ ] 浏览器不会收到后来被判失败的初稿内容；
- [ ] 核验关闭时回答行为与任务 03 基线一致；
- [ ] 核验器异常路径有测试并带明确状态；
- [ ] 完整后端测试通过。

## 针对性验证

```powershell
uv run --with pytest python -m pytest -q tests/test_verification.py tests/test_answer.py tests/test_api.py
```

测试必须确认“第二版复核”确实发生，并限制答案模型最多调用两次、核验模型最多调用两次。

## 交付给任务 05 的接口

交付一个只接收“原始用户问题 + 最终证据”并返回可发送答案的统一回答流程。Adaptive Router
只负责选择取证路径，不得复制核验逻辑。

