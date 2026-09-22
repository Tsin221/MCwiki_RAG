# Three.js 问答前端实施计划

状态：问答界面和当前 Shader 背景已实施；原计划中的浮空岛、雾效与粒子未接入

更新时间：2026-07-30

## 1. 实施原则

先完成可靠的二维流式问答，再加入三维背景。任何 Three.js 问题都不能阻塞
`POST /answers`、SSE 流式文本、来源卡片或匿名 Cookie。

第一版建议使用：

- `three`：三维渲染基础；
- `@react-three/fiber`：在 React 中管理 Three.js 场景；
- `@react-three/drei`：只选用确实需要的辅助组件；
- React + TypeScript + Vite：页面和构建基础。

不在文档阶段锁定具体依赖版本。实施时根据当时官方文档选择兼容版本并提交锁文件。

## 2. 建议目录

```text
frontend/src/
├─ App.tsx
├─ components/
│  ├─ chat/
│  │  ├─ ChatMessage.tsx
│  │  ├─ QuestionComposer.tsx
│  │  └─ SourceCard.tsx
│  └─ scene/
│     ├─ AmbientScene.tsx
│     ├─ FloatingIsland.tsx
│     ├─ SceneEffects.tsx
│     └─ SceneFallback.tsx
├─ hooks/
│  ├─ useAnswerStream.ts
│  ├─ useReducedMotion.ts
│  └─ useSceneQuality.ts
├─ lib/
│  └─ api.ts
├─ types/
│  ├─ api.ts
│  └─ scene.ts
└─ styles/
   ├─ app.css
   └─ tokens.css
```

三维组件不能导入 API Key、DeepSeek 配置或检索实现。它们只接收有限的页面状态，
例如 `idle | retrieving | generating | complete | error`。

## 3. 实施顺序

问答骨架、静态视觉、当前 Shader 背景和状态联动已完成。实际文件结构与最初建议略有差异：场景实现拆分为
`AmbientScene`、`SceneLayer` 和 `SceneFallback`。`FloatingIsland` 与
`SceneParticles` 已编写，但目前未接入 `AmbientScene`。

### 阶段 A：问答骨架

- 初始化 React + TypeScript + Vite；
- 实现正常 DOM 布局；
- 完成输入、发送、SSE 流式回答、引用和来源；
- 完成错误、中断和证据不足状态；
- 验证匿名 Cookie 请求行为。

验收：关闭 JavaScript 三维模块或不支持 WebGL 时，完整问答流程仍可使用。

### 阶段 B：静态视觉基线

- 建立颜色、间距、字体和层级变量；
- 实现深蓝黑渐变背景和内容面板；
- 完成桌面端、平板和 360px 移动端布局；
- 创建 `SceneFallback`，作为 WebGL 不可用时的固定背景。

验收：不加载 Three.js 时页面已经具备完整、可读的视觉基线。

### 阶段 C：Three.js 场景

- 异步加载 Canvas；
- 实现当前使用的 `SunsetShaderPlane` 动态背景；浮空方块岛与粒子组件暂未接入；
- 设置桌面与移动端质量等级；
- 保证 Canvas 不参与交互和页面语义；
- 处理初始化失败、页面隐藏和组件卸载。

验收：三维场景不会遮挡 UI，不影响首个问题的提交。

### 阶段 D：状态联动

- 将问答状态映射为有限的场景状态；
- 实现检索、生成、完成和错误的轻量动效；
- 确保 DOM 文字仍是主要状态反馈；
- 为 `prefers-reduced-motion` 提供无动画路径。

验收：快速切换状态、请求失败和流中断时没有动画残留或资源泄漏。

### 阶段 E：验证与收尾

- 运行前端组件测试和生产构建；
- 在真实浏览器完成至少 3 个端到端问题；
- 检查 360px、常见桌面分辨率和高 DPI 屏幕；
- 检查键盘导航、焦点、颜色对比和减少动态效果；
- 记录桌面端与移动端帧率；
- 检查控制台、网络请求和 Three.js 资源释放。

验收：满足设计文档中的 12 项完成标准。

## 4. 测试策略

自动化测试不验证 WebGL 像素结果，重点验证产品契约：

- 场景模块加载失败时渲染静态背景；
- 减少动态效果时不启用视差和持续动画；
- 问答状态正确映射到场景状态；
- 场景不会改变问题提交与 SSE 数据处理；
- 三维模块未加载时输入框仍可立即使用；
- 页面卸载时注销监听器。

真实浏览器检查：

- WebGL 正常路径；
- WebGL 初始化失败路径；
- 360px 移动端降级路径；
- `prefers-reduced-motion` 路径；
- 请求进行中、完成、证据不足、错误和中断；
- 页面切到后台再返回；
- 连续提出多个问题后无明显内存持续增长。

## 5. 工期

在原有问答 Web MVP 基础上，Three.js 第一版预计增加 1～2 个工作日：

| 工作 | 预估 |
|---|---:|
| DeepSeek、SSE、匿名 Cookie 后端 | 1～1.5 天 |
| React 流式问答与引用界面 | 1～1.5 天 |
| Three.js 场景与响应式降级 | 1～1.5 天 |
| 联调、浏览器验收和基础评测 | 0.5～1.5 天 |
| 总计 | 约 4～6 个工作日 |

如果后续要求定制建模、复杂 Shader、大量动画或像参考图一样进行完整视觉导演，需另行
预留 2～4 个工作日，不计入本次 MVP。

## 6. 主要风险与控制

| 风险 | 控制措施 |
|---|---|
| Three.js 包体拖慢首屏 | 异步分包，先显示 DOM 和静态背景 |
| 低端手机掉帧 | 降低 DPR、粒子和几何复杂度 |
| 动画干扰阅读 | 低速低振幅，支持减少动态效果 |
| Canvas 遮挡控件 | 固定背景层并使用 `pointer-events: none` |
| 场景错误拖垮页面 | 错误边界和静态 CSS 降级 |
| 状态联动造成耦合 | 只传递有限场景状态枚举 |
| 第三方素材存在授权问题 | 第一版使用程序化几何体和自有材质 |
| 开发时间失控 | 不做三维文字、导航、物理和重型后处理 |

## 7. 实际实施边界

本轮已经创建前端工程、安装并锁定 Three.js 相关依赖、接入 DeepSeek 流式问答，并
完成前后端联调。仍未实施的范围保持不变：

- 三维文字、三维导航、物理模拟和复杂相机控制；
- 大型 GLTF 场景、重型后处理和高成本自定义 Shader；
- 依赖三维交互才能完成的产品功能；
- 登录、会话持久化、联网搜索和 Agentic RAG。

完整回答质量评测仍由 `docs/specs/qa-web-mvp.md` 的任务 7 跟踪，不属于 Three.js
视觉实现的未完成项。
