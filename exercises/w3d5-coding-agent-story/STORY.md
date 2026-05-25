# SGLang Coding Agent 面试故事大纲

## 1. Coding Agent 设计原则

### Agent-Computer Interface
- 来源锚点: SWE-agent ACI
- 面试表达: Coding agent 的能力不只来自模型本身，还来自它能看到什么上下文、能执行哪些命令、观察结果如何回流，以及错误信息是否足够可行动。
- SGLang 映射: 在 SGLang 量化适配中，我会让 agent 先用搜索理解现有 FP8/W8A8 抽象，再把改动限制在清晰的接口边界内。

### Long-horizon workspace
- 来源锚点: Devin-style workflow
- 面试表达: 长任务 agent 需要持续维护计划、代码状态、测试反馈和待办，而不是每轮重新开始。
- SGLang 映射: MXFP4/MXFP8 适配横跨模型加载、kernel contract、离线/在线量化和测试，我把任务拆成可验证阶段，让 agent 每步产出都能被 review。

### Tests as environment feedback
- 来源锚点: Coding-agent eval loop
- 面试表达: coding agent 的关键闭环是执行测试、读取失败、修复，再运行更小范围的验证。
- SGLang 映射: 我不会只看 agent 写出的代码是否像样，而是要求它补单测/小规模 e2e，并用 trace 或 CI 结果定位 scale、shape、dtype 这类边界问题。

### Human-led architecture
- 来源锚点: Practical agent use
- 面试表达: agent 擅长探索和局部实现，但架构抽象、性能权衡、跨模块协作仍要人主导。
- SGLang 映射: 我把 agent 当成高效初级工程师：给清晰 spec、让它探索和实现，但由我决定抽象边界、性能方向和最终 merge 标准。

## 2. STAR 故事主线

### Situation
- SGLang 量化适配涉及模型加载、scale 布局、NPU kernel contract 和多模型路径。
- 任务不是单点 bugfix，而是需要在复杂代码库中稳定扩展量化能力。

### Task
- 目标是把 MXFP4/MXFP8 相关能力接入既有架构，并尽量复用已有 FP8/W8A8 模式。
- 同时要保证可测试、可 review，不能为了快而破坏现有推理路径。

### Action
- 我先写清楚 spec：目标、约束、参考 PR、不得改动的边界和验收命令。
- 让 agent 探索代码库并解释现有抽象，我通过它的 trace 判断它是否真的理解。
- 实现时拆成小步：入口、参数转换、kernel contract、测试和文档分别推进。
- 我持续 review 架构和性能方向，让 agent 做局部实现、测试补全和文档同步。

### Result
- agent 显著提升了复杂仓库探索、重复模式迁移和测试生成效率。
- 我也明确了边界：架构决策、性能瓶颈定位、跨模块协调仍需要工程师主导。
- 这段经历让我能把 coding agent 讲成真实工程 workflow，而不是泛泛说提升效率。

## 3. 项目证据

### 清晰任务规约
- 证据: 先写 task spec：目标是把 MXFP4/MXFP8 适配到 SGLang 的不同模型路径，约束是不破坏既有 W8A8/FP8 路径，并保留在线/离线量化可扩展性。
- 价值: 这能降低 agent 过度发散和误改公共抽象的概率。

### 代码库探索
- 证据: 让 agent 用搜索和文件阅读梳理量化模块、kernel contract、dense 与 MoE 路径的差异，再输出修改计划。
- 价值: 真实大型仓库里，先理解现有抽象比直接写代码更重要。

### 逐步 review
- 证据: 实现过程中按阶段 review：模型侧入口、scale 处理、kernel 调用参数、测试覆盖分别检查，而不是等最后一次性验收。
- 价值: 早期方向错了会让后续代码全部返工，阶段性 review 更稳。

### 跨场景复用
- 证据: 做完一个 diffusion/MXFP 路径后，把同一套适配思路推广到 Dense LLM 和 MoE LLM。
- 价值: 这体现 agent 在模式复用、重复工程和测试生成上的效率优势。

## 4. 失败模式与控制

### 过度抽象
- 例子: agent 为了显得优雅，把简单 dtype/scale 分支包装成多层新抽象。
- 控制: 要求它用最小改动复用现有模式，并说明为什么需要新增抽象。

### 边界 case 漏测
- 例子: W4A4 与 W4A8 的 scale 处理不同，第一版容易混在一起。
- 控制: 把 shape、dtype、scale layout 写进测试矩阵，并要求失败样例可复现。

### 性能直觉不足
- 例子: 逻辑正确但 kernel 参数组织导致额外转换或内存开销。
- 控制: 用 profiling 或小规模 benchmark 定位瓶颈，再把具体问题交给 agent 修复。

### 权限与副作用失控
- 例子: coding agent 自动执行大范围改动或触碰无关文件。
- 控制: 限定工作目录、明确禁止改动路径，提交前只 stage 目标文件。

## 5. Agent 安全速记

- Prompt injection: 代码库、issue、日志里的文本都可能诱导 agent 忽略原始指令；需要把外部文本当作数据而不是指令。
- Excessive agency: 自动提交、发布、删除分支、改 CI 配置这类动作必须有人确认。
- Sensitive information disclosure: `.env`、token、内部 benchmark 结果默认不能进入 prompt、trace 或 commit。
- Supply-chain risk: agent 建议新增依赖时要检查维护状态、许可证、安装来源和是否真的必要。
- Insecure output handling: agent 生成的 shell、SQL、Python 片段要审查执行边界，尤其是删除、网络和凭据操作。

## 6. 90 秒背诵版

我在 SGLang 量化适配里使用 coding agent 的方式，不是让它替我做架构决策，而是把它放在一个清晰的工程闭环里。第一步我会写 task spec，明确目标、约束、参考实现和不能碰的边界；第二步让 agent 搜索代码库，解释现有量化抽象和 kernel contract，我通过它的 trace 判断它是否真的理解；第三步按模型入口、scale 处理、kernel 调用、测试和文档拆成小步实现，每一步我都 review。这样做的收益是复杂仓库探索、重复模式迁移和测试生成会快很多。但我也遇到过失败模式，比如过度抽象、边界 case 漏测、性能直觉不足，所以我会用最小改动约束、测试矩阵和 profiling 反馈来控制它。我的结论是：coding agent 不能替代工程师判断，但在 well-defined、可验证的工程任务上，可以非常明显地放大工程师产出。

## 7. 追问演练

- 如果面试官问：你怎么保证 coding agent 不乱改？先回答 spec、边界、stage 范围、review 和测试。
- 如果面试官追问：agent 犯过什么错？用过度抽象、W4A4/W4A8 scale 混淆、性能直觉不足三个例子。
- 如果面试官问：你和普通 Copilot 用户有什么不同？强调任务分解、trace review、测试闭环和架构边界由人主导。
- 如果面试官问：为什么这和 SWE-agent/Devin 有关？回答 ACI、长任务 workspace、环境反馈和人类验收。
