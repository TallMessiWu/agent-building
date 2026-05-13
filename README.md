# 🤖 Agent 动手实践：从零构建生产级 AI Agent

> **一句话**: 四周时间，零框架，从 ReAct 循环 → MCP 工具治理 → 弹性容错 → LangGraph 状态机，
> 逐层构建一个能写进简历的生产级 Agent 系统。

---

## 架构演进

```
Week 1 Day 1          Week 1 Day 2            Week 1 Day 3          Week 2+ (WIP)
┌──────────────┐     ┌──────────────┐        ┌──────────────┐      ┌───────────────┐
│ 纯 Function  │ →  │  MCP 协议    │   →   │  弹性 Agent  │  →  │  LangGraph    │
│  Calling     │     │  标准化工具   │        │  三层防御     │      │  状态图编排    │
└──────────────┘     └──────────────┘        └──────────────┘      └───────────────┘
 工具硬编码在        MCP Server 暴露        重试/熔断/降级         图结构管理
 Agent 代码中        工具，Client 发现       让 Agent 自愈         多分支决策
```

## 目录结构

```
agent-building/
├── README.md                          # ← 你在这里
├── CLAUDE.md                          # 开发环境 & 进度追踪
├── Agent_4周学习计划.xlsx              # 结构化学习路线
│
├── exercises/                         # 日常编程练习（按时间线）
│   ├── w1d1-function-calling/         # Day 1: 纯 Function Calling
│   │   ├── agent.py                   #   完整脚本（6 测试用例）
│   │   └── agent.ipynb                #   交互式 Notebook（13 步骤）
│   ├── w1d2-mcp-server/               # Day 2: MCP 协议改造
│   │   ├── agent.py                   #   MCP Server + Client（6 测试用例）
│   │   └── agent.ipynb                #   交互式 Notebook（13 步骤）
│   ├── w1d3-resilient-agent/          # Day 3: 弹性 Agent
│   │   ├── agent.py                   #   重试/熔断/降级 + 混沌模式
│   │   └── agent.ipynb                #   交互式 Notebook（45 cells）
│   └── w2d1-langgraph/                # Day 4: LangGraph StateGraph ✅
│   │   ├── agent.py                   #   StateGraph + SQLite Checkpoint（4 测试用例）
│   │   └── agent.ipynb                #   交互式 Notebook（10 cells）
│   └── w2d2-langgraph-hitl/           # Day 5: LangGraph HITL + Subgraph ✅
│   │   ├── agent.py                   #   interrupt_before 断点 + 恢复 + 子图（3 模式）
│   │   └── agent.ipynb                #   交互式 Notebook（11 cells）
│   └── w2d3-autogen/                  # Day 6: AutoGen Multi-Agent ✅
│       ├── agent.py                   #   SelectorGroupChat + 工具调用 + 框架对比（4 模式）
│       └── agent.ipynb                #   交互式 Notebook（10 cells）
│
├── react-agent/                       # 📦 子模块 → TallMessiWu/react-agent
│   ├── agent.py                       #   完整 ReAct Agent：GitHub API + 弹性三层 + 分析报告
│   └── agent.ipynb                    #   交互式 Notebook（8 步骤拆解）
├── research-assistant/                # ★ 项目 2: 长期记忆研究助手
└── multi-agent-collab/                # ★ 项目 3: Multi-Agent 协作写作
```

## 学习时间线

| 日期 | 主题 | 文件 | 八股题 | 核心交付 |
|------|------|------|--------|---------|
| 05-06 | **Function Calling** | `exercises/w1d1-function-calling/` | #1-11 | ReAct 循环、eval 沙箱、tool_choice |
| 05-07 | **MCP 协议改造** | `exercises/w1d2-mcp-server/` | #12-13 | Server/Client 架构、list_tools/call_tool、MCP→OpenAI 格式转换 |
| 05-08 | **弹性 Agent** | `exercises/w1d3-resilient-agent/` | #14 | 指数退避重试、Circuit Breaker 三态熔断、结果校验、混沌工程 |
| 05-09 | ★ **项目 1** ✅ | `react-agent/` | #5,7,8 | 手写 ReAct Agent，完成 GitHub repo 分析报告 |
| 05-10 | ★ **项目 1 完成** ✅ | 📦 子模块 `react-agent/` | #14, 总复习1-16 | GitHub 仓库独立发布 + 简历段落 + 周复盘 |
| 05-11 | ✅ **LangGraph** | `exercises/w2d1-langgraph/` | #24,27 | StateGraph、Checkpoint 持久化、reasoning_content 兼容 |
| 05-12 | ✅ **HITL + Subgraph** | `exercises/w2d2-langgraph-hitl/` | #26 | interrupt_before 断点恢复、人工审批、工具参数校验子图 |
| 05-13 | ✅ **AutoGen** | `exercises/w2d3-autogen/` | #25 | SelectorGroupChat 多 Agent 对话、LLM 选择发言人、三框架对比 |

## 快速开始

```bash
# 克隆（含子模块）
git clone --recurse-submodules https://github.com/TallMessiWu/agent-building.git

# 安装依赖
uv sync

# 日常练习
uv run python exercises/w1d1-function-calling/agent.py
uv run python exercises/w1d2-mcp-server/agent.py
uv run python exercises/w1d2-mcp-server/agent.py serve    # MCP Server 独立模式
uv run python exercises/w1d3-resilient-agent/agent.py
uv run python exercises/w1d3-resilient-agent/agent.py --chaos  # 混沌模式 🌪️
uv run python exercises/w2d1-langgraph/agent.py               # LangGraph StateGraph
uv run python exercises/w2d1-langgraph/agent.py --memory      # + Checkpoint 记忆演示
uv run python exercises/w2d2-langgraph-hitl/agent.py           # LangGraph HITL（自动模式）
uv run python exercises/w2d2-langgraph-hitl/agent.py --hitl    # HITL 模式（交互式审批）
uv run python exercises/w2d2-langgraph-hitl/agent.py --subgraph # 子图示例
uv run python exercises/w2d3-autogen/agent.py                  # AutoGen 基础两 Agent 对话
uv run python exercises/w2d3-autogen/agent.py --group          # SelectorGroupChat（Researcher+Writer+Critic）
uv run python exercises/w2d3-autogen/agent.py --tool           # 多 Agent + 工具调用
uv run python exercises/w2d3-autogen/agent.py --compare        # 三框架完整对比

# 简历项目
uv run python react-agent/agent.py                           # ★ ReAct Agent — GitHub 仓库分析

# 交互式学习
uv run jupyter notebook exercises/w1d1-function-calling/agent.ipynb
uv run jupyter notebook exercises/w1d2-mcp-server/agent.ipynb
uv run jupyter notebook exercises/w1d3-resilient-agent/agent.ipynb
uv run jupyter notebook react-agent/agent.ipynb              # ReAct Agent 交互式讲解
uv run jupyter notebook exercises/w2d1-langgraph/agent.ipynb # LangGraph 交互式讲解
uv run jupyter notebook exercises/w2d2-langgraph-hitl/agent.ipynb # HITL + Subgraph 交互式讲解
uv run jupyter notebook exercises/w2d3-autogen/agent.ipynb      # AutoGen Multi-Agent 交互式讲解
```

## 技术亮点（面试可说）

| 亮点 | 位置 | 为什么值钱 |
|------|------|-----------|
| **受限 eval 沙箱** | `w1d1/agent.py` — calculate() | `__builtins__` 置空 + 白名单，面试高频考点 |
| **MCP 协议落地** | `w1d2/agent.py` — Server 定义 | Anthropic 主导的开放协议，Agent 工具标准 |
| **Circuit Breaker** | `w1d3/agent.py` — CircuitBreaker 类 | 分布式系统经典模式，三态状态机实现 |
| **指数退避 + 抖动** | `w1d3/agent.py` — resilient_execute_tool() | 避免雷鸣羊群效应，生产级重试必备 |
| **混沌工程** | `w1d3/agent.py` — chaos_inject() | Netflix Chaos Monkey 思路，主动注入故障验证弹性 |
| **结果校验层** | `w1d3/agent.py` — validate_*_result() | 防止垃圾数据污染 LLM 上下文 |
| **GitHub ReAct Agent** | `react-agent/agent.py` — 完整 ReAct 循环 | 真实 API 集成 + 弹性三层 + 结构化报告生成 |
| **LangGraph StateGraph** | `w2d1/agent.py` — build_graph() | 图结构控制流、Checkpoint 持久化、多用户会话隔离 |
| **HITL 断点恢复** | `w2d2/agent.py` — interrupt_before | 零侵入实现人工审批，app.invoke(None) 从断点继续执行 |
| **Subgraph 子图** | `w2d2/agent.py` — build_validation_subgraph() | 独立 StateGraph 封装横切关注点，支持独立暂停/恢复 |
| **AutoGen SelectorGroupChat** | `w2d3/agent.py` — SelectorGroupChat | 对话抽象下的 Multi-Agent 协作，LLM 动态选择发言人，涌现式合作 |
| **三框架对比** | `w2d3/agent.py` — show_framework_comparison() | LangGraph vs AutoGen vs CrewAI 完整维度对比表（面试题 25） |

## 技术栈

`Python 3.14` · `OpenAI SDK (DeepSeek API)` · `MCP Python SDK` · `LangGraph` · `langchain-core` · `asyncio` · `Jupyter Notebook` · `uv`
