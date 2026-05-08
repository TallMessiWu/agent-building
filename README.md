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
│   └── w1d3-resilient-agent/          # Day 3: 弹性 Agent
│       ├── agent.py                   #   重试/熔断/降级 + 混沌模式
│       └── agent.ipynb                #   交互式 Notebook（45 cells）
│
├── react-agent/                       # ★ 项目 1: 手写 ReAct Agent
├── research-assistant/                # ★ 项目 2: 长期记忆研究助手
└── multi-agent-collab/                # ★ 项目 3: Multi-Agent 协作写作
```

## 学习时间线

| 日期 | 主题 | 文件 | 八股题 | 核心交付 |
|------|------|------|--------|---------|
| 05-06 | **Function Calling** | `exercises/w1d1-function-calling/` | #1-11 | ReAct 循环、eval 沙箱、tool_choice |
| 05-07 | **MCP 协议改造** | `exercises/w1d2-mcp-server/` | #12-13 | Server/Client 架构、list_tools/call_tool、MCP→OpenAI 格式转换 |
| 05-08 | **弹性 Agent** | `exercises/w1d3-resilient-agent/` | #14 | 指数退避重试、Circuit Breaker 三态熔断、结果校验、混沌工程 |
| 05-09 | ★ **项目 1** | `react-agent/` | #5,7,8 | 手写 ReAct Agent，完成 GitHub repo 分析报告 |
| 05-11+ | **LangGraph** (WIP) | `exercises/w2d1-langgraph/` | #24,27 | StateGraph、Checkpoint 持久化 |

## 快速开始

```bash
# 安装依赖
uv sync

# 日常练习
uv run python exercises/w1d1-function-calling/agent.py
uv run python exercises/w1d2-mcp-server/agent.py
uv run python exercises/w1d2-mcp-server/agent.py serve    # MCP Server 独立模式
uv run python exercises/w1d3-resilient-agent/agent.py
uv run python exercises/w1d3-resilient-agent/agent.py --chaos  # 混沌模式 🌪️

# 交互式学习
uv run jupyter notebook exercises/w1d1-function-calling/agent.ipynb
uv run jupyter notebook exercises/w1d2-mcp-server/agent.ipynb
uv run jupyter notebook exercises/w1d3-resilient-agent/agent.ipynb
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

## 技术栈

`Python 3.14` · `OpenAI SDK (DeepSeek API)` · `MCP Python SDK` · `asyncio` · `Jupyter Notebook` · `uv`
