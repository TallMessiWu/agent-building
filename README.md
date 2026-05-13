# 🤖 Agent 动手实践：从零构建生产级 AI Agent

> 四周时间，零框架手写 ReAct → MCP 工具治理 → 弹性容错 → LangGraph 状态机 → Multi-Agent 协作。
> **完整学习看板** → **[tallmessiwu.github.io/agent-building](https://tallmessiwu.github.io/agent-building/)**

## 快速开始

```bash
git clone --recurse-submodules https://github.com/TallMessiWu/agent-building.git
uv sync

# 日常练习（按时间线推进）
uv run python exercises/w1d1-function-calling/agent.py
uv run python exercises/w1d2-mcp-server/agent.py
uv run python exercises/w1d3-resilient-agent/agent.py
uv run python exercises/w2d1-langgraph/agent.py
uv run python exercises/w2d2-langgraph-hitl/agent.py
uv run python exercises/w2d3-autogen/agent.py

# 交互式 Notebook
uv run jupyter notebook exercises/w1d1-function-calling/agent.ipynb
```

## 目录结构

```
agent-building/
├── README.md
├── CLAUDE.md
├── Agent_4周学习计划.xlsx
├── docs/index.html                # → GitHub Pages 学习看板
│
├── exercises/                     # 每日编程练习（6 天）
│   ├── w1d1-function-calling/     # Day 1: Function Calling
│   ├── w1d2-mcp-server/           # Day 2: MCP 协议
│   ├── w1d3-resilient-agent/      # Day 3: 重试/熔断/降级
│   ├── w2d1-langgraph/            # Day 4: StateGraph + Checkpoint
│   ├── w2d2-langgraph-hitl/       # Day 5: HITL + Subgraph
│   └── w2d3-autogen/              # Day 6: AutoGen Multi-Agent
│
├── react-agent/                   # 📦 项目 1: 手写 ReAct Agent
├── research-assistant/            # 📦 项目 2: 长期记忆研究助手
└── multi-agent-collab/            # 📦 项目 3: Multi-Agent 协作写作
```

> 📅 完整学习时间线、41 道八股题库、必读清单、项目追踪、模拟面试反馈 → **[网站版学习看板](https://tallmessiwu.github.io/agent-building/)**

## 技术亮点

| 亮点 | 位置 | 说明 |
|------|------|------|
| **受限 eval 沙箱** | `w1d1/agent.py` | `__builtins__` 置空 + 白名单，面试高频考点 |
| **MCP 协议落地** | `w1d2/agent.py` | Anthropic 主导的 Agent 工具标准协议 |
| **Circuit Breaker** | `w1d3/agent.py` | 三态状态机熔断器，分布式系统经典模式 |
| **指数退避 + 抖动** | `w1d3/agent.py` | 避免雷鸣羊群效应，生产级重试 |
| **混沌工程** | `w1d3/agent.py` | 主动注入故障验证弹性 |
| **LangGraph StateGraph** | `w2d1/agent.py` | 图结构控制流 + SQLite Checkpoint 持久化 |
| **HITL 断点恢复** | `w2d2/agent.py` | 零侵入人工审批 + 子图横切关注点 |
| **AutoGen Multi-Agent** | `w2d3/agent.py` | LLM 动态选择发言人 + 三框架完整对比 |

## 技术栈

`Python 3.14` · `OpenAI SDK (DeepSeek API)` · `MCP Python SDK` · `LangGraph` · `AutoGen` · `Chroma` · `uv`
