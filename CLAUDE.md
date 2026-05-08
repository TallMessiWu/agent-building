# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Agent 动手实践项目，主题为纯手写 Function Calling Agent（零框架），使用 DeepSeek API（OpenAI 兼容接口）实现 ReAct 模式下的工具调用循环。

整个仓库分为两部分：
- **`exercises/`** — 日常编程练习，按 `w<周>d<天>-<主题>` 组织
- **根目录 `*-agent/`** — 三个简历项目（react-agent / research-assistant / multi-agent-collab）

## 开发环境

- **Python 版本**: 3.14
- **包管理器**: `uv` — 所有依赖安装和脚本运行均通过 `uv run` 或 `uv add`
- **API**: DeepSeek API（OpenAI 兼容接口），base_url: `https://api.deepseek.com`
- **API Key**: 通过 `.env` 文件中的 `API_KEY` 环境变量加载（python-dotenv）

## 常用命令

```bash
# 安装依赖
uv sync

# 添加新依赖
uv add <package>

# 运行练习脚本
uv run python exercises/w1d1-function-calling/agent.py
uv run python exercises/w1d2-mcp-server/agent.py
uv run python exercises/w1d2-mcp-server/agent.py serve    # MCP Server 独立模式
uv run python exercises/w1d3-resilient-agent/agent.py
uv run python exercises/w1d3-resilient-agent/agent.py --chaos  # 混沌模式

# 启动 Jupyter Notebook
uv run jupyter notebook exercises/w1d1-function-calling/agent.ipynb
uv run jupyter notebook exercises/w1d2-mcp-server/agent.ipynb
uv run jupyter notebook exercises/w1d3-resilient-agent/agent.ipynb
```

## 提交规范

提交时**必须**调用 `gitmoji-commit` skill：`Skill(skill: "gitmoji-commit")`。该 skill 会自动分析暂存区变更、生成符合 Gitmoji 规范的中文提交信息并执行本地提交（不推送）。

## ⚠️ 每日提交前必做

**每天学习结束、提交代码前，更新 `README.md`**（进度以 README 为准，CLAUDE.md 不重复维护进度表）：

- 在"学习时间线"表格中标记当天完成状态
- 如有新文件，在"目录结构"中体现

> Claude 启动时自动读取 README.md，立刻知道当前进度和昨天产出的文件位置。

---

## 架构

整个项目遵循 **ReAct (Reasoning + Acting)** 模式，四个层级加一个入口：

1. **Tool Schema 层** — `WEATHER_TOOL` / `CALCULATOR_TOOL` 用 JSON Schema 定义工具签名（name + description + parameters + required）。`TOOLS` 列表是发送给 LLM 的工具注册表。

2. **工具执行层** — `get_weather(city, unit)` 和 `calculate(expression)` 是真正干活的函数。`TOOL_EXECUTORS` 字典做函数名→实现的映射，`execute_tool(name, args)` 是统一调度入口。

3. **Agent 循环** (`run_agent`) — 核心。消息历史初始化 `[system, user]` → 调 LLM → 若返回 `tool_calls` 则执行工具并追加结果到 history → 循环；若返回 `content` 则终止。关键参数: `tool_choice="auto"`, `temperature=0.0`, `max_turns=10`。

4. **安全沙箱** — `calculate` 中的 `eval` 将 `__builtins__` 置空，仅暴露 `math` 模块公开函数。这是面试常考点。

## 文件说明

```
exercises/
├── w1d1-function-calling/             # Day 1: 纯 Function Calling
│   ├── agent.py                       #   完整脚本，包含 6 个测试用例
│   └── agent.ipynb                    #   交互式 Notebook，分步骤讲解
├── w1d2-mcp-server/                   # Day 2: MCP 协议改造
│   ├── agent.py                       #   MCP Server + Client，6 个测试用例
│   └── agent.ipynb                    #   交互式 Notebook
└── w1d3-resilient-agent/              # Day 3: 弹性 Agent
    ├── agent.py                       #   重试/熔断/降级 + 混沌模式
    └── agent.ipynb                    #   交互式 Notebook（45 cells）

react-agent/                           # ★ 项目 1: 手写 ReAct Agent（05-09~10）
research-assistant/                    # ★ 项目 2: 长期记忆研究助手（05-16~17）
multi-agent-collab/                    # ★ 项目 3: Multi-Agent 协作写作（05-23~24）
```
