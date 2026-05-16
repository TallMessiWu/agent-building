# 项目 2 · 长期记忆研究助手

> LangGraph + Chroma 实现的研究助手 Agent，解决"对话结束即遗忘"痛点：跨会话沉淀研究片段，下一次新会话能召回过往。

## 1. 设计目标

| 维度 | 项目 1 (react-agent) | 项目 2 (本项目) |
| --- | --- | --- |
| 任务形态 | 单轮分析 | 多轮、多会话研究 |
| 记忆能力 | 无 | 短期 (state) + 长期 (Chroma) |
| 工具集 | 4 个 GitHub 工具 | 5 个研究工具 (web/pdf/memory) |
| 编排 | 手写 ReAct 循环 | LangGraph StateGraph |
| 持久化 | 无 | SqliteSaver + Chroma 双轨 |

## 2. 架构

```
                 ┌──────────────────────────────────────────────┐
                 │              LangGraph StateGraph             │
                 │                                                │
   START ──▶ chatbot (DeepSeek) ──▶ tools_condition ──▶ tools ──┐ │
                 ▲                                              │ │
                 │           ToolNode                            │ │
                 │   ┌──────────────────────────────────────────┘ │
                 │   ▼                                              │
                 │  search_web / fetch_url / read_pdf               │
                 │  save_memory / search_memory                     │
                 └──────────────────────────────────────────────┘
                                 │
               ┌─────────────────┴────────────────────┐
               ▼                                       ▼
   ┌────────────────────┐                  ┌──────────────────────┐
   │ SqliteSaver        │                  │ Chroma (Persistent)  │
   │ checkpoints.db     │                  │ chroma/              │
   │ 短期：thread 内    │                  │ 长期：跨 thread/进程  │
   │ 状态快照 + 断点    │                  │ 向量库 + hybrid 重排  │
   └────────────────────┘                  └──────────────────────┘
```

## 3. 5 个工具

| 工具 | 作用 | 关键参数 |
| --- | --- | --- |
| `search_web` | DuckDuckGo 搜索网页 | `query`, `max_results` |
| `fetch_url` | 抓取 URL 正文 (BeautifulSoup 清洗) | `url` |
| `read_pdf` | 读本地 PDF (pypdf) | `path`, `max_pages` |
| `save_memory` | 写入长期记忆 (Chroma) | `title`, `content`, `tags` |
| `search_memory` | 从长期记忆 hybrid 召回 | `query`, `top_k` |

`save_memory` / `search_memory` 是本项目灵魂：Agent 自主判断何时沉淀、何时召回，跨会话不丢失上下文。

## 4. 运行方式

```bash
# 在线 demo（需要 .env 中的 API_KEY，DeepSeek 兼容 OpenAI 接口）
uv run python research-assistant/agent.py

# 离线自测：验证 5 工具骨架，不依赖 LLM
uv run python research-assistant/agent.py --self-test

# 查看 LangGraph 拓扑 (mermaid)
uv run python research-assistant/agent.py --graph

# 检查已沉淀的长期记忆
uv run python research-assistant/agent.py --memory-dump

# 清空长期记忆
uv run python research-assistant/agent.py --reset-memory
```

## 5. 跨会话演示

`run_agent_demo()` 会跑两个 thread：

- **Session A** (`thread_id="session-A"`)：让 Agent 研究 "LangGraph 长期记忆方案"，并主动 `save_memory`
- **Session B** (`thread_id="session-B"`)：新 thread，**state 完全独立**，提问 "上次研究过 LangGraph 的长期记忆方案，能复述一下吗？"
  - Agent 必须先 `search_memory` 查到 Session A 写入的条目，才能回答
  - 这是 RAG 作为长期记忆最朴素的工程形态

跑完后用 `--memory-dump` 可以看到 Chroma 里多出来的研究片段。

## 6. Hybrid 检索

`search_memory` 不是纯向量召回：

1. Chroma 用 hash embedding 算 cosine 距离取候选 top (3·k)
2. 候选用 `_lexical_score` 做词项重叠 + 短语命中重排
3. 最终 score = `0.6 * lexical + 0.4 * vector`

教学版用 hash embedding 跑通链路；生产替换为 OpenAI embedding / BGE 即可。

## 7. STAR 简历段落

> **S** (情境)：纯 LLM 没有跨会话记忆，反复回答相同问题、上下文断裂。
> **T** (任务)：构建可持久化研究助手，支持网页搜索、PDF 阅读、跨 session 记忆。
> **A** (行动)：基于 LangGraph 编排 ReAct 循环，封装 5 个工具 (search_web / fetch_url / read_pdf / save_memory / search_memory)；用 Chroma + 自研 hash embedding + hybrid 重排实现长期记忆，用 SqliteSaver 做短期断点恢复。
> **R** (结果)：500 行单文件实现；Session A 沉淀的研究片段，Session B 新 thread 能完整召回并复述，跑通真实 DeepSeek API。

## 8. 文件结构

```
research-assistant/
├── agent.py          # 主程序（约 500 行）
├── README.md         # 本文档
├── chroma/           # 长期记忆持久化目录（自动创建）
└── checkpoints.db    # SqliteSaver 检查点（自动创建）
```

## 9. 依赖

通过根目录 `pyproject.toml` 管理：

- `langgraph` + `langgraph-checkpoint-sqlite` — 图编排 + 短期记忆
- `chromadb` — 长期记忆向量库
- `pypdf` — PDF 解析
- `beautifulsoup4` + `httpx` — 网页抓取
- `ddgs` — DuckDuckGo 搜索
- `openai` — DeepSeek API (OpenAI 兼容)
