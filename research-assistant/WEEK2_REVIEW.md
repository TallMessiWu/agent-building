# 第 2 周复盘 · LangGraph / Multi-Agent / RAG 记忆

## 本周产出

- `exercises/w2d1-langgraph/`: 用 LangGraph StateGraph 重写 ReAct agent，理解 state / node / edge / conditional edge。
- `exercises/w2d2-langgraph-hitl/`: 加入 checkpoint 与 HITL，中断后可恢复。
- `exercises/w2d3-autogen/`: 跑通 AutoGen multi-agent conversation，并整理框架对比。
- `exercises/w2d4-crewai/`: 跑通 CrewAI role-based agent，理解 role / goal / backstory / task 的协作模型。
- `exercises/w2d5-rag-agent/`: 用 Chroma 搭最小 RAG，并接入 LangGraph。
- `research-assistant/`: 完成小项目 2 v2.0，支持网页搜索、网页抓取、PDF 阅读、长期记忆、记忆检索、上下文打包。

## 项目 2 关键设计

短期记忆使用 LangGraph state + `SqliteSaver`，解决同一 thread 内断点恢复。长期记忆使用 Chroma 持久化 collection，解决跨 thread / 跨进程召回。两者职责不同：checkpoint 保存对话运行状态，向量库保存可复用研究知识。

检索没有只依赖向量相似度，而是使用 hash embedding 做候选召回，再用 `_lexical_score` 做词项重叠和短语命中重排。教学版可离线跑通完整链路，生产版可以替换为 OpenAI embedding / BGE / E5。

今天补上的 `build_context_pack` 把《Lost in the Middle》的结论落到工程实现里：最高置信证据放在 prompt 前部，补充证据倒序放到尾部做 anchor；每条证据保留首尾，避免长文本截断时丢掉结论或来源。

## 面试讲法

**S**: 普通 LLM 对话结束即遗忘，研究类任务会反复丢上下文。

**T**: 做一个能跨 session 沉淀研究片段的助手，并能在新会话里召回旧结论。

**A**: 用 LangGraph 编排 ReAct 循环，工具层封装 `search_web` / `fetch_url` / `read_pdf` / `save_memory` / `search_memory` / `build_context_pack`；用 Chroma 做长期记忆，`SqliteSaver` 做短期 checkpoint，hybrid rerank 提高召回稳定性。

**R**: 离线自测覆盖写入、召回、上下文打包、PDF 错误处理和网页联通；在线 demo 可让 Session A 写入记忆，Session B 新 thread 召回并组织上下文后回答。

## 下周衔接

第 3 周进入 Reflexion / Plan-Execute / Multi-Agent 评估。可以直接复用 `research-assistant` 的记忆工具作为 Multi-Agent 协作写作项目的 `Researcher` 子系统，再加入 `Writer` 和 `Critic` 角色。
