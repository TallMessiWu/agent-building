"""
Agent 动手任务 - 2026-05-13 (第2周 周三)
============================================
主题：AutoGen — Multi-Agent 对话 + 框架选择
工具：查天气(get_weather) + 算数(calculate)
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 25 食用）：
  - AutoGen 对话抽象: Agent 通过自然语言消息交流，行为涌现
  - SelectorGroupChat: LLM 动态选择下一个发言的 Agent
  - AssistantAgent: 带 system prompt 的 LLM 驱动 Agent
  - 三种框架抽象对比: LangGraph(图/状态机) vs AutoGen(对话) vs CrewAI(角色)
  - 八股题 25: 三框架选择标准

与 LangGraph 的关系：
  LangGraph (w2d1/w2d2): 工程师视角 — 显式定义状态机，精细控制
  AutoGen (今天):     研究者视角 — Agent 以自然语言对话协作，涌现行为

运行方式：
  uv run python exercises/w2d3-autogen/agent.py                  # 基础两 Agent 对话
  uv run python exercises/w2d3-autogen/agent.py --group          # SelectorGroupChat（3 Agent 协作）
  uv run python exercises/w2d3-autogen/agent.py --tool           # 多 Agent + 工具调用
  uv run python exercises/w2d3-autogen/agent.py --compare        # 三框架对比输出

参考资料：
  AutoGen 官方文档: https://microsoft.github.io/autogen/
  SelectorGroupChat: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/
"""

import asyncio
import json
import math
import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import (
    UserMessage, SystemMessage, AssistantMessage,
    FunctionExecutionResult, FunctionExecutionResultMessage,
)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第一部分：工具定义（与 w1d1-w2d2 保持一致）                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的实时天气信息。返回温度、天气状况、湿度、风速。"""
    weather_db = {
        "北京":    {"temp_c": 22, "condition": "晴",     "humidity": 40, "wind": "北风 3级"},
        "上海":    {"temp_c": 25, "condition": "多云",   "humidity": 68, "wind": "东南风 2级"},
        "广州":    {"temp_c": 29, "condition": "雷阵雨", "humidity": 85, "wind": "南风 4级"},
        "深圳":    {"temp_c": 28, "condition": "阴",     "humidity": 78, "wind": "东风 3级"},
        "杭州":    {"temp_c": 24, "condition": "小雨",   "humidity": 72, "wind": "东北风 2级"},
        "成都":    {"temp_c": 21, "condition": "阴",     "humidity": 75, "wind": "无持续风向 1级"},
        "tokyo":   {"temp_c": 18, "condition": "晴",     "humidity": 50, "wind": "北风 2级"},
        "london":  {"temp_c": 13, "condition": "小雨",   "humidity": 80, "wind": "西风 5级"},
        "new york":{"temp_c": 16, "condition": "多云",   "humidity": 55, "wind": "西南风 4级"},
        "sydney":  {"temp_c": 20, "condition": "晴",     "humidity": 45, "wind": "东风 3级"},
        "paris":   {"temp_c": 15, "condition": "阴",     "humidity": 70, "wind": "西南风 3级"},
    }
    key = city.strip().lower()
    data = weather_db.get(key, {
        "temp_c": 20, "condition": "暂无数据", "humidity": 60, "wind": "未知",
    })
    temp = data["temp_c"]
    unit_label = "°C"
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
        unit_label = "°F"
    return json.dumps({
        "city": city, "temperature": temp, "unit": unit_label,
        "condition": data["condition"], "humidity": f"{data['humidity']}%",
        "wind": data["wind"],
    }, ensure_ascii=False)


def calculate(expression: str) -> str:
    """安全执行数学表达式计算。"""
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result, "error": None}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"expression": expression, "result": None, "error": str(e)}, ensure_ascii=False)


TOOLS = [get_weather, calculate]
TOOL_EXECUTORS = {t.__name__: t for t in TOOLS}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": t.__name__,
            "description": t.__doc__.split("\n")[0] if t.__doc__ else "",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                } if t.__name__ == "get_weather" else {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["city"] if t.__name__ == "get_weather" else ["expression"],
            },
        },
    }
    for t in TOOLS
]


def execute_tool(name: str, args: dict) -> str:
    """执行工具调用并返回结果。"""
    fn = TOOL_EXECUTORS.get(name)
    if not fn:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第二部分：模型客户端工厂                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def make_model_client() -> OpenAIChatCompletionClient:
    """创建 DeepSeek API 兼容的 OpenAI 客户端。"""
    return OpenAIChatCompletionClient(
        model="deepseek-v4-flash",
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
            "family": "deepseek",
        },
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第三部分：工具驱动的手写 Agent 循环（AutoGen 工具层兼容）     ║
# ╚══════════════════════════════════════════════════════════════════╝

async def tool_agent_loop(user_input: str) -> str:
    """手写 ReAct 循环：直接调用 DeepSeek API，带工具执行。

    这个函数演示了一个关键概念：
    AutoGen 框架内部也做同样的事——LLM → tool_calls → 执行 → 回传结果 → 循环。
    理解这个循环后，再看 AutoGen 的封装就一目了然。

    注：AutoGen 0.7 的 AssistantAgent 目前工具支持还在演进中，
    因此这里用手写循环来展示 multi-agent + 工具调用的完整流程。
    """
    import openai as openai_module

    client = openai_module.OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
    )

    messages = [
        {"role": "system", "content": (
            "你是一个具备工具调用能力的智能助手。\n"
            "工具：get_weather（查天气）、calculate（算数）\n"
            "规则：需要天气数据就调 get_weather，需要计算就调 calculate，禁止心算。\n"
            "工具返回后用中文向用户转述结果。"
        )},
        {"role": "user", "content": user_input},
    ]

    for _ in range(10):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # 构建 assistant message，保留 reasoning_content（DeepSeek 要求）
            assistant_msg: dict = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                assistant_msg["reasoning_content"] = rc
            messages.append(assistant_msg)

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                print(f"  → {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
                preview = result[:120] + "..." if len(result) > 120 else result
                print(f"  ← {preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content or ""

    return "（超出最大轮次 10）"


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第四部分：AutoGen 多 Agent 演示                               ║
# ╚══════════════════════════════════════════════════════════════════╝

async def run_basic_demo():
    """模式 1：基础两 Agent 对话 —— AutoGen 最简示例。

    这个演示展示了 AutoGen 的核心模式：
    - 两个 AssistantAgent 相互对话
    - SelectorGroupChat 管理回合
    - LLM 选择下一个发言人
    """
    client = make_model_client()

    # Agent 1: 提问者（用户代理）
    asker = AssistantAgent(
        name="asker",
        model_client=client,
        system_message="""你是 asker，一位对 AI Agent 技术好奇的开发者。
请直接用中文提出一个问题（不要自我介绍，不要解释你在做什么）。
问题围绕：AI Agent 框架选型（LangGraph vs AutoGen vs CrewAI）。
只提一个问题，不要多嘴，不要追问。""",
    )

    # Agent 2: 回答者（专家）
    expert = AssistantAgent(
        name="expert",
        model_client=client,
        system_message="""你是 expert，一位 AI 架构专家。
用中文简洁回答 asker 的问题：
1. 先给结论再解释（3-5 句）
2. 使用技术术语但确保清晰
回答完毕后在末尾说 "回答完毕。"
不要自我介绍、不要解释你在做什么。""",
    )

    # 用 SelectorGroupChat 管理多 Agent 对话
    # 这是 AutoGen 的核心抽象 —— Agent 间通过自然语言交流
    team = SelectorGroupChat(
        participants=[asker, expert],
        model_client=client,
        selector_prompt="""你是对话主持人。按规则选择下一个发言人：

流程：先选 asker（提问），然后选 expert（回答），之后选 TERMINATE。
规则：每轮只选一个 Agent，不要重复选同一个。

可用 Agent：asker, expert
""",
        termination_condition=TextMentionTermination(text="回答完毕"),
    )

    try:
        task = "请开始一轮关于 AI Agent 框架选择的对话。asker 先问，expert 回答。"
        result = await team.run(task=task)

        print("\n" + "─" * 60)
        print("  [对话记录]")
        print("─" * 60)
        for msg in result.messages:
            if hasattr(msg, "source") and hasattr(msg, "content"):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)[:200]
                print(f"  [{msg.source}]: {content}")
                print()
    finally:
        await client.close()


async def run_group_demo():
    """模式 2：SelectorGroupChat — 3 Agent 协作（Researcher + Writer + Critic）。

    这演示了 AutoGen 最有价值的场景：多角色分工协作。
    三个 Agent 通过自然语言协作完成一篇短文，与项目 3 的 multi-agent-collab 呼应。
    """
    client = make_model_client()

    researcher = AssistantAgent(
        name="researcher",
        model_client=client,
        system_message="""你是 researcher，资料研究员。用中文工作。
不要自我介绍，不要解释你在做什么。

接收写作主题后，直接提供 3-5 个具体要点/数据/论据。
只提供素材，不写完整段落。

格式：
【研究素材】
1. <要点1>
2. <要点2>
...
末尾说 "研究完毕" """,
    )

    writer = AssistantAgent(
        name="writer",
        model_client=client,
        system_message="""你是 writer，内容写手。用中文工作。
不要自我介绍，不要解释你在做什么。

基于 researcher 的素材，撰写一篇 150-200 字短文。
包含标题 + 正文 + 一句话总结。

格式：
【文稿】
标题：<标题>
正文：<内容>
总结：<一句话>
末尾说 "写作完毕" """,
    )

    critic = AssistantAgent(
        name="critic",
        model_client=client,
        system_message="""你是 critic，审稿人。用中文工作。
不要自我介绍，不要解释你在做什么。

阅读 writer 的文稿，给出 2-3 条具体改进建议。
关注：事实准确性、逻辑连贯性、表达清晰度。
批评要具体到句。

格式：
【审稿意见】
1. <问题+建议>
2. <问题+建议>
末尾说 "审稿完毕，可以终止" """,
    )

    team = SelectorGroupChat(
        participants=[researcher, writer, critic],
        model_client=client,
        selector_prompt="""你是团队协调人。请按以下流程选择下一个发言的 Agent：

1. 用户提出主题后 → 选 researcher（搜集素材）
2. researcher 完成后 → 选 writer（撰写文稿）
3. writer 完成后 → 选 critic（审稿）
4. critic 完成后 → 选 TERMINATE

可用 Agent：researcher, writer, critic
当前谁还没有完成任务？按顺序推进。""",
        termination_condition=TextMentionTermination(text="可以终止"),
        max_turns=5,
    )

    try:
        task = "请三位协作完成一篇关于「AI Agent 为什么需要工具调用」的科普短文"
        result = await team.run(task=task)

        print("\n" + "─" * 60)
        print("  [Multi-Agent 协作记录]")
        print("─" * 60)
        for msg in result.messages:
            if hasattr(msg, "source") and hasattr(msg, "content"):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)[:300]
                print(f"  [{msg.source}]: {content}")
                print()
    finally:
        await client.close()


async def run_tool_demo():
    """模式 3：多 Agent + 工具调用。

    演示 AutoGen 风格的多 Agent 对话 + 工具执行。
    使用手写循环 + Tool schema 注入 DeepSeek API，
    但 Agent 分工由 system prompt 模拟 multi-agent 协作效果。
    """
    print("  AutoGen 多 Agent + 工具调用")
    print("  " + "=" * 50)
    print("  注：使用手写 ReAct 循环 + DeepSeek tool_calls")
    print("  Agent 分工由 system prompt 模拟 multi-agent 效果")
    print()

    test_cases = [
        ("weather_agent", "北京和上海今天天气分别怎么样？"),
        ("math_agent",   "北京比广州凉快几度？帮我算一下温差"),
        ("multi_agent",  "如果今天是夏天，广州的温度再升高 5 度是多少？然后换算成华氏度"),
    ]

    for agent_role, query in test_cases:
        print(f"  [{agent_role}] 用户: {query}")
        result = await tool_agent_loop(query)
        print(f"  [{agent_role}] Agent: {result}\n")


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第五部分：框架对比                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def show_framework_comparison():
    """三框架对比表 —— 配合八股题 25。"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           LangGraph · AutoGen · CrewAI — 三框架深度对比                   ║
║                      （八股题 25 完整答案）                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────┬─────────────────┬─────────────────┬────────────────┐  ║
║  │     维度      │   LangGraph     │    AutoGen      │    CrewAI      │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 核心抽象      │   图 (Graph)    │   对话 (Chat)   │   角色 (Role)  │  ║
║  │              │  Node + Edge    │  Agent Message  │  Role + Task   │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 设计哲学      │   工程师视角     │   研究者视角     │   业务视角      │  ║
║  │              │  显式状态机      │  涌现行为        │  角色建模       │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 可控性        │   ★★★ 最高     │   ★★☆ 中等     │   ★★☆ 中等    │  ║
║  │              │   每步显式定义   │   LLM 选发言者    │   角色+任务     │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 上手难度      │   ★★★ 最陡     │   ★★☆ 中等     │   ★☆☆ 最易    │  ║
║  │              │   概念多( reducer│   对话抽象直观    │   YAML 配置     │  ║
║  │              │   /checkpoint)  │                  │                │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ HITL 支持     │   ★★★ 原生     │   ★★☆ 支持     │   ★☆☆ 有限    │  ║
║  │              │   interrupt_    │   UserProxyAgent │   需自定义      │  ║
║  │              │   before/after  │                  │                │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 生产级        │   ★★★ 首选     │   ★★☆ 可以     │   ★☆☆ 不推荐  │  ║
║  │              │   checkpoint    │   稳定但不成熟    │   快速原型      │  ║
║  │              │   + tracing     │                  │                │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 适用场景      │  状态复杂的     │   多 Agent 研究  │   业务流程类    │  ║
║  │              │   生产级 Agent  │   快速原型验证   │   Demo/教学    │  ║
║  │              │   需要精细控制  │   涌现行为探索   │   角色分工明确  │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────┤  ║
║  │ 生态系统      │   LangChain     │   微软研究院     │   社区驱动      │  ║
║  │              │   文档最完善     │   论文支撑       │   模板丰富      │  ║
║  └──────────────┴─────────────────┴─────────────────┴────────────────┘  ║
║                                                                          ║
║  选型建议（面试这么说）：                                                  ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │ 1. 生产级核心 Agent → LangGraph（精细控制 + checkpoint）          │    ║
║  │ 2. 多 Agent 研究/原型 → AutoGen（对话涌现 + 快速验证）            │    ║
║  │ 3. 业务 Demo/教学 → CrewAI（角色建模直观 + 上手最快）              │    ║
║  │ 4. 遵循 Anthropic 原则：先用最简方案，只在真正需要时加复杂度       │    ║
║  │ 5. 框架是脚手架——学习阶段用，产品核心自己写 loop                    │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  本周实战体会（三个框架都跑过后）：                                        ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │ w2d1 LangGraph: 用 StateGraph 定义节点和边,显式控制流程            │    ║
║  │ w2d2 LangGraph: interrupt_before 一行实现 HITL,框架价值体现       │    ║
║  │ w2d3 AutoGen:   让 Agent 自己决定谁发言,涌现行为 vs 显式控制       │    ║
║  │ 结论: 图抽象适合"我知道流程是什么",对话抽象适合"让 Agent 自己商量" │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第六部分：主入口                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    show_group = "--group" in sys.argv
    show_tool = "--tool" in sys.argv
    show_compare = "--compare" in sys.argv

    if show_compare:
        show_framework_comparison()
        return

    print("=" * 60)
    print("  AutoGen Multi-Agent — 对话抽象下的多 Agent 协作")
    print("=" * 60)

    if show_group:
        print("\n【模式】SelectorGroupChat — Researcher + Writer + Critic")
        print("-" * 60)
        asyncio.run(run_group_demo())
    elif show_tool:
        print("\n【模式】多 Agent + 工具调用")
        print("-" * 60)
        asyncio.run(run_tool_demo())
    else:
        print("\n【模式】基础两 Agent 对话 — asker vs expert")
        print("-" * 60)
        asyncio.run(run_basic_demo())

    print("-" * 60)
    print("  使用 --group    体验 SelectorGroupChat（Researcher+Writer+Critic）")
    print("  使用 --tool     体验多 Agent + 工具调用")
    print("  使用 --compare  查看三框架完整对比（八股题 25）")


if __name__ == "__main__":
    main()
