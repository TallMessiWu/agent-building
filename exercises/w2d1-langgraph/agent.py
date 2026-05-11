"""
Agent 动手任务 - 2026-05-11 (第2周 周一)
============================================
主题：LangGraph — StateGraph + Checkpoint 持久化
工具：查天气(get_weather) + 算数(calculate)
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 24、27 食用）：
  - LangGraph 将 Agent 控制流建模为有向图（StateGraph）
  - 节点（Node）= 一次 LLM 调用或一次工具执行
  - 边（Edge）= 控制流转移（固定边 or 条件边）
  - Checkpoint 让对话历史持久化到磁盘，实现跨会话记忆

架构对比：
  前三天（手写 while 循环）:
    while True → if tool_calls → execute → messages.append → repeat

  今天（LangGraph StateGraph）:
    START → chatbot ──[有 tool_calls]──→ tools → chatbot → ...
                   └──[无 tool_calls]──→ END

  ⚠ 注意：DeepSeek 思维模型会在响应中附带 reasoning_content，
         再次调用 API 时必须原样传回。LangChain 默认不保留此字段，
         因此本脚本在 chatbot 节点内直接使用 openai SDK，手动做
         LangChain ↔ OpenAI 格式转换，以保留 reasoning_content。

运行方式：
  uv run python exercises/w2d1-langgraph/agent.py           # 基础测试
  uv run python exercises/w2d1-langgraph/agent.py --memory  # 额外展示 Checkpoint 持久化
  uv run python exercises/w2d1-langgraph/agent.py --graph   # 打印图拓扑（Mermaid 格式，可复制到 mermaid.live 查看）

参考资料：
  LangGraph 文档:    https://langchain-ai.github.io/langgraph/
  LangGraph 教程:    https://langchain-ai.github.io/langgraph/tutorials/introduction/
  Checkpoint 概念:   https://langchain-ai.github.io/langgraph/concepts/persistence/
"""

import json
import math
import os
import sys
from typing import Annotated

import openai as openai_module
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════╗
# ║       第一部分：工具定义（LangChain @tool 装饰器）            ║
# ╚══════════════════════════════════════════════════════════════╝
# 与前三天手写 JSON Schema 相比，@tool 自动从函数签名和 docstring
# 生成 Tool Schema，减少样板代码，保持信息同源（描述就在代码旁边）。


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的实时天气信息。返回温度、天气状况、湿度、风速。

    Args:
        city: 城市名称，支持中文或英文，例如：北京、Tokyo、London
        unit: 温度单位，celsius（摄氏度）或 fahrenheit（华氏度），默认 celsius
    """
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


@tool
def calculate(expression: str) -> str:
    """安全执行数学表达式计算。

    支持：四则运算(+ - * /)、幂运算(**)、三角函数(sin/cos/tan)、
    平方根(sqrt)、对数(log/log10)、绝对值(abs)。

    Args:
        expression: 数学表达式字符串，如 '(2+3)*4'、'sqrt(144)'、'2**10'
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result, "error": None}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"expression": expression, "result": None, "error": str(e)}, ensure_ascii=False)


TOOLS = [get_weather, calculate]

# 从 @tool 函数自动生成 OpenAI 格式的 Tool Schema
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema.model_json_schema(),
        },
    }
    for t in TOOLS
]


# ╔══════════════════════════════════════════════════════════════╗
# ║       第二部分：状态定义（State = 图的"记忆单元"）            ║
# ╚══════════════════════════════════════════════════════════════╝
# LangGraph 的核心概念：图中所有节点共享同一个 State 对象。
# 每个节点返回 State 的"增量更新"，而非完整替换。
#
# add_messages 是内置 Reducer：把新消息追加到列表，而非覆盖。
# 等同于前三天手写的 messages.append(...)。
#
# 【八股题 24】StateGraph vs 手写循环的关键区别：
#   手写循环: messages 是函数内局部变量，调用结束即消失
#   StateGraph: State 由 Checkpoint 持久化，跨调用、跨进程依然存在

class State(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """你是一个具备工具调用能力的智能助手。你拥有以下工具：

1. get_weather — 查询任意城市的实时天气（温度、天气状况、湿度、风速）
2. calculate   — 执行数学表达式计算（支持四则运算、幂运算、三角函数等）

行为准则：
- 用户询问天气相关信息时，主动调用 get_weather
- 用户需要数值计算时，调用 calculate，禁止自行心算
- 收到工具返回结果后，用流畅的中文向用户转述
- 保持回答简洁、信息密度高"""


# ╔══════════════════════════════════════════════════════════════╗
# ║     第三部分：LangChain ↔ OpenAI 消息格式互转                ║
# ╚══════════════════════════════════════════════════════════════╝
# DeepSeek 思维模型在 AIMessage 中返回 reasoning_content，
# 下一轮调用时必须原样传回。这里手动处理格式转换，确保不丢失。

def _to_openai_format(messages: list) -> list:
    """将 LangChain 消息列表转换为 OpenAI API 格式，保留 reasoning_content。"""
    result = []
    for msg in messages:
        if msg.type == "system":
            result.append({"role": "system", "content": msg.content})
        elif msg.type == "human":
            result.append({"role": "user", "content": msg.content or ""})
        elif msg.type == "ai":
            d: dict = {"role": "assistant", "content": msg.content or ""}
            # 保留 DeepSeek 思维模型的 reasoning_content
            rc = (msg.additional_kwargs or {}).get("reasoning_content")
            if rc:
                d["reasoning_content"] = rc
            if msg.tool_calls:
                d["content"] = None
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif msg.type == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
    return result


def _from_openai_response(msg) -> AIMessage:
    """将 OpenAI 响应消息转为 LangChain AIMessage，保留 reasoning_content。"""
    kwargs: dict = {}
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        kwargs["additional_kwargs"] = {"reasoning_content": rc}

    if msg.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "type": "tool_call",
            }
            for tc in msg.tool_calls
        ]
        return AIMessage(content=msg.content or "", tool_calls=tool_calls, **kwargs)

    return AIMessage(content=msg.content or "", **kwargs)


# ╔══════════════════════════════════════════════════════════════╗
# ║        第四部分：图构建（Graph = 控制流骨架）                 ║
# ╚══════════════════════════════════════════════════════════════╝

def build_graph(checkpointer=None):
    """构建并编译 LangGraph Agent 图。

    图拓扑（与前三天 while 循环完全等价，只是用图表达）：

      START ──→ chatbot ──[有 tool_calls]──→ tools ──┐
                       └──[无 tool_calls]──→ END      │
                ↑                                     │
                └─────────────────────────────────────┘

    checkpointer: 传入 SqliteSaver 则启用持久化，None 则无状态运行。
    """
    raw_client = openai_module.OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
    )

    def chatbot(state: State) -> dict:
        """LLM 节点：注入 System Prompt，调用 DeepSeek API。"""
        messages = state["messages"]
        if not messages or messages[0].type != "system":
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        openai_msgs = _to_openai_format(messages)
        response = raw_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=openai_msgs,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        return {"messages": [_from_openai_response(response.choices[0].message)]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")

    return graph.compile(checkpointer=checkpointer)


# ╔══════════════════════════════════════════════════════════════╗
# ║        第五部分：辅助函数 — 会话封装 + 格式化输出             ║
# ╚══════════════════════════════════════════════════════════════╝

def chat(app, user_input: str, thread_id: str = "default", verbose: bool = True) -> str:
    """单次对话调用封装。

    thread_id 是 Checkpoint 的会话标识：
      - 相同 thread_id → 共享消息历史（多轮对话记忆）
      - 不同 thread_id → 完全隔离（多用户独立会话）
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    final = result["messages"][-1].content
    if verbose:
        _print_messages(user_input, result["messages"], final)
    return final


def _print_messages(query: str, messages: list, answer: str):
    print(f"  用户: {query}")
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                print(f"  → 工具调用: {tc['name']}({args_str})")
        elif msg.type == "tool":
            preview = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
            print(f"  ← 工具返回: {preview}")
    print(f"  Agent: {answer}\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║              第六部分：测试用例                               ║
# ╚══════════════════════════════════════════════════════════════╝

def run_basic_tests(app):
    """基础功能：单工具、并行调用、多步推理。"""
    print("\n" + "=" * 60)
    print("【测试 1】单工具 — 天气查询")
    print("=" * 60)
    chat(app, "北京今天天气怎么样？", thread_id="basic-1")

    print("=" * 60)
    print("【测试 2】单工具 — 数学计算")
    print("=" * 60)
    chat(app, "帮我算一下 2 的 20 次方", thread_id="basic-2")

    print("=" * 60)
    print("【测试 3】并行调用 — 同时查两个城市")
    print("=" * 60)
    chat(app, "上海和广州的天气分别怎么样？", thread_id="basic-3")

    print("=" * 60)
    print("【测试 4】多步推理 — 先查天气再计算")
    print("=" * 60)
    chat(app, "北京比成都热几度？如果成都再降 3 度，成都变成多少度？", thread_id="basic-4")


def run_memory_tests(app):
    """
    Checkpoint 持久化测试：展示多轮对话记忆和会话隔离。

    【八股题 27】Checkpoint 的核心价值：
      传统手写 Agent: 每次 run_agent() 从空历史开始，无跨轮记忆
      LangGraph:      相同 thread_id 自动接续上一轮 State，
                      跨进程、跨重启都能恢复（持久化到 SQLite）
    """
    print("\n" + "=" * 60)
    print("【Checkpoint 测试】多轮对话 — 同一 thread_id 保留历史")
    print("=" * 60)

    tid = "memory-demo"

    print("--- 第 1 轮 ---")
    chat(app, "北京今天天气怎么样？", thread_id=tid)

    print("--- 第 2 轮（引用第 1 轮查到的温度做换算）---")
    chat(app, "刚才北京的温度换算成华氏度是多少？", thread_id=tid)

    print("--- 第 3 轮（跨轮引用细节）---")
    chat(app, "北京的风力适合放风筝吗？", thread_id=tid)

    print("\n" + "=" * 60)
    print("【Checkpoint 测试】会话隔离 — 不同 thread_id 互不干扰")
    print("=" * 60)
    print("--- 全新会话（不知道前面聊了什么）---")
    chat(app, "刚才我们聊到北京了吗？", thread_id="new-session")


def main():
    show_memory = "--memory" in sys.argv
    show_graph = "--graph" in sys.argv

    db_path = "exercises/w2d1-langgraph/checkpoints.db"

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        if show_graph:
            print("=" * 60)
            print("LangGraph 图拓扑 — Mermaid 格式")
            print("（可复制到 https://mermaid.live 查看或粘贴到支持 Mermaid 的 Markdown 编辑器）")
            print("=" * 60)
            print()
            print(app.get_graph(xray=True).draw_mermaid())
            print()
            print("=" * 60)
            print("图例说明")
            print("=" * 60)
            print("  __start__ → chatbot:  入口，首次输入从 chatbot 开始")
            print("  chatbot → tools:      条件边，LLM 返回 tool_calls 时走这条路")
            print("  tools → chatbot:      固定边，工具执行完回到 LLM")
            print("  chatbot → __end__:    条件边，LLM 直接文本回复时结束")
            print()
            print("🔍 节点内部逻辑：")
            print("  chatbot — 注入 System Prompt → 调 DeepSeek API > 返回 AIMessage")
            print("  tools   — 读取最后一条消息的 tool_calls → 执行@tool函数 → 返回 ToolMessage")
            return

        print("=" * 60)
        print("LangGraph Agent — StateGraph + SQLite Checkpoint")
        print("=" * 60)

        run_basic_tests(app)

        if show_memory:
            run_memory_tests(app)


if __name__ == "__main__":
    main()
