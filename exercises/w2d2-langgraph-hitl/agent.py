"""
Agent 动手任务 - 2026-05-12 (第2周 周二)
============================================
主题：LangGraph — HITL (Human-in-the-Loop) + 断点恢复 + 子图
工具：查天气(get_weather) + 算数(calculate)
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 26 食用）：
  - interrupt_before: 在指定节点前自动暂停，零侵入实现 HITL
  - HITL 断点恢复: 暂停 → 人工审批 → 继续执行，State 在断点处完整保留
  - Subgraph: 将横切关注点封装为独立子图，在主图中作为节点复用
  - 框架 vs 手写: HITL 是框架价值的典型案例——手写要做到状态
    冻结/恢复/回滚极其复杂，LangGraph 的 Checkpoint 原生支持

与昨天的关系：
  昨天 (w2d1): StateGraph + SQLite Checkpoint — State 持久化到磁盘
  今天 (w2d2): 在此基础上加 HITL 断点，让执行可以在任意节点前暂停

图拓扑（HITL 模式）:

    START ──→ chatbot ──[有 tool_calls]──→ ⚠ PAUSE ⚠ ──→ tools ──┐
                     └──[无 tool_calls]──→ END                     │
              ↑                                                    │
              └────────────────────────────────────────────────────┘

    其中 ⚠ PAUSE 由 interrupt_before=["tools"] 实现——
    在进入 tools 节点前冻结 State，等待人工审批后继续

运行方式：
  uv run python exercises/w2d2-langgraph-hitl/agent.py             # 自动模式
  uv run python exercises/w2d2-langgraph-hitl/agent.py --hitl      # HITL 模式（交互式审批）
  uv run python exercises/w2d2-langgraph-hitl/agent.py --subgraph  # 子图示例
  uv run python exercises/w2d2-langgraph-hitl/agent.py --graph     # 打印图拓扑

参考资料：
  LangGraph HITL:      https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
  LangGraph Subgraph:   https://langchain-ai.github.io/langgraph/how-tos/subgraph/
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第一部分：工具定义（同 w2d1，@tool 装饰器自动生成 Schema）    ║
# ╚══════════════════════════════════════════════════════════════════╝

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
TOOL_EXECUTORS = {t.name: t for t in TOOLS}

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


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第二部分：State + 消息格式转换（同 w2d1）                    ║
# ╚══════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第三部分：图构建 — interrupt_before 实现 HITL 断点            ║
# ╚══════════════════════════════════════════════════════════════════╝
# 【八股题 26 核心】为什么用框架而不是手写？
#
#   HITL 是框架价值的教科书案例。手写一个「工具执行前暂停」需要：
#   1. 序列化整个对话历史到存储（你昨天的 Checkpoint）
#   2. 冻结执行位置（当前在第几步、哪个分支）
#   3. 提供恢复接口（从断点继续，而非从头开始）
#   4. 支持 State 修改（审批人可能拒绝/修改参数）
#   5. 跨进程恢复（重启后仍能继续）
#
#   用 LangGraph: interrupt_before=["tools"] 一行搞定。
#   不用 LangGraph: 你需要自己实现上述 5 点，轻松上千行。

def build_graph(interrupt_before_tools: bool = False):
    """构建 LangGraph Agent 图。

    Args:
        interrupt_before_tools: True 则在 tools 节点前暂停，等待人工审批。

    图拓扑:
      START → chatbot ──[tools_condition]──→ tools → chatbot → ...
                      └──[tools_condition]──→ END
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

    # ★ HITL 关键: interrupt_before 让图在进入指定节点前自动暂停
    #   暂停时 State 被 Checkpoint 完整保存，等待人工恢复
    interrupt_before = ["tools"] if interrupt_before_tools else None
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=interrupt_before)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第四部分：Subgraph 示例 — 工具参数校验子图                     ║
# ╚══════════════════════════════════════════════════════════════════╝
# 子图的核心思想：将一段可复用的逻辑封装为一个独立的 StateGraph，
# 主图通过 add_node("name", subgraph) 将其作为普通节点嵌入。
# LangGraph 自动做状态映射（根据字段名匹配）。
#
# 本例：构建独立的"工具参数校验"子图——
#   将参数校验这个横切关注点从主图中剥离为独立图，
#   既可以在主图中作为节点嵌入，也可以独立测试和复用。

class ValidationState(TypedDict):
    """子图 State：包含待校验的工具调用和校验结果。"""
    tool_call: dict      # {"name": ..., "args": {...}}
    valid: bool
    reason: str


def validate_tool_call(state: ValidationState) -> dict:
    """校验节点：检查工具调用参数是否安全、合法。"""
    tc = state["tool_call"]
    name = tc.get("name", "")
    args = tc.get("args", {})

    if name == "calculate":
        expr = args.get("expression", "")
        dangerous = ["__", "import", "exec", "open", "file", "compile", "eval"]
        for pattern in dangerous:
            if pattern in expr.lower():
                return {"valid": False, "reason": f"拒绝: 表达式含危险模式 '{pattern}'"}
    elif name == "get_weather":
        city = args.get("city", "")
        if not city.strip():
            return {"valid": False, "reason": "拒绝: 城市名称为空"}
        if len(city) > 100:
            return {"valid": False, "reason": "拒绝: 城市名称过长"}

    return {"valid": True, "reason": "参数校验通过"}


def build_validation_subgraph():
    """构建并编译工具参数校验子图。"""
    sg = StateGraph(ValidationState)
    sg.add_node("validate", validate_tool_call)
    sg.add_edge(START, "validate")
    sg.add_edge("validate", END)
    return sg.compile()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第五部分：会话封装 — auto_chat / hitl_chat                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def _has_pending_tool_calls(state: dict) -> bool:
    """检查 State 中是否有未执行的工具调用（即图是否在 HITL 断点处）。"""
    msgs = state.get("messages", [])
    if not msgs:
        return False
    last = msgs[-1]
    return hasattr(last, "tool_calls") and last.tool_calls


def _print_tool_calls(tool_calls: list):
    """格式化打印工具调用列表。"""
    for i, tc in enumerate(tool_calls):
        args_str = json.dumps(tc["args"], ensure_ascii=False)
        print(f"  [{i}] {tc['name']}({args_str})")


def auto_chat(app, user_input: str, thread_id: str = "default") -> str:
    """自动模式：无断点，直接执行全图（等价于 w2d1 的 chat 函数）。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    final = result["messages"][-1].content
    print(f"  用户: {user_input}")
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                print(f"  → 工具调用: {tc['name']}({args_str})")
        elif msg.type == "tool":
            preview = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
            print(f"  ← 工具返回: {preview}")
    print(f"  Agent: {final}\n")
    return final


def hitl_chat(app, user_input: str, thread_id: str = "default", auto_approve: bool = False) -> str:
    """HITL 模式：在工具执行前暂停，等待人工审批后继续。

    这是今天最关键的代码。interrupt_before=["tools"] 使得每次 LLM 产生 tool_calls
    后，图在 tools 节点前自动冻结。调用者通过检查 State 中的未执行 tool_calls
    来检测是否被中断，并通过 app.invoke(None, config) 恢复执行。

    完整流程（用户 → Agent → 回复）可能触发 0~N 次中断：
      用户输入 → chatbot → [第1次中断] → 审批 → tools → chatbot
                         → [第2次中断] → 审批 → tools → chatbot
                         → ... → 最终回复
    """
    config = {"configurable": {"thread_id": thread_id}}

    print(f"  用户: {user_input}")

    # 第一步：发送用户消息。如果 LLM 产生了 tool_calls，
    # 图会在 tools 节点前中断，返回包含 tool_calls 的 State。
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # 第二步：循环处理中断点。每次 LLM 产生 tool_calls 都会触发一次中断。
    interrupt_count = 0
    while _has_pending_tool_calls(result):
        interrupt_count += 1
        last_msg = result["messages"][-1]

        print(f"\n  ╔════════════════════════════════════════╗")
        print(f"  ║  ⚠ 断点 #{interrupt_count}: 工具调用待审批     ║")
        print(f"  ╚════════════════════════════════════════╝")
        _print_tool_calls(last_msg.tool_calls)

        if auto_approve:
            print("  → 自动批准，继续执行...")
        else:
            choice = input("  [回车=批准 / s=跳过] ").strip().lower()
            if choice == "s":
                # 跳过：更新 State 告知 LLM 被跳过，然后恢复
                print("  → 已跳过，通知 LLM 换方案...")
                app.update_state(config, {
                    "messages": [
                        HumanMessage(
                            content=f"系统提示：以下工具调用被人为跳过，请换一种方式回答："
                                    f"{', '.join(tc['name'] for tc in last_msg.tool_calls)}"
                        )
                    ]
                })

        # 核心操作：app.invoke(None, config) 从断点恢复执行
        # 传入 None 表示"没有新输入，继续跑完剩余节点"
        result = app.invoke(None, config=config)

    final = result["messages"][-1].content
    status = f"(经 {interrupt_count} 次审批)" if interrupt_count else "(无需工具)"
    print(f"\n  Agent {status}: {final}\n")
    return final


# ╔══════════════════════════════════════════════════════════════════╗
# ║  第六部分：测试用例                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_auto_tests():
    """自动模式：验证基础功能正常（同 w2d1 测试）。"""
    app = build_graph(interrupt_before_tools=False)

    print("\n" + "=" * 60)
    print("【测试 1】单工具 — 天气查询")
    print("=" * 60)
    auto_chat(app, "北京今天天气怎么样？", thread_id="auto-1")

    print("=" * 60)
    print("【测试 2】并行调用 — 同时查两个城市")
    print("=" * 60)
    auto_chat(app, "上海和广州的天气分别怎么样？", thread_id="auto-2")

    print("=" * 60)
    print("【测试 3】多步推理 — 查天气 + 计算温差")
    print("=" * 60)
    auto_chat(app, "北京比成都热几度？如果成都再降 3 度呢？", thread_id="auto-3")


def run_hitl_tests():
    """HITL 模式：演示人工审批流程。"""
    app = build_graph(interrupt_before_tools=True)

    print("\n" + "=" * 60)
    print("【HITL 测试 1】单工具查询 — 需人工审批")
    print("=" * 60)
    print("  场景: 用户查北京天气，LLM 请求调 get_weather('北京')")
    print("  在审批时按回车即可批准，或输入 s 跳过")
    hitl_chat(app, "北京今天天气怎么样？", thread_id="hitl-1")

    print("=" * 60)
    print("【HITL 测试 2】多步推理 — 每轮工具调用都需审批")
    print("=" * 60)
    print("  场景: LLM 先并行查两城市天气，再计算温差 ——")
    print("  第一轮: 查天气 ×2 → 审批")
    print("  第二轮: 计算 ×2 → 审批")
    hitl_chat(app, "北京比成都热几度？如果成都再降 3 度呢？", thread_id="hitl-2")

    print("=" * 60)
    print("【HITL 测试 3】多轮对话记忆 — 同一 thread_id 保留上下文")
    print("=" * 60)
    print("  场景: 引用上一轮查到的北京温度做华氏度换算")
    hitl_chat(app, "刚才北京的温度换算成华氏度是多少？", thread_id="hitl-2")


def run_subgraph_demo():
    """子图示例：独立运行校验子图并展示概念。"""
    print("\n" + "=" * 60)
    print("【Subgraph 示例】工具参数校验子图")
    print("=" * 60)

    sg = build_validation_subgraph()

    test_cases = [
        ({"name": "calculate", "args": {"expression": "2 + 3"}}, True),
        ({"name": "calculate", "args": {"expression": "__import__('os')"}}, False),
        ({"name": "calculate", "args": {"expression": "open('/etc/passwd')"}}, False),
        ({"name": "get_weather", "args": {"city": "北京"}}, True),
        ({"name": "get_weather", "args": {"city": ""}}, False),
    ]

    for tc, expect_valid in test_cases:
        result = sg.invoke({"tool_call": tc})
        status = "✅" if result["valid"] else "❌"
        match = "✓" if result["valid"] == expect_valid else "✗ 预期不符!"
        print(f"  {status} {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
        print(f"     → {result['reason']}  {match}")

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │  💡 子图核心概念（面试可能问到）         │")
    print("  ├─────────────────────────────────────────┤")
    print("  │  1. 子图 = 独立 StateGraph，有自己的     │")
    print("  │     State 类型和内部节点                 │")
    print("  │  2. 主图通过 add_node('x', subgraph)     │")
    print("  │     将子图作为普通节点嵌入               │")
    print("  │  3. LangGraph 按字段名自动做状态映射     │")
    print("  │  4. 典型用途: 参数校验、权限检查、       │")
    print("  │     日志审计、多阶段流水线               │")
    print("  │  5. Subgraph vs 普通函数:                │")
    "  │     函数是无状态的，Subgraph 有独立       │"
    "  │     Checkpoint，可以独立暂停/恢复/回放    │"
    print("  └─────────────────────────────────────────┘")


def main():
    show_hitl = "--hitl" in sys.argv
    show_subgraph = "--subgraph" in sys.argv
    show_graph = "--graph" in sys.argv

    if show_graph:
        app = build_graph(interrupt_before_tools=True)
        print("=" * 60)
        print("LangGraph 图拓扑（HITL 模式）— Mermaid 格式")
        print("（可复制到 https://mermaid.live 查看）")
        print("=" * 60)
        print()
        print(app.get_graph(xray=True).draw_mermaid())
        print()
        print("=" * 60)
        print("图例说明")
        print("=" * 60)
        print("  __start__ → chatbot:     入口，用户消息从 chatbot 开始")
        print("  chatbot → tools:         条件边，LLM 返回 tool_calls 时触发")
        print("  chatbot → __end__:       条件边，LLM 直接文本回复时结束")
        print("  tools → chatbot:         固定边，工具执行完回到 LLM 分析结果")
        print()
        print("  ⚠ interrupt_before=['tools']:")
        print("     chatbot→tools 的边执行后，tools 节点运行前，")
        print("     图自动暂停。State 被 Checkpoint 完整保存。")
        print("     此时可以: inspect State → 审批 → invoke(None) 恢复")
        return

    if show_subgraph:
        run_subgraph_demo()
        return

    print("=" * 60)
    print("LangGraph Agent — HITL + 断点恢复 + 子图")
    print("=" * 60)

    if show_hitl:
        run_hitl_tests()
    else:
        run_auto_tests()
        print("-" * 60)
        print("💡 使用 --hitl     体验人工审批断点模式")
        print("💡 使用 --subgraph 查看子图示例")
        print("💡 使用 --graph    查看图拓扑（Mermaid）")


if __name__ == "__main__":
    main()
