"""
Agent 动手任务 - 2026-05-07 (第1周 周四)
============================================
主题：Function Calling Agent → MCP Server 改造
工具：查天气(get_weather) + 算数(calculate)
协议：MCP (Model Context Protocol) over stdio
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 12, 13 食用）：
  - MCP 协议：标准化的 Client-Server 工具暴露协议
  - MCP vs Function Calling：MCP 是工具的"USB-C 接口"，解耦工具实现与 Agent 逻辑
  - Tool Schema 标准化：Server 统一管理工具定义，Client 发现并转换为 LLM 格式

架构对比：
  昨天（纯 FC）: 用户 → Agent Loop → LLM → tool_calls → 本地执行工具 → 回传 LLM
  今天（MCP）:   用户 → Agent Loop → LLM → tool_calls → MCP Client → MCP Server → 执行 → 回传

运行方式：
  uv run python exercises/w1d2-mcp-server/agent.py            # 客户端模式：启动 Server + 运行测试
  uv run python exercises/w1d2-mcp-server/agent.py serve      # 服务端模式：独立启动 MCP Server（供调试用）

==================== 参考资料 ====================
  MCP 官方文档:    https://modelcontextprotocol.io
  MCP Python SDK:  https://github.com/modelcontextprotocol/python-sdk
  MCP 规范:        https://spec.modelcontextprotocol.io
  DeepSeek API:    https://api-docs.deepseek.com/
  OpenAI FC:       https://platform.openai.com/docs/guides/function-calling
==================== 参考资料 ====================
"""

import asyncio
import json
import math
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from mcp.types import Tool, TextContent

load_dotenv()


# ╔══════════════════════════════════════════════════════════════╗
# ║            第一部分：工具实现（真正干活的代码）               ║
# ╚══════════════════════════════════════════════════════════════╝
# 与昨天完全相同的工具函数——MCP 改造不改这部分！
# MCP Server 只是给这些函数包了一层标准化的"壳"。


def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询城市天气（当前为模拟数据）。

    如需接入真实 API，可替换为：
      - wttr.in（免费，无需注册）:  requests.get(f"https://wttr.in/{city}?format=j1")
      - OpenWeatherMap:            https://openweathermap.org/api
      - 和风天气（国内）:          https://dev.qweather.com/
    """
    weather_db = {
        "北京":    {"temp_c": 22, "condition": "晴",      "humidity": 40, "wind": "北风 3级"},
        "上海":    {"temp_c": 25, "condition": "多云",    "humidity": 68, "wind": "东南风 2级"},
        "广州":    {"temp_c": 29, "condition": "雷阵雨",  "humidity": 85, "wind": "南风 4级"},
        "深圳":    {"temp_c": 28, "condition": "阴",      "humidity": 78, "wind": "东风 3级"},
        "杭州":    {"temp_c": 24, "condition": "小雨",    "humidity": 72, "wind": "东北风 2级"},
        "成都":    {"temp_c": 21, "condition": "阴",      "humidity": 75, "wind": "无持续风向 1级"},
        "武汉":    {"temp_c": 26, "condition": "多云",    "humidity": 62, "wind": "南风 2级"},
        "tokyo":   {"temp_c": 18, "condition": "晴",      "humidity": 50, "wind": "北风 2级"},
        "london":  {"temp_c": 13, "condition": "小雨",    "humidity": 80, "wind": "西风 5级"},
        "new york":{"temp_c": 16, "condition": "多云",    "humidity": 55, "wind": "西南风 4级"},
        "sydney":  {"temp_c": 20, "condition": "晴",      "humidity": 45, "wind": "东风 3级"},
        "paris":   {"temp_c": 15, "condition": "阴",      "humidity": 70, "wind": "西南风 3级"},
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

    return {
        "city": city,
        "temperature": temp,
        "unit": unit_label,
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "wind": data["wind"],
    }


def calculate(expression: str) -> dict:
    """安全执行数学表达式。

    安全设计（面试常考！）：
      用受限 eval——__builtins__ 置空，只暴露 math 模块的安全函数，
      彻底阻断 os.system / __import__ / open 等危险调用。
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum,
    })

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


TOOL_EXECUTORS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def execute_tool(name: str, args: dict) -> str:
    """工具统一调度入口——与昨天完全一致。MCP Server 只负责转发到这里。"""
    func = TOOL_EXECUTORS.get(name)
    if func is None:
        return json.dumps({"error": f"未知工具 '{name}'"}, ensure_ascii=False)
    try:
        result = func(**args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║          第二部分：MCP Server（工具的标准"包装壳"）          ║
# ╚══════════════════════════════════════════════════════════════╝
#
# 【八股题 12】MCP 协议核心概念：
#   MCP (Model Context Protocol) 是 Anthropic 提出的开放协议，
#   定义了 LLM 应用如何与外部工具、资源进行标准化的 Client-Server 交互。
#
#   三大核心原语：
#     ① Tools：    LLM 可调用的工具（本文件的主角）
#     ② Resources：服务端暴露的数据资源（文件、数据库记录等）
#     ③ Prompts：  预定义的提示词模板
#
#   传输层：
#     - stdio：    本地进程通信（本文件使用的方式）
#     - HTTP/SSE： 远程服务通信
#
#   MCP 工具定义的 JSON Schema 完全兼容 OpenAI Function Calling 格式！
#   这意味着：
#     - MCP Server 定义一次工具 → 任何 MCP Client 都能发现和调用
#     - Client 把 MCP Tool Schema 转成 LLM 需要的 Tool Calling 格式
#     - 工具实现和 Agent 逻辑完全解耦
#
# 【八股题 13】MCP vs Function Calling 的关系：
#   - Function Calling 是 LLM 的能力：LLM 输出"我想调这个函数"
#   - MCP 是工具的标准化协议：定义工具如何被发现、描述、调用
#   - MCP 解决的是"工具从哪来、怎么管"的问题
#   - Function Calling 解决的是"LLM 怎么表达调用意图"的问题
#   - 二者互补：MCP 负责工具供应链，FC 负责工具消费端

server = Server("agent-tools")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """MCP 核心原语①：列出所有可用工具。

    Client 通过此接口发现 Server 提供了哪些工具、各自需要什么参数。
    返回的 Tool 对象包含完整的 JSON Schema，Client 可直接转换为 LLM 格式。
    """
    return [
        Tool(
            name="get_weather",
            description=(
                "查询指定城市的实时天气信息。"
                "返回数据包含：温度、天气状况、湿度、风速。"
                "适用场景：用户询问'某地天气怎么样'、'某地热不热'、'需要带伞吗'等。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，支持中文或英文，例如：北京、上海、Tokyo、London",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位。celsius=摄氏度，fahrenheit=华氏度。不传则默认摄氏度。",
                    },
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="calculate",
            description=(
                "执行数学表达式计算。"
                "支持：四则运算(+ - * /)、幂运算(**)、三角函数(sin/cos/tan)、"
                "平方根(sqrt)、对数(log/log10)、绝对值(abs)、取整等。"
                "当用户需要精确数值计算或复杂运算时，必须调用此工具，禁止直接心算。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式字符串，如 '(2+3)*4'、'sqrt(144)'、'2**10'",
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """MCP 核心原语②：执行工具调用。

    Client 发送工具名 + 参数 → Server 执行 → 返回结果。
    这里是 MCP Server 的"最后一公里"——把请求转发给真正的工具函数。
    """
    result_json = execute_tool(name, arguments)
    return [TextContent(type="text", text=result_json)]


async def run_server():
    """启动 MCP Server——通过 stdio 与 Client 通信。

    stdio_server() 创建一个上下文，从 stdin 读 JSON-RPC 请求，
    从 stdout 写 JSON-RPC 响应。这是本地 MCP 最常用的传输方式。
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║          第三部分：MCP → OpenAI Tool Schema 转换             ║
# ╚══════════════════════════════════════════════════════════════╝
#
# MCP Tool 与 OpenAI Function Calling Tool 的格式几乎一致！
# 区别仅在外层包装：
#   MCP Tool: { name, description, inputSchema }
#   OpenAI:   { type: "function", function: { name, description, parameters } }
#
# inputSchema 就是 JSON Schema，直接映射到 parameters 字段即可。


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """将 MCP Tool 列表转换为 OpenAI Function Calling 格式。

    这是 MCP 与 LLM 之间的"翻译层"——也是理解 MCP 价值的关键：
    MCP 定义了一次工具 Schema，Client 可以把它转成任何 LLM 需要的格式。
    OpenAI、Anthropic、Gemini 的格式不同，但 MCP Tool Schema 是统一的。
    """
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return openai_tools


# ╔══════════════════════════════════════════════════════════════╗
# ║          第四部分：MCP Agent 循环（Client 端）               ║
# ╚══════════════════════════════════════════════════════════════╝
#
# 与昨天的 Agent 循环核心逻辑完全一致：
#   用户输入 → LLM 决策 → 需要工具？→ 调工具 → 回传结果 → 再决策 → 最终回复
#
# 关键区别（也是今天的核心学习点）：
#   昨天：工具在本地，直接调 execute_tool()
#   今天：工具在 MCP Server，通过 session.call_tool() 远程调用
#
# 这个区别看似简单，但意义重大：
#   - 工具实现可以独立部署、独立更新
#   - 多个 Agent 可以共享同一个 MCP Server
#   - 工具可以用任何语言实现（只要遵循 MCP 协议）

SYSTEM_PROMPT = """你是一个具备工具调用能力的智能助手。你拥有以下工具：

1. get_weather — 查询任意城市的实时天气
2. calculate   — 执行数学表达式计算

行为准则：
- 用户询问天气相关信息时，主动调用 get_weather
- 用户需要数值计算时，调用 calculate，禁止自行心算
- 收到工具返回结果后，用自然流畅的中文向用户转述
- 如果用户同时问了天气和计算，可以一次调用多个工具（并行）
- 保持回答简洁、信息密度高"""


async def run_agent(
    session: ClientSession,
    llm_client: AsyncOpenAI,
    user_query: str,
    model: str = "deepseek-v4-flash",
    max_turns: int = 10,
    verbose: bool = True,
) -> str:
    """Agent 主循环 —— 通过 MCP Session 调用工具。

    与昨天 run_agent() 的区别清单（面试重点）：
      ① 工具发现：从 MCP Server 动态获取（list_tools），而非硬编码 TOOLS 列表
      ② 工具执行：通过 session.call_tool()，而非本地 execute_tool()
      ③ 格式转换：MCP Tool → OpenAI format（mcp_tools_to_openai）
      ④ 工具结果：从 CallToolResult.content[0].text 提取，而非直接 JSON 字符串

    参数:
        session:    MCP ClientSession（已初始化）
        llm_client: DeepSeek 异步客户端
        user_query: 用户输入
        model:      模型 ID
        max_turns:  最大交互轮次
        verbose:    是否打印调试日志

    返回:
        Agent 的最终文本回复
    """
    # ── 第0步：从 MCP Server 动态发现工具 ──
    # 这是 MCP 架构的关键！工具不是硬编码的，而是从 Server 查询得到。
    tools_result = await session.list_tools()
    mcp_tools = tools_result.tools
    openai_tools = mcp_tools_to_openai(mcp_tools)

    if verbose:
        print(f"  [MCP] 发现 {len(mcp_tools)} 个工具: "
              f"{', '.join(t.name for t in mcp_tools)}")

    # ── 初始化消息历史 ──
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for turn in range(1, max_turns + 1):
        # ── 第①步：调用 LLM ──
        response = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message

        # ── 第②步：分岔路口 ──
        if msg.tool_calls:
            # ====== 分支 A：LLM 决定调用工具 ======
            if verbose:
                names = [tc.function.name for tc in msg.tool_calls]
                print(f"\n  [轮次 {turn}] >> 调用工具: {', '.join(names)}")

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                if verbose:
                    args_preview = json.dumps(tool_args, ensure_ascii=False)
                    print(f"           IN: {tool_name}({args_preview})")

                # —— MCP 调用（与昨天的关键区别！）——
                # 昨天: result_json = execute_tool(tool_name, tool_args)
                # 今天: 通过 MCP session 远程调用工具
                result = await session.call_tool(tool_name, tool_args)
                result_text = result.content[0].text

                if verbose:
                    print(f"           OUT: {result_text}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        else:
            # ====== 分支 B：LLM 直接文本回复 → 最终答案 ======
            if verbose:
                print(f"  [轮次 {turn}] DONE - 最终回复")
            return msg.content

    return "处理超时，请将问题拆分为更小的子问题后重试。"


# ╔══════════════════════════════════════════════════════════════╗
# ║             第五部分：主程序 & 学习测试用例                  ║
# ╚══════════════════════════════════════════════════════════════╝

async def main_client():
    """客户端主程序——启动 MCP Server，连接，运行测试用例。

    这就是 MCP 的完整 Client-Server 交互流程：
      1. 用 stdio_client() 启动 Server 子进程
      2. 建立 ClientSession
      3. 初始化 MCP 握手（initialize）
      4. 运行 Agent 循环（Agent 内部会 list_tools + call_tool）
    """

    # Windows 终端编码兼容
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 65)
    print("   Agent 动手任务 | 2026-05-07 | 第1周周四")
    print("   工具: get_weather + calculate | 协议: MCP over stdio")
    print("   模式: MCP Client-Server | 核心: Tool Discovery + Agent Loop")
    print("=" * 65)

    # ── Server 启动参数 ──
    # Client 通过 stdio 启动 Server 子进程，二者通过 JSON-RPC 通信
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", __file__, "serve"],
        env=None,  # 继承当前环境变量（含 API_KEY）
    )

    # ── 初始化 DeepSeek 异步客户端 ──
    llm_client = AsyncOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
    )

    with open(os.devnull, "w") as errlog:
        async with stdio_client(server_params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"  [MCP] 已连接到 MCP Server: {session.get_server_capabilities()}\n")

                # ── 测试用例矩阵 ──
                tests = [
                    # [场景1] 单工具 - 天气
                    (
                        "北京今天天气怎么样？",
                        "单工具·天气查询",
                    ),
                    # [场景2] 单工具 - 计算
                    (
                        "请帮我计算 (456 + 789) * 23 / 7 的结果",
                        "单工具·数学计算",
                    ),
                    # [场景3] 并行调用——一次查两个城市
                    (
                        "上海和广州现在的天气分别怎么样？",
                        "并行调用·多城市天气",
                    ),
                    # [场景4] 多步推理——先查天气再计算温差
                    (
                        "北京现在多少度？如果北京比上海热 5 度，上海应该是多少度？",
                        "多步推理·天气→计算",
                    ),
                    # [场景5] 英文输入
                    (
                        "What's the weather in Tokyo and New York? Answer in Chinese.",
                        "多语言·英文城市名查询",
                    ),
                    # [场景6] 复杂计算
                    (
                        "一个圆的半径是 7.5，请计算它的面积和周长",
                        "计算·几何公式",
                    ),
                ]

                for query, description in tests:
                    print(f"\n{'─' * 65}")
                    print(f"  [场景]: {description}")
                    print(f"  [用户]: {query}")
                    print(f"{'─' * 65}")

                    try:
                        answer = await run_agent(session, llm_client, query, verbose=True)
                        print(f"\n  [Agent]: {answer}")
                    except Exception as e:
                        print(f"\n  [ERROR]: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Server 模式——通过 stdio 与 Client 通信
        asyncio.run(run_server())
    else:
        # Client 模式——启动 Server 子进程 + 运行测试
        asyncio.run(main_client())
