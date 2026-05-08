"""
Agent 动手任务 - 2026-05-06 (第1周 周三)
============================================
主题：纯手写 Function Calling Agent（零框架）
工具：查天气(get_weather) + 算数(calculate)
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 10, 11 食用）：
  - Tool Calling：让 LLM 能调用外部工具的核心机制
  - Agent 循环：感知 → 决策 → 执行 → 反馈 → 再决策（ReAct 模式）
  - Tool Schema：工具的能力说明书（JSON Schema 格式），是 LLM 与工具的"合同"

==================== 参考资料 ====================
  OpenAI Function Calling:  https://platform.openai.com/docs/guides/function-calling
  Anthropic Tool Use:       https://docs.anthropic.com/en/docs/build-with-claude/tool-use
  DeepSeek API 文档:        https://api-docs.deepseek.com/
  JSON Schema 规范:         https://json-schema.org/
  Building Effective Agents:https://www.anthropic.com/research/building-effective-agents
  Writing Effective Tools:  https://www.anthropic.com/engineering/writing-tools-for-agents
==================== 参考资料 ====================
"""

import json
import math
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ╔══════════════════════════════════════════════════════════════╗
# ║           第一部分：Tool Schema 定义（核心中的核心）         ║
# ╚══════════════════════════════════════════════════════════════╝
#
# 【八股题 10】Function Calling 底层流程：
#   ① 开发者定义工具的"函数签名" → name + description + parameters(JSON Schema)
#   ② 每次请求时，把工具列表 tools[] 一并发送给 LLM
#   ③ LLM 根据用户意图，二选一：
#      - 直接回复文本（不需要工具）      → response.choices[0].message.content
#      - 返回一个 tool_calls 数组       → response.choices[0].message.tool_calls
#   ④ 开发者在本地执行工具，把结果作为 tool role 消息追加到对话历史
#   ⑤ LLM 再次收到结果，综合后生成最终的自然语言回复
#
# 【八股题 11】Tool Schema 设计三原则：
#   原则1: description 越详细越好——LLM 靠它判断"这个工具能解决用户问题吗？"
#   原则2: parameters 按严格 JSON Schema 定义，required 数组明确必填项
#   原则3: 命名语义化——name 动词+名词，property 名字见名知义

# ── 工具 ①：天气查询 ──
# 设计要点：多参数（city 必填 + unit 可选）、enum 枚举约束
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "查询指定城市的实时天气信息。"
            "返回数据包含：温度、天气状况、湿度、风速。"
            "适用场景：用户询问'某地天气怎么样'、'某地热不热'、'需要带伞吗'等。"
        ),
        "parameters": {
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
            "required": ["city"],  # unit 是可选的——演示 required 的正确用法
        },
    },
}

# ── 工具 ②：数学计算 ──
# 设计要点：单参数、安全执行约束
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "执行数学表达式计算。"
            "支持：四则运算(+ - * /)、幂运算(**)、三角函数(sin/cos/tan)、"
            "平方根(sqrt)、对数(log/log10)、绝对值(abs)、取整等。"
            "当用户需要精确数值计算或复杂运算时，必须调用此工具，禁止直接心算。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式字符串，如 '(2+3)*4'、'sqrt(144)'、'2**10'",
                }
            },
            "required": ["expression"],
        },
    },
}

# 全局工具注册表——新增工具只需加到这里
TOOLS = [WEATHER_TOOL, CALCULATOR_TOOL]


# ╔══════════════════════════════════════════════════════════════╗
# ║              第二部分：工具实现（真正干活的代码）             ║
# ╚══════════════════════════════════════════════════════════════╝
# LLM 不执行代码——它只输出"我想调用 get_weather(city='北京')"。
# 真正的函数调用、HTTP 请求、计算，都在这部分完成。

def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    查询城市天气（当前为模拟数据）。

    如需接入真实 API，可替换为：
      - wttr.in（免费，无需注册）:  requests.get(f"https://wttr.in/{city}?format=j1")
      - OpenWeatherMap:            https://openweathermap.org/api
      - 和风天气（国内）:          https://dev.qweather.com/
    """
    # ---- 模拟天气数据库 ----
    # 实际项目中这里是一个 HTTP 请求
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
        "temp_c": 20,
        "condition": "暂无数据",
        "humidity": 60,
        "wind": "未知",
    })

    # 温度单位换算
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
    """
    安全执行数学表达式。

    安全设计（面试常考！）：
      用受限 eval——__builtins__ 置空，只暴露 math 模块的安全函数，
      彻底阻断 os.system / __import__ / open 等危险调用。

    生产环境替代方案：
      - numexpr 库（更快更安全）: https://github.com/pydata/numexpr
      - 沙箱子进程执行
    """
    # 构建受限命名空间：只包含 math 的公开函数
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    # 补充几个常用内置函数
    allowed.update({
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum,
    })

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


# 工具路由表——新增工具时只需加一行映射
TOOL_EXECUTORS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def execute_tool(name: str, args: dict) -> str:
    """工具统一调度入口。接收 LLM 的调用请求 → 执行 → 返回 JSON 字符串。"""
    func = TOOL_EXECUTORS.get(name)
    if func is None:
        return json.dumps({"error": f"未知工具 '{name}'"}, ensure_ascii=False)
    try:
        result = func(**args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║              第三部分：Agent 循环（LLM 的大脑回路）          ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Agent 循环就是著名的 ReAct 模式的工程实现：
#
#   ┌───────────┐      ┌───────────┐      ┌──────────────┐
#   │  用户输入   │ ───→ │    LLM    │ ───→ │ tool_calls?  │
#   └───────────┘      └───────────┘      └──┬───────┬───┘
#                          ↑                │ YES   │ NO
#                          │          ┌─────┘       └──────┐
#                          │          ▼                     ▼
#                          │   ┌───────────┐       ┌───────────┐
#                          │   │  执行工具   │       │  返回文本   │
#                          │   └─────┬─────┘       │   (结束)    │
#                          │         │             └───────────┘
#                          └─── 追加结果 ──┘
#                    （循环，直到 LLM 不再调用工具）

SYSTEM_PROMPT = """你是一个具备工具调用能力的智能助手。你拥有以下工具：

1. get_weather — 查询任意城市的实时天气
2. calculate   — 执行数学表达式计算

行为准则：
- 用户询问天气相关信息时，主动调用 get_weather
- 用户需要数值计算时，调用 calculate，禁止自行心算
- 收到工具返回结果后，用自然流畅的中文向用户转述
- 如果用户同时问了天气和计算，可以一次调用多个工具（并行）
- 保持回答简洁、信息密度高"""


def run_agent(
    client: OpenAI,
    user_query: str,
    model: str = "deepseek-v4-flash",
    max_turns: int = 10,
    verbose: bool = True,
) -> str:
    """
    Agent 主循环 —— ReAct (Reasoning + Acting) 模式的核心实现。

    参数:
        client:     OpenAI 客户端实例（已配置 DeepSeek endpoint）
        user_query: 用户输入的自然语言问题
        model:      DeepSeek 模型 ID（deepseek-v4-flash / deepseek-v4-pro）
        max_turns:  安全上限——最大 LLM 交互轮次，防止死循环
        verbose:    是否打印调试日志

    返回:
        Agent 的最终文本回复
    """
    # 初始化消息历史——这三条是发给 LLM 的完整上下文
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for turn in range(1, max_turns + 1):
        # ---- 第①步：调用 LLM，让它看消息历史 + 可用工具 ----
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",     # "auto" = LLM 自行判断是否调工具
            temperature=0.0,        # 低温度 → 工具调用更稳定、可预测
        )

        msg = response.choices[0].message

        # ---- 第②步：分岔路口 —— text 还是 tool_calls？ ----
        if msg.tool_calls:
            # ====== 分支 A：LLM 决定调用工具 ======
            if verbose:
                names = [tc.function.name for tc in msg.tool_calls]
                print(f"\n  [轮次 {turn}] >> 调用工具: {', '.join(names)}")

            # 关键！把 assistant 的 tool_calls 消息原样塞回历史
            # 这样 LLM 下一轮才知道"我刚才叫你调了这个工具"
            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                if verbose:
                    args_preview = json.dumps(tool_args, ensure_ascii=False)
                    print(f"           IN: {tool_name}({args_preview})")

                # 执行工具
                result_json = execute_tool(tool_name, tool_args)

                if verbose:
                    print(f"           OUT: {result_json}")

                # 把工具结果作为 tool 角色消息追加
                # tool_call_id 必须对上——这是 LLM 用来关联"哪个调用对应哪个结果"的
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })
            # 本轮结束，回到循环开头——LLM 收到工具结果后会决定下一步

        else:
            # ====== 分支 B：LLM 直接文本回复 → 这就是最终答案 ======
            if verbose:
                print(f"  [轮次 {turn}] DONE - 最终回复")
            return msg.content

    # 理论上不会走到这里（LLM 一般 2-5 轮就结束）
    return "处理超时，请将问题拆分为更小的子问题后重试。"


# ╔══════════════════════════════════════════════════════════════╗
# ║             第四部分：主程序 & 学习测试用例                  ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    """程序入口——初始化客户端，依次运行测试用例"""

    # Windows 终端默认 GBK 编码，LLM 回复含 Unicode 符号会乱码
    # 强制 stdout 使用 UTF-8，配合 chcp 65001 或现代终端使用
    # Jupyter 中 sys.stdout 是 OutStream，无 reconfigure 方法——需做能力检测
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── 初始化 DeepSeek 客户端 ──
    # DeepSeek API 完全兼容 OpenAI SDK——只需换 base_url + api_key
    # 参考: https://api-docs.deepseek.com/zh-cn/
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
    )

    print("=" * 65)
    print("   Agent 动手任务 | 2026-05-06 | 第1周周三")
    print("   工具: get_weather + calculate | API: DeepSeek")
    print("   模式: 零框架纯手写 | 核心: Agent Loop + Tool Schema")
    print("=" * 65)

    # ── 测试用例矩阵 ──
    # 从简单到复杂：单工具 → 并行多工具 → 串行多工具
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
        # [场景3] 可能触发并行工具调用（一次查两个城市天气）
        (
            "上海和广州现在的天气分别怎么样？",
            "并行调用·多城市天气",
        ),
        # [场景4] 串行调用——先查天气再计算温差
        (
            "北京现在多少度？如果北京比上海热 5 度，上海应该是多少度？",
            "多步推理·天气→计算",
        ),
        # [场景5] 英文输入——测试 LLM 对非中文城市名的处理
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
            answer = run_agent(client, query, verbose=True)
            print(f"\n  [Agent]: {answer}")
        except Exception as e:
            print(f"\n  [ERROR]: {e}")


if __name__ == "__main__":
    main()
