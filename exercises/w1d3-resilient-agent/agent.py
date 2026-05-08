"""
Agent 动手任务 - 2026-05-08 (第1周 周五)
============================================
主题：Agent 健壮性 —— 错误重试 + 工具结果验证 + 降级策略
工具：查天气(get_weather) + 算数(calculate)
API：DeepSeek API（OpenAI 兼容接口）

核心概念（配合八股题 14 食用）：
  - 工具调用不是 100% 可靠的——网络抖动、API 超时、格式异常都会导致失败
  - 生产级 Agent 需要三层防御：重试 → 熔断 → 降级
  - 工具结果验证是防止"垃圾数据污染 LLM 上下文"的最后一道防线
  - Anthropic 官方指南《Writing Effective Tools》的核心理念落地

架构对比：
  前天（纯 FC）:   用户 → Agent Loop → LLM → tool_calls → 本地执行 → 回传 → 最终答案
  昨天（MCP）:     用户 → Agent Loop → LLM → tool_calls → MCP Client/Server → 执行 → 回传
  今天（弹性 FC）: 用户 → Agent Loop → LLM → tool_calls → [重试→校验→熔断→降级] → 回传 → 最终答案

运行方式：
  uv run python exercises/w1d3-resilient-agent/agent.py           # 默认：运行全部测试（正常模式）
  uv run python exercises/w1d3-resilient-agent/agent.py chaos     # 混沌模式：注入随机故障，展示弹性能力

==================== 参考资料 ====================
  Anthropic: Writing Effective Tools   https://www.anthropic.com/engineering/writing-tools-for-agents
  OpenAI Function Calling:             https://platform.openai.com/docs/guides/function-calling
  DeepSeek API:                        https://api-docs.deepseek.com/
  重试策略最佳实践:                     https://learn.microsoft.com/en-us/azure/architecture/patterns/retry
  熔断器模式:                           https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
==================== 参考资料 ====================
"""

import asyncio
import json
import math
import os
import random
import time
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Windows 终端 GBK 编码兼容
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════╗
# ║     第一部分：工具实现（真正干活的代码——与前两天完全一致）    ║
# ╚══════════════════════════════════════════════════════════════╝
# 工具逻辑本身不需要改动。弹性层是包裹在工具外部的"保护壳"，
# 就像给裸机装上保险丝 + 稳压器——工具本身不变，变的是调用方式。


def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询城市天气（当前为模拟数据）。

    如需接入真实 API，可替换为：
      - wttr.in（免费）:       requests.get(f"https://wttr.in/{city}?format=j1")
      - OpenWeatherMap:         https://openweathermap.org/api
      - 和风天气（国内）:       https://dev.qweather.com/
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
      用受限 eval——__builtins__ 置空，只暴露 math 模块的安全函数。
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

# ╔══════════════════════════════════════════════════════════════╗
# ║  第二部分：弹性配置 —— 所有可调整的健壮性参数集中管理        ║
# ╚══════════════════════════════════════════════════════════════╝

RESILIENCE_CONFIG = {
    # 重试策略
    "max_retries": 3,          # 最大重试次数
    "base_delay": 0.3,         # 基础等待秒数（指数退避：delay * 2^attempt）
    "max_delay": 5.0,          # 单次等待上限
    "retryable_errors": (      # 可重试的错误类型关键字
        "timeout", "connection", "rate_limit", "server_error",
        "server error", "internal", "temporary", "unavailable",
    ),

    # 熔断器
    "circuit_breaker_threshold": 3,   # 连续失败 N 次后熔断
    "circuit_breaker_reset": 15,      # 熔断后 N 秒尝试半开

    # 超时
    "tool_timeout": 8,         # 单次工具调用超时秒数

    # 混沌模式（仅用于演示/测试）
    "chaos_enabled": False,    # 是否注入随机故障
    "chaos_fail_rate": 0.25,   # 随机故障概率
    "chaos_timeout_rate": 0.1, # 随机超时概率
    "chaos_corrupt_rate": 0.1, # 随机返回格式损坏概率
}


# ╔══════════════════════════════════════════════════════════════╗
# ║       第三部分：弹性层核心组件（今天的核心学习内容）          ║
# ╚══════════════════════════════════════════════════════════════╝
#
# 【八股题 14】Agent 健壮性设计（面试高频！）：
#
#   三层防御体系（由外到内）：
#     ① 重试层 (Retry)：   识别可恢复错误 → 指数退避重试 → 达到上限后向上抛
#     ② 熔断层 (Circuit)： 统计连续失败 → 超过阈值断开 → 定时半开探测
#     ③ 降级层 (Fallback)： 返回默认值/缓存/错误提示 → 让 Agent 优雅告知用户
#
#   关键面试考点：
#     - 什么错误该重试？→ 瞬态错误（network/timeout/503），非业务错误（invalid input）
#     - 为什么用指数退避？→ 避免「雷鸣羊群效应」，给服务恢复时间
#     - 熔断器三态：关闭(正常) → 打开(熔断) → 半开(探测)
#     - 工具结果为什么要校验？→ LLM 会根据结果回复用户，垃圾进垃圾出


# ─── 3.1 可重试错误判断 ───

def is_retryable(error_message: str) -> bool:
    """判断错误是否属于可重试的瞬态错误。

    核心原则：只重试「可能自己恢复」的错误，不重试「注定失败」的错误。
    例如：timeout 可能下次成功，但 'invalid expression' 重试 100 次也没用。
    """
    msg_lower = error_message.lower()
    return any(kw in msg_lower for kw in RESILIENCE_CONFIG["retryable_errors"])


# ─── 3.2 工具结果校验器 ───

def validate_weather_result(result: dict) -> tuple[bool, str]:
    """校验天气查询结果的格式完整性。

    为什么要校验？
      - 工具可能返回不完整数据（API 变更、解析错误）
      - LLM 会根据返回值组织回复——如果数据残缺，LLM 会"脑补"虚假信息
      - 提前校验能防止垃圾数据进入对话上下文
    """
    required_fields = ["city", "temperature", "condition"]
    for field in required_fields:
        if field not in result:
            return False, f"缺少必填字段: {field}"
    if not isinstance(result.get("temperature"), (int, float)):
        return False, f"temperature 类型异常: {type(result.get('temperature'))}"
    if result.get("temperature", 0) < -90 or result.get("temperature", 0) > 60:
        return False, f"temperature 值异常: {result['temperature']}（地球表面温度应在 -90~60°C）"
    return True, ""


def validate_calculate_result(result: dict) -> tuple[bool, str]:
    """校验计算结果的格式完整性。

    重点检查：
      - result 字段存在且为数字（不是 None/NaN/Inf）
      - expression 字段存在（用于调试回溯）
    """
    if "result" not in result:
        return False, "缺少必填字段: result"
    r = result["result"]
    if r is not None:
        if not isinstance(r, (int, float)):
            return False, f"result 类型异常: {type(r)}"
        if isinstance(r, float) and (math.isnan(r) or math.isinf(r)):
            return False, f"result 值为 NaN 或 Infinity"
    if "expression" not in result:
        return False, "缺少必填字段: expression"
    return True, ""


RESULT_VALIDATORS = {
    "get_weather": validate_weather_result,
    "calculate": validate_calculate_result,
}


# ─── 3.3 降级策略（Fallback） ───

FALLBACK_RESULTS = {
    "get_weather": {
        "city": "未知",
        "temperature": "N/A",
        "unit": "°C",
        "condition": "天气数据暂时不可用",
        "humidity": "N/A",
        "wind": "N/A",
        "_fallback": True,
    },
    "calculate": {
        "expression": "N/A",
        "result": None,
        "error": "计算服务暂时不可用，请稍后重试",
        "_fallback": True,
    },
}


# ─── 3.4 混沌注入（Chaos Engineering） ───

class ChaosException(Exception):
    """混沌注入异常——仅用于测试/演示。"""
    pass


def chaos_inject(tool_name: str):
    """根据混沌配置注入随机故障。

    三种故障模式：
      1. 超时：模拟网络延迟或服务 hang
      2. 执行失败：模拟服务端 500 错误
      3. 格式损坏：模拟 API 返回格式变更
    """
    cfg = RESILIENCE_CONFIG
    if not cfg["chaos_enabled"]:
        return None  # 无注入，正常执行

    roll = random.random()

    if roll < cfg["chaos_timeout_rate"]:
        raise ChaosException(
            f"[混沌注入] {tool_name} 模拟超时 (timeout after {cfg['tool_timeout']}s)"
        )
    elif roll < cfg["chaos_timeout_rate"] + cfg["chaos_fail_rate"]:
        msgs = [
            f"connection reset by peer",
            f"500 Internal Server Error",
            f"rate_limit exceeded, please retry later",
            f"service temporarily unavailable",
        ]
        raise ChaosException(f"[混沌注入] {tool_name} 模拟故障: {random.choice(msgs)}")
    elif roll < cfg["chaos_timeout_rate"] + cfg["chaos_fail_rate"] + cfg["chaos_corrupt_rate"]:
        return "corrupt"  # 信号：返回格式损坏

    return None


# ─── 3.5 带弹性的工具执行器（整合所有层） ───

class CircuitBreaker:
    """熔断器——防止对已经"坏了"的服务持续发起请求。

    三个状态：
      CLOSED  → 正常通行，统计失败次数
      OPEN    → 直接拒绝，不再尝试（快速失败）
      HALF_OPEN → 允许一次探测，成功则 CLOSED，失败则 OPEN

    这是分布式系统容错的经典模式（Martin Fowler, 2014）。
    """

    def __init__(self, threshold: int, reset_timeout: float):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    def allow_request(self) -> bool:
        """判断当前是否允许发起请求。"""
        now = time.time()
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if now - self.last_failure_time >= self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        # HALF_OPEN: 只允许一次探测
        return True

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"

    def status(self) -> str:
        return f"[熔断器: {self.state} | 连续失败: {self.failure_count}/{self.threshold}]"


# 为每个工具维护独立的熔断器
CIRCUIT_BREAKERS: dict[str, CircuitBreaker] = {
    name: CircuitBreaker(
        threshold=RESILIENCE_CONFIG["circuit_breaker_threshold"],
        reset_timeout=RESILIENCE_CONFIG["circuit_breaker_reset"],
    )
    for name in TOOL_EXECUTORS
}


async def resilient_execute_tool(
    name: str,
    args: dict,
    verbose: bool = True,
) -> str:
    """带完整弹性保护的工具执行入口。

    执行链路：
      ① 熔断器检查 → 如果已熔断，直接降级
      ② 重试循环 → 指数退避，最多 N 次
      ③ 混沌注入 → （仅演示模式）模拟故障
      ④ 实际执行 → 调用真正的工具函数
      ⑤ 结果校验 → 检查返回格式完整性
      ⑥ 校验失败 → 视为执行错误，进入重试
      ⑦ 全部失败 → 降级返回 fallback

    这个函数是今天最核心的学习点——它展示了如何把 3 层防御
    整合到一个统一的工具调度入口。
    """
    cb = CIRCUIT_BREAKERS[name]

    # ── 第0步：熔断器检查 ──
    if not cb.allow_request():
        if verbose:
            print(f"  ⚡ {name} 已熔断，直接降级")
        return json.dumps(FALLBACK_RESULTS.get(name, {"error": "服务不可用"}), ensure_ascii=False)

    # ── 第1步：重试循环 ──
    last_error = None
    cfg = RESILIENCE_CONFIG

    for attempt in range(cfg["max_retries"] + 1):
        try:
            # ── 混沌注入（仅演示） ──
            chaos_result = chaos_inject(name)
            if chaos_result == "corrupt":
                # 模拟返回格式损坏：返回一个残缺的 dict
                if name == "get_weather":
                    corrupt = {"city": args.get("city", "?"), "temperature": "ERROR"}
                else:
                    corrupt = {"result": float("nan")}
                result = corrupt
            elif isinstance(chaos_result, ChaosException):
                raise chaos_result
            else:
                # ── 第2步：实际执行工具 ──
                func = TOOL_EXECUTORS[name]
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(func, **args),
                        timeout=cfg["tool_timeout"],
                    )
                except asyncio.TimeoutError:
                    raise ChaosException(
                        f"{name} 执行超时 (>{cfg['tool_timeout']}s)"
                    )

            # ── 第3步：结果校验 ──
            validator = RESULT_VALIDATORS.get(name)
            if validator and isinstance(result, dict):
                is_valid, err_msg = validator(result)
                if not is_valid:
                    raise ChaosException(
                        f"{name} 返回格式校验失败: {err_msg} | 原始数据: {json.dumps(result, ensure_ascii=False)}"
                    )

            # ── 成功！ ──
            cb.record_success()
            if verbose and attempt > 0:
                print(f"  ✓ {name} 重试成功（第 {attempt} 次）")
            return json.dumps(result, ensure_ascii=False)

        except ChaosException as e:
            last_error = str(e)
        except Exception as e:
            last_error = f"未知异常: {type(e).__name__}: {e}"

        cb.record_failure()

        # 判断是否值得重试
        if not is_retryable(last_error):
            if verbose:
                print(f"  ✗ {name} 不可重试错误，放弃: {last_error[:80]}")
            break

        if attempt < cfg["max_retries"]:
            delay = min(cfg["base_delay"] * (2 ** attempt), cfg["max_delay"])
            # 添加随机抖动，避免多个 client 同时重试（雷鸣羊群效应）
            jitter = delay * 0.2 * random.random()
            wait = delay + jitter
            if verbose:
                print(f"  ↻ {name} 第 {attempt + 1}/{cfg['max_retries']} 次重试 "
                      f"（等待 {wait:.2f}s）: {last_error[:60]}")
            await asyncio.sleep(wait)

    # ── 第4步：全部失败 → 降级 ──
    if verbose:
        print(f"  ▼ {name} 全部重试失败({cb.status()})，使用降级数据")
    fallback = FALLBACK_RESULTS.get(name, {"error": last_error, "_fallback": True})
    # 对于天气，尝试保留用户请求的城市名
    if name == "get_weather":
        fallback["city"] = args.get("city", "未知")
    return json.dumps(fallback, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║          第四部分：Agent 循环（带弹性工具调用）              ║
# ╚══════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = """你是一个具备工具调用能力的智能助手。你拥有以下工具：

1. get_weather — 查询任意城市的实时天气（温度、天气状况、湿度、风速）
2. calculate   — 执行数学表达式计算（支持四则运算、幂运算、三角函数等）

行为准则：
- 用户询问天气相关信息时，主动调用 get_weather
- 用户需要数值计算时，调用 calculate，禁止自行心算
- 收到工具返回结果后，用流畅的中文向用户转述
- 如果工具返回中包含 "_fallback": true 或 "暂时不可用"，向用户如实说明数据异常，不要编造数据
- 保持回答简洁、信息密度高"""

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
            "required": ["city"],
        },
    },
}

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

TOOLS = [WEATHER_TOOL, CALCULATOR_TOOL]


async def run_agent(
    llm_client: AsyncOpenAI,
    user_query: str,
    model: str = "deepseek-v4-flash",
    max_turns: int = 10,
    verbose: bool = True,
) -> str:
    """Agent 主循环 —— 使用弹性工具调用。

    与前两天 Agent 循环的唯一区别：
      昨天:  result_json = execute_tool(name, args)   ← 裸调用，一次失败就炸
      今天:  result_json = await resilient_execute_tool(name, args)  ← 弹性调用

    其余逻辑（消息管理、LLM 交互、终止判断）完全一致。
    这正是「非侵入式弹性改造」的优雅之处——工具调度层独立升级，
    Agent 循环主体无需改动。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for turn in range(1, max_turns + 1):
        response = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            if verbose:
                names = [tc.function.name for tc in msg.tool_calls]
                print(f"\n[轮次 {turn}] LLM 调用工具: {', '.join(names)}")

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"  → {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                # ── 关键区别：使用弹性执行器而非裸调用 ──
                result_json = await resilient_execute_tool(
                    tool_name, tool_args, verbose=verbose
                )

                if verbose:
                    preview = result_json[:120] + "..." if len(result_json) > 120 else result_json
                    print(f"  ← {preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })

        else:
            if verbose:
                print(f"\n[轮次 {turn}] LLM 最终回复")
            return msg.content

    return "处理超时，请将问题拆分为更小的子问题。"


# ╔══════════════════════════════════════════════════════════════╗
# ║              第五部分：测试用例                                ║
# ╚══════════════════════════════════════════════════════════════╝

async def main():
    llm_client = AsyncOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
    )

    chaos = "--chaos" in sys.argv or "chaos" in sys.argv
    if chaos:
        RESILIENCE_CONFIG["chaos_enabled"] = True
        print("=" * 60)
        print("⚡ 混沌模式已启用")
        print(f"   故障注入率: {RESILIENCE_CONFIG['chaos_fail_rate']:.0%}")
        print(f"   超时注入率: {RESILIENCE_CONFIG['chaos_timeout_rate']:.0%}")
        print(f"   格式损坏率: {RESILIENCE_CONFIG['chaos_corrupt_rate']:.0%}")
        print("=" * 60)

    test_cases = [
        # 基础测试（混沌模式下会触发弹性机制）
        ("场景 A: 单工具 — 天气查询", "北京今天天气怎么样？"),
        ("场景 B: 单工具 — 数学计算", "帮我算一下 2 的 20 次方"),
        ("场景 C: 并行调用 — 同时查两个城市", "上海和广州的天气分别怎么样？"),
        ("场景 D: 多步推理 — 先查天气再计算", "北京现在多少度？如果成都比北京冷3度，成都多少度？"),
        ("场景 E: 英文输入", "What's the weather in Tokyo? Answer in Chinese please."),
        ("场景 F: 三角函数", "计算 sin(pi/6) + cos(pi/3) + tan(pi/4)"),
    ]

    for title, query in test_cases:
        print(f"\n{'=' * 60}")
        print(f"📋 {title}")
        print(f"👤 用户: {query}")
        print(f"{'=' * 60}")

        answer = await run_agent(llm_client, query, verbose=True)
        print(f"\n🤖 Agent 最终回复:\n{answer}")
        print()

    # 重置混沌配置，展示弹性统计
    if chaos:
        print("\n" + "=" * 60)
        print("📊 混沌模式运行统计")
        print("=" * 60)
        for name, cb in CIRCUIT_BREAKERS.items():
            print(f"  {name}: {cb.status()}")


if __name__ == "__main__":
    asyncio.run(main())
