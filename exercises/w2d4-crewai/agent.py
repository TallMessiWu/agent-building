"""
Agent 动手任务 - 2026-05-14 (第2周 周四)
============================================
主题：CrewAI — Role-based Agent + 三框架对比表
工具：查天气(get_weather) + 算数(calculate)
API：DeepSeek API（OpenAI 兼容接口，经由 LiteLLM）

核心概念（配合八股题 24-27 食用）：
  - CrewAI 抽象：Agent(role/goal/backstory) + Task(description/expected_output) + Crew
  - Process.sequential：任务按顺序执行；Process.hierarchical：manager 调度
  - role-based 哲学：从「角色」切入而非状态机/对话——业务建模视角
  - 三框架抽象：LangGraph(图) vs AutoGen(对话) vs CrewAI(角色)
  - 八股题 24: LangGraph 与 LangChain 关系
  - 八股题 25: 三框架选择
  - 八股题 26: 为什么手写不用框架
  - 八股题 27: LangGraph 三要素 State/Node/Edge

与前两天的对比：
  LangGraph (w2d1/w2d2): 显式图+状态机，工程师视角
  AutoGen (w2d3):     自然语言对话，研究者视角
  CrewAI (今天):      角色+任务+协作，业务视角——最接近现实公司组织

运行方式：
  uv run python exercises/w2d4-crewai/agent.py                # 基础：研究员+写手协作
  uv run python exercises/w2d4-crewai/agent.py --tool         # 角色 + 工具调用
  uv run python exercises/w2d4-crewai/agent.py --hierarchy    # 层级模式（manager 调度）
  uv run python exercises/w2d4-crewai/agent.py --compare      # 三框架完整对比表

参考资料：
  CrewAI 官方文档: https://docs.crewai.com
  CrewAI quickstart: https://docs.crewai.com/quickstart
"""

import json
import math
import os
import sys
from typing import Type

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第一部分：工具定义（与 w1d1-w2d3 保持一致）                    ║
# ║     CrewAI 工具走 BaseTool 子类化路径，args_schema 用 pydantic     ║
# ╚══════════════════════════════════════════════════════════════════╝

_WEATHER_DB = {
    "北京":    {"temp_c": 22, "condition": "晴",     "humidity": 40, "wind": "北风 3级"},
    "上海":    {"temp_c": 25, "condition": "多云",   "humidity": 68, "wind": "东南风 2级"},
    "广州":    {"temp_c": 29, "condition": "雷阵雨", "humidity": 85, "wind": "南风 4级"},
    "深圳":    {"temp_c": 28, "condition": "阴",     "humidity": 78, "wind": "东风 3级"},
    "杭州":    {"temp_c": 24, "condition": "小雨",   "humidity": 72, "wind": "东北风 2级"},
    "成都":    {"temp_c": 21, "condition": "阴",     "humidity": 75, "wind": "无持续风向 1级"},
    "tokyo":   {"temp_c": 18, "condition": "晴",     "humidity": 50, "wind": "北风 2级"},
    "london":  {"temp_c": 13, "condition": "小雨",   "humidity": 80, "wind": "西风 5级"},
    "new york":{"temp_c": 16, "condition": "多云",   "humidity": 55, "wind": "西南风 4级"},
}


class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名称，如「北京」「tokyo」")
    unit: str = Field("celsius", description="温度单位：celsius 或 fahrenheit")


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "查询指定城市的实时天气信息。返回温度、天气状况、湿度、风速。"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str, unit: str = "celsius") -> str:
        key = city.strip().lower()
        data = _WEATHER_DB.get(key, {
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


class CalcInput(BaseModel):
    expression: str = Field(..., description="数学表达式，如 '2+3*5'、'math.sqrt(16)'")


class CalculateTool(BaseTool):
    name: str = "calculate"
    description: str = "安全执行数学表达式计算。支持 math 模块函数（sqrt、log、sin 等）。"
    args_schema: Type[BaseModel] = CalcInput

    def _run(self, expression: str) -> str:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})
        try:
            result = eval(expression, {"__builtins__": {}}, allowed)
            return json.dumps({"expression": expression, "result": result, "error": None}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"expression": expression, "result": None, "error": str(e)}, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第二部分：LLM 客户端 — DeepSeek 经 LiteLLM 接入                ║
# ║     CrewAI 内部用 LiteLLM，所以模型名前缀要写 "deepseek/"          ║
# ╚══════════════════════════════════════════════════════════════════╝

def make_llm() -> LLM:
    """构造 CrewAI 的 LLM 对象，指向 DeepSeek API。"""
    return LLM(
        model="deepseek/deepseek-chat",
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.0,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第三部分：基础演示 — Researcher + Writer 顺序协作              ║
# ║     展示 CrewAI 的核心：role/goal/backstory + Task                ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_basic_demo():
    """模式 1：Researcher + Writer 顺序协作 —— CrewAI 入门示例。

    展示 CrewAI 核心抽象：
    - Agent：role（角色名）+ goal（目标）+ backstory（背景故事，决定行为风格）
    - Task：description（任务说明）+ expected_output（期望产出格式）
    - Crew：把 Agent 和 Task 编排成团队，Process.sequential 顺序执行
    """
    llm = make_llm()

    researcher = Agent(
        role="资料研究员",
        goal="为指定主题搜集 3-5 个具体、可信、有信息密度的论据",
        backstory=(
            "你是一位严谨的研究员，擅长把模糊问题拆成可验证的具体点。"
            "你不写完整段落，只列要点、数据、原理。"
            "你的素材必须可被复用：每一条都要能独立成为一个论据。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="技术博客作者",
        goal="把研究员提供的素材改写成一篇 150-200 字的中文短文，结构清晰",
        backstory=(
            "你是一位写技术博客的工程师，擅长用类比和生活例子讲清楚概念。"
            "你不会堆术语，而是先讲一个故事再点出原理。"
            "你的稿子必须有标题、正文、总结三部分。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    topic = "为什么 AI Agent 需要工具调用（Function Calling）？"

    research_task = Task(
        description=f"针对主题「{topic}」搜集 3-5 个具体论据/原理/案例。",
        expected_output=(
            "Markdown 列表，每条以「- 」开头。\n"
            "每条不超过 30 字，包含具体原理或场景。"
        ),
        agent=researcher,
    )

    write_task = Task(
        description=f"基于研究员的素材，撰写一篇 150-200 字的中文短文，主题「{topic}」。",
        expected_output=(
            "三部分结构：\n"
            "【标题】<10字内>\n"
            "【正文】<150-180字>\n"
            "【总结】<一句话>"
        ),
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "─" * 60)
    print("  [Crew 协作产出]")
    print("─" * 60)
    print(result.raw if hasattr(result, "raw") else str(result))


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第四部分：工具调用演示 — 天气查询员 + 数据分析师                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_tool_demo():
    """模式 2：带工具的 Agent — 角色分工 + tool 调用。

    演示 CrewAI 怎么把 tool 挂到 Agent 上：
    - weather_agent 拿 get_weather 工具，负责取数
    - analyst       拿 calculate 工具，负责算账
    - 任务间通过 context 串联（Task 2 能拿到 Task 1 的产出）
    """
    llm = make_llm()
    weather_tool = WeatherTool()
    calc_tool = CalculateTool()

    weather_agent = Agent(
        role="天气信息员",
        goal="精确查询用户问到的所有城市天气，返回温度、湿度、天气状况",
        backstory=(
            "你只负责查天气，不做计算、不做分析。"
            "用户问几个城市，你就调几次 get_weather 工具。"
            "你的产出必须是结构化的，每个城市一行。"
        ),
        tools=[weather_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst = Agent(
        role="数据分析师",
        goal="基于天气信息员提供的数据，用 calculate 工具做温度对比、单位换算等计算",
        backstory=(
            "你只负责算账，绝不心算。任何数值计算都要走 calculate 工具。"
            "你不查天气——所有数据从上一位同事那里获取。"
            "你的产出必须给出结论 + 计算依据。"
        ),
        tools=[calc_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    fetch_task = Task(
        description="查询北京、广州、上海三个城市的当前天气。",
        expected_output="三行数据，每行格式：城市名 - 温度 - 天气 - 湿度",
        agent=weather_agent,
    )

    analysis_task = Task(
        description=(
            "基于上一步的天气数据，回答两个问题：\n"
            "1) 三个城市温差最大的是哪两个？相差多少摄氏度？\n"
            "2) 把最热城市的温度换算成华氏度（公式 F = C*9/5+32），用 calculate 工具算。"
        ),
        expected_output="结论 + 两次 calculate 调用的具体表达式和结果",
        agent=analyst,
        context=[fetch_task],
    )

    crew = Crew(
        agents=[weather_agent, analyst],
        tasks=[fetch_task, analysis_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "─" * 60)
    print("  [Tool-using Crew 产出]")
    print("─" * 60)
    print(result.raw if hasattr(result, "raw") else str(result))


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第五部分：层级模式 — Manager 自动调度                          ║
# ║     Process.hierarchical 让 LLM 做 PM，自动决定谁干啥               ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_hierarchy_demo():
    """模式 3：Process.hierarchical — Manager Agent 自动调度。

    与 sequential 模式的区别：
    - sequential: 开发者预先指定 Agent 顺序
    - hierarchical: 内置 manager Agent 看 Task 描述，自己决定派给谁

    这是 CrewAI 与 LangGraph 最大的哲学差异：
    LangGraph 要你画图，CrewAI 让 LLM 自己当 PM。
    """
    llm = make_llm()
    weather_tool = WeatherTool()
    calc_tool = CalculateTool()

    weather_agent = Agent(
        role="weather_specialist",
        goal="为任何涉及城市天气的子任务取数",
        backstory="只查天气，工具是 get_weather。",
        tools=[weather_tool],
        llm=llm,
        allow_delegation=False,
    )

    math_agent = Agent(
        role="math_specialist",
        goal="为任何涉及数值计算的子任务做计算",
        backstory="只算数，工具是 calculate，禁止心算。",
        tools=[calc_tool],
        llm=llm,
        allow_delegation=False,
    )

    main_task = Task(
        description=(
            "请回答用户问题：「广州今天比北京热多少度？温差换算成华氏度是多少？」\n"
            "提示：先查两个城市天气，再算摄氏温差，再换算到华氏度。"
        ),
        expected_output="最终回答 + 各步骤的工具调用记录",
    )

    crew = Crew(
        agents=[weather_agent, math_agent],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "─" * 60)
    print("  [Hierarchical Crew 产出（manager 自动调度）]")
    print("─" * 60)
    print(result.raw if hasattr(result, "raw") else str(result))


# ╔══════════════════════════════════════════════════════════════════╗
# ║     第六部分：三框架对比表 — 八股题 24-27 完整答案                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def show_framework_comparison():
    """三框架对比表 —— 配合八股题 24-27，本周收官梳理。"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LangGraph · AutoGen · CrewAI —— 三框架终极对比                       ║
║              （八股题 24/25/26/27 一并解决）                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────┬─────────────────┬─────────────────┬────────────────────┐  ║
║  │     维度     │   LangGraph     │    AutoGen      │     CrewAI         │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 核心抽象     │   图 Graph      │   对话 Chat     │   角色 Role        │  ║
║  │              │  StateGraph     │  GroupChat      │  Agent + Task      │  ║
║  │              │  Node + Edge    │  Agent Message  │  + Crew + Process  │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 设计哲学     │   工程师视角    │   研究者视角    │   业务视角         │  ║
║  │              │  显式状态机     │  涌现行为       │  组织/职责建模     │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 可控性       │   ★★★ 最高     │   ★★☆ 中等     │   ★★☆ 中等        │  ║
║  │              │   每步显式定义  │   LLM 选发言    │   manager 调度     │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 上手难度     │   ★★★ 最陡     │   ★★☆ 中等     │   ★☆☆ 最易        │  ║
║  │              │   reducer/      │   Selector +    │   role/goal/       │  ║
║  │              │   checkpoint    │   GroupChat     │   backstory 即写完 │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ HITL 支持    │   ★★★ 原生     │   ★★☆ 支持     │   ★☆☆ 有限        │  ║
║  │              │   interrupt_    │   UserProxy     │   人工 callback    │  ║
║  │              │   before/after  │   Agent         │                    │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 工具集成     │   ★★★ 灵活     │   ★★★ 灵活     │   ★★★ 简单        │  ║
║  │              │   ToolNode 或   │   AssistantAgent│   BaseTool 子类    │  ║
║  │              │   自定义节点    │   tools=[...]   │   挂到 Agent       │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 状态持久化   │   ★★★ 内置     │   ★☆☆ 弱       │   ★☆☆ 需自定义    │  ║
║  │              │   Checkpointer  │   消息历史      │   外接 DB          │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 生产级       │   ★★★ 首选     │   ★★☆ 可以     │   ★☆☆ 原型为主    │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 典型场景     │  状态复杂的生产 │  多 Agent 研究  │   业务流程 Demo    │  ║
║  │              │  级 Agent       │  快速原型       │   角色清晰的协作   │  ║
║  ├──────────────┼─────────────────┼─────────────────┼────────────────────┤  ║
║  │ 生态系统     │   LangChain     │   微软研究院    │   社区驱动         │  ║
║  │              │   文档最完善    │   论文支撑      │   模板丰富         │  ║
║  └──────────────┴─────────────────┴─────────────────┴────────────────────┘  ║
║                                                                              ║
║  ▼ 八股题 24：LangGraph 与 LangChain 关系                                     ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │ LangChain 是组件库（Chain/Prompt/Model 等高阶封装），LangGraph 是   │      ║
║  │ 编排层（Graph/State/Checkpoint），二者由 LangChain AI 同一家维护。  │      ║
║  │ LangChain 早期的 AgentExecutor 不可控，LangGraph 是它的替代方案。   │      ║
║  │ 一句话：LangChain 给积木，LangGraph 给图纸。                        │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║  ▼ 八股题 25：三框架怎么选                                                    ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │ 1. 生产级核心 Agent → LangGraph                                     │      ║
║  │    精细控制 + Checkpoint + HITL 原生支持                            │      ║
║  │ 2. 多 Agent 研究/原型 → AutoGen                                     │      ║
║  │    对话涌现 + 微软背书 + 适合论文复现                                │      ║
║  │ 3. 业务 Demo / 教学 / 角色明确的协作 → CrewAI                       │      ║
║  │    role-based 直观 + 上手最快 + 业务方能看懂                        │      ║
║  │ 4. 不确定 → 先手写 ReAct，确认瓶颈后再选框架                         │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║  ▼ 八股题 26：为什么有时候手写不用框架                                        ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │ 1. 框架隐藏 prompt 细节，Bug 难定位（w1d3 的重试/熔断手写最清晰）   │      ║
║  │ 2. 锁定 LLM API：框架升级慢于 OpenAI / Anthropic                    │      ║
║  │ 3. 增加依赖体积和冷启动时间，serverless 场景敏感                    │      ║
║  │ 4. ReAct 循环本身只有 30 行，简单场景框架反成累赘                    │      ║
║  │ 5. Anthropic《Building Effective Agents》原则：先用最简方案         │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║  ▼ 八股题 27：LangGraph 三要素 State/Node/Edge                                ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │ State：TypedDict 描述全局状态，reducer 控制字段合并方式（如         │      ║
║  │        add_messages 把新消息追加而非覆盖）                          │      ║
║  │ Node：纯函数 (state) -> state_update，无副作用                      │      ║
║  │ Edge：固定边 add_edge / 条件边 add_conditional_edges                │      ║
║  │ 入口 END：用 START / END 两个常量标识起止                            │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║  ▼ 本周实战体会（三个框架都跑过后）                                            ║
║  ┌────────────────────────────────────────────────────────────────────┐      ║
║  │ w2d1/w2d2 LangGraph：节点+边显式，HITL 一行 interrupt_before        │      ║
║  │ w2d3 AutoGen：     SelectorGroupChat 让 LLM 自选发言者              │      ║
║  │ w2d4 CrewAI：      role/goal/backstory 即写完 Agent 行为            │      ║
║  │ 结论：抽象层级递增——图(细) → 对话(中) → 角色(粗)。                 │      ║
║  │      抽象越高越易上手，越易失控；产品核心还是要自己写 loop。         │      ║
║  └────────────────────────────────────────────────────────────────────┘      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ╔══════════════════════════════════════════════════════════════════╗
# ║     主入口                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    show_tool = "--tool" in sys.argv
    show_hierarchy = "--hierarchy" in sys.argv
    show_compare = "--compare" in sys.argv

    if show_compare:
        show_framework_comparison()
        return

    print("=" * 60)
    print("  CrewAI — Role-based Multi-Agent 协作")
    print("=" * 60)

    if show_hierarchy:
        print("\n【模式】Process.hierarchical — manager 自动调度")
        print("-" * 60)
        run_hierarchy_demo()
    elif show_tool:
        print("\n【模式】带工具的 Agent — weather_specialist + math_specialist")
        print("-" * 60)
        run_tool_demo()
    else:
        print("\n【模式】基础顺序协作 — 资料研究员 + 技术博客作者")
        print("-" * 60)
        run_basic_demo()

    print("-" * 60)
    print("  --tool       体验 Agent + 工具调用")
    print("  --hierarchy  体验 Process.hierarchical（manager 自动调度）")
    print("  --compare    查看三框架完整对比（八股题 24-27）")


if __name__ == "__main__":
    main()
