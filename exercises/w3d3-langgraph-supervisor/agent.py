"""
Agent 动手任务 - 2026-05-20 (第3周 周三)
============================================
主题：Multi-Agent 模式 + LangGraph Supervisor
工具：离线 supervisor 路由 + Researcher / Writer / Critic 三角色协作

核心概念（配合八股题 7 食用）：
  - Multi-Agent：把一个复杂任务拆给多个专业角色，而不是让单个 Agent 全包
  - Supervisor：中心调度节点，负责决定下一步由哪个 worker 执行
  - Worker：只处理自己的职责边界，输出结构化中间结果
  - 反馈闭环：Critic 不通过时回到 Writer 修订，通过后终止
  - 工程边界：生产系统里 supervisor 必须有停止条件，避免多 Agent 互相空转

运行方式：
  uv run python exercises/w3d3-langgraph-supervisor/agent.py
  uv run python exercises/w3d3-langgraph-supervisor/agent.py --graph
  uv run python exercises/w3d3-langgraph-supervisor/agent.py --explain
  uv run python exercises/w3d3-langgraph-supervisor/agent.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TASK = "写一段 180 字以内的短文：为什么 AI Agent 需要工具调用？"
Role = Literal["supervisor", "researcher", "writer", "critic"]
Route = Literal["researcher", "writer", "critic", "finish"]


@dataclass(frozen=True)
class AgentReport:
    role: Role
    content: str
    passed: bool | None = None


class SupervisorState(TypedDict, total=False):
    task: str
    reports: list[AgentReport]
    evidence: list[str]
    draft: str
    critique: str
    approved: bool
    revision_count: int
    next_role: Route
    final_answer: str
    trace: list[str]


def initial_state(task: str = TASK) -> SupervisorState:
    """构造图的初始状态，避免每个节点都判断缺省字段。"""
    return {
        "task": task,
        "reports": [],
        "evidence": [],
        "draft": "",
        "critique": "",
        "approved": False,
        "revision_count": 0,
        "next_role": "researcher",
        "final_answer": "",
        "trace": [],
    }


def latest_report(state: SupervisorState, role: Role) -> AgentReport | None:
    """从后往前找到某个角色最近一次输出。"""
    for report in reversed(state.get("reports", [])):
        if report.role == role:
            return report
    return None


def choose_next_role(state: SupervisorState) -> Route:
    """Supervisor 的核心决策：根据当前状态决定下一个 worker。"""
    if not state.get("evidence"):
        return "researcher"
    if not state.get("draft"):
        return "writer"
    if state.get("approved"):
        return "finish"

    last = state.get("reports", [])[-1]
    if last.role == "writer":
        return "critic"
    if last.role == "critic" and state.get("revision_count", 0) >= 2:
        return "finish"
    if last.role == "critic":
        return "writer"
    if state.get("revision_count", 0) >= 2:
        return "finish"
    return "writer"


def append_trace(state: SupervisorState, event: str) -> list[str]:
    """返回新的 trace 列表，保持节点更新是显式的。"""
    return [*state.get("trace", []), event]


def append_report(state: SupervisorState, report: AgentReport) -> list[AgentReport]:
    """返回新的 reports 列表，避免在原列表上原地修改。"""
    return [*state.get("reports", []), report]


def supervisor_node(state: SupervisorState) -> SupervisorState:
    """中心调度节点：只做路由，不做具体业务。"""
    route = choose_next_role(state)
    update: SupervisorState = {
        "next_role": route,
        "trace": append_trace(state, f"supervisor -> {route}"),
    }
    if route == "finish":
        update["final_answer"] = state.get("draft", "")
    return update


def researcher_node(state: SupervisorState) -> SupervisorState:
    """Researcher 负责提供事实素材，不写成稿。"""
    evidence = [
        "E1: LLM 的参数知识有截止日期，实时数据和私有数据必须通过工具获取。",
        "E2: 工具调用把搜索、数据库、计算、代码执行等能力接入 Agent 循环。",
        "E3: 工具返回可记录、可回放、可评估，比纯自然语言推理更容易审计。",
    ]
    content = "研究素材：\n" + "\n".join(evidence)
    report = AgentReport(role="researcher", content=content)
    return {
        "evidence": evidence,
        "reports": append_report(state, report),
        "trace": append_trace(state, "researcher: collected 3 evidence items"),
    }


def writer_node(state: SupervisorState) -> SupervisorState:
    """Writer 负责根据素材写短文；第一次故意漏引用，展示 critic 反馈闭环。"""
    revision_count = state.get("revision_count", 0) + 1
    if revision_count == 1:
        draft = (
            "AI Agent 需要工具调用，因为模型本身只能根据已有参数生成回答，"
            "遇到实时信息、私有数据或精确计算时容易失真。接入搜索、数据库和计算工具后，"
            "Agent 能先获取外部证据，再基于结果完成任务，可靠性和可审计性都会提升。"
        )
    else:
        draft = (
            "AI Agent 需要工具调用：模型参数有知识截止日期，实时和私有数据要靠外部工具获取[E1]；"
            "搜索、数据库、计算和代码执行能补齐模型能力边界[E2]；"
            "工具结果可记录、回放和评估，使 Agent 比纯文本推理更可审计[E3]。"
        )

    report = AgentReport(role="writer", content=draft)
    return {
        "draft": draft,
        "revision_count": revision_count,
        "reports": append_report(state, report),
        "trace": append_trace(state, f"writer: produced draft v{revision_count}"),
    }


def critic_node(state: SupervisorState) -> SupervisorState:
    """Critic 只按验收标准判断文稿，不重写文稿。"""
    draft = state.get("draft", "")
    missing = [tag for tag in ("[E1]", "[E2]", "[E3]") if tag not in draft]
    too_long = len(draft) > 180
    approved = not missing and not too_long

    if approved:
        critique = "通过：短文包含 E1/E2/E3 证据引用，长度满足 180 字以内。"
    else:
        problems = []
        if missing:
            problems.append(f"缺少证据引用: {', '.join(missing)}")
        if too_long:
            problems.append("超过 180 字限制")
        critique = "不通过：" + "；".join(problems) + "。请 Writer 修订。"

    report = AgentReport(role="critic", content=critique, passed=approved)
    return {
        "critique": critique,
        "approved": approved,
        "reports": append_report(state, report),
        "trace": append_trace(state, f"critic: {'approved' if approved else 'requested revision'}"),
    }


def route_after_supervisor(state: SupervisorState) -> Route:
    """LangGraph 条件边读取 supervisor 写入的 next_role。"""
    return state["next_role"]


def build_graph():
    """构建 LangGraph supervisor 拓扑。"""
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "finish": END,
        },
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    return graph.compile()


def run_supervisor(task: str = TASK) -> SupervisorState:
    """运行一次完整的 supervisor 多 Agent 协作。"""
    app = build_graph()
    return app.invoke(initial_state(task))


def print_trace(state: SupervisorState) -> None:
    """把多 Agent 执行轨迹打印成适合学习的格式。"""
    print(f"任务：{state['task']}\n")
    print("=== 路由轨迹 ===")
    for event in state["trace"]:
        print(f"- {event}")

    print("\n=== 角色输出 ===")
    for report in state["reports"]:
        suffix = "" if report.passed is None else f" | passed={report.passed}"
        print(f"\n[{report.role}{suffix}]\n{report.content}")

    print("\n=== 最终答案 ===")
    print(state["final_answer"])


def graph_to_mermaid() -> str:
    """输出 LangGraph 拓扑，方便复制到 Mermaid 查看。"""
    return build_graph().get_graph(xray=True).draw_mermaid()


def explain() -> None:
    print(
        "\n".join([
            "Supervisor 模式把多 Agent 协作拆成一个调度者和多个 worker。",
            "Supervisor 不直接完成任务，而是观察状态并选择下一个角色。",
            "Worker 只负责单一职责：Researcher 找素材，Writer 写稿，Critic 验收。",
            "这种模式比纯群聊更可控，因为每条边、停止条件和重试次数都写在图里。",
            "生产系统里要限制最大修订次数，并把 worker 输出结构化，否则很容易空转。",
        ])
    )


def self_test() -> None:
    state = run_supervisor()
    assert state["approved"] is True
    assert state["revision_count"] == 2
    assert "[E1]" in state["final_answer"]
    assert "[E2]" in state["final_answer"]
    assert "[E3]" in state["final_answer"]
    assert len(state["final_answer"]) <= 180
    assert "supervisor -> researcher" in state["trace"]
    assert "supervisor -> writer" in state["trace"]
    assert "supervisor -> critic" in state["trace"]
    graph = graph_to_mermaid()
    assert "supervisor" in graph
    assert "researcher" in graph
    assert "writer" in graph
    assert "critic" in graph
    print("✅ self-test passed: supervisor routed workers, revision loop converged, graph is valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Supervisor 离线多 Agent 练习")
    parser.add_argument("--graph", action="store_true", help="打印 Mermaid 图拓扑")
    parser.add_argument("--explain", action="store_true", help="打印核心概念速记")
    parser.add_argument("--self-test", action="store_true", help="运行离线断言测试")
    args = parser.parse_args()

    if args.graph:
        print(graph_to_mermaid())
    elif args.explain:
        explain()
    elif args.self_test:
        self_test()
    else:
        print_trace(run_supervisor())


if __name__ == "__main__":
    main()
