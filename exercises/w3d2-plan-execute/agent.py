"""
Agent 动手任务 - 2026-05-19 (第3周 周二)
============================================
主题：Plan-and-Execute / LLMCompiler 思路
工具：离线任务规划器 + DAG 调度器 + Mermaid 拓扑输出

核心概念（配合八股题 4 食用）：
  - ReAct：边想边做，适合短任务和需要频繁观察的场景
  - Plan-and-Execute：先拆计划，再逐步执行，适合多步骤任务和可审计流程
  - LLMCompiler 思路：把任务编译成依赖图，找出可并行节点，减少串行等待
  - 工程边界：计划必须有输入、输出、依赖和失败处理，不能只是自然语言清单

运行方式：
  uv run python exercises/w3d2-plan-execute/agent.py
  uv run python exercises/w3d2-plan-execute/agent.py --graph
  uv run python exercises/w3d2-plan-execute/agent.py --explain
  uv run python exercises/w3d2-plan-execute/agent.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TASK = "为 research-assistant 增加一次论文调研回答的质量评估"


@dataclass(frozen=True)
class PlanStep:
    id: str
    title: str
    action: str
    depends_on: tuple[str, ...] = ()


@dataclass
class ExecutionResult:
    step_id: str
    title: str
    output: str


@dataclass
class ExecutionState:
    task: str
    completed: dict[str, ExecutionResult] = field(default_factory=dict)


Executor = Callable[[PlanStep, ExecutionState], str]


def build_plan(task: str = TASK) -> list[PlanStep]:
    """把用户目标编译成有依赖关系的计划，而不是线性 todo。"""
    return [
        PlanStep(
            id="inspect",
            title="确认评估对象",
            action="阅读 research-assistant 的输入输出接口和已有 self-test。",
        ),
        PlanStep(
            id="rubric",
            title="设计评估标准",
            action="定义 groundedness、citation、coverage、conciseness 四个维度。",
            depends_on=("inspect",),
        ),
        PlanStep(
            id="fixtures",
            title="准备样例数据",
            action="构造一个论文问题、三条证据和一个候选回答。",
            depends_on=("inspect",),
        ),
        PlanStep(
            id="score",
            title="实现打分器",
            action="按 rubric 检查回答是否引用证据、是否覆盖关键点。",
            depends_on=("rubric", "fixtures"),
        ),
        PlanStep(
            id="report",
            title="生成评估报告",
            action="汇总分数、失败原因和下一步修复建议。",
            depends_on=("score",),
        ),
    ]


def ready_steps(plan: list[PlanStep], completed: set[str]) -> list[PlanStep]:
    return [
        step for step in plan
        if step.id not in completed and all(dep in completed for dep in step.depends_on)
    ]


def default_executor(step: PlanStep, state: ExecutionState) -> str:
    """离线模拟执行器：真实项目中可替换成工具调用、代码运行或 LLM 调用。"""
    if step.id == "inspect":
        return "确认目标接口：输入 question/context/answer，输出结构化评估结果。"
    if step.id == "rubric":
        return "rubric=groundedness,citation,coverage,conciseness；每项 0/1。"
    if step.id == "fixtures":
        return "fixtures=论文问题 + context pack top evidence + 候选回答。"
    if step.id == "score":
        rubric = state.completed["rubric"].output
        fixtures = state.completed["fixtures"].output
        return f"score=3/4；已使用 {rubric} 与 {fixtures}；缺少 conciseness 约束。"
    if step.id == "report":
        score = state.completed["score"].output
        return f"report: {score} 建议加入 120 字限制与证据 ID 引用检查。"
    raise ValueError(f"未知步骤: {step.id}")


def execute_plan(
    plan: list[PlanStep],
    executor: Executor = default_executor,
    task: str = TASK,
) -> list[list[ExecutionResult]]:
    """按 DAG 分批执行；同一批里的步骤没有依赖关系，可并行。"""
    state = ExecutionState(task=task)
    batches: list[list[ExecutionResult]] = []

    while len(state.completed) < len(plan):
        batch_steps = ready_steps(plan, set(state.completed))
        if not batch_steps:
            pending = [step.id for step in plan if step.id not in state.completed]
            raise RuntimeError(f"计划存在循环依赖或缺失依赖: {pending}")

        batch: list[ExecutionResult] = []
        for step in batch_steps:
            output = executor(step, state)
            result = ExecutionResult(step_id=step.id, title=step.title, output=output)
            batch.append(result)

        for result in batch:
            state.completed[result.step_id] = result
        batches.append(batch)

    return batches


def plan_to_mermaid(plan: list[PlanStep]) -> str:
    lines = ["graph TD"]
    for step in plan:
        label = f"{step.id}[{step.title}]"
        if not step.depends_on:
            lines.append(f"  start((任务)) --> {label}")
        for dep in step.depends_on:
            lines.append(f"  {dep} --> {label}")
    return "\n".join(lines)


def print_run(batches: list[list[ExecutionResult]]) -> None:
    print(f"任务：{TASK}\n")
    for index, batch in enumerate(batches, start=1):
        ids = ", ".join(item.step_id for item in batch)
        print(f"=== 执行批次 {index}: {ids} ===")
        for item in batch:
            print(f"- {item.title}: {item.output}")
        print()


def explain() -> None:
    print(
        "\n".join([
            "Plan-and-Execute 把复杂任务拆成先规划、后执行两阶段。",
            "它比 ReAct 更适合多步骤、可审计、可回放的任务，但计划错误会被放大。",
            "LLMCompiler 的关键思路是把自然语言计划变成依赖图：",
            "1. 无依赖节点可以并行执行。",
            "2. 下游节点只消费上游结构化输出。",
            "3. 失败时可以只重跑受影响子图，而不是整条轨迹。",
        ])
    )


def self_test() -> None:
    plan = build_plan()
    batches = execute_plan(plan)
    batch_ids = [[result.step_id for result in batch] for batch in batches]
    assert batch_ids == [["inspect"], ["rubric", "fixtures"], ["score"], ["report"]]
    graph = plan_to_mermaid(plan)
    assert "rubric --> score" in graph
    assert "fixtures --> score" in graph
    assert "score --> report" in graph
    final = batches[-1][0].output
    assert "证据 ID" in final
    print("✅ self-test passed: Plan DAG, parallel batches, and final report are valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-and-Execute / LLMCompiler 离线练习")
    parser.add_argument("--graph", action="store_true", help="打印 Mermaid 计划依赖图")
    parser.add_argument("--explain", action="store_true", help="打印核心概念速记")
    parser.add_argument("--self-test", action="store_true", help="运行离线断言测试")
    args = parser.parse_args()

    plan = build_plan()
    if args.graph:
        print(plan_to_mermaid(plan))
    elif args.explain:
        explain()
    elif args.self_test:
        self_test()
    else:
        print_run(execute_plan(plan))


if __name__ == "__main__":
    main()
