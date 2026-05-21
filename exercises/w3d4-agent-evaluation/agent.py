"""
Agent 动手任务 - 2026-05-21 (第3周 周四)
============================================
主题：Agent 评估脚本 v1
工具：离线评估数据集 + 任务成功率 + 轨迹评估 + LLM-as-judge 风格 rubric + 本地 trace 记录

今天的重点不是再写一个更复杂的 Agent，而是学会“怎么判断 Agent 是否真的做好了”。
生产里的 Agent 评估通常会同时看三层：

1. Outcome / 成功率：
   最终答案有没有满足任务要求，例如长度、格式、是否引用证据。

2. Trajectory / 轨迹质量：
   Agent 是否按合理路径行动，例如有没有先研究再写作再批改，是否死循环。

3. Judge / 语义质量：
   单靠规则难以判断“回答是否有用”，所以常用 LLM-as-judge 按 rubric 打分。
   本练习为了离线稳定运行，用确定性规则模拟 judge 的评分逻辑；接入 DeepSeek、
   Langfuse 或 Braintrust 时，只需要替换 judge / trace sink 这两层。

运行方式：
  uv run python exercises/w3d4-agent-evaluation/agent.py
  uv run python exercises/w3d4-agent-evaluation/agent.py --case good
  uv run python exercises/w3d4-agent-evaluation/agent.py --trace-json exercises/w3d4-agent-evaluation/trace.local.json
  uv run python exercises/w3d4-agent-evaluation/agent.py --self-test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


CaseId = Literal["good", "missing_citation", "too_long", "bad_trajectory"]

REQUIRED_EVIDENCE = ("[E1]", "[E2]", "[E3]")
EXPECTED_ROUTE = ("researcher", "writer", "critic")
MAX_ANSWER_CHARS = 180
MAX_REVISIONS = 2


@dataclass(frozen=True)
class AgentRun:
    """一条可评估的 Agent 样本。

    真实项目里它可能来自 LangGraph trace、Langfuse trace、Braintrust dataset run，
    或线上日志回放。这里先把字段收敛到最小集合，方便理解评估闭环。
    """

    case_id: CaseId
    task: str
    final_answer: str
    route: list[str]
    tool_observations: list[str]
    revision_count: int


@dataclass(frozen=True)
class OutcomeScore:
    """最终答案层面的硬规则评估。"""

    passed: bool
    length_ok: bool
    citations_ok: bool
    missing_citations: list[str]
    answer_chars: int


@dataclass(frozen=True)
class TrajectoryScore:
    """执行轨迹层面的过程评估。

    这类指标能发现“答案碰巧对了，但过程不可控”的问题。例如 Agent 没有查资料、
    反复走同一节点、跳过 critic，都会让生产系统难以审计。
    """

    passed: bool
    has_expected_route: bool
    no_dead_loop: bool
    used_tools: bool
    notes: list[str]


@dataclass(frozen=True)
class JudgeScore:
    """LLM-as-judge 风格的语义评分。

    字段设计成可替换：以后可以把 deterministic_judge 换成真实 LLM 调用，
    只要仍返回 0-1 的维度分和总分即可。
    """

    score: float
    coverage: float
    groundedness: float
    conciseness: float
    auditability: float
    rationale: str


@dataclass(frozen=True)
class EvaluationResult:
    """单条样本的完整评估结果。"""

    case_id: str
    outcome: OutcomeScore
    trajectory: TrajectoryScore
    judge: JudgeScore
    success: bool


@dataclass
class TraceEvent:
    """本地 trace 事件，字段对齐 Langfuse/Braintrust 的常见概念。

    - name: 一个 span / event 的名称
    - input/output: 这一步输入输出
    - scores: 评估分数，方便后续按 case 聚合
    - metadata: 任意调试信息
    """

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float | bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class LocalTraceRecorder:
    """离线 trace recorder。

    生产接 Langfuse / Braintrust 时，一般会把这里的 record() 映射到：
      - Langfuse: trace -> span -> score
      - Braintrust: experiment log -> scores -> metadata

    本地 JSON 的好处是稳定、可回放、无需 API key，适合学习和 CI。
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        scores: dict[str, float | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                name=name,
                input=input or {},
                output=output or {},
                scores=scores or {},
                metadata=metadata or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"events": [asdict(event) for event in self.events]}

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_dataset() -> list[AgentRun]:
    """构造小型离线数据集。

    真实 eval 会用几十到几千条任务，这里保留 4 条代表性样本：
    - good: 正例
    - missing_citation: 结论像对，但缺引用
    - too_long: 内容完整但违反长度约束
    - bad_trajectory: 答案过关，但执行路径跳过关键角色
    """

    task = "写一段 180 字以内的短文：为什么 AI Agent 需要工具调用？"
    return [
        AgentRun(
            case_id="good",
            task=task,
            final_answer=(
                "AI Agent 需要工具调用：模型参数有知识截止日期，实时和私有数据要靠外部工具获取[E1]；"
                "搜索、数据库、计算和代码执行能补齐模型能力边界[E2]；"
                "工具结果可记录、回放和评估，使 Agent 比纯文本推理更可审计[E3]。"
            ),
            route=["researcher", "writer", "critic"],
            tool_observations=["E1", "E2", "E3"],
            revision_count=2,
        ),
        AgentRun(
            case_id="missing_citation",
            task=task,
            final_answer=(
                "AI Agent 需要工具调用，因为模型本身不能访问实时数据、私有系统和精确计算环境。"
                "工具能把搜索、数据库和代码执行接入任务循环，让结果更可靠，也更容易审计。"
            ),
            route=["researcher", "writer", "critic"],
            tool_observations=["E1", "E2", "E3"],
            revision_count=1,
        ),
        AgentRun(
            case_id="too_long",
            task=task,
            final_answer=(
                "AI Agent 需要工具调用：模型参数有知识截止日期，实时和私有数据要靠外部工具获取[E1]；"
                "搜索、数据库、计算和代码执行能补齐模型能力边界[E2]；工具结果可记录、回放和评估，"
                "使 Agent 比纯文本推理更可审计[E3]。同时，工具还能把企业内部系统、权限校验、"
                "状态更新和工作流执行接入 Agent，使它从聊天助手变成能真正完成业务动作的软件系统；"
                "但如果没有评估闭环，团队很难知道这些动作在不同任务、不同工具失败条件下是否稳定可靠。"
            ),
            route=["researcher", "writer", "critic"],
            tool_observations=["E1", "E2", "E3"],
            revision_count=2,
        ),
        AgentRun(
            case_id="bad_trajectory",
            task=task,
            final_answer=(
                "AI Agent 需要工具调用：模型参数有知识截止日期，实时和私有数据要靠外部工具获取[E1]；"
                "搜索、数据库、计算和代码执行能补齐模型能力边界[E2]；"
                "工具结果可记录、回放和评估，使 Agent 比纯文本推理更可审计[E3]。"
            ),
            route=["writer", "writer", "critic", "writer", "critic"],
            tool_observations=[],
            revision_count=4,
        ),
    ]


def evaluate_outcome(run: AgentRun) -> OutcomeScore:
    """检查最终答案是否满足可规则化的验收条件。"""

    missing = [tag for tag in REQUIRED_EVIDENCE if tag not in run.final_answer]
    answer_chars = len(run.final_answer)
    length_ok = answer_chars <= MAX_ANSWER_CHARS
    citations_ok = len(missing) == 0
    return OutcomeScore(
        passed=length_ok and citations_ok,
        length_ok=length_ok,
        citations_ok=citations_ok,
        missing_citations=missing,
        answer_chars=answer_chars,
    )


def contains_subsequence(route: list[str], expected: tuple[str, ...]) -> bool:
    """判断 route 是否按顺序经过 expected 节点。

    不要求完全相等，因为真实 Agent 可能多轮修订；只要求关键阶段顺序正确。
    """

    cursor = 0
    for step in route:
        if cursor < len(expected) and step == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def evaluate_trajectory(run: AgentRun) -> TrajectoryScore:
    """评估执行轨迹是否可控。"""

    notes: list[str] = []
    has_expected_route = contains_subsequence(run.route, EXPECTED_ROUTE)
    no_dead_loop = run.revision_count <= MAX_REVISIONS
    used_tools = len(run.tool_observations) > 0

    if not has_expected_route:
        notes.append("缺少 researcher -> writer -> critic 的关键路径")
    if not no_dead_loop:
        notes.append(f"revision_count={run.revision_count}，超过上限 {MAX_REVISIONS}")
    if not used_tools:
        notes.append("没有记录工具观察，最终答案不可回放")

    return TrajectoryScore(
        passed=has_expected_route and no_dead_loop and used_tools,
        has_expected_route=has_expected_route,
        no_dead_loop=no_dead_loop,
        used_tools=used_tools,
        notes=notes,
    )


def deterministic_judge(run: AgentRun, outcome: OutcomeScore, trajectory: TrajectoryScore) -> JudgeScore:
    """离线版 LLM-as-judge。

    真实 LLM judge 通常会收到 task、answer、trace、rubric，然后输出 JSON 分数。
    为了让本练习在没有 API key 时也能跑，我们把 rubric 写成确定性函数。
    """

    coverage = sum(tag in run.final_answer for tag in REQUIRED_EVIDENCE) / len(REQUIRED_EVIDENCE)
    groundedness = 1.0 if outcome.citations_ok and run.tool_observations else 0.4 if outcome.citations_ok else 0.2
    conciseness = 1.0 if outcome.length_ok else max(0.0, 1 - (outcome.answer_chars - MAX_ANSWER_CHARS) / 120)
    auditability = 1.0 if trajectory.passed else 0.5 if trajectory.has_expected_route else 0.2
    score = statistics.fmean([coverage, groundedness, conciseness, auditability])

    rationale_parts = [
        f"coverage={coverage:.2f}",
        f"groundedness={groundedness:.2f}",
        f"conciseness={conciseness:.2f}",
        f"auditability={auditability:.2f}",
    ]
    if trajectory.notes:
        rationale_parts.append("trajectory_notes=" + " / ".join(trajectory.notes))

    return JudgeScore(
        score=round(score, 3),
        coverage=round(coverage, 3),
        groundedness=round(groundedness, 3),
        conciseness=round(conciseness, 3),
        auditability=round(auditability, 3),
        rationale="; ".join(rationale_parts),
    )


def evaluate_run(run: AgentRun, recorder: LocalTraceRecorder | None = None) -> EvaluationResult:
    """评估单条样本，并把过程写入 trace。"""

    recorder = recorder or LocalTraceRecorder()
    recorder.record(
        "agent.run",
        input={"task": run.task},
        output={"final_answer": run.final_answer, "route": run.route},
        metadata={"case_id": run.case_id, "revision_count": run.revision_count},
    )

    outcome = evaluate_outcome(run)
    recorder.record(
        "eval.outcome",
        input={"final_answer": run.final_answer},
        output=asdict(outcome),
        scores={"outcome_passed": outcome.passed},
        metadata={"case_id": run.case_id},
    )

    trajectory = evaluate_trajectory(run)
    recorder.record(
        "eval.trajectory",
        input={"route": run.route, "tool_observations": run.tool_observations},
        output=asdict(trajectory),
        scores={"trajectory_passed": trajectory.passed},
        metadata={"case_id": run.case_id},
    )

    judge = deterministic_judge(run, outcome, trajectory)
    recorder.record(
        "eval.judge",
        input={"rubric": "coverage + groundedness + conciseness + auditability"},
        output=asdict(judge),
        scores={"judge_score": judge.score},
        metadata={"case_id": run.case_id},
    )

    success = outcome.passed and trajectory.passed and judge.score >= 0.8
    recorder.record(
        "eval.summary",
        output={"success": success},
        scores={"success": success, "judge_score": judge.score},
        metadata={"case_id": run.case_id},
    )

    return EvaluationResult(
        case_id=run.case_id,
        outcome=outcome,
        trajectory=trajectory,
        judge=judge,
        success=success,
    )


def evaluate_dataset(
    runs: list[AgentRun],
    *,
    recorder: LocalTraceRecorder | None = None,
) -> list[EvaluationResult]:
    """评估整个数据集。"""

    recorder = recorder or LocalTraceRecorder()
    return [evaluate_run(run, recorder) for run in runs]


def summarize_results(results: list[EvaluationResult]) -> dict[str, float]:
    """聚合数据集指标，模拟 eval dashboard 的核心数字。"""

    total = len(results)
    success_rate = sum(result.success for result in results) / total
    outcome_rate = sum(result.outcome.passed for result in results) / total
    trajectory_rate = sum(result.trajectory.passed for result in results) / total
    avg_judge_score = statistics.fmean(result.judge.score for result in results)
    return {
        "total": float(total),
        "success_rate": round(success_rate, 3),
        "outcome_pass_rate": round(outcome_rate, 3),
        "trajectory_pass_rate": round(trajectory_rate, 3),
        "avg_judge_score": round(avg_judge_score, 3),
    }


def select_runs(case_id: str | None) -> list[AgentRun]:
    dataset = build_dataset()
    if case_id is None:
        return dataset
    selected = [run for run in dataset if run.case_id == case_id]
    if not selected:
        valid = ", ".join(run.case_id for run in dataset)
        raise SystemExit(f"未知 case: {case_id}，可选值: {valid}")
    return selected


def print_report(results: list[EvaluationResult], summary: dict[str, float]) -> None:
    """打印适合学习和面试复盘的评估报告。"""

    print("=== Agent Eval Report ===")
    print(
        "summary: "
        f"success_rate={summary['success_rate']:.3f}, "
        f"outcome_pass_rate={summary['outcome_pass_rate']:.3f}, "
        f"trajectory_pass_rate={summary['trajectory_pass_rate']:.3f}, "
        f"avg_judge_score={summary['avg_judge_score']:.3f}"
    )
    print()

    for result in results:
        print(f"[{result.case_id}] success={result.success} judge={result.judge.score:.3f}")
        print(
            "  outcome: "
            f"passed={result.outcome.passed}, "
            f"chars={result.outcome.answer_chars}, "
            f"missing={result.outcome.missing_citations or '-'}"
        )
        print(
            "  trajectory: "
            f"passed={result.trajectory.passed}, "
            f"notes={result.trajectory.notes or '-'}"
        )
        print(f"  judge rationale: {result.judge.rationale}")


def self_test() -> None:
    recorder = LocalTraceRecorder()
    results = evaluate_dataset(build_dataset(), recorder=recorder)
    summary = summarize_results(results)

    by_case = {result.case_id: result for result in results}
    assert by_case["good"].success is True
    assert by_case["missing_citation"].outcome.passed is False
    assert "[E1]" in by_case["missing_citation"].outcome.missing_citations
    assert by_case["too_long"].outcome.length_ok is False
    assert by_case["bad_trajectory"].trajectory.passed is False
    assert summary["success_rate"] == 0.25
    assert any(event.name == "eval.judge" for event in recorder.events)
    print("✅ self-test passed: outcome, trajectory, judge, trace recorder all work.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 评估脚本 v1：成功率 + 轨迹 + judge + trace")
    parser.add_argument("--case", choices=[run.case_id for run in build_dataset()], help="只评估某个样本")
    parser.add_argument("--trace-json", type=Path, help="导出本地 trace JSON，结构可映射到 Langfuse/Braintrust")
    parser.add_argument("--self-test", action="store_true", help="运行离线断言测试")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    recorder = LocalTraceRecorder()
    runs = select_runs(args.case)
    results = evaluate_dataset(runs, recorder=recorder)
    summary = summarize_results(results)
    print_report(results, summary)

    if args.trace_json:
        recorder.save_json(args.trace_json)
        print(f"\ntrace saved: {args.trace_json}")


if __name__ == "__main__":
    main()
