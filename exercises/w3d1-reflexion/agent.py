"""
Agent 动手任务 - 2026-05-18 (第3周 周一)
============================================
主题：Reflexion + Self-Refine
工具：离线模拟 LLM 生成、批判、反思、再生成循环

核心概念（配合八股题 5、6、7、8 食用）：
  - Reflexion：把失败轨迹转成文字反思，写入记忆，下一轮显式引用
  - Self-Refine：同一个模型拆成 generator / critic / refiner 三个角色
  - 与 ReAct 的关系：ReAct 管「想-做-观察」；Reflexion 管「失败后如何学」
  - 工程边界：反思必须可执行、可验证，不能只是“下次更仔细”

运行方式：
  uv run python exercises/w3d1-reflexion/agent.py
  uv run python exercises/w3d1-reflexion/agent.py --explain
  uv run python exercises/w3d1-reflexion/agent.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TASK = "用 120 字以内回答：如何降低 RAG Agent 的幻觉？"

RUBRIC = {
    "grounding": "要求回答必须提到基于检索证据生成，而不是只依赖模型记忆",
    "citation": "要求回答必须提到引用、证据片段或来源检查",
    "fallback": "要求回答必须提到检索不足时拒答、追问或降级",
    "evaluation": "要求回答必须提到评估闭环，如 groundedness / faithfulness",
    "concise": "要求回答不超过 120 个中文字符",
}


@dataclass(frozen=True)
class Critique:
    score: int
    missing: list[str]
    notes: list[str]


@dataclass(frozen=True)
class Reflection:
    lessons: list[str]


@dataclass(frozen=True)
class Attempt:
    turn: int
    answer: str
    critique: Critique
    reflection: Reflection


def generate_initial_answer(task: str) -> str:
    """模拟第一次回答：方向正确，但缺少可验证约束。"""
    return "降低幻觉要让 Agent 使用 RAG，先检索知识库，再把上下文交给模型回答。"


def critique_answer(answer: str) -> Critique:
    """用确定性 rubric 扮演 critic，避免本练习依赖真实 API。"""
    missing: list[str] = []
    notes: list[str] = []

    checks = {
        "grounding": ["证据", "检索", "上下文", "知识库"],
        "citation": ["引用", "来源", "证据片段"],
        "fallback": ["拒答", "追问", "降级", "不足"],
        "evaluation": ["评估", "groundedness", "faithfulness", "回归"],
    }
    for key, keywords in checks.items():
        if not any(word in answer for word in keywords):
            missing.append(key)
            notes.append(RUBRIC[key])

    if len(answer) > 120:
        missing.append("concise")
        notes.append(RUBRIC["concise"])

    score = max(0, 5 - len(missing))
    return Critique(score=score, missing=missing, notes=notes)


def reflect(critique: Critique, old_lessons: list[str]) -> Reflection:
    """把 critic 的缺口改写成下一轮可执行的记忆。"""
    lesson_templates = {
        "grounding": "回答 RAG 可靠性问题时，先强调答案必须绑定检索证据。",
        "citation": "最终答案要包含引用/来源校验，方便用户追踪证据。",
        "fallback": "检索证据不足时不要硬答，应拒答、追问或触发降级策略。",
        "evaluation": "把离线评估或线上回归写进闭环，避免只靠 prompt 约束。",
        "concise": "先删泛泛而谈的形容词，保留工程动作。",
    }
    lessons = list(old_lessons)
    for key in critique.missing:
        lesson = lesson_templates[key]
        if lesson not in lessons:
            lessons.append(lesson)
    return Reflection(lessons=lessons)


def refine_answer(previous_answer: str, reflection: Reflection) -> str:
    """模拟 refiner：只根据反思记忆补齐缺口。"""
    draft = "降低 RAG Agent 幻觉：先用高召回检索取证据，再让模型只基于证据作答。"

    if any("引用" in lesson or "来源" in lesson for lesson in reflection.lessons):
        draft += "输出附引用或证据片段。"
    if any("不足" in lesson or "拒答" in lesson for lesson in reflection.lessons):
        draft += "证据不足时拒答、追问或降级。"
    if any("评估" in lesson or "回归" in lesson for lesson in reflection.lessons):
        draft += "用 groundedness/faithfulness 回归评估闭环。"

    return draft[:120]


def run_reflexion(task: str = TASK, max_turns: int = 3) -> list[Attempt]:
    answer = generate_initial_answer(task)
    lessons: list[str] = []
    attempts: list[Attempt] = []

    for turn in range(1, max_turns + 1):
        critique = critique_answer(answer)
        reflection = reflect(critique, lessons)
        attempts.append(Attempt(turn, answer, critique, reflection))
        lessons = reflection.lessons

        if critique.score == 5:
            break
        answer = refine_answer(answer, reflection)

    return attempts


def print_trace(attempts: list[Attempt]) -> None:
    print(f"任务：{TASK}\n")
    for attempt in attempts:
        print(f"=== 第 {attempt.turn} 轮 ===")
        print(f"答案：{attempt.answer}")
        print(f"评分：{attempt.critique.score}/5")
        if attempt.critique.missing:
            print("缺口：")
            for note in attempt.critique.notes:
                print(f"- {note}")
        else:
            print("缺口：无")
        print("反思记忆：")
        for lesson in attempt.reflection.lessons or ["无"]:
            print(f"- {lesson}")
        print()


def explain() -> None:
    print(
        "\n".join([
            "Reflexion 不是让模型把同一题再想一遍，而是把失败原因写成可复用记忆。",
            "Self-Refine 把一次回答拆成 generator -> critic -> refiner。",
            "生产实现时要注意三点：",
            "1. critic 要有 rubric，否则反思会变成空话。",
            "2. reflection 要能转成下一轮约束，例如必须引用证据、证据不足要拒答。",
            "3. 每轮都要有停止条件，避免无意义自我循环。",
        ])
    )


def self_test() -> None:
    attempts = run_reflexion()
    final = attempts[-1]
    assert final.critique.score == 5
    assert "引用" in final.answer or "证据片段" in final.answer
    assert "拒答" in final.answer or "追问" in final.answer
    assert "groundedness" in final.answer
    print("✅ self-test passed: Reflexion loop converged to a rubric-complete answer.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflexion + Self-Refine 离线练习")
    parser.add_argument("--explain", action="store_true", help="打印核心概念速记")
    parser.add_argument("--self-test", action="store_true", help="运行离线断言测试")
    args = parser.parse_args()

    if args.explain:
        explain()
    elif args.self_test:
        self_test()
    else:
        print_trace(run_reflexion())


if __name__ == "__main__":
    main()
