"""
Agent hands-on task - 2026-05-22 (Week 3 Friday)
================================================
Topic: Coding Agent design + SGLang project interview story outline

This day is less about building another runtime agent and more about turning
coding-agent papers/blog ideas into an interview-ready engineering story.

The script keeps the outline deterministic and offline:
  uv run python exercises/w3d5-coding-agent-story/agent.py
  uv run python exercises/w3d5-coding-agent-story/agent.py --section pitch
  uv run python exercises/w3d5-coding-agent-story/agent.py --format json
  uv run python exercises/w3d5-coding-agent-story/agent.py --export-md exercises/w3d5-coding-agent-story/STORY.md
  uv run python exercises/w3d5-coding-agent-story/agent.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SectionName = Literal[
    "principles",
    "story",
    "failure_modes",
    "safety",
    "pitch",
    "rehearsal",
]


@dataclass(frozen=True)
class CodingAgentPrinciple:
    """A paper/blog idea translated into an engineering talking point."""

    name: str
    source_anchor: str
    interview_point: str
    sglang_mapping: str


@dataclass(frozen=True)
class ProjectEvidence:
    """Concrete evidence that makes the SGLang story credible."""

    theme: str
    evidence: str
    why_it_matters: str


@dataclass(frozen=True)
class FailureMode:
    """A realistic agent failure mode and the human control used to fix it."""

    risk: str
    example: str
    control: str


@dataclass(frozen=True)
class StorySection:
    title: str
    bullets: list[str]


@dataclass(frozen=True)
class StoryPack:
    principles: list[CodingAgentPrinciple]
    evidence: list[ProjectEvidence]
    star_story: list[StorySection]
    failure_modes: list[FailureMode]
    safety_notes: list[str]
    pitch_90s: str
    rehearsal_prompts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_principles() -> list[CodingAgentPrinciple]:
    return [
        CodingAgentPrinciple(
            name="Agent-Computer Interface",
            source_anchor="SWE-agent ACI",
            interview_point=(
                "Coding agent 的能力不只来自模型本身，还来自它能看到什么上下文、"
                "能执行哪些命令、观察结果如何回流，以及错误信息是否足够可行动。"
            ),
            sglang_mapping=(
                "在 SGLang 量化适配中，我会让 agent 先用搜索理解现有 FP8/W8A8 "
                "抽象，再把改动限制在清晰的接口边界内。"
            ),
        ),
        CodingAgentPrinciple(
            name="Long-horizon workspace",
            source_anchor="Devin-style workflow",
            interview_point=(
                "长任务 agent 需要持续维护计划、代码状态、测试反馈和待办，而不是每轮重新开始。"
            ),
            sglang_mapping=(
                "MXFP4/MXFP8 适配横跨模型加载、kernel contract、离线/在线量化和测试，"
                "我把任务拆成可验证阶段，让 agent 每步产出都能被 review。"
            ),
        ),
        CodingAgentPrinciple(
            name="Tests as environment feedback",
            source_anchor="Coding-agent eval loop",
            interview_point=(
                "coding agent 的关键闭环是执行测试、读取失败、修复，再运行更小范围的验证。"
            ),
            sglang_mapping=(
                "我不会只看 agent 写出的代码是否像样，而是要求它补单测/小规模 e2e，"
                "并用 trace 或 CI 结果定位 scale、shape、dtype 这类边界问题。"
            ),
        ),
        CodingAgentPrinciple(
            name="Human-led architecture",
            source_anchor="Practical agent use",
            interview_point=(
                "agent 擅长探索和局部实现，但架构抽象、性能权衡、跨模块协作仍要人主导。"
            ),
            sglang_mapping=(
                "我把 agent 当成高效初级工程师：给清晰 spec、让它探索和实现，"
                "但由我决定抽象边界、性能方向和最终 merge 标准。"
            ),
        ),
    ]


def build_evidence() -> list[ProjectEvidence]:
    return [
        ProjectEvidence(
            theme="清晰任务规约",
            evidence=(
                "先写 task spec：目标是把 MXFP4/MXFP8 适配到 SGLang 的不同模型路径，"
                "约束是不破坏既有 W8A8/FP8 路径，并保留在线/离线量化可扩展性。"
            ),
            why_it_matters="这能降低 agent 过度发散和误改公共抽象的概率。",
        ),
        ProjectEvidence(
            theme="代码库探索",
            evidence=(
                "让 agent 用搜索和文件阅读梳理量化模块、kernel contract、"
                "dense 与 MoE 路径的差异，再输出修改计划。"
            ),
            why_it_matters="真实大型仓库里，先理解现有抽象比直接写代码更重要。",
        ),
        ProjectEvidence(
            theme="逐步 review",
            evidence=(
                "实现过程中按阶段 review：模型侧入口、scale 处理、kernel 调用参数、"
                "测试覆盖分别检查，而不是等最后一次性验收。"
            ),
            why_it_matters="早期方向错了会让后续代码全部返工，阶段性 review 更稳。",
        ),
        ProjectEvidence(
            theme="跨场景复用",
            evidence=(
                "做完一个 diffusion/MXFP 路径后，把同一套适配思路推广到 Dense LLM 和 MoE LLM。"
            ),
            why_it_matters="这体现 agent 在模式复用、重复工程和测试生成上的效率优势。",
        ),
    ]


def build_star_story() -> list[StorySection]:
    return [
        StorySection(
            title="Situation",
            bullets=[
                "SGLang 量化适配涉及模型加载、scale 布局、NPU kernel contract 和多模型路径。",
                "任务不是单点 bugfix，而是需要在复杂代码库中稳定扩展量化能力。",
            ],
        ),
        StorySection(
            title="Task",
            bullets=[
                "目标是把 MXFP4/MXFP8 相关能力接入既有架构，并尽量复用已有 FP8/W8A8 模式。",
                "同时要保证可测试、可 review，不能为了快而破坏现有推理路径。",
            ],
        ),
        StorySection(
            title="Action",
            bullets=[
                "我先写清楚 spec：目标、约束、参考 PR、不得改动的边界和验收命令。",
                "让 agent 探索代码库并解释现有抽象，我通过它的 trace 判断它是否真的理解。",
                "实现时拆成小步：入口、参数转换、kernel contract、测试和文档分别推进。",
                "我持续 review 架构和性能方向，让 agent 做局部实现、测试补全和文档同步。",
            ],
        ),
        StorySection(
            title="Result",
            bullets=[
                "agent 显著提升了复杂仓库探索、重复模式迁移和测试生成效率。",
                "我也明确了边界：架构决策、性能瓶颈定位、跨模块协调仍需要工程师主导。",
                "这段经历让我能把 coding agent 讲成真实工程 workflow，而不是泛泛说提升效率。",
            ],
        ),
    ]


def build_failure_modes() -> list[FailureMode]:
    return [
        FailureMode(
            risk="过度抽象",
            example="agent 为了显得优雅，把简单 dtype/scale 分支包装成多层新抽象。",
            control="要求它用最小改动复用现有模式，并说明为什么需要新增抽象。",
        ),
        FailureMode(
            risk="边界 case 漏测",
            example="W4A4 与 W4A8 的 scale 处理不同，第一版容易混在一起。",
            control="把 shape、dtype、scale layout 写进测试矩阵，并要求失败样例可复现。",
        ),
        FailureMode(
            risk="性能直觉不足",
            example="逻辑正确但 kernel 参数组织导致额外转换或内存开销。",
            control="用 profiling 或小规模 benchmark 定位瓶颈，再把具体问题交给 agent 修复。",
        ),
        FailureMode(
            risk="权限与副作用失控",
            example="coding agent 自动执行大范围改动或触碰无关文件。",
            control="限定工作目录、明确禁止改动路径，提交前只 stage 目标文件。",
        ),
    ]


def build_safety_notes() -> list[str]:
    return [
        "Prompt injection: 代码库、issue、日志里的文本都可能诱导 agent 忽略原始指令；需要把外部文本当作数据而不是指令。",
        "Excessive agency: 自动提交、发布、删除分支、改 CI 配置这类动作必须有人确认。",
        "Sensitive information disclosure: `.env`、token、内部 benchmark 结果默认不能进入 prompt、trace 或 commit。",
        "Supply-chain risk: agent 建议新增依赖时要检查维护状态、许可证、安装来源和是否真的必要。",
        "Insecure output handling: agent 生成的 shell、SQL、Python 片段要审查执行边界，尤其是删除、网络和凭据操作。",
    ]


def build_pitch() -> str:
    return (
        "我在 SGLang 量化适配里使用 coding agent 的方式，不是让它替我做架构决策，"
        "而是把它放在一个清晰的工程闭环里。第一步我会写 task spec，明确目标、约束、"
        "参考实现和不能碰的边界；第二步让 agent 搜索代码库，解释现有量化抽象和 kernel contract，"
        "我通过它的 trace 判断它是否真的理解；第三步按模型入口、scale 处理、kernel 调用、测试和文档拆成小步实现，"
        "每一步我都 review。这样做的收益是复杂仓库探索、重复模式迁移和测试生成会快很多。"
        "但我也遇到过失败模式，比如过度抽象、边界 case 漏测、性能直觉不足，所以我会用最小改动约束、"
        "测试矩阵和 profiling 反馈来控制它。我的结论是：coding agent 不能替代工程师判断，"
        "但在 well-defined、可验证的工程任务上，可以非常明显地放大工程师产出。"
    )


def build_rehearsal_prompts() -> list[str]:
    return [
        "如果面试官问：你怎么保证 coding agent 不乱改？先回答 spec、边界、stage 范围、review 和测试。",
        "如果面试官追问：agent 犯过什么错？用过度抽象、W4A4/W4A8 scale 混淆、性能直觉不足三个例子。",
        "如果面试官问：你和普通 Copilot 用户有什么不同？强调任务分解、trace review、测试闭环和架构边界由人主导。",
        "如果面试官问：为什么这和 SWE-agent/Devin 有关？回答 ACI、长任务 workspace、环境反馈和人类验收。",
    ]


def build_story_pack() -> StoryPack:
    return StoryPack(
        principles=build_principles(),
        evidence=build_evidence(),
        star_story=build_star_story(),
        failure_modes=build_failure_modes(),
        safety_notes=build_safety_notes(),
        pitch_90s=build_pitch(),
        rehearsal_prompts=build_rehearsal_prompts(),
    )


def render_markdown(pack: StoryPack, *, section: SectionName | None = None) -> str:
    lines: list[str] = ["# SGLang Coding Agent 面试故事大纲", ""]

    if section in (None, "principles"):
        lines.extend(["## 1. Coding Agent 设计原则", ""])
        for item in pack.principles:
            lines.extend(
                [
                    f"### {item.name}",
                    f"- 来源锚点: {item.source_anchor}",
                    f"- 面试表达: {item.interview_point}",
                    f"- SGLang 映射: {item.sglang_mapping}",
                    "",
                ]
            )

    if section in (None, "story"):
        lines.extend(["## 2. STAR 故事主线", ""])
        for story_section in pack.star_story:
            lines.append(f"### {story_section.title}")
            lines.extend(f"- {bullet}" for bullet in story_section.bullets)
            lines.append("")

        lines.extend(["## 3. 项目证据", ""])
        for evidence in pack.evidence:
            lines.extend(
                [
                    f"### {evidence.theme}",
                    f"- 证据: {evidence.evidence}",
                    f"- 价值: {evidence.why_it_matters}",
                    "",
                ]
            )

    if section in (None, "failure_modes"):
        lines.extend(["## 4. 失败模式与控制", ""])
        for failure in pack.failure_modes:
            lines.extend(
                [
                    f"### {failure.risk}",
                    f"- 例子: {failure.example}",
                    f"- 控制: {failure.control}",
                    "",
                ]
            )

    if section in (None, "safety"):
        lines.extend(["## 5. Agent 安全速记", ""])
        lines.extend(f"- {note}" for note in pack.safety_notes)
        lines.append("")

    if section in (None, "pitch"):
        lines.extend(["## 6. 90 秒背诵版", "", pack.pitch_90s, ""])

    if section in (None, "rehearsal"):
        lines.extend(["## 7. 追问演练", ""])
        lines.extend(f"- {prompt}" for prompt in pack.rehearsal_prompts)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def validate_story_pack(pack: StoryPack) -> list[str]:
    errors: list[str] = []
    required_pitch_terms = ["task spec", "trace", "测试", "review", "SGLang"]
    for term in required_pitch_terms:
        if term not in pack.pitch_90s:
            errors.append(f"90 秒背诵版缺少关键词: {term}")

    if len(pack.principles) < 4:
        errors.append("Coding Agent 设计原则少于 4 条")
    if len(pack.failure_modes) < 3:
        errors.append("失败模式少于 3 条")
    if not any("Prompt injection" in note for note in pack.safety_notes):
        errors.append("安全速记缺少 prompt injection")
    if not any(section.title == "Action" and len(section.bullets) >= 3 for section in pack.star_story):
        errors.append("STAR Action 部分不够具体")

    return errors


def self_test() -> None:
    pack = build_story_pack()
    errors = validate_story_pack(pack)
    assert not errors, "; ".join(errors)

    markdown = render_markdown(pack)
    assert "# SGLang Coding Agent 面试故事大纲" in markdown
    assert "90 秒背诵版" in markdown
    assert "W4A4" in markdown and "W4A8" in markdown
    assert len(pack.pitch_90s) <= 520
    print("✅ self-test passed: story pack, safety notes, pitch, and markdown rendering are valid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang 项目 Coding Agent 面试故事大纲生成器")
    parser.add_argument(
        "--section",
        choices=["principles", "story", "failure_modes", "safety", "pitch", "rehearsal"],
        help="只输出某个章节",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="输出格式")
    parser.add_argument("--export-md", type=Path, help="把完整 Markdown 大纲写入指定文件")
    parser.add_argument("--self-test", action="store_true", help="运行离线断言测试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    pack = build_story_pack()
    errors = validate_story_pack(pack)
    if errors:
        raise SystemExit("故事大纲校验失败:\n" + "\n".join(f"- {error}" for error in errors))

    if args.export_md:
        args.export_md.parent.mkdir(parents=True, exist_ok=True)
        args.export_md.write_text(render_markdown(pack), encoding="utf-8")
        print(f"story outline exported: {args.export_md}")
        return

    if args.format == "json":
        print(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2))
        return

    print(render_markdown(pack, section=args.section))


if __name__ == "__main__":
    main()
