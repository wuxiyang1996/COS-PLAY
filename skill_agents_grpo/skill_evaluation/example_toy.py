"""
Toy example demonstrating the LLM-agentic Skill Evaluation module.

Creates a small synthetic skill bank with three skills of varying quality,
runs the full evaluation pipeline, and prints the results.

By default uses a mock LLM for offline testing.  To use a real LLM, pass
``--live`` and ensure ``API_func.ask_model`` is configured.

Usage::

    python -m skill_agents.skill_evaluation.example_toy          # mock LLM
    python -m skill_agents.skill_evaluation.example_toy --live   # real LLM
"""

from __future__ import annotations

import argparse
import json
import logging
import re

from skill_agents_grpo.skill_bank.bank import SkillBankMVP
from skill_agents_grpo.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents_grpo.skill_evaluation import run_skill_evaluation, SkillEvaluationConfig
from skill_agents_grpo.skill_evaluation.config import LLMJudgeConfig


# ── Mock LLM for offline testing ─────────────────────────────────────

def _detect_target_skill(prompt: str) -> str:
    """Identify which skill the prompt is evaluating (the target, not peers)."""
    m = re.search(r"Skill ID:\s*(\S+)", prompt)
    return m.group(1) if m else ""


def _mock_ask_model(prompt: str, **kwargs) -> str:
    """Return a plausible JSON response based on prompt keywords."""
    prompt_lower = prompt.lower()
    target = _detect_target_skill(prompt)

    if "coherence" in prompt_lower:
        if target == "misc_action":
            return json.dumps({
                "score": 3,
                "evidence": [
                    "Instances show 4 completely different effect signatures",
                    "The skill conflates pick-up, navigation, and cooking actions",
                    "Support counts are uneven — some effects only fire for 1/4 instances",
                ],
                "issues": ["Multiple unrelated behaviors under one label"],
                "recommendation": "SPLIT",
            })
        elif target == "pick_up":
            return json.dumps({
                "score": 9,
                "evidence": [
                    "All 9 instances produce holding(obj) and remove on_table(obj)",
                    "Durations are consistent at ~8 timesteps",
                    "The only variation is whether grasp_event fires (67% of instances)",
                ],
                "issues": [],
                "recommendation": "KEEP",
            })
        else:
            return json.dumps({
                "score": 8,
                "evidence": [
                    "All instances produce at(dest) and remove at(src)",
                    "Consistent duration and effect pattern",
                ],
                "issues": [],
                "recommendation": "KEEP",
            })

    if "discriminability" in prompt_lower:
        if target == "misc_action":
            return json.dumps({
                "score": 3,
                "evidence": [
                    "Shares holding(obj), on_table(obj) with pick_up",
                    "Shares at(dest), at(src) with navigate_to",
                    "Only unique effect is cooked(dish) and ui_click",
                ],
                "issues": ["High overlap with both pick_up and navigate_to"],
                "recommendation": "SPLIT",
            })
        elif target == "navigate_to":
            return json.dumps({
                "score": 6,
                "evidence": [
                    "at(dest) and at(src) are fairly common predicates",
                    "But the consistent pattern distinguishes it from misc_action",
                ],
                "issues": [],
                "recommendation": "KEEP",
            })
        else:
            return json.dumps({
                "score": 8,
                "evidence": [
                    "holding(obj) + on_table(obj) + grasp_event is a unique combination",
                    "No other skill produces grasp_event",
                ],
                "issues": [],
                "recommendation": "KEEP",
            })

    if "composability" in prompt_lower:
        return json.dumps({
            "score": 7,
            "evidence": [
                "Has clear predecessors and successors",
                "Some self-transition observed but not excessive",
            ],
            "issues": [],
            "recommendation": "KEEP",
        })

    if "generalization" in prompt_lower:
        if target == "misc_action":
            return json.dumps({
                "score": 4,
                "evidence": [
                    "Only appears in 2 trajectories with 4 total instances",
                    "Pass rate of 0.30 suggests the contract barely fits the data",
                ],
                "issues": ["Low instance count and poor pass rate"],
                "recommendation": "REFINE",
            })
        return json.dumps({
            "score": 8,
            "evidence": [
                "Appears across multiple trajectories with consistent pass rates",
            ],
            "issues": [],
            "recommendation": "KEEP",
        })

    if "utility" in prompt_lower:
        if target == "misc_action":
            return json.dumps({
                "score": 4,
                "evidence": [
                    "6 effects but they are incoherent — not a focused sub-task",
                    "Present in successful episodes but not clearly causal",
                ],
                "issues": ["Effect magnitude is misleading due to incoherence"],
                "recommendation": "REFINE",
            })
        return json.dumps({
            "score": 8,
            "evidence": [
                "Produces meaningful state changes that advance the task",
                "Present in successful episodes",
            ],
            "issues": [],
            "recommendation": "KEEP",
        })

    if "granularity" in prompt_lower:
        if target == "misc_action":
            return json.dumps({
                "score": 4,
                "evidence": [
                    "Duration of 3 timesteps is reasonable but 6 literals is high for such a short skill",
                    "Likely conflates multiple granularities",
                ],
                "issues": ["Over-specified for its duration"],
                "recommendation": "SPLIT",
            })
        return json.dumps({
            "score": 9,
            "evidence": [
                "Duration is appropriate for a reusable sub-task",
                "Contract size is well-calibrated",
            ],
            "issues": [],
            "recommendation": "KEEP",
        })

    # Holistic pass
    if "holistic" in prompt_lower or "synthesise" in prompt_lower:
        if target == "misc_action":
            return json.dumps({
                "score": 3,
                "evidence": [
                    "Low coherence and discriminability dominate",
                    "The skill conflates unrelated behaviors",
                ],
                "recommendation": "SPLIT",
                "merge_with": None,
                "reasoning": (
                    "This skill is a grab-bag of pick-up, navigation, and cooking "
                    "actions. It should be decomposed into its constituent behaviors."
                ),
            })
        elif target == "navigate_to":
            return json.dumps({
                "score": 7,
                "evidence": [
                    "Good coherence and generalization",
                    "Moderate discriminability due to overlap with misc_action",
                ],
                "recommendation": "KEEP",
                "merge_with": None,
                "reasoning": (
                    "A solid navigation skill. Once misc_action is split, "
                    "discriminability should improve."
                ),
            })
        else:
            return json.dumps({
                "score": 9,
                "evidence": [
                    "Excellent coherence, discriminability, and generalization",
                    "A well-formed pick-and-place skill",
                ],
                "recommendation": "KEEP",
                "merge_with": None,
                "reasoning": "High-quality skill ready for downstream use.",
            })

    return json.dumps({
        "score": 5,
        "evidence": ["Default mock response"],
        "issues": [],
        "recommendation": "KEEP",
    })


# ── Toy data builders ────────────────────────────────────────────────

def _make_segment(
    seg_id: str,
    traj_id: str,
    t_start: int,
    t_end: int,
    skill_label: str,
    eff_add: set | None = None,
    eff_del: set | None = None,
    eff_event: set | None = None,
) -> SegmentRecord:
    return SegmentRecord(
        seg_id=seg_id,
        traj_id=traj_id,
        t_start=t_start,
        t_end=t_end,
        skill_label=skill_label,
        eff_add=eff_add or set(),
        eff_del=eff_del or set(),
        eff_event=eff_event or set(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Evaluation toy example")
    parser.add_argument("--live", action="store_true", help="Use real LLM via API_func")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    bank = SkillBankMVP()

    # ── Skill A: high-quality pick-and-place skill ────────────────
    contract_a = SkillEffectsContract(
        skill_id="pick_up", version=2,
        eff_add={"holding(obj)"}, eff_del={"on_table(obj)"},
        eff_event={"grasp_event"},
        support={"holding(obj)": 8, "on_table(obj)": 8, "grasp_event": 7},
        n_instances=8,
    )
    report_a = VerificationReport(
        skill_id="pick_up", n_instances=8,
        eff_add_success_rate={"holding(obj)": 0.95},
        eff_del_success_rate={"on_table(obj)": 0.90},
        eff_event_rate={"grasp_event": 0.88},
        overall_pass_rate=0.88,
    )
    bank.add_or_update(contract_a, report_a)

    # ── Skill B: medium-quality navigation skill ─────────────────
    contract_b = SkillEffectsContract(
        skill_id="navigate_to", version=1,
        eff_add={"at(dest)"}, eff_del={"at(src)"},
        support={"at(dest)": 5, "at(src)": 4},
        n_instances=5,
    )
    report_b = VerificationReport(
        skill_id="navigate_to", n_instances=5,
        eff_add_success_rate={"at(dest)": 0.70},
        eff_del_success_rate={"at(src)": 0.60},
        overall_pass_rate=0.60,
    )
    bank.add_or_update(contract_b, report_b)

    # ── Skill C: low-quality / incoherent skill ──────────────────
    contract_c = SkillEffectsContract(
        skill_id="misc_action", version=1,
        eff_add={"holding(obj)", "at(dest)", "cooked(dish)"},
        eff_del={"on_table(obj)", "at(src)"},
        eff_event={"ui_click"},
        support={
            "holding(obj)": 2, "at(dest)": 1, "cooked(dish)": 1,
            "on_table(obj)": 2, "at(src)": 1, "ui_click": 3,
        },
        n_instances=4,
    )
    report_c = VerificationReport(
        skill_id="misc_action", n_instances=4,
        eff_add_success_rate={"holding(obj)": 0.40, "at(dest)": 0.25, "cooked(dish)": 0.25},
        eff_del_success_rate={"on_table(obj)": 0.30, "at(src)": 0.25},
        eff_event_rate={"ui_click": 0.50},
        overall_pass_rate=0.30,
        worst_segments=["seg_c1", "seg_c3"],
    )
    bank.add_or_update(contract_c, report_c)

    # ── Build synthetic segments ──────────────────────────────────
    segments: list[SegmentRecord] = []

    for t in range(3):
        for i in range(3):
            segments.append(_make_segment(
                f"seg_a_{t}_{i}", f"traj_{t}",
                t_start=i * 10, t_end=i * 10 + 8,
                skill_label="pick_up",
                eff_add={"holding(obj)"}, eff_del={"on_table(obj)"},
                eff_event={"grasp_event"} if i % 2 == 0 else set(),
            ))

    for t in range(2):
        for i in range(3):
            segments.append(_make_segment(
                f"seg_b_{t}_{i}", f"traj_{t}",
                t_start=30 + i * 15, t_end=30 + i * 15 + 12,
                skill_label="navigate_to",
                eff_add={"at(dest)"}, eff_del={"at(src)"},
            ))

    sigs = [
        ({"holding(obj)"}, {"on_table(obj)"}, {"ui_click"}),
        ({"at(dest)"}, {"at(src)"}, set()),
        ({"cooked(dish)"}, set(), {"ui_click"}),
        ({"holding(obj)"}, set(), {"ui_click"}),
    ]
    for i, (ea, ed, ee) in enumerate(sigs):
        segments.append(_make_segment(
            f"seg_c_{i}", f"traj_{i % 2}",
            t_start=80 + i * 5, t_end=80 + i * 5 + 3,
            skill_label="misc_action",
            eff_add=ea, eff_del=ed, eff_event=ee,
        ))

    episode_outcomes = {"traj_0": True, "traj_1": True, "traj_2": False}

    # ── Configure ────────────────────────────────────────────────
    llm_cfg = LLMJudgeConfig()
    if not args.live:
        llm_cfg.ask_model_fn = _mock_ask_model

    config = SkillEvaluationConfig(llm=llm_cfg)

    # ── Run evaluation ───────────────────────────────────────────
    summary = run_skill_evaluation(
        bank, segments,
        config=config,
        episode_outcomes=episode_outcomes,
    )

    # ── Print results ────────────────────────────────────────────
    print()
    print(summary.format_for_llm())
    print()

    for skill_id, report in sorted(summary.skill_reports.items()):
        print(report.format_for_llm())
        print()


if __name__ == "__main__":
    main()
