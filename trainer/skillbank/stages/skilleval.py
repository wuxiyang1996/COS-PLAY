"""
SkillEval: quality gating for bank updates.

Evaluates proposed bank changes on multiple criteria:
  - Pass rate: fraction of segment instances satisfying the contract
  - Support: minimum instance count per skill
  - Discriminability: margin between best and runner-up skill
  - Complexity: contract size (number of literals)

Returns accept/reject decision with diagnostic report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.skillbank.stages.stage2_decode import DecodeResult, NEW_LABEL
from trainer.skillbank.stages.stage3_contracts import ContractLearningResult

logger = logging.getLogger(__name__)


@dataclass
class SkillEvalConfig:
    """Gating thresholds for SkillEval."""

    min_pass_rate: float = 0.6
    min_support: int = 3
    max_new_rate: float = 0.3
    margin_regression_tol: float = 0.1
    confusion_threshold: float = 0.3
    max_literals_per_skill: int = 50


@dataclass
class SkillReport:
    """Per-skill quality report."""

    skill_id: str
    pass_rate: float = 0.0
    support: int = 0
    mean_margin: float = 0.0
    n_confusers: int = 0
    n_literals: int = 0
    issues: List[str] = field(default_factory=list)


@dataclass
class SkillEvalResult:
    """Result of SkillEval gating."""

    accepted: bool = True
    overall_pass_rate: float = 0.0
    new_rate: float = 0.0
    mean_margin: float = 0.0
    per_skill: List[SkillReport] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "overall_pass_rate": self.overall_pass_rate,
            "new_rate": self.new_rate,
            "mean_margin": self.mean_margin,
            "n_skills_evaluated": len(self.per_skill),
            "rejection_reasons": self.rejection_reasons,
        }


def evaluate_bank_update(
    decode_results: List[DecodeResult],
    contract_result: ContractLearningResult,
    bank: Any,
    config: Optional[SkillEvalConfig] = None,
    prev_metrics: Optional[Dict[str, float]] = None,
) -> SkillEvalResult:
    """Evaluate whether a proposed bank update should be accepted.

    Args:
        decode_results: decode output from Stage 2
        contract_result: contract learning output from Stage 3
        bank: proposed bank (after updates)
        config: gating thresholds
        prev_metrics: metrics from the previous bank version (for regression detection)

    Returns:
        SkillEvalResult with accept/reject decision.
    """
    cfg = config or SkillEvalConfig()
    result = SkillEvalResult()

    total_segments = 0
    new_segments = 0
    all_margins: List[float] = []
    all_pass_rates: List[float] = []

    confusion_map: Dict[str, List[str]] = {}

    for dr in decode_results:
        for seg in dr.segments:
            total_segments += 1
            if seg.skill_label == NEW_LABEL:
                new_segments += 1
            all_margins.append(seg.margin)
            if seg.confusers:
                confusion_map.setdefault(seg.skill_label, []).extend(seg.confusers)

    result.new_rate = new_segments / max(total_segments, 1)
    result.mean_margin = sum(all_margins) / len(all_margins) if all_margins else 0.0

    for skill_id, lc in contract_result.contracts.items():
        report = SkillReport(
            skill_id=skill_id,
            pass_rate=lc.pass_rate,
            support=lc.n_instances,
            n_literals=len(lc.eff_add) + len(lc.eff_del),
        )

        skill_margins = [
            seg.margin for dr in decode_results
            for seg in dr.segments if seg.skill_label == skill_id
        ]
        report.mean_margin = sum(skill_margins) / len(skill_margins) if skill_margins else 0.0
        report.n_confusers = len(set(confusion_map.get(skill_id, [])))

        if lc.pass_rate < cfg.min_pass_rate:
            report.issues.append(f"pass_rate {lc.pass_rate:.2f} < {cfg.min_pass_rate}")
        if lc.n_instances < cfg.min_support:
            report.issues.append(f"support {lc.n_instances} < {cfg.min_support}")
        if report.n_literals > cfg.max_literals_per_skill:
            report.issues.append(f"too many literals ({report.n_literals})")

        all_pass_rates.append(lc.pass_rate)
        result.per_skill.append(report)

    result.overall_pass_rate = (
        sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 1.0
    )

    if result.new_rate > cfg.max_new_rate:
        result.accepted = False
        result.rejection_reasons.append(
            f"NEW rate {result.new_rate:.2f} > {cfg.max_new_rate}"
        )

    if result.overall_pass_rate < cfg.min_pass_rate:
        result.accepted = False
        result.rejection_reasons.append(
            f"overall pass rate {result.overall_pass_rate:.2f} < {cfg.min_pass_rate}"
        )

    if prev_metrics:
        prev_margin = prev_metrics.get("mean_margin", 0.0)
        if result.mean_margin < prev_margin - cfg.margin_regression_tol:
            result.accepted = False
            result.rejection_reasons.append(
                f"margin regressed {result.mean_margin:.3f} vs {prev_margin:.3f}"
            )

    if result.accepted:
        logger.info("SkillEval: ACCEPTED (pass=%.2f, new=%.2f, margin=%.3f)",
                     result.overall_pass_rate, result.new_rate, result.mean_margin)
    else:
        logger.warning("SkillEval: REJECTED — %s", "; ".join(result.rejection_reasons))

    return result
