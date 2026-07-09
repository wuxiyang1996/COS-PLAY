"""
Bank diff reports between versions.

Computes and logs the difference between two bank snapshots, including:
  - Added/removed/modified skills
  - Contract effect changes
  - Support and pass rate changes
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SkillDiff:
    """Diff for a single skill between two bank versions."""

    skill_id: str
    change_type: str  # "added" | "removed" | "modified" | "unchanged"
    old_version: int = 0
    new_version: int = 0
    eff_add_added: List[str] = field(default_factory=list)
    eff_add_removed: List[str] = field(default_factory=list)
    eff_del_added: List[str] = field(default_factory=list)
    eff_del_removed: List[str] = field(default_factory=list)
    old_n_instances: int = 0
    new_n_instances: int = 0
    old_pass_rate: Optional[float] = None
    new_pass_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v}
        return d


@dataclass
class BankDiffReport:
    """Complete diff report between two bank versions."""

    old_version: int = 0
    new_version: int = 0
    timestamp: float = field(default_factory=time.time)
    skill_diffs: List[SkillDiff] = field(default_factory=list)
    n_added: int = 0
    n_removed: int = 0
    n_modified: int = 0
    n_unchanged: int = 0
    old_total_skills: int = 0
    new_total_skills: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "timestamp": self.timestamp,
            "n_added": self.n_added,
            "n_removed": self.n_removed,
            "n_modified": self.n_modified,
            "n_unchanged": self.n_unchanged,
            "old_total_skills": self.old_total_skills,
            "new_total_skills": self.new_total_skills,
            "skill_diffs": [d.to_dict() for d in self.skill_diffs],
        }

    @property
    def churn_rate(self) -> float:
        """Fraction of skills that changed (added + removed + modified)."""
        total = max(self.old_total_skills + self.n_added, 1)
        return (self.n_added + self.n_removed + self.n_modified) / total


def compute_bank_diff(
    old_bank: Any,
    new_bank: Any,
    old_version: int = 0,
    new_version: int = 0,
) -> BankDiffReport:
    """Compute the diff between two bank snapshots.

    Args:
        old_bank: previous SkillBankMVP
        new_bank: proposed SkillBankMVP
        old_version: version number of old bank
        new_version: version number of new bank

    Returns:
        BankDiffReport with per-skill diffs and aggregate counts.
    """
    old_ids = set(getattr(old_bank, "skill_ids", []))
    new_ids = set(getattr(new_bank, "skill_ids", []))

    report = BankDiffReport(
        old_version=old_version,
        new_version=new_version,
        old_total_skills=len(old_ids),
        new_total_skills=len(new_ids),
    )

    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids

    for sid in sorted(added):
        nc = new_bank.get_contract(sid) if hasattr(new_bank, "get_contract") else None
        report.skill_diffs.append(SkillDiff(
            skill_id=sid,
            change_type="added",
            new_version=getattr(nc, "version", 1) if nc else 1,
            new_n_instances=getattr(nc, "n_instances", 0) if nc else 0,
            eff_add_added=sorted(getattr(nc, "eff_add", set()) or set()) if nc else [],
            eff_del_added=sorted(getattr(nc, "eff_del", set()) or set()) if nc else [],
        ))
    report.n_added = len(added)

    for sid in sorted(removed):
        oc = old_bank.get_contract(sid) if hasattr(old_bank, "get_contract") else None
        report.skill_diffs.append(SkillDiff(
            skill_id=sid,
            change_type="removed",
            old_version=getattr(oc, "version", 0) if oc else 0,
            old_n_instances=getattr(oc, "n_instances", 0) if oc else 0,
        ))
    report.n_removed = len(removed)

    for sid in sorted(common):
        oc = old_bank.get_contract(sid) if hasattr(old_bank, "get_contract") else None
        nc = new_bank.get_contract(sid) if hasattr(new_bank, "get_contract") else None

        if oc is None or nc is None:
            report.n_unchanged += 1
            continue

        old_add = getattr(oc, "eff_add", set()) or set()
        new_add = getattr(nc, "eff_add", set()) or set()
        old_del = getattr(oc, "eff_del", set()) or set()
        new_del = getattr(nc, "eff_del", set()) or set()

        add_added = sorted(new_add - old_add)
        add_removed = sorted(old_add - new_add)
        del_added = sorted(new_del - old_del)
        del_removed = sorted(old_del - new_del)

        changed = bool(add_added or add_removed or del_added or del_removed)

        if changed:
            old_report = old_bank.get_report(sid) if hasattr(old_bank, "get_report") else None
            new_report = new_bank.get_report(sid) if hasattr(new_bank, "get_report") else None

            report.skill_diffs.append(SkillDiff(
                skill_id=sid,
                change_type="modified",
                old_version=oc.version,
                new_version=nc.version,
                eff_add_added=add_added,
                eff_add_removed=add_removed,
                eff_del_added=del_added,
                eff_del_removed=del_removed,
                old_n_instances=oc.n_instances,
                new_n_instances=nc.n_instances,
                old_pass_rate=old_report.overall_pass_rate if old_report else None,
                new_pass_rate=new_report.overall_pass_rate if new_report else None,
            ))
            report.n_modified += 1
        else:
            report.n_unchanged += 1

    return report


class DiffLogger:
    """Persistent diff logger that writes reports to a directory."""

    def __init__(self, diff_dir: str = "runs/skillbank/diffs"):
        self.diff_dir = Path(diff_dir)
        self.diff_dir.mkdir(parents=True, exist_ok=True)
        self._reports: List[BankDiffReport] = []

    def log_diff(self, report: BankDiffReport) -> str:
        """Write a diff report to disk and return the file path."""
        self._reports.append(report)
        filename = f"diff_v{report.old_version}_to_v{report.new_version}.json"
        path = self.diff_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info(
            "Diff v%d→v%d: +%d -%d ~%d (churn=%.2f)",
            report.old_version, report.new_version,
            report.n_added, report.n_removed, report.n_modified,
            report.churn_rate,
        )
        return str(path)

    @property
    def reports(self) -> List[BankDiffReport]:
        return list(self._reports)
