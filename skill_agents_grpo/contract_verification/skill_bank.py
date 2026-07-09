"""
Step 8 — Persistent Skill Bank with versioning.

The ``SkillBank`` stores all skills and their contracts, supports:
  - Adding / updating / deprecating skills.
  - Version tracking with full history.
  - Persistence via JSONL (one JSON object per line, append-friendly).
  - Providing a ``compat_fn`` to Stage 2's ``SegmentScorer`` so that
    contract compatibility is used during decoding.
  - Querying active (non-deprecated) skills and their contracts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from skill_agents_grpo.contract_verification.schemas import (
    SegmentRecord,
    SkillContract,
    VerificationReport,
    UpdateAction,
)
from skill_agents_grpo.contract_verification.config import ContractVerificationConfig


class SkillBank:
    """Persistent, versioned skill bank.

    Each skill has a ``SkillContract`` and optional verification metadata.
    Modifications increment the version and are logged for reproducibility.

    Integration with Stage 2
    ------------------------
    Call ``bank.compat_fn`` to get a scorer function compatible with
    ``SegmentScorer(compat_fn=bank.compat_fn)``.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._contracts: Dict[str, SkillContract] = {}
        self._reports: Dict[str, VerificationReport] = {}
        self._history: List[dict] = []
        self._path = path

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def skill_ids(self) -> List[str]:
        """All skill ids (including deprecated)."""
        return list(self._contracts.keys())

    @property
    def active_skill_ids(self) -> List[str]:
        """Non-deprecated skill ids."""
        return [s for s, c in self._contracts.items() if not c.deprecated]

    @property
    def active_contracts(self) -> Dict[str, SkillContract]:
        return {s: c for s, c in self._contracts.items() if not c.deprecated}

    def get_contract(self, skill_id: str) -> Optional[SkillContract]:
        return self._contracts.get(skill_id)

    def get_report(self, skill_id: str) -> Optional[VerificationReport]:
        return self._reports.get(skill_id)

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._contracts

    def __len__(self) -> int:
        return len(self._contracts)

    # ── Mutations ────────────────────────────────────────────────────

    def add_or_update(
        self,
        contract: SkillContract,
        report: Optional[VerificationReport] = None,
    ) -> None:
        """Add a new skill or update an existing one."""
        self._contracts[contract.skill_id] = contract
        if report is not None:
            self._reports[contract.skill_id] = report
        self._log_event("add_or_update", contract.skill_id, contract.version)

    def deprecate(self, skill_id: str) -> None:
        """Mark a skill as deprecated (e.g. after SPLIT)."""
        if skill_id in self._contracts:
            self._contracts[skill_id].deprecated = True
            self._contracts[skill_id].bump_version()
            self._log_event("deprecate", skill_id, self._contracts[skill_id].version)

    def remove(self, skill_id: str) -> None:
        """Remove a skill entirely (use deprecate instead when possible)."""
        self._contracts.pop(skill_id, None)
        self._reports.pop(skill_id, None)
        self._log_event("remove", skill_id, -1)

    def apply_action(
        self,
        action: UpdateAction,
        contract: Optional[SkillContract] = None,
        children: Optional[List[SkillContract]] = None,
        reports: Optional[List[VerificationReport]] = None,
    ) -> None:
        """Apply a verified update action to the bank.

        Parameters
        ----------
        action : UpdateAction
            The decision from ``updates.decide_action``.
        contract : SkillContract, optional
            The (possibly modified) contract for KEEP/REFINE actions.
        children : list[SkillContract], optional
            Child contracts for SPLIT actions.
        reports : list[VerificationReport], optional
            Reports for new/child contracts.
        """
        if action.action == "KEEP":
            if contract:
                self.add_or_update(contract)
        elif action.action == "REFINE":
            if contract:
                self.add_or_update(contract)
        elif action.action == "SPLIT":
            self.deprecate(action.skill_id)
            if children:
                for i, child in enumerate(children):
                    rpt = reports[i] if reports and i < len(reports) else None
                    self.add_or_update(child, rpt)
        elif action.action == "MATERIALIZE_NEW":
            if children:
                for i, child in enumerate(children):
                    rpt = reports[i] if reports and i < len(reports) else None
                    self.add_or_update(child, rpt)

        self._log_event(action.action, action.skill_id, 0, action.to_dict())

    # ── Stage 2 integration ─────────────────────────────────────────

    def compat_fn(
        self,
        skill: str,
        predicates_start: Optional[dict],
        predicates_end: Optional[dict],
    ) -> float:
        """Contract compatibility scorer for ``SegmentScorer``.

        Plug into Stage 2:
            ``SegmentScorer(compat_fn=bank.compat_fn)``

        Returns a score in [-1, 1] based on how well the start/end
        predicates satisfy the skill's contract.
        """
        contract = self._contracts.get(skill)
        if contract is None or contract.deprecated:
            return 0.0
        P_start = predicates_start or {}
        P_end = predicates_end or {}
        p_start_float = {k: (float(v) if not isinstance(v, float) else v)
                         for k, v in P_start.items()}
        p_end_float = {k: (float(v) if not isinstance(v, float) else v)
                       for k, v in P_end.items()}
        return contract.compat_score(p_start_float, p_end_float)

    def get_skill_names(self) -> List[str]:
        """Active skill names for Stage 2 pipeline input."""
        return self.active_skill_ids

    # ── Action language export ──────────────────────────────────────

    def to_action_language(
        self,
        fmt: str = "pddl",
        domain_name: str = "skill_domain",
        **kwargs,
    ) -> str:
        """Export all active skills in a formal action language format.

        Parameters
        ----------
        fmt : str
            One of ``"pddl"``, ``"strips"``, ``"sas"``, ``"compact"``.
        domain_name : str
            Used as the PDDL domain name (``"pddl"`` format only).
        **kwargs
            Passed to the underlying formatter (e.g. ``include_inv``).

        Returns
        -------
        str
            Full action language representation of the bank.
        """
        from skill_agents_grpo.contract_verification.action_language import (
            bank_to_action_language,
        )
        return bank_to_action_language(
            self, fmt=fmt, domain_name=domain_name, **kwargs,
        )

    def save_action_language(
        self,
        filepath: str,
        fmt: str = "pddl",
        domain_name: str = "skill_domain",
        **kwargs,
    ) -> None:
        """Save all active skills to a file in action language format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_action_language(fmt=fmt, domain_name=domain_name, **kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, filepath: Optional[str] = None) -> None:
        """Save bank to JSONL (one contract per line)."""
        path = Path(filepath or self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for skill_id, contract in self._contracts.items():
                entry = {
                    "contract": contract.to_dict(),
                    "report": self._reports[skill_id].to_dict() if skill_id in self._reports else None,
                }
                f.write(json.dumps(entry, default=str) + "\n")

    def load(self, filepath: Optional[str] = None) -> None:
        """Load bank from JSONL."""
        path = Path(filepath or self._path)
        if not path.exists():
            return
        self._contracts.clear()
        self._reports.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                contract = SkillContract.from_dict(entry["contract"])
                self._contracts[contract.skill_id] = contract
                if entry.get("report"):
                    self._reports[contract.skill_id] = VerificationReport.from_dict(entry["report"])

    # ── History ──────────────────────────────────────────────────────

    def _log_event(
        self,
        event_type: str,
        skill_id: str,
        version: int,
        details: Optional[dict] = None,
    ) -> None:
        self._history.append({
            "event": event_type,
            "skill_id": skill_id,
            "version": version,
            "timestamp": time.time(),
            "details": details,
        })

    @property
    def history(self) -> List[dict]:
        return list(self._history)

    def summary(self) -> Dict[str, dict]:
        """Compact summary of the bank for logging / agent context."""
        result: Dict[str, dict] = {}
        for skill_id, contract in self._contracts.items():
            if contract.deprecated:
                continue
            result[skill_id] = {
                "version": contract.version,
                "pre": sorted(contract.pre),
                "eff_add": sorted(contract.eff_add),
                "eff_del": sorted(contract.eff_del),
                "inv": sorted(contract.inv),
                "n_pre": len(contract.pre),
                "n_eff": len(contract.eff_add) + len(contract.eff_del),
            }
            if skill_id in self._reports:
                result[skill_id]["pass_rate"] = self._reports[skill_id].overall_pass_rate
                result[skill_id]["n_instances"] = self._reports[skill_id].n_instances
        return result
