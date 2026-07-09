"""
Step 6 — Persistent Skill Bank (MVP).

``_skills: Dict[str, Skill]`` is the **single source of truth**.
Each ``Skill`` owns its contract, protocol, and sub-episode pointers.
``get_contract()`` and ``get_report()`` are convenience accessors for
downstream code that still speaks in contract terms.

Fully backward-compatible with old ``skill_bank.jsonl`` files that only
contain ``contract`` and ``report`` (auto-migrated to ``Skill`` on load).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from skill_agents_grpo.stage3_mvp.schemas import (
    Skill,
    SkillEffectsContract,
    SubEpisodeRef,
    VerificationReport,
)


def _effects_compat_score(
    contract: SkillEffectsContract,
    predicates_start: Optional[dict],
    predicates_end: Optional[dict],
    p_thresh: float = 0.5,
    missing_penalty: float = -0.5,
    contradiction_penalty: float = -1.0,
) -> float:
    """Compute effects-based compatibility between a contract and observed predicates.

    This is the core closed-loop signal from Stage 3 → Stage 2.  It rewards
    segments whose observed predicate changes match the skill's learned effects
    and penalises mismatches.

    Parameters
    ----------
    contract : SkillEffectsContract
        The skill's effects contract (eff_add, eff_del, eff_event).
    predicates_start, predicates_end : dict or None
        Predicate dicts at segment start/end.  Values are floats in [0, 1]
        (probabilities) or bools.
    p_thresh : float
        Threshold above which a predicate is considered "true".
    missing_penalty : float
        Score contribution when an expected effect literal is absent.
    contradiction_penalty : float
        Score contribution when an effect is directly contradicted.

    Returns
    -------
    float
        Normalised score in approx. [-1, +1].  0.0 when contract is empty.
    """
    n_clauses = len(contract.eff_add or set()) + len(contract.eff_del or set()) + len(contract.eff_event or set())
    if n_clauses == 0:
        return 0.0

    P_end = predicates_end or {}
    # Convert bools to floats
    P_end_f = {k: (float(v) if not isinstance(v, (float, int)) else float(v))
               for k, v in P_end.items()}

    score = 0.0

    # eff_add: predicate should be true at end
    for lit in (contract.eff_add or set()):
        val = P_end_f.get(lit)
        if val is not None and val >= p_thresh:
            score += 1.0  # match
        elif val is not None and val < p_thresh:
            score += contradiction_penalty  # contradicted
        else:
            score += missing_penalty  # not observed

    # eff_del: predicate should be false at end
    for lit in (contract.eff_del or set()):
        val = P_end_f.get(lit)
        if val is not None and val < p_thresh:
            score += 1.0  # match (deleted = now false)
        elif val is not None and val >= p_thresh:
            score += contradiction_penalty  # still true = contradicted
        else:
            score += missing_penalty  # not observed

    # eff_event: event should have occurred (check if present in end predicates)
    for lit in (contract.eff_event or set()):
        val = P_end_f.get(lit)
        if val is not None and val >= p_thresh:
            score += 1.0
        else:
            score += missing_penalty

    return score / n_clauses


class SkillBankMVP:
    """Persistent skill bank.

    ``_skills`` is the single source of truth.  Every skill is a ``Skill``
    object that owns its ``contract``, ``protocol``, and sub-episode
    pointers.  ``_reports`` is a side-car kept for verification diagnostics.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._skills: Dict[str, Skill] = {}
        self._reports: Dict[str, VerificationReport] = {}
        self._history: List[dict] = []
        self._path = path

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def skill_ids(self) -> List[str]:
        return list(self._skills.keys())

    def get_contract(self, skill_id: str) -> Optional[SkillEffectsContract]:
        """Convenience accessor — delegates to ``Skill.contract``."""
        skill = self._skills.get(skill_id)
        return skill.contract if skill is not None else None

    def get_report(self, skill_id: str) -> Optional[VerificationReport]:
        return self._reports.get(skill_id)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    # ── Mutations ────────────────────────────────────────────────────

    def add_or_update(
        self,
        contract: SkillEffectsContract,
        report: Optional[VerificationReport] = None,
    ) -> None:
        """Store or update a skill's contract and report."""
        if report is not None:
            self._reports[contract.skill_id] = report
        existing = self._skills.get(contract.skill_id)
        if existing is None:
            self._skills[contract.skill_id] = Skill.from_contract(contract)
        else:
            existing.contract = contract
            existing.updated_at = time.time()
        self._log("add_or_update", contract.skill_id, contract.version)

    def add_or_update_skill(self, skill: Skill) -> None:
        """Store or update a full Skill object."""
        self._skills[skill.skill_id] = skill
        self._log("add_or_update_skill", skill.skill_id, skill.version)

    def remove(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)
        self._reports.pop(skill_id, None)
        self._log("remove", skill_id, -1)

    # ── Sub-episode management ──────────────────────────────────────

    def ingest_sub_episode(self, skill_id: str, sub_ep: SubEpisodeRef) -> bool:
        """Append a sub-episode reference to an existing skill.

        Returns True if the skill exists, False otherwise.
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            return False
        skill.sub_episodes.append(sub_ep)
        skill.n_instances = len(skill.sub_episodes)
        skill.updated_at = time.time()
        if skill.contract:
            skill.contract.n_instances = skill.n_instances
        return True

    def drop_sub_episode(self, skill_id: str, episode_id: str, seg_start: int) -> bool:
        """Remove a specific sub-episode from a skill."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return False
        before = len(skill.sub_episodes)
        skill.sub_episodes = [
            se for se in skill.sub_episodes
            if not (se.episode_id == episode_id and se.seg_start == seg_start)
        ]
        removed = before - len(skill.sub_episodes)
        if removed > 0:
            skill.n_instances = len(skill.sub_episodes)
            skill.updated_at = time.time()
        return removed > 0

    def get_sub_episodes(self, skill_id: str) -> List[SubEpisodeRef]:
        """Return sub-episodes for a skill (empty list if not found)."""
        skill = self._skills.get(skill_id)
        return list(skill.sub_episodes) if skill else []

    def recompute_stats(self, skill_id: str) -> None:
        """Recompute n_instances from current sub-episodes."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.n_instances = len(skill.sub_episodes)
        if skill.contract:
            skill.contract.n_instances = skill.n_instances
            skill.contract.updated_at = time.time()
        skill.updated_at = time.time()

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, filepath: Optional[str] = None) -> None:
        """Save bank to JSONL.

        Each line: ``{"skill": {...}, "report": {...|null}}``.
        The contract is inside ``skill.contract`` — no separate key.
        Old-format files (top-level ``"contract"``) are still loadable.
        """
        path = Path(filepath or self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for skill_id, skill in self._skills.items():
                entry = {
                    "skill": skill.to_dict(),
                    "report": (
                        self._reports[skill_id].to_dict()
                        if skill_id in self._reports
                        else None
                    ),
                }
                f.write(json.dumps(entry, default=str) + "\n")

    def load(self, filepath: Optional[str] = None) -> None:
        """Load bank from JSONL (backward-compatible with old format).

        Handles two formats:
          - New: ``{"skill": {...}, "report": ...}`` (contract inside skill)
          - Old: ``{"contract": {...}, "report": ..., "skill"?: ...}``
        """
        path = Path(filepath or self._path)
        if not path.exists():
            return
        self._skills.clear()
        self._reports.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                if entry.get("skill"):
                    skill = Skill.from_dict(entry["skill"])
                    # Old format may have top-level contract that's more recent
                    if entry.get("contract") and skill.contract is None:
                        skill.contract = SkillEffectsContract.from_dict(entry["contract"])
                    self._skills[skill.skill_id] = skill
                elif entry.get("contract"):
                    # Legacy format: only contract + report, no Skill wrapper
                    contract = SkillEffectsContract.from_dict(entry["contract"])
                    self._skills[contract.skill_id] = Skill.from_contract(contract)
                else:
                    continue

                sid = next(reversed(self._skills))
                if entry.get("report"):
                    self._reports[sid] = VerificationReport.from_dict(entry["report"])

    # ── Stage 2 integration ─────────────────────────────────────────

    def compat_fn(
        self,
        skill: str,
        predicates_start: Optional[dict],
        predicates_end: Optional[dict],
    ) -> float:
        """Effects-based contract compatibility scorer for ``SegmentScorer``."""
        contract = self.get_contract(skill)
        if contract is None:
            return 0.0
        return _effects_compat_score(contract, predicates_start, predicates_end)

    def get_skill_names(self) -> List[str]:
        """Active skill names for Stage 2 pipeline input."""
        return list(self._skills.keys())

    # ── Summary / debug ─────────────────────────────────────────────

    def summary(self) -> Dict[str, dict]:
        """Compact per-skill summary for logging / inspection."""
        result: Dict[str, dict] = {}
        for skill_id, skill in self._skills.items():
            contract = skill.contract
            info: dict = {
                "version": skill.version,
                "has_protocol": bool(skill.protocol.steps),
                "n_sub_episodes": len(skill.sub_episodes),
                "n_with_summary": sum(
                    1 for se in skill.sub_episodes if se.summary
                ),
                "success_rate": round(skill.success_rate, 3),
                "retired": skill.retired,
            }
            if contract is not None:
                info["n_eff_add"] = len(contract.eff_add)
                info["n_eff_del"] = len(contract.eff_del)
                info["n_eff_event"] = len(contract.eff_event)
                info["total_literals"] = contract.total_literals
                info["n_instances"] = contract.n_instances
            if skill_id in self._reports:
                info["pass_rate"] = self._reports[skill_id].overall_pass_rate
            result[skill_id] = info
        return result

    def get_evidence_view(self, skill_id: str) -> Optional[Dict]:
        """Return the evidence-store view for a skill (pointers + summaries)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        return skill.to_evidence_view()

    def get_skills_for_decision_agent(self) -> List[Dict]:
        """Return decision-agent-safe views of all active skills."""
        views = []
        for sid in self.skill_ids:
            skill = self._skills.get(sid)
            if skill is not None and not skill.retired:
                views.append(skill.to_decision_agent_view())
        return views

    # ── History ──────────────────────────────────────────────────────

    def _log(self, event: str, skill_id: str, version: int) -> None:
        self._history.append({
            "event": event,
            "skill_id": skill_id,
            "version": version,
            "timestamp": time.time(),
        })

    @property
    def history(self) -> List[dict]:
        return list(self._history)
