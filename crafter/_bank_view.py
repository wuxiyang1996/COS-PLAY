"""`BankView` — read-only snapshot of (active ∪ candidate ∪ draft) skills.

Spec: implementation_notes/legacy/crafter-harness-orchestrator-roles.md §"Read
scope". The Crafter's per-episode pass needs to reason across stores
(e.g. "this candidate skill subsumes that active skill") without
violating the two architectural invariants enforced on the package:

  1. The crafter never imports ``skill_bank.stores`` directly.
  2. The crafter never holds a ``SkillLifecycleManager`` reference
     anywhere except inside ``crafter.service.SkillCrafterService``.

So the service is the only crafter component that can construct a
``BankView``; component proposers (``Composer``, ``Generalizer``,
``Hypothesizer``, ``Repairer``) receive a frozen view as a parameter,
read it, and return proposals. They never re-fetch from the lifecycle
manager themselves.

The view is a *snapshot in time*: ``taken_at`` records when it was
built. Mutations to the underlying stores after that point are
invisible to anyone holding the view, which is the desired semantics
for a single Crafter pass — the pass should not see its own writes
turn into inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from data_structure.extensions.skill_record import SkillRecord


@dataclass(frozen=True)
class BankView:
    """A frozen, read-only multi-store snapshot.

    Built only by ``SkillCrafterService._take_bank_view``. Never
    constructed directly inside a proposer.
    """

    actives: Dict[str, SkillRecord] = field(default_factory=dict)
    candidates: Dict[str, SkillRecord] = field(default_factory=dict)
    drafts: Dict[str, SkillRecord] = field(default_factory=dict)
    taken_at: float = 0.0

    # -- lookup ----------------------------------------------------------

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        """Return the record from any store, preferring active > candidate > draft."""
        for d in (self.actives, self.candidates, self.drafts):
            r = d.get(skill_id)
            if r is not None:
                return r
        return None

    def status_of(self, skill_id: str) -> Optional[str]:
        if skill_id in self.actives:
            return "active"
        if skill_id in self.candidates:
            return "candidate"
        if skill_id in self.drafts:
            return "draft"
        return None

    # -- iteration -------------------------------------------------------

    def actives_iter(self) -> Iterator[SkillRecord]:
        return iter(self.actives.values())

    def candidates_iter(self) -> Iterator[SkillRecord]:
        return iter(self.candidates.values())

    def drafts_iter(self) -> Iterator[SkillRecord]:
        return iter(self.drafts.values())

    def all_iter(self) -> Iterator[SkillRecord]:
        for r in self.actives.values():
            yield r
        for r in self.candidates.values():
            yield r
        for r in self.drafts.values():
            yield r

    # -- subsumption heuristic ------------------------------------------

    def subsumed_pairs(
        self, *, candidate_ids: Iterable[str]
    ) -> List[Tuple[str, str, str]]:
        """Detect candidates that subsume an active skill.

        For each ``candidate_id`` in ``candidate_ids`` whose record is
        present in the candidate store, look for an active skill where:

          (a) the candidate explicitly lists the active in its
              ``parent_skill_ids`` (the Bank Agent's mining flow always
              fills this when a refine derives a candidate from an
              active), AND
          (b) the candidate's contract is *at least as strong* as the
              active's: every effect / evidence-role / success-criterion
              in the active is also covered by the candidate.

        Heuristic-only: the gate (G3 replay + G5 non-regression) makes
        the actual call. A false positive here costs one rejected
        ``RetireProposal`` at gate time, which is cheap.

        Returns a list of ``(candidate_id, active_id, rationale)``
        triples ready for ``RetireProposal`` construction.
        """
        out: List[Tuple[str, str, str]] = []
        for cid in candidate_ids:
            cand = self.candidates.get(cid)
            if cand is None:
                continue
            for parent_id in cand.parent_skill_ids:
                active = self.actives.get(parent_id)
                if active is None:
                    continue
                ok, reason = _subsumes(cand, active)
                if not ok:
                    continue
                out.append((cid, active.skill_id, reason))
        return out

    # -- diagnostics -----------------------------------------------------

    def size_summary(self) -> Dict[str, int]:
        return {
            "n_active": len(self.actives),
            "n_candidate": len(self.candidates),
            "n_draft": len(self.drafts),
        }


def take_bank_view(repository) -> BankView:
    """Build a ``BankView`` from a ``skill_bank.repository.SkillRepository``.

    Imported at call-time inside ``SkillCrafterService`` to keep this
    module free of any direct ``skill_bank`` import (mechanical
    invariant: only ``service.py`` may name the lifecycle / repository
    types).
    """
    actives = {r.skill_id: r for r in repository.runnable(include_shadow=True)}
    candidates = {r.skill_id: r for r in repository.candidates()}
    drafts = {r.skill_id: r for r in repository.drafts()}
    return BankView(
        actives=actives,
        candidates=candidates,
        drafts=drafts,
        taken_at=time.time(),
    )


# ---- subsumption helpers --------------------------------------------------


def _subsumes(candidate: SkillRecord, active: SkillRecord) -> Tuple[bool, str]:
    """True iff ``candidate`` strictly covers ``active``'s contract.

    The check is intentionally conservative — we want very few false
    positives because every fired rule produces a ``RetireProposal``
    that the gate has to evaluate. The four criteria (in
    intersection):

      1. ``effects_add`` of active ⊆ ``effects_add`` of candidate
      2. ``effects_del`` of active ⊆ ``effects_del`` of candidate
      3. ``expected_evidence_roles`` of active ⊆ candidate's
      4. ``success_criteria`` of active ⊆ candidate's

    Tightening any criterion is a one-line change here; loosening is
    discouraged (let the gate's G5 non-regression stage do the loose
    work).
    """
    ac = active.contract
    cc = candidate.contract
    if not _is_subset(ac.effects_add, cc.effects_add):
        return False, ""
    if not _is_subset(ac.effects_del, cc.effects_del):
        return False, ""
    if not _is_subset(ac.expected_evidence_roles, cc.expected_evidence_roles):
        return False, ""
    if not _is_subset(ac.success_criteria, cc.success_criteria):
        return False, ""
    return True, (
        f"candidate {candidate.skill_id} carries effects_add⊇{list(ac.effects_add)}, "
        f"effects_del⊇{list(ac.effects_del)}, evidence_roles⊇"
        f"{list(ac.expected_evidence_roles)}, success_criteria⊇"
        f"{list(ac.success_criteria)} of active {active.skill_id}; "
        f"propose retiring the active in favour of the strictly-stronger candidate."
    )


def _is_subset(small, big) -> bool:
    big_set = set(big)
    return all(item in big_set for item in small)


__all__ = ["BankView", "take_bank_view"]
