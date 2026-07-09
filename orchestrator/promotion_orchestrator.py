"""`PromotionOrchestrator` — atomic promotion / rollback transactions.

Spec: PLAN-PIPELINE-ORCHESTRATOR §7, PLAN-UNIFIED-SKILL-GATE §6.

A "promotion" is the act of taking a CANDIDATE record + its passing
`SkillEvaluationRecord` and moving it to ACTIVE / SHADOW / PROVISIONAL.
This must be atomic: either the bank state, the snapshot, and the
release record all advance together, or none of them do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.enums import (
    TRANSFER_TARGET_DOMAINS,
    GateStage,
    GateVerdict,
    SkillStatus,
)
from data_structure.extensions.run_release import RunRelease
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillRecord
from orchestrator.artifact_store import ArtifactStore
from orchestrator.snapshot_manager import SnapshotManager
from skill_bank.lifecycle import LifecycleError, SkillLifecycleManager


@dataclass
class PromotionPlan:
    """One atomic batch of (skill, target_status) decisions."""

    transitions: List[tuple[SkillRecord, SkillStatus, SkillEvaluationRecord, str]] = field(
        default_factory=list
    )

    def add(
        self,
        *,
        skill: SkillRecord,
        target_status: SkillStatus,
        evaluation: SkillEvaluationRecord,
        rationale: str,
    ) -> None:
        self.transitions.append((skill, target_status, evaluation, rationale))

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class PromotionResult:
    release: Optional[RunRelease]
    promoted_skill_ids: List[str]
    deprecated_skill_ids: List[str]
    rolled_back: bool
    rollback_reason: Optional[str] = None


class PromotionOrchestrator:
    def __init__(
        self,
        *,
        lifecycle: SkillLifecycleManager,
        snapshot_manager: SnapshotManager,
        artifact_store: ArtifactStore,
    ) -> None:
        self._lifecycle = lifecycle
        self._snapshots = snapshot_manager
        self._artifacts = artifact_store

    # -- promotion ---------------------------------------------------------

    def promote(
        self,
        plan: PromotionPlan,
        *,
        adapter_signature: List[str],
        config_payload: Dict[str, Any],
        notes: str = "",
        bank_snapshot_id: Optional[str] = None,
    ) -> PromotionResult:
        # Enforce the gate-binding invariant: each transition's evaluation
        # must (a) PASS or LIMITED_PASS, (b) match the skill's content hash.
        for skill, target, evaluation, _ in plan.transitions:
            if evaluation.verdict is None:
                raise LifecycleError(
                    f"Cannot promote {skill.skill_id!r}: evaluation has no verdict."
                )
            verdict = evaluation.verdict.final_verdict
            if verdict == GateVerdict.FAIL:
                raise LifecycleError(
                    f"Cannot promote {skill.skill_id!r}: gate verdict=FAIL."
                )
            if verdict == GateVerdict.LIMITED_PASS and target == SkillStatus.ACTIVE:
                raise LifecycleError(
                    f"Cannot promote {skill.skill_id!r} to ACTIVE on LIMITED_PASS: "
                    f"only PROVISIONAL is allowed."
                )
            if evaluation.skill_content_hash != skill.content_hash():
                raise LifecycleError(
                    f"Cannot promote {skill.skill_id!r}: content hash drift "
                    f"(eval={evaluation.skill_content_hash[:10]}, "
                    f"current={skill.content_hash()[:10]})."
                )

        # Day-9a: capture pre-transition status_before on every
        # evaluation. Done before persisting so the audit trail is
        # internally consistent: the row that names a skill's
        # transition records what status it was leaving.
        for skill, _target, evaluation, _ in plan.transitions:
            if evaluation.status_before is None:
                evaluation.status_before = skill.status

        # Persist evaluations first (audit trail).
        for _, _, evaluation, _ in plan.transitions:
            self._artifacts.put_evaluation(evaluation)

        # PLAN-UNIFIED-SKILL-GATE Stage 3a: write any verified target
        # domains into the SkillRecord *before* the status transition, so
        # the lifecycle manager's ACTIVE invariant (PLAN-SKILL-BANK §0.4)
        # sees the up-to-date `verified_domains`. The gate is the only
        # producer of these entries; the lifecycle manager is the only
        # writer.
        for skill, target, evaluation, rationale in plan.transitions:
            self._record_transfer_verifications(
                skill=skill,
                evaluation=evaluation,
                rationale=rationale,
            )

        # Apply transitions atomically via the lifecycle manager.
        transitions = [
            (skill.skill_id, target, rationale)
            for skill, target, _ev, rationale in plan.transitions
        ]
        self._lifecycle.transition_many(transitions)

        # Take snapshot.
        active_records = self._lifecycle.repository.runnable(include_shadow=True)
        snapshot = self._snapshots.take(
            active_records=active_records,
            adapter_signature=adapter_signature,
            config_payload=config_payload,
            notes=notes,
        )

        # Day-9a: post-transition anchor pin —
        #   * `status_after`     ← target_status from the plan
        #   * `bank_snapshot_id` ← caller-supplied or the just-minted snapshot
        # then re-persist so the audit row reflects the final shape.
        effective_snapshot_id = bank_snapshot_id or snapshot.get("snapshot_id")
        for _skill, target, evaluation, _ in plan.transitions:
            if evaluation.status_after is None:
                evaluation.status_after = target
            if evaluation.bank_snapshot_id is None:
                evaluation.bank_snapshot_id = effective_snapshot_id
            self._artifacts.put_evaluation(evaluation)

        # Mint release.
        promoted_ids = [s.skill_id for s, t, _e, _r in plan.transitions if t in {SkillStatus.ACTIVE, SkillStatus.PROVISIONAL, SkillStatus.SHADOW}]
        deprecated_ids = [s.skill_id for s, t, _e, _r in plan.transitions if t in {SkillStatus.DEPRECATED, SkillStatus.ROLLED_BACK, SkillStatus.REJECTED}]
        release = RunRelease(
            parent_release_id=self._latest_release_id(),
            bank_snapshot_path=f"snapshots/{snapshot['snapshot_id']}.json",
            adapter_snapshot_paths={},
            config_snapshot_path="",
            promoted_skill_ids=promoted_ids,
            deprecated_skill_ids=deprecated_ids,
            gate_evaluation_ids=[ev.evaluation_id for _s, _t, ev, _r in plan.transitions],
            notes=notes,
            created_at=time.time(),
        )
        self._artifacts.put_release(release)
        # Day-9a: surface the GateRunner reproducibility-anchor union
        # in the audit row so the timeline of "(snapshot, eval_suite,
        # adapter_versions) → release" is searchable.
        anchor_union: Dict[str, Any] = {}
        adapter_versions_union: Dict[str, str] = {}
        eval_suite_ids: set[str] = set()
        ontology_versions: set[str] = set()
        for _s, _t, ev, _r in plan.transitions:
            if ev.eval_suite_id:
                eval_suite_ids.add(ev.eval_suite_id)
            if ev.ontology_version:
                ontology_versions.add(ev.ontology_version)
            for k, v in (ev.adapter_versions or {}).items():
                adapter_versions_union.setdefault(k, v)
        if eval_suite_ids:
            anchor_union["eval_suite_ids"] = sorted(eval_suite_ids)
        if ontology_versions:
            anchor_union["ontology_versions"] = sorted(ontology_versions)
        if adapter_versions_union:
            anchor_union["adapter_versions"] = adapter_versions_union
        self._artifacts.append_audit(
            {
                "kind": "release",
                "release_id": release.release_id,
                "snapshot_id": snapshot["snapshot_id"],
                "promoted_skill_ids": promoted_ids,
                "deprecated_skill_ids": deprecated_ids,
                "reproducibility_anchors": anchor_union,
            }
        )
        return PromotionResult(
            release=release,
            promoted_skill_ids=promoted_ids,
            deprecated_skill_ids=deprecated_ids,
            rolled_back=False,
        )

    # -- rollback ----------------------------------------------------------

    def rollback(
        self,
        *,
        skill_id: str,
        reason: str,
    ) -> PromotionResult:
        record = self._lifecycle.get(skill_id)
        if record is None:
            raise LifecycleError(f"Unknown skill {skill_id!r}")
        if record.status not in {SkillStatus.ACTIVE, SkillStatus.PROVISIONAL, SkillStatus.SHADOW, SkillStatus.DEPRECATED}:
            raise LifecycleError(
                f"Skill {skill_id!r} is in status {record.status.value}, "
                f"not eligible for rollback."
            )
        target = SkillStatus.ROLLED_BACK
        # If still ACTIVE/PROVISIONAL/SHADOW, must first deprecate then roll back.
        if record.status != SkillStatus.DEPRECATED:
            self._lifecycle.transition(
                skill_id, to_status=SkillStatus.DEPRECATED, rationale=f"pre-rollback: {reason}"
            )
        self._lifecycle.transition(skill_id, to_status=target, rationale=reason)
        self._artifacts.append_audit(
            {"kind": "rollback", "skill_id": skill_id, "reason": reason}
        )
        return PromotionResult(
            release=None,
            promoted_skill_ids=[],
            deprecated_skill_ids=[skill_id],
            rolled_back=True,
            rollback_reason=reason,
        )

    # -- helpers -----------------------------------------------------------

    def _record_transfer_verifications(
        self,
        *,
        skill: SkillRecord,
        evaluation: SkillEvaluationRecord,
        rationale: str,
    ) -> None:
        """Mirror the gate's Stage 3a outcome into the bank record.

        Reads the TRANSFER stage verdict from the evaluation, extracts the
        per-target pass-rate metrics emitted by `GateService._run_transfer`
        (`pass_rate.<target>` / `k_used.<target>`), intersects them with
        `eligible_domains ∩ TRANSFER_TARGET_DOMAINS`, and asks the
        lifecycle manager to persist the result.
        """
        verdict = evaluation.verdict
        if verdict is None:
            return
        stage = verdict.stage_for(GateStage.TRANSFER)
        if stage is None or stage.verdict == GateVerdict.FAIL:
            return
        verified = [
            d
            for d in verdict.eligible_domains
            if d in TRANSFER_TARGET_DOMAINS and d not in skill.verified_domains
        ]
        if not verified:
            return
        per_target_metrics: Dict[str, Dict[str, float]] = {}
        for tgt in verified:
            pr = stage.metrics.get(f"pass_rate.{tgt}")
            ku = stage.metrics.get(f"k_used.{tgt}")
            if pr is None and ku is None:
                continue
            metrics: Dict[str, float] = {}
            if pr is not None:
                metrics["pass_rate"] = float(pr)
            if ku is not None:
                metrics["k_used"] = float(ku)
            per_target_metrics[tgt] = metrics
        self._lifecycle.record_transfer_verification(
            skill.skill_id,
            verified_targets=verified,
            evaluation_id=evaluation.evaluation_id,
            per_target_metrics=per_target_metrics,
            rationale=f"stage_3a:{rationale}" if rationale else "stage_3a",
        )

    def _latest_release_id(self) -> Optional[str]:
        rels = self._artifacts.list_releases()
        if not rels:
            return None
        rels.sort(key=lambda r: r.get("created_at") or 0)
        return rels[-1]["release_id"]


__all__ = ["PromotionOrchestrator", "PromotionPlan", "PromotionResult"]
