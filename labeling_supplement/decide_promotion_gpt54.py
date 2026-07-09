#!/usr/bin/env python
"""
Offline ``PromotionOrchestrator`` mirror — turn Crafter proposals into
transactional promotion decisions, audit records, and a frozen
``RunRelease`` per ``(corpus, source)`` pair.

This is the third sibling driver under ``labeling_supplement/`` (after
``decide_skill_crafting_gpt54.py`` for the Crafter and the still-pending
Harness ``GateRunner`` mirror) called for in
``implementation_notes/legacy/crafter-harness-orchestrator-roles.md`` §4 and
§6.2.  It replays the live offline ``PromotionOrchestrator`` semantics —
the *transactional* component that takes a ``SkillEvaluationRecord``
plus a candidate ``SkillRecord`` and decides among
``{PROMOTE, REJECT, DEFER, ROLLBACK}`` — against the deterministic
artefacts produced by the prior two stages.

What it consumes
----------------
*Required.*

* ``--proposals-run`` — a ``crafter_proposals_out/run_<ts>`` directory
  (the output of ``decide_skill_crafting_gpt54.py``).  Each
  ``<corpus>/<source>/proposals.jsonl`` line is the offline crafter
  proposal schema (``proposal_kind``,
  ``target_skill_id`` / ``source_skill_id`` / ``components``, etc.).
* ``--bank-run`` — a ``labeling/skill_bank_out/run_<ts>`` directory.
  Used to hydrate the original ``SkillRecord`` for each *base skill*
  the proposals reference (so patches/retires can resolve their
  subject) and to seed the bank snapshot the promotion transaction
  takes against.

*Optional.*

* ``--actions-run`` — a ``labeling/skill_actions_out/run_<ts>``
  directory.  Used **only** for the per-source ``actor_batch_metrics``
  signal that the live ``PromotionOrchestrator.rollback`` path consumes
  (PLAN-PIPELINE-ORCHESTRATOR §3a.4).  Without it, ``ROLLBACK`` cannot
  fire — only ``PROMOTE`` / ``REJECT`` / ``DEFER`` are reachable.
* ``--gate-verdicts-run`` — *forward-compatibility hook* for the
  pending Harness ``GateRunner`` mirror.  Each
  ``<corpus>/<source>/gate_verdicts.jsonl`` line, if provided, is
  consumed verbatim (one ``SkillEvaluationRecord`` per proposal_id).
  When omitted (the Phase-1 default today), this driver runs the LIVE
  ``orchestrator.GateService`` inline against a freshly-seeded temp
  bank to produce equivalent verdicts.

What it does
------------
For every ``(corpus, source)`` pair under ``--proposals-run``:

  1. Spin up an ephemeral live stack (``SkillStore`` × 4 stores +
     ``SkillRepository`` + ``SkillLifecycleManager`` +
     ``ArtifactStore`` + ``SnapshotManager`` +
     ``GateService(harness=...)`` +
     ``PromotionOrchestrator(...)``) under a tempdir.  The Phase-1
     ``GateService`` runs without a real ``AdapterRegistry`` because
     Stage-1/2/3a fall back to ``LIMITED_PASS`` / legacy n-domains
     paths in absence of replay seeds / shadow logs / live target
     adapters; the rule-only Stage-0 + Stage-3a-legacy + Stage-4 chain
     is exactly what the live Phase-B MVP exercises in
     ``tests/test_smoke.py::test_smoke_end_to_end``.

  2. Seed the bank: for each skill in
     ``<bank-run>/<corpus>/<source>/skill_bank.jsonl``, hydrate a
     ``SkillRecord`` and ``ingest_draft → CANDIDATE`` it through the
     lifecycle manager.  This is the **read** side of §6.2 #1 — the
     bank snapshot id is the candidate-store directory name, the same
     content-addressed string the live ``SnapshotManager`` mints.

  3. For each proposal in
     ``<proposals-run>/<corpus>/<source>/proposals.jsonl``:

       a) Translate the offline schema into a live
          ``BankMutationProposal`` (see ``_translate_proposal``).  This
          is the *only* place the two schemas intersect; downstream
          code uses the live types end-to-end.
       b) Materialise the *subject skill* the gate evaluates against.
          For ``patch`` / ``retire``: re-use the seeded
          ``CANDIDATE`` record.  For ``compose`` / ``transfer`` /
          ``hypothesize``: synthesise a fresh candidate
          ``SkillRecord`` (the mirror's stand-in for what the live
          ``SkillCrafterService`` would have minted) and
          ``ingest_draft → CANDIDATE`` it.  Subjects always carry
          ``feasible_domains == DOMAINS`` because every offline
          proposal declares all five target domains per
          PLAN-SKILL-CRAFTER §2.5 — that's what lets Stage 0's
          general-protocol invariant pass.
       c) Build a ``SkillEvaluationRecord``.  Three modes are
          available (CLI ``--gate-mode``):

            * ``offline-synthetic`` (DEFAULT) — runs the LIVE Stage-0
              static checks inline, then attaches ``LIMITED_PASS``
              synthetic verdicts for Stages 1–4 (no replay seeds, no
              shadow log, no live target adapters, no baseline/post
              score).  The aggregate is ``LIMITED_PASS`` whenever
              Stage 0 passes, which the ``PromotionOrchestrator`` maps
              to ``PROVISIONAL``.  This is the documented Phase-1
              outcome ("everything stops at PROVISIONAL until
              Phase-2"; see
              ``implementation_notes/legacy/crafter-harness-orchestrator-roles.md``
              §7.1 mismatch #4).
            * ``live`` — calls the LIVE
              ``orchestrator.GateService.evaluate(...)`` end-to-end.
              In the absence of replay seeds / target adapters this
              currently emits a Stage-3a ``FAIL`` for every proposal,
              so it is most useful as a diagnostic; promotions are
              expected to be rare.
            * ``external`` — implied when ``--gate-verdicts-run`` is
              provided.  Each proposal's verdict is read verbatim
              from ``<gate-verdicts-run>/<corpus>/<source>/gate_verdicts.jsonl``;
              this is the integration hook for the pending Harness
              ``GateRunner`` mirror.
       d) Decide the target lifecycle status:

            * ``PASS``         → ``ACTIVE``       (PROMOTE)
            * ``LIMITED_PASS`` → ``PROVISIONAL``  (PROMOTE)
            * ``FAIL``         → ``REJECTED``     (REJECT)
            * ``RetireProposal`` always maps to
              ``DEPRECATED → ROLLED_BACK`` regardless of verdict
              (the orchestrator's ``rollback`` path).

       e) Build a ``PromotionPlan`` and call
          ``PromotionOrchestrator.promote(plan)`` for all PASS /
          LIMITED_PASS subjects in one batch.  For FAIL subjects we
          still ``put_evaluation`` to the audit trail but transition
          the candidate to ``REJECTED`` directly via the lifecycle
          manager (the live ``promote()`` refuses FAIL verdicts; this
          is the offline equivalent of the manual reject path).

  4. After the per-source batch:

       * If ``--actions-run`` was provided, compute crude
         per-skill ``actor_batch_metrics`` from
         ``<actions-run>/<corpus>/<source>/_run_summary.json``
         (selection count, mean reward proxy).  Any skill that just
         got ``PROMOTED`` to ACTIVE but whose metric drops below
         ``--rollback-min-pass-rate`` triggers
         ``PromotionOrchestrator.rollback(skill_id, reason="post-promotion regression")``.
       * Mint a final ``RunRelease`` (the orchestrator already does
         this inside ``promote()``; here we surface the latest one).

What it writes
--------------
``<output_dir>/<corpus>/<source>/``::

    gate_verdicts.jsonl          # SkillEvaluationRecord per proposal
    promotion_decisions.jsonl    # one decision per proposal: PROMOTE / REJECT / DEFER / ROLLBACK
    audit.jsonl                  # ArtifactStore.audit.jsonl mirror — one event per promote/reject/rollback
    release.json                 # latest RunRelease (or null if no promotions)
    bank_snapshots/<id>.json     # snapshot(s) the promotions took
    defer_followups.jsonl        # back-edge for the next Crafter run (§6.2 #4); empty in Phase 1
    _promotion_summary.json      # per-source stats

``<output_dir>/_run_meta.json``  — full argv + thresholds + linked input runs.
``<output_dir>/_run_summary.json`` — aggregate across pairs.

Schema of one ``promotion_decisions.jsonl`` line::

    {
      "proposal_id":    "...",
      "subject_skill_id": "...",
      "subject_skill_content_hash": "...",
      "evaluation_id":  "...",
      "final_verdict":  "pass | limited_pass | fail",
      "decision":       "PROMOTE | REJECT | DEFER | ROLLBACK",
      "from_status":    "candidate",
      "target_status":  "active | provisional | rejected | deprecated | rolled_back",
      "release_id":     "...",          # null on REJECT / DEFER
      "from_snapshot":  "...",
      "to_snapshot":    "...",          # null on REJECT / DEFER
      "rationale":      "...",
      "audit_event_kind": "release | rejection | rollback",
      "triggered_by":   "gate_pass | gate_fail | non_regression_fail | retire_proposal",
      "linked_eval_record": "..."
    }

Why this driver is in the rule-light tier
-----------------------------------------
The live ``PromotionOrchestrator`` is itself deterministic plumbing —
it consumes a ``SkillEvaluationRecord`` and applies a fixed
state-machine (``promote_orchestrator.py:86-130`` enforces verdict +
content-hash drift checks, then calls the lifecycle manager).  No LLM
fires inside the offline orchestrator surface; the rule path here is
just the live code with on-disk JSONL inputs.  The ``_gpt54`` suffix is
preserved for naming-consistency with sibling drivers — the **frozen
teacher** identifier (``Qwen/Qwen3.5-35B-A3B`` per
``common/models.BACKBONE_TEACHER_MODEL``) is logged into
``_run_meta.json`` for audit, but no model is invoked.

What this driver explicitly does NOT do
---------------------------------------
Per ``implementation_notes/legacy/crafter-harness-orchestrator-roles.md`` §6.3
("No driver imports another driver's code"):

  * No import of ``decide_skill_crafting_gpt54.py`` or
    ``reflect_per_episode_gpt54.py``.  The shared API is the on-disk
    JSONL.
  * No mutation of input directories.  ``--proposals-run``,
    ``--bank-run``, ``--actions-run`` are read-only.
  * No live env: ``GateService`` Stages 1 / 2 always return
    ``LIMITED_PASS`` here (no replay seeds, no shadow log) — those
    stages get *real* signal only once the Harness ``GateRunner``
    mirror lands and feeds its output through ``--gate-verdicts-run``.
  * No teacher / judge model invocation.  Both
    ``BACKBONE_TEACHER_MODEL`` and ``BACKBONE_JUDGE_MODEL`` are
    captured into ``_run_meta.json`` only.

Usage
-----

::

    # Default: process every (corpus, source) under --proposals-run.
    python labeling_supplement/decide_promotion_gpt54.py \\
        --proposals-run labeling_supplement/crafter_proposals_out/run_<ts> \\
        --bank-run      labeling/skill_bank_out/run_<ts> \\
        --actions-run   labeling/skill_actions_out/run_<ts>

    # Smoke: one source, deterministic.
    python labeling_supplement/decide_promotion_gpt54.py \\
        --proposals-run labeling_supplement/crafter_proposals_out/run_<ts> \\
        --bank-run      labeling/skill_bank_out/run_<ts> \\
        --corpus env_wrappers --source twenty_forty_eight -v

The companion bash dispatcher ``run_decide_promotion.sh`` fans this out
one worker per ``(corpus, source)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Path setup so the script runs from any cwd (mirror sibling drivers).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Project imports — these load the LIVE Orchestrator code.
# ---------------------------------------------------------------------------
from common.enums import (
    DOMAINS,
    SOURCE_DOMAINS,
    TRANSFER_TARGET_DOMAINS,
    GateStage,
    GateVerdict,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.models import BACKBONE_JUDGE_MODEL, BACKBONE_TEACHER_MODEL
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
    proposal_to_json,
)
from data_structure.extensions.gate_verdict import GateVerdictPayload, StageVerdict
from data_structure.extensions.run_release import RunRelease
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness import AdapterRegistry, RewardLogger, SkillHarness
from orchestrator import (
    ArtifactStore,
    GateService,
    OrchestratorConfig,
    PromotionOrchestrator,
    PromotionPlan,
    SnapshotManager,
)
from skill_bank import LifecycleError, SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName

logger = logging.getLogger("labeling_supplement.decide_promotion")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PROPOSALS_RUN = (
    CODEBASE_ROOT / "labeling_supplement" / "crafter_proposals_out" / "run_20260430_073444"
)
DEFAULT_BANK_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_bank_out" / "run_20260430_030637"
)
DEFAULT_ACTIONS_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_actions_out" / "run_20260430_064325"
)
DEFAULT_OUTPUT_ROOT = (
    CODEBASE_ROOT / "labeling_supplement" / "promotion_decisions_out"
)

CORPORA = ("gym_v", "env_wrappers")

# Crude per-skill regression thresholds for the §3.3 / §3a.4 rollback path.
# Conservative on purpose — single-domain cold-start data has very little
# signal here so we only want to trigger ROLLBACK when the metric is clearly
# broken (no selections at all on a freshly-promoted ACTIVE skill).
DEFAULTS = dict(
    rollback_min_selections=1,        # < this on a newly-active skill -> ROLLBACK
    rollback_min_pass_rate=0.50,      # < this -> ROLLBACK
)


# ═══════════════════════════════════════════════════════════════════════
# Bank loading (mirror of the read paths in reflect_per_episode_gpt54.py
# but kept self-contained per §6.3 "no driver imports another driver's code")
# ═══════════════════════════════════════════════════════════════════════

# Map between the on-disk evidence_role string and a SkillType + canonical
# expected_evidence_roles value. Kept in sync with reflect_per_episode_gpt54
# semantically without importing it.
_ROLE_TO_SKILL_TYPE: Dict[str, SkillType] = {
    "GATHER":  SkillType.GROUNDING,
    "VERIFY":  SkillType.REASONING,
    "REASON":  SkillType.REASONING,
    "COMMIT":  SkillType.ACTION,
}


def _safe_skill_id(skill_id: str) -> str:
    """Cold-start labels use ``OPERATOR/SUBGOAL`` (e.g. ``COMMIT/ATTACK``);
    the on-disk ``SkillStore`` flat-filename layout rejects ``/`` in IDs.
    Mirror the offline-mirror convention of mapping ``/`` → ``__``."""
    return skill_id.replace("/", "__")


def _wrap_protocol_steps(raw_steps: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in raw_steps or []:
        if isinstance(s, dict):
            out.append(dict(s))
        elif isinstance(s, str):
            out.append({"action": "EXEC", "payload": {}, "notes": s})
        else:
            out.append({"action": "EXEC", "payload": {}, "notes": str(s)})
    return out


def _record_from_bank_entry(
    entry: Dict[str, Any],
    *,
    target_domains_for_subject: Tuple[str, ...] = DOMAINS,
) -> SkillRecord:
    """Hydrate a `SkillRecord` from one ``skill_bank.jsonl`` envelope.

    The cold-start corpus carries single-domain skills
    (``applicable_domains == ["gymv"]``), which fail Stage 0's
    ``feasible_domains < 2`` invariant.  Every offline crafter proposal
    declares **all five domains** as its ``target_domains`` per
    PLAN-SKILL-CRAFTER §2.5 — i.e., the proposal asserts the subject
    is feasible there once promoted.  We honour that by widening the
    subject's ``feasible_domains`` to the proposal's stated set, which
    is precisely what the live ``PromotionOrchestrator`` would see if
    the Crafter had attached the same metadata at proposal time.
    The original on-disk ``applicable_domains`` field is retained as
    ``source_domains`` (game lineage) so the gate's source-domain
    check (PLAN-SKILL-BANK §0.4) keeps firing.
    """
    skill = entry.get("skill") or {}
    contract = skill.get("contract") or {}
    role = (skill.get("evidence_role") or "COMMIT").upper()
    skill_type = _ROLE_TO_SKILL_TYPE.get(role, SkillType.MIXED)

    on_disk_domains = list(skill.get("applicable_domains") or []) or ["gymv"]
    feasible = sorted(set(target_domains_for_subject) | set(on_disk_domains))
    # We deliberately leave ``source_domains`` and
    # ``transfer_target_domains`` empty on seeded bank skills.  In the
    # offline mirror they are NEVER directly the gate's subject —
    # patches mint a fresh REPAIRED candidate via ``_make_patch_subject``
    # that points at this skill via ``parent_skill_ids``; retires
    # consume this skill but the ``RetireProposal`` path does not use
    # source/target metadata.  Phase 2 (when target adapters land) will
    # populate these fields from a real harness probe.

    # Day-2 lift can emit ``protocol`` either as the legacy dict
    # ``{"steps":[<NL>], "preconditions":[…], …}`` or as a list of
    # typed hops ``[{"action", "payload", "notes"}, …]``. Mirror the
    # `_record_from_bank_entry` shim in ``trainer/coevolution/_crafter_hook.py``.
    raw_protocol = skill.get("protocol")
    if isinstance(raw_protocol, list):
        protocol_steps = list(raw_protocol)
        protocol_blob: Mapping[str, Any] = {}
    elif isinstance(raw_protocol, Mapping):
        protocol_blob = raw_protocol
        protocol_steps = list(protocol_blob.get("steps") or [])
    else:
        protocol_blob = {}
        protocol_steps = []

    rec = SkillRecord.new(
        name=skill.get("name", skill.get("skill_id", "_unknown")),
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=feasible,
        protocol=_wrap_protocol_steps(protocol_steps),
        contract=SkillContract(
            preconditions=list(protocol_blob.get("preconditions") or []),
            effects_add=list(contract.get("eff_add") or []),
            effects_del=list(contract.get("eff_del") or []),
            expected_evidence_roles=[role] if role else [],
            success_criteria=list(protocol_blob.get("success_criteria") or []),
            abort_criteria=list(protocol_blob.get("abort_criteria") or []),
        ),
    )
    raw_id = skill.get("skill_id") or rec.skill_id
    object.__setattr__(rec, "skill_id", _safe_skill_id(raw_id))
    return rec


def _seed_bank_as_candidates(
    lifecycle: SkillLifecycleManager,
    bank_path: Path,
    *,
    target_domains_for_subject: Tuple[str, ...] = DOMAINS,
) -> Tuple[int, int, Dict[str, SkillRecord]]:
    """Load ``skill_bank.jsonl`` and seed every entry as CANDIDATE.

    Returns ``(n_seeded, n_skipped, by_id_map)``.  The map lets the
    promotion loop look up any base skill referenced by a proposal
    (``target_skill_id`` / ``base_skill_id`` / ``components``).
    """
    if not bank_path.exists():
        return 0, 0, {}
    n = 0
    skipped = 0
    by_id: Dict[str, SkillRecord] = {}
    with bank_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            try:
                rec = _record_from_bank_entry(
                    entry, target_domains_for_subject=target_domains_for_subject,
                )
                lifecycle.ingest_draft(rec)
                lifecycle.transition(
                    rec.skill_id,
                    to_status=SkillStatus.CANDIDATE,
                    rationale="seed-from-bank-snapshot",
                )
                by_id[rec.skill_id] = rec
                n += 1
            except (LifecycleError, ValueError) as exc:
                logger.debug(
                    "skip bank seed %s: %s",
                    entry.get("skill", {}).get("skill_id"), exc,
                )
                skipped += 1
    return n, skipped, by_id


# ═══════════════════════════════════════════════════════════════════════
# Offline-proposal → live BankMutationProposal translation
# ═══════════════════════════════════════════════════════════════════════
#
# The offline crafter (decide_skill_crafting_gpt54.py) writes proposals
# with field names like ``target_skill_id`` / ``patch_kind`` /
# ``compose_op`` / ``retire_reason``.  The live schema in
# ``data_structure/extensions/bank_mutation_proposal.py`` uses
# ``base_skill_id`` / ``recovery_strategy`` / ``component_skill_ids`` /
# ``reason``.  This translator is the only seam where the two schemas
# meet — every downstream call site uses the live types.

@dataclass
class _OfflineProposal:
    """Raw view of one offline crafter proposal — the on-disk JSONL row."""
    raw: Dict[str, Any]
    proposal_kind: str       # patch | compose | transfer | retire
    proposal_id: str
    proposer: str            # composer | generalizer | hypothesizer | reflector
    target_domains: List[str]
    rationale: str

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "_OfflineProposal":
        return cls(
            raw=dict(row),
            proposal_kind=str(row.get("proposal_kind") or "").lower(),
            proposal_id=str(row.get("proposal_id") or ""),
            proposer=str(row.get("proposer") or ""),
            target_domains=list(row.get("target_domains") or []),
            rationale=str(row.get("rationale") or ""),
        )


def _translate_proposal(
    op: _OfflineProposal,
    *,
    teacher_model: str,
) -> Tuple[BankMutationProposal, str]:
    """Turn an offline proposal into a live ``BankMutationProposal``.

    Returns ``(live_proposal, base_or_subject_skill_id_hint)``.  The
    second element is the offline proposal's *primary* skill reference
    — for patches/retires it's the existing skill we're acting on; for
    compose/transfer it's the first parent (we use it to seed the new
    candidate's ``parent_skill_ids``).  IDs are run through
    ``_safe_skill_id`` so they match the keys we used when seeding.
    """
    raw = op.raw
    target_domains = list(op.target_domains)
    common_kwargs = dict(
        proposal_id=op.proposal_id,
        rationale=op.rationale,
        target_domains=target_domains,
        teacher_model=teacher_model,
        proposed_at=time.time(),
    )

    if op.proposal_kind == "patch":
        target_id = _safe_skill_id(str(raw.get("target_skill_id") or ""))
        return (
            PatchProposal(
                **common_kwargs,
                base_skill_id=target_id,
                patched_protocol=[],                       # offline mirror has no protocol delta
                patched_contract=None,
                recovery_strategy=str(raw.get("patch_kind") or "protocol_patch"),
                parent_skill_ids=[target_id] if target_id else [],
                seed_failure_ids=[],
            ),
            target_id,
        )

    if op.proposal_kind == "retire":
        target_id = _safe_skill_id(str(raw.get("target_skill_id") or ""))
        return (
            RetireProposal(
                **common_kwargs,
                target_skill_id=target_id,
                reason=str(raw.get("retire_reason") or ""),
            ),
            target_id,
        )

    if op.proposal_kind == "compose":
        components = [_safe_skill_id(c) for c in (raw.get("components") or [])]
        first = components[0] if components else ""
        return (
            ComposeProposal(
                **common_kwargs,
                name=f"compose__{'__'.join(components) or 'unknown'}"[:120],
                component_skill_ids=components,
                composed_protocol=[],
                contract=SkillContract(),
                parent_skill_ids=components,
            ),
            first,
        )

    if op.proposal_kind == "transfer":
        source_id = _safe_skill_id(str(raw.get("source_skill_id") or ""))
        source_domain = str(raw.get("source_domain") or "gymv")
        new_per_target = raw.get("new_adapter_per_target") or {}
        target_domain = next(
            (d for d in TRANSFER_TARGET_DOMAINS if d in new_per_target), ""
        )
        return (
            GeneralizeProposal(
                **common_kwargs,
                name=f"transfer__{source_id}__to__{target_domain or 'any'}",
                base_skill_id=source_id,
                abstracted_protocol=[],
                contract=SkillContract(),
                source_domain=source_domain,
                target_domain=target_domain,
                slot_remap=dict((raw.get("slot_remap_per_target") or {})
                                 .get(target_domain, {})),
                demo_selection={},
                demo_episode_ids=[],
                k_shot_budget=5,
                parent_skill_ids=[source_id] if source_id else [],
            ),
            source_id,
        )

    # Unknown kinds become benign hypothesis stubs — never silently dropped.
    return (
        HypothesisProposal(
            **common_kwargs,
            name=f"unknown_kind__{op.proposal_kind or 'empty'}",
            novel_protocol=[],
            contract=SkillContract(),
            source_failure_pattern_ids=[],
        ),
        "",
    )


# ═══════════════════════════════════════════════════════════════════════
# Subject-skill construction for compose / transfer / hypothesis paths
# ═══════════════════════════════════════════════════════════════════════
#
# Patches and retires re-use the seeded CANDIDATE bank skill as the
# gate's subject.  Compose / transfer / hypothesis proposals invent a
# *new* skill — the offline mirror's stand-in for what the live
# SkillCrafterService would have minted.  We construct the candidate
# below so it carries:
#   * ``feasible_domains == proposal.target_domains`` (Stage 0 needs ≥2),
#   * ``source_domains`` from any parent skill that already had one,
#   * ``transfer_target_domains`` = ``feasible_domains ∩ TRANSFER_TARGET_DOMAINS``
#     so Stage 3a's source/target asymmetry path fires (PLAN-SKILL-BANK §0.4).
# Without these the gate either FAILs Stage 0 trivially or falls back
# to the legacy n-domains check on Stage 3a, neither of which is the
# behaviour we want to surface.

_PROPOSAL_KIND_TO_SKILL_TYPE: Dict[str, SkillType] = {
    "compose":  SkillType.MIXED,
    "transfer": SkillType.MIXED,
    "hypothesize": SkillType.REASONING,
}


def _make_compose_subject(
    op: _OfflineProposal,
    bank_by_id: Mapping[str, SkillRecord],
) -> SkillRecord:
    """Subject for a compose proposal.  ``source_type == CRAFTED``
    matches ``ComposeProposal.source_type`` exactly (Stage 0 source-type
    check).  ``source_domains`` is left empty — composed skills earn
    source-domain lineage through Stage 1 / Stage 4 verification, not
    by inheritance from their components.

    When the proposal carries a concrete ``composed_protocol`` /
    ``contract`` payload we adopt it verbatim (production behaviour).
    For Phase-1 offline crafter outputs the payload is currently empty,
    so we synthesize a placeholder protocol from the component list —
    *and bump ``version`` with a deterministic proposal suffix* — so
    sibling proposals on the same components do not collide on
    ``content_hash`` (which deliberately excludes ``proposal_id`` /
    ``parent_skill_ids`` / ``source_type``).
    """
    components = [_safe_skill_id(c) for c in (op.raw.get("components") or [])]
    feasible = sorted(set(op.target_domains) & set(DOMAINS)) or list(DOMAINS)
    role = (op.raw.get("evidence_role") or "COMMIT").upper()

    payload_protocol = list(op.raw.get("composed_protocol") or [])
    payload_contract_raw = op.raw.get("contract") or op.raw.get("composed_contract")
    contract = (
        SkillContract(**payload_contract_raw)
        if isinstance(payload_contract_raw, dict)
        else SkillContract(
            preconditions=[],
            effects_add=[],
            effects_del=[],
            expected_evidence_roles=[role] if role else ["COMMIT"],
            success_criteria=[],
            abort_criteria=[],
        )
    )
    protocol = payload_protocol or [
        {"action": "EXEC", "payload": {}, "notes": f"step:{c}"}
        for c in components
    ] or [{"action": "EXEC", "payload": {}, "notes": "compose"}]
    version = "v1" if payload_protocol else f"v1.offline.{op.proposal_id[-8:]}"
    rec = SkillRecord.new(
        name=op.raw.get("name") or f"compose__{'__then__'.join(components) or 'unknown'}"[:120],
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.CRAFTED,
        feasible_domains=feasible,
        contract=contract,
        protocol=protocol,
        proposal_id=op.proposal_id,
        parent_skill_ids=components,
        source_domains=[],
        transfer_target_domains=[],
    )
    rec.version = version
    return rec


def _make_transfer_subject(
    op: _OfflineProposal,
    bank_by_id: Mapping[str, SkillRecord],
) -> SkillRecord:
    """Subject for a transfer proposal.

    The translated live proposal is a ``GeneralizeProposal`` with both
    ``source_domain`` and ``target_domain`` populated, whose
    ``source_type`` property therefore returns ``FEW_SHOT_ADAPTED``
    (PLAN-UNIFIED-SKILL-GATE §7 Stage 3a).  Stage 0 enforces
    ``proposal.source_type == skill.source_type``, so the subject's
    ``source_type`` must match.  We also leave ``source_domains`` empty
    here — the few-shot adapted skill is what *earns* a source-domain
    entry through Stage 3a verification, not what asserts one up
    front.

    When the proposal supplies a concrete ``abstracted_protocol`` /
    ``contract`` payload we adopt it (production behaviour).  In
    Phase-1 the offline crafter omits the payload, so we clone from
    the parent and bump ``version`` with a deterministic proposal
    suffix to avoid sibling-hash collisions on the same parent.
    """
    src_id = _safe_skill_id(str(op.raw.get("source_skill_id") or ""))
    parent = bank_by_id.get(src_id)
    feasible = sorted(set(op.target_domains) & set(DOMAINS)) or list(DOMAINS)
    role = (op.raw.get("evidence_role") or "COMMIT").upper()

    payload_protocol = list(op.raw.get("abstracted_protocol") or [])
    payload_contract_raw = op.raw.get("contract")
    contract = (
        SkillContract(**payload_contract_raw)
        if isinstance(payload_contract_raw, dict)
        else (parent.contract if parent else SkillContract(expected_evidence_roles=[role]))
    )
    protocol = (
        payload_protocol
        or (list(parent.protocol) if parent else
            [{"action": "EXEC", "payload": {}, "notes": "transfer"}])
    )
    version = "v1" if payload_protocol else f"v1.offline.{op.proposal_id[-8:]}"
    rec = SkillRecord.new(
        name=op.raw.get("name") or f"transfer__{parent.name if parent else src_id}"[:120],
        skill_type=parent.skill_type if parent else SkillType.MIXED,
        source_type=SkillSourceType.FEW_SHOT_ADAPTED,
        feasible_domains=feasible,
        contract=contract,
        protocol=protocol,
        proposal_id=op.proposal_id,
        parent_skill_ids=[src_id] if src_id else [],
        source_domains=[],
        transfer_target_domains=[],
    )
    rec.version = version
    return rec


def _make_patch_subject(
    op: _OfflineProposal,
    base: SkillRecord,
) -> SkillRecord:
    """Synthesize a fresh ``REPAIRED`` candidate descended from ``base``.

    The live ``SkillCrafterService.repair`` path mints a new versioned
    candidate (``source_type == REPAIRED``) that points at the original
    via ``parent_skill_ids``; the original stays in place to serve as
    the rollback target.  We mirror that here so Stage 0's
    ``source_type`` check (proposal=REPAIRED) passes against the
    subject (skill=REPAIRED) instead of FAILing against the seeded
    MINED record.

    When the proposal carries a ``patched_protocol`` / ``patched_contract``
    payload we adopt it (production behaviour: the patch *changes* the
    skill body).  Phase-1 offline crafter outputs leave the payload
    empty, so we clone the base and bump ``version`` with a
    deterministic proposal suffix — sibling patches on the same base
    (e.g., R2 warrant + R3 precondition + R5 transfer on
    ``COMMIT/MERGE``) would otherwise collapse to the same
    ``content_hash`` (which deliberately ignores
    ``proposal_id``/``parent_skill_ids``/``source_type``).
    """
    feasible = sorted(set(op.target_domains) & set(DOMAINS)) or list(base.feasible_domains)
    role = (op.raw.get("evidence_role") or "COMMIT").upper()

    payload_protocol = list(op.raw.get("patched_protocol") or [])
    payload_contract_raw = op.raw.get("patched_contract")
    contract = (
        SkillContract(**payload_contract_raw)
        if isinstance(payload_contract_raw, dict)
        else (base.contract or SkillContract(expected_evidence_roles=[role]))
    )
    protocol = (
        payload_protocol
        or list(base.protocol)
        or [{"action": "EXEC", "payload": {}, "notes": "patch"}]
    )
    version = "v1" if payload_protocol else f"v1.offline.{op.proposal_id[-8:]}"
    rec = SkillRecord.new(
        name=f"patch__{base.name}"[:120],
        skill_type=base.skill_type,
        source_type=SkillSourceType.REPAIRED,
        feasible_domains=feasible,
        contract=contract,
        protocol=protocol,
        proposal_id=op.proposal_id,
        parent_skill_ids=[base.skill_id],
        source_domains=[],
        transfer_target_domains=[],
    )
    rec.version = version
    return rec


def _make_hypothesis_subject(op: _OfflineProposal) -> SkillRecord:
    """Subject for a hypothesis proposal.  ``HypothesisProposal.source_type``
    returns ``TEACHER`` when ``teacher_model`` is set, else ``CRAFTED``;
    we always set ``teacher_model`` in ``_translate_proposal``, so the
    subject is pinned to ``TEACHER``.

    When the proposal supplies a concrete ``novel_protocol`` /
    ``contract`` payload (production), we adopt it.  In Phase-1 the
    offline crafter emits empty payloads, so we synthesize a
    placeholder protocol and bump ``version`` with a deterministic
    proposal suffix to avoid sibling-hash collisions.
    """
    feasible = sorted(set(op.target_domains) & set(DOMAINS)) or list(DOMAINS)
    payload_protocol = list(op.raw.get("novel_protocol") or [])
    payload_contract_raw = op.raw.get("contract")
    contract = (
        SkillContract(**payload_contract_raw)
        if isinstance(payload_contract_raw, dict)
        else SkillContract(expected_evidence_roles=["REASON"])
    )
    protocol = payload_protocol or [
        {"action": "EXEC", "payload": {}, "notes": "hypothesis"}
    ]
    version = "v1" if payload_protocol else f"v1.offline.{op.proposal_id[-8:]}"
    rec = SkillRecord.new(
        name=op.raw.get("name") or f"hypothesis__{op.proposal_id[:24]}",
        skill_type=SkillType.REASONING,
        source_type=SkillSourceType.TEACHER,
        feasible_domains=feasible,
        contract=contract,
        protocol=protocol,
        proposal_id=op.proposal_id,
        source_domains=[],
        transfer_target_domains=[],
    )
    rec.version = version
    return rec


# ═══════════════════════════════════════════════════════════════════════
# External gate-verdicts loading (forward-compat for Harness mirror)
# ═══════════════════════════════════════════════════════════════════════

def _load_external_verdicts(
    gate_run: Optional[Path],
    corpus: str,
    source: str,
) -> Dict[str, SkillEvaluationRecord]:
    """When the Harness ``GateRunner`` mirror lands and writes
    ``gate_verdicts.jsonl`` under its own output dir, we consume those
    verdicts verbatim instead of running ``GateService`` inline.

    Returns an empty dict when ``--gate-verdicts-run`` is not provided —
    the Phase-1 default, in which case we fall through to the inline
    GateService path.
    """
    if gate_run is None:
        return {}
    p = gate_run / corpus / source / "gate_verdicts.jsonl"
    if not p.exists():
        return {}

    out: Dict[str, SkillEvaluationRecord] = {}
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                ev = _eval_record_from_json(d)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("skip malformed gate verdict: %s", exc)
                continue
            out[ev.proposal_id] = ev
    return out


def _eval_record_from_json(d: Dict[str, Any]) -> SkillEvaluationRecord:
    v = d.get("verdict") or {}
    stages: List[StageVerdict] = []
    for s in v.get("stages") or []:
        try:
            stages.append(StageVerdict(
                stage=GateStage(s.get("stage")),
                verdict=GateVerdict(s.get("verdict")),
                metrics={k: float(val) for k, val in (s.get("metrics") or {}).items()},
                failures=list(s.get("failures") or []),
                notes=str(s.get("notes") or ""),
            ))
        except Exception:
            continue
    payload = GateVerdictPayload(
        proposal_id=v.get("proposal_id") or d.get("proposal_id", ""),
        skill_id=v.get("skill_id") or d.get("skill_id", ""),
        skill_content_hash=v.get("skill_content_hash") or d.get("skill_content_hash", ""),
        stages=stages,
        final_verdict=GateVerdict(v.get("final_verdict") or "fail"),
        rationale=str(v.get("rationale") or ""),
        eligible_domains=list(v.get("eligible_domains") or []),
        notes=v.get("notes"),
    )
    return SkillEvaluationRecord(
        evaluation_id=d.get("evaluation_id", ""),
        proposal_id=d.get("proposal_id", ""),
        skill_id=d.get("skill_id", ""),
        skill_content_hash=d.get("skill_content_hash", ""),
        episode_ids=list(d.get("episode_ids") or []),
        verdict=payload,
        metrics={k: float(val) for k, val in (d.get("metrics") or {}).items()},
        failure_class_distribution=dict(d.get("failure_class_distribution") or {}),
        transfer_labels=dict(d.get("transfer_labels") or {}),
        judge_model=d.get("judge_model"),
        seed=d.get("seed"),
        started_at=d.get("started_at"),
        finished_at=d.get("finished_at"),
    )


# ═══════════════════════════════════════════════════════════════════════
# Optional: actor batch metrics from --actions-run (rollback signal)
# ═══════════════════════════════════════════════════════════════════════

def _load_actor_batch_metrics(
    actions_run: Optional[Path],
    corpus: str,
    source: str,
) -> Dict[str, Dict[str, float]]:
    """Pull crude per-skill metrics out of the cold-start actor batch
    summary.

    Honours §6.2 #2: the offline rollback signal comes from the
    cold-start summary files emitted by ``cold_start_labeling`` (we
    never re-run a live actor).  Returns
    ``{skill_id_safe: {"selections": float, "pass_rate": float}}``.

    Backwards-compatible filename / field detection:

    * filename — ``_skill_actions_summary.json`` (current cold-start
      output), with fallback to ``_run_summary.json`` (older runs).
    * selections histogram — ``selection_histogram`` (current), with
      fallback to ``skill_selection_histogram``.
    * per-skill pass rate — ``skill_pass_rates`` / ``pass_rates``
      (preferred when present).
    * if no per-skill pass rate is present, derive a global
      ``1 − n_failed/n_with_skill`` proxy and apply it uniformly so a
      fully-failing batch can still drive a rollback while a healthy
      0-failure batch leaves promoted skills untouched.

    Empty dict if no summary file is found.
    """
    if actions_run is None:
        return {}
    base = actions_run / corpus / source
    candidate_files = [
        base / "_skill_actions_summary.json",
        base / "_run_summary.json",
    ]
    summary_path: Optional[Path] = next(
        (p for p in candidate_files if p.exists()), None,
    )
    if summary_path is None:
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    sel_hist = (
        data.get("selection_histogram")
        or data.get("skill_selection_histogram")
        or {}
    )
    per_skill_rates = data.get("skill_pass_rates") or data.get("pass_rates") or {}

    n_failed = float(data.get("n_failed") or 0.0)
    n_with_skill = float(data.get("n_with_skill") or 0.0)
    global_pass_rate: Optional[float] = None
    if n_with_skill > 0:
        global_pass_rate = max(0.0, 1.0 - (n_failed / n_with_skill))

    for sid, n in sel_hist.items():
        safe = _safe_skill_id(str(sid))
        out.setdefault(safe, {})["selections"] = float(n or 0)

    for sid, pr in per_skill_rates.items():
        safe = _safe_skill_id(str(sid))
        out.setdefault(safe, {})["pass_rate"] = float(pr or 0.0)

    if global_pass_rate is not None:
        for safe, m in out.items():
            m.setdefault("pass_rate", global_pass_rate)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Gate evaluation — three modes: offline-synthetic, live, external
# ═══════════════════════════════════════════════════════════════════════

GATE_MODE_OFFLINE_SYNTHETIC = "offline-synthetic"
GATE_MODE_LIVE = "live"
GATE_MODE_EXTERNAL = "external"
# Block B3 — "w/o lifecycle gating" ablation.  Auto-passes every
# proposal at every stage (PASS, not LIMITED_PASS), driving DRAFT →
# ACTIVE directly with no gate logic.  Keeps the rest of the
# Promotion subprocess machinery (transactional release, audit,
# writeback) intact so downstream analysis paths still work.
GATE_MODE_PERMISSIVE = "permissive"
# ``offline-with-llm-judge`` extends ``offline-synthetic`` with a single
# 35B-A3B (BACKBONE_JUDGE_MODEL) call per proposal — the synthetic
# stages still fire (so the verdict surface is unchanged for downstream
# consumers), and the LLM judge's verdict is appended as an extra
# StageVerdict.  An LLM-graded ``fail`` flips the aggregate to FAIL ⇒
# REJECT, otherwise behaviour matches the synthetic floor (PROMOTE to
# PROVISIONAL on LIMITED_PASS).  See ``_llm_skill_judge.py`` for the
# call shape; routing to the local 35B is via ``VLLM_BASE_URL_MAP`` /
# ``API_func.ask_model``.
GATE_MODE_OFFLINE_LLM_JUDGE = "offline-with-llm-judge"


def _build_gate_service() -> GateService:
    """Construct a GateService with a *minimal* live harness — only used
    when ``--gate-mode live`` is selected.  The default ``offline-synthetic``
    mode never calls this."""
    registry = AdapterRegistry()
    harness = SkillHarness(registry, reward_logger=RewardLogger())
    return GateService(
        harness=harness,
        thresholds=OrchestratorConfig().gate_thresholds,
    )


# ---------------------------------------------------------------------------
# Rule-based Stage 0 (static) — mirror of orchestrator.gate_service._run_static
# kept self-contained so we don't have to instantiate a full GateService for
# the offline-synthetic path.  The check list is the same one the live code
# runs; only the protocol-non-empty rule is relaxed for retires (PLAN-UNIFIED-
# SKILL-GATE §7 Stage 0).
# ---------------------------------------------------------------------------

def _rule_stage_0_static(
    skill: SkillRecord, proposal: BankMutationProposal,
) -> StageVerdict:
    failures: List[str] = []
    if len(set(skill.feasible_domains)) < 2:
        failures.append("feasible_domains < 2 (general-protocol invariant)")
    for d in skill.feasible_domains:
        if d not in DOMAINS:
            failures.append(f"unknown_domain={d!r}")
    if (not skill.contract.expected_evidence_roles
            and skill.skill_type.value != "action"):
        failures.append("contract.expected_evidence_roles empty (G0)")
    if not isinstance(proposal, RetireProposal) and not skill.protocol:
        failures.append("skill.protocol is empty")
    if proposal.source_type != skill.source_type:
        failures.append(
            f"source_type mismatch: proposal={proposal.source_type.value}, "
            f"skill={skill.source_type.value}"
        )
    if isinstance(proposal, ComposeProposal) and not proposal.component_skill_ids:
        failures.append("ComposeProposal.component_skill_ids is empty")
    if (isinstance(proposal, (GeneralizeProposal, PatchProposal))
            and not getattr(proposal, "base_skill_id", "")):
        failures.append("base_skill_id is empty")
    verdict = GateVerdict.PASS if not failures else GateVerdict.FAIL
    return StageVerdict(stage=GateStage.STATIC, verdict=verdict, failures=failures)


def _build_synthetic_evaluation(
    *,
    proposal: BankMutationProposal,
    skill: SkillRecord,
    judge_model: str,
) -> SkillEvaluationRecord:
    """Phase-1 offline mirror verdict: rule-based Stage 0 + LIMITED_PASS
    placeholders for Stages 1–4.

    This is the *documented* Phase-1 behaviour: nothing reaches ACTIVE
    until the Harness mirror lands and a real
    ``orchestrator.eval_suite.run`` is plumbed through — but
    ``LIMITED_PASS`` correctly maps to ``PROVISIONAL`` so the
    transactional spine of the offline ``PromotionOrchestrator`` (snapshot,
    audit, release) is exercised end-to-end on the cold-start corpus.
    """
    stages: List[StageVerdict] = []

    stages.append(_rule_stage_0_static(skill, proposal))

    stages.append(StageVerdict(
        stage=GateStage.REPLAY,
        verdict=GateVerdict.LIMITED_PASS,
        metrics={"n_seeds": 0.0},
        notes="offline-mirror: no replay seeds available",
    ))

    stages.append(StageVerdict(
        stage=GateStage.SHADOW,
        verdict=GateVerdict.LIMITED_PASS,
        notes="offline-mirror: no shadow log available",
    ))

    # Stage 3a — synthetic LIMITED_PASS scaffolding.  We emit metrics
    # tagged ``offline_mirror`` so the per-target counters in the gate
    # dashboard never confuse this run with a real few-shot probe.  The
    # diagnostics list the four target domains that *would* be probed
    # once the Harness mirror lands.
    targets = [
        d for d in (skill.transfer_target_domains or skill.feasible_domains)
        if d in TRANSFER_TARGET_DOMAINS
    ]
    transfer_metrics: Dict[str, float] = {
        "n_targets": float(len(targets)),
        "n_verified_targets": 0.0,
        "min_verified_targets": 1.0,
    }
    for tgt in targets:
        transfer_metrics[f"pass_rate.{tgt}"] = 0.0
        transfer_metrics[f"k_used.{tgt}"] = 0.0
    stages.append(StageVerdict(
        stage=GateStage.TRANSFER,
        verdict=GateVerdict.LIMITED_PASS,
        metrics=transfer_metrics,
        notes=(
            f"offline-mirror: deferred to harness GateRunner mirror; "
            f"targets={targets}"
        ),
    ))

    stages.append(StageVerdict(
        stage=GateStage.NON_REGRESSION,
        verdict=GateVerdict.LIMITED_PASS,
        notes="offline-mirror: no baseline/post score available",
    ))

    # Aggregate per the live GateService rules.
    any_fail = any(s.verdict == GateVerdict.FAIL for s in stages)
    any_limited = any(s.verdict == GateVerdict.LIMITED_PASS for s in stages)
    if any_fail:
        failing = [s.stage.value for s in stages if s.verdict == GateVerdict.FAIL]
        rationale = f"failed_stages={failing}"
        final_verdict = GateVerdict.FAIL
        eligible: List[str] = []
    elif any_limited:
        rationale = "promotion_to_provisional_only (offline-mirror synthetic)"
        final_verdict = GateVerdict.LIMITED_PASS
        eligible = list(skill.source_domains) or list(skill.feasible_domains)
    else:
        rationale = "all_stages_pass"
        final_verdict = GateVerdict.PASS
        eligible = list(skill.source_domains) or list(skill.feasible_domains)

    payload = GateVerdictPayload(
        proposal_id=proposal.proposal_id,
        skill_id=skill.skill_id,
        skill_content_hash=skill.content_hash(),
        stages=stages,
        final_verdict=final_verdict,
        rationale=rationale,
        eligible_domains=eligible,
        notes="offline-mirror:synthetic",
    )

    now = time.time()
    flat_metrics = {
        f"{s.stage.value}.{k}": float(v)
        for s in stages for k, v in s.metrics.items()
    }
    return SkillEvaluationRecord(
        evaluation_id=f"eval-offline-{proposal.proposal_id[-12:]}-{int(now*1000)%10**8}",
        proposal_id=proposal.proposal_id,
        skill_id=skill.skill_id,
        skill_content_hash=skill.content_hash(),
        episode_ids=[],
        verdict=payload,
        metrics=flat_metrics,
        failure_class_distribution={},
        transfer_labels={},
        judge_model=judge_model,
        seed=None,
        started_at=now,
        finished_at=now,
    )


def _build_permissive_evaluation(
    *,
    proposal: BankMutationProposal,
    skill: SkillRecord,
    judge_model: str,
) -> SkillEvaluationRecord:
    """Block B3: auto-PASS every stage for the "w/o lifecycle gating"
    ablation.  Drives DRAFT → ACTIVE directly so the §5.5 ablation
    can isolate the gate's contribution from the crafter and harness.

    Differs from ``_build_synthetic_evaluation`` in three ways:
      * Stage 0 is auto-PASS (we don't even run ``_rule_stage_0_static``).
      * Stages 1/2/3a/4 are PASS instead of LIMITED_PASS.
      * The aggregate verdict is PASS — promotes to ACTIVE rather than
        capping at PROVISIONAL.

    This is intentionally lossy (skips static checks too) — the
    ablation is "no gate at all", not "no LLM judge".
    """
    stages: List[StageVerdict] = [
        StageVerdict(
            stage=GateStage.STATIC,
            verdict=GateVerdict.PASS,
            notes="permissive: gate bypassed (block B3 ablation)",
        ),
        StageVerdict(
            stage=GateStage.REPLAY, verdict=GateVerdict.PASS,
            notes="permissive: gate bypassed",
        ),
        StageVerdict(
            stage=GateStage.SHADOW, verdict=GateVerdict.PASS,
            notes="permissive: gate bypassed",
        ),
        StageVerdict(
            stage=GateStage.TRANSFER, verdict=GateVerdict.PASS,
            notes="permissive: gate bypassed",
        ),
        StageVerdict(
            stage=GateStage.NON_REGRESSION, verdict=GateVerdict.PASS,
            notes="permissive: gate bypassed",
        ),
    ]
    eligible = list(skill.source_domains) or list(skill.feasible_domains)
    payload = GateVerdictPayload(
        proposal_id=proposal.proposal_id,
        skill_id=skill.skill_id,
        skill_content_hash=skill.content_hash(),
        stages=stages,
        final_verdict=GateVerdict.PASS,
        rationale="permissive_bypass",
        eligible_domains=eligible,
        notes="permissive:auto-promote",
    )
    now = time.time()
    return SkillEvaluationRecord(
        evaluation_id=f"eval-perm-{proposal.proposal_id[-12:]}-{int(now*1000)%10**8}",
        proposal_id=proposal.proposal_id,
        skill_id=skill.skill_id,
        skill_content_hash=skill.content_hash(),
        episode_ids=[],
        verdict=payload,
        metrics={},
        failure_class_distribution={},
        transfer_labels={},
        judge_model=judge_model,
        seed=None,
        started_at=now,
        finished_at=now,
    )


def _evaluate_proposal_live(
    gate: GateService,
    *,
    proposal: BankMutationProposal,
    skill: SkillRecord,
) -> SkillEvaluationRecord:
    """Run the live ``GateService.evaluate`` end-to-end.  Used only when
    ``--gate-mode live`` is selected.  Stages 1 / 2 / 4 fall back to
    ``LIMITED_PASS`` (no replay seeds / shadow log / baseline scores);
    Stage 0 fires verbatim and Stage 3a falls back to its few-shot path,
    which currently FAILs in the offline mirror because no target
    adapters are registered."""
    return gate.evaluate(
        proposal=proposal,
        skill=skill,
        replay_seeds=(),
        shadow_log=None,
        baseline_score=None,
        post_score=None,
        few_shot_demos=None,
    )


def _evaluate_proposal_offline_with_llm_judge(
    *,
    proposal: BankMutationProposal,
    skill: SkillRecord,
    judge_model: str,
    corpus_hint: Optional[str] = None,
    source_hint: Optional[str] = None,
    enable_thinking: bool = False,
    max_tokens: int = 256,
) -> SkillEvaluationRecord:
    """``offline-synthetic`` floor + one extra 35B LLM-judge stage.

    Always runs the same synthetic stages as
    :func:`_build_synthetic_evaluation` first (so all dashboards keyed
    off the per-stage shape keep working), then issues a single
    ``API_func.ask_model`` call against ``judge_model`` and appends the
    parsed verdict as a sixth ``GateStage.STATIC`` entry tagged
    ``llm-judge:`` in its ``notes``.  The aggregate ``final_verdict``
    is recomputed across the augmented stage list using the same
    aggregation rules as the synthetic path:

    * any FAIL  ⇒ FAIL          (REJECT)
    * any LIMITED_PASS ⇒ LIMITED_PASS (PROMOTE→PROVISIONAL)
    * else PASS                  (PROMOTE→ACTIVE)

    Failure-mode contract (per :mod:`_llm_skill_judge`): if the LLM
    call raises or returns garbage, the judge stage degrades to
    ``LIMITED_PASS`` (no override of the synthetic floor) and the
    Promotion driver continues unaffected.
    """
    base_record = _build_synthetic_evaluation(
        proposal=proposal, skill=skill, judge_model=judge_model,
    )

    from labeling_supplement._llm_skill_judge import (
        build_stage_verdict,
        judge_proposal,
    )

    # ``corpus_hint`` is the bank corpus (e.g. ``env_wrappers``) and
    # ``source_hint`` is the per-game directory name (e.g.
    # ``gymv_thunder_force_iii``).  We pass the source hint to the
    # prompt so the judge can reason about feasibility against the
    # actual game's action vocabulary rather than the corpus-level
    # group.
    game_hint = source_hint or corpus_hint
    outcome = judge_proposal(
        proposal=proposal,
        skill=skill,
        model=judge_model,
        game_hint=game_hint,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    extra_stage = build_stage_verdict(outcome)

    payload = base_record.verdict
    augmented_stages: List[StageVerdict] = list(payload.stages) + [extra_stage]

    any_fail    = any(s.verdict == GateVerdict.FAIL         for s in augmented_stages)
    any_limited = any(s.verdict == GateVerdict.LIMITED_PASS for s in augmented_stages)
    if any_fail:
        failing = [s.stage.value for s in augmented_stages if s.verdict == GateVerdict.FAIL]
        rationale = f"failed_stages={failing}; llm_judge={outcome.rationale!r}"
        final_verdict = GateVerdict.FAIL
        eligible: List[str] = []
    elif any_limited:
        rationale = (
            f"promotion_to_provisional_only "
            f"(offline-mirror synthetic + llm-judge:{outcome.verdict.value})"
        )
        final_verdict = GateVerdict.LIMITED_PASS
        eligible = list(skill.source_domains) or list(skill.feasible_domains)
    else:
        rationale = "all_stages_pass (incl. llm-judge)"
        final_verdict = GateVerdict.PASS
        eligible = list(skill.source_domains) or list(skill.feasible_domains)

    new_payload = GateVerdictPayload(
        proposal_id=payload.proposal_id,
        skill_id=payload.skill_id,
        skill_content_hash=payload.skill_content_hash,
        stages=augmented_stages,
        final_verdict=final_verdict,
        rationale=rationale,
        eligible_domains=eligible,
        notes="offline-with-llm-judge",
    )

    flat_metrics = {
        f"{s.stage.value}.{k}": float(v)
        for s in augmented_stages
        for k, v in s.metrics.items()
    }

    base_record.verdict = new_payload
    base_record.metrics = flat_metrics
    base_record.judge_model = judge_model
    return base_record


# ═══════════════════════════════════════════════════════════════════════
# Decision and persistence per (corpus, source)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _PromotionDecision:
    proposal_id: str
    subject_skill_id: str
    subject_skill_content_hash: str
    evaluation_id: str
    final_verdict: str                             # "pass" | "limited_pass" | "fail"
    decision: str                                  # PROMOTE | REJECT | DEFER | ROLLBACK
    from_status: str
    target_status: str
    release_id: Optional[str]
    from_snapshot: Optional[str]
    to_snapshot: Optional[str]
    rationale: str
    audit_event_kind: str                          # release | rejection | rollback
    triggered_by: str                              # gate_pass | gate_fail | non_regression_fail | retire_proposal
    linked_eval_record: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "proposal_id":                self.proposal_id,
            "subject_skill_id":           self.subject_skill_id,
            "subject_skill_content_hash": self.subject_skill_content_hash,
            "evaluation_id":              self.evaluation_id,
            "final_verdict":              self.final_verdict,
            "decision":                   self.decision,
            "from_status":                self.from_status,
            "target_status":              self.target_status,
            "release_id":                 self.release_id,
            "from_snapshot":              self.from_snapshot,
            "to_snapshot":                self.to_snapshot,
            "rationale":                  self.rationale,
            "audit_event_kind":           self.audit_event_kind,
            "triggered_by":               self.triggered_by,
            "linked_eval_record":         self.linked_eval_record,
        }


def _missing_base_decision(
    op: "_OfflineProposal",
    subject_hint_id: str,
    bank_path: Path,
) -> "_PromotionDecision":
    """Synthetic REJECT decision for proposals whose ``base_skill_id`` /
    ``target_skill_id`` does not resolve in the seeded bank.  Recorded
    so the audit trail still carries the proposal even when the
    subject is missing."""
    return _PromotionDecision(
        proposal_id=op.proposal_id,
        subject_skill_id=subject_hint_id,
        subject_skill_content_hash="",
        evaluation_id="",
        final_verdict="fail",
        decision="REJECT",
        from_status="missing",
        target_status=SkillStatus.REJECTED.value,
        release_id=None,
        from_snapshot=None,
        to_snapshot=None,
        rationale=(
            f"Subject skill {subject_hint_id!r} not found in bank "
            f"snapshot {bank_path}."
        ),
        audit_event_kind="rejection",
        triggered_by="missing_base_skill",
        linked_eval_record="",
    )


@dataclass
class _SourceRunResult:
    corpus: str
    source: str
    n_proposals: int
    decisions: List[_PromotionDecision] = field(default_factory=list)
    evaluations: List[SkillEvaluationRecord] = field(default_factory=list)
    by_decision: Counter = field(default_factory=Counter)
    by_verdict:  Counter = field(default_factory=Counter)
    by_kind:     Counter = field(default_factory=Counter)
    by_target_status: Counter = field(default_factory=Counter)
    n_rollbacks: int = 0
    n_defers: int = 0
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    release_id: Optional[str] = None
    snapshot_ids: List[str] = field(default_factory=list)
    n_skills_seeded: int = 0
    release: Optional[RunRelease] = None


def _decide_per_source(
    *,
    corpus: str,
    source: str,
    proposals_run: Path,
    bank_run: Path,
    actions_run: Optional[Path],
    gate_run: Optional[Path],
    gate_mode: str,
    output_root: Path,
    cfg: Dict[str, Any],
    teacher_model: str,
    judge_model: str,
    enable_thinking: bool = False,
    judge_max_tokens: int = 256,
) -> _SourceRunResult:
    t0 = time.time()
    res = _SourceRunResult(corpus=corpus, source=source, n_proposals=0)
    proposals_path = proposals_run / corpus / source / "proposals.jsonl"
    bank_path = bank_run / corpus / source / "skill_bank.jsonl"
    out_src = output_root / corpus / source
    out_src.mkdir(parents=True, exist_ok=True)
    snap_dir = out_src / "bank_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    if not proposals_path.exists():
        res.error = f"missing-proposals: {proposals_path}"
        _write_per_source(res, out_src, cfg, teacher_model, judge_model)
        return res
    if not bank_path.exists():
        res.error = f"missing-bank: {bank_path}"
        _write_per_source(res, out_src, cfg, teacher_model, judge_model)
        return res

    # ── temp live stack ──
    temp_root = Path(tempfile.mkdtemp(prefix=f"promotion_mirror_{corpus}_{source}_"))
    try:
        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(temp_root / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(temp_root / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(temp_root / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(temp_root / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(temp_root / "artifacts"))
        snapshots = SnapshotManager(artifacts)
        gate: Optional[GateService] = (
            _build_gate_service() if gate_mode == GATE_MODE_LIVE else None
        )
        promoter = PromotionOrchestrator(
            lifecycle=lifecycle,
            snapshot_manager=snapshots,
            artifact_store=artifacts,
        )

        # The proposals always declare all five domains as their target
        # set (PLAN-SKILL-CRAFTER §2.5); we mirror that into the seeded
        # subjects so Stage 0's general-protocol invariant passes.
        n_seeded, _, bank_by_id = _seed_bank_as_candidates(
            lifecycle, bank_path, target_domains_for_subject=DOMAINS,
        )
        res.n_skills_seeded = n_seeded

        external_verdicts = _load_external_verdicts(gate_run, corpus, source)
        actor_metrics = _load_actor_batch_metrics(actions_run, corpus, source)

        # ── load proposals ──
        proposals: List[_OfflineProposal] = []
        with proposals_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    proposals.append(_OfflineProposal.from_json(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        res.n_proposals = len(proposals)

        # ── per-proposal: translate, materialise subject, evaluate ──
        plan = PromotionPlan()
        # Tracks (proposal_id, subject_id, target_status, eval, kind)
        plan_meta: List[Tuple[str, str, SkillStatus, SkillEvaluationRecord, str]] = []
        # Decisions for proposals that DON'T enter the promote() batch
        # (FAIL → REJECTED, retires → DEPRECATED → ROLLED_BACK).
        side_decisions: List[_PromotionDecision] = []

        for op in proposals:
            res.by_kind[op.proposal_kind] += 1
            live_proposal, subject_hint_id = _translate_proposal(
                op, teacher_model=teacher_model,
            )

            # ── materialise the subject skill the gate will see ──
            subject: Optional[SkillRecord] = None
            from_status = SkillStatus.CANDIDATE.value
            if op.proposal_kind == "retire":
                # Retire targets the existing seeded skill in place.
                subject = bank_by_id.get(subject_hint_id)
                if subject is None:
                    side_decisions.append(_missing_base_decision(
                        op, subject_hint_id, bank_path,
                    ))
                    continue
            elif op.proposal_kind == "patch":
                # Patches mint a fresh REPAIRED descendant — Stage 0's
                # source_type check would otherwise FAIL against the
                # original MINED record.  This mirrors the live
                # SkillCrafterService.repair flow.
                base = bank_by_id.get(subject_hint_id)
                if base is None:
                    side_decisions.append(_missing_base_decision(
                        op, subject_hint_id, bank_path,
                    ))
                    continue
                subject = _make_patch_subject(op, base)
                try:
                    lifecycle.ingest_draft(subject)
                    lifecycle.transition(
                        subject.skill_id,
                        to_status=SkillStatus.CANDIDATE,
                        rationale=f"patch-subject-for-{op.proposal_id}",
                    )
                except LifecycleError:
                    pass
            elif op.proposal_kind == "compose":
                subject = _make_compose_subject(op, bank_by_id)
                lifecycle.ingest_draft(subject)
                lifecycle.transition(
                    subject.skill_id,
                    to_status=SkillStatus.CANDIDATE,
                    rationale=f"compose-subject-for-{op.proposal_id}",
                )
                from_status = SkillStatus.CANDIDATE.value
            elif op.proposal_kind == "transfer":
                subject = _make_transfer_subject(op, bank_by_id)
                try:
                    lifecycle.ingest_draft(subject)
                    lifecycle.transition(
                        subject.skill_id,
                        to_status=SkillStatus.CANDIDATE,
                        rationale=f"transfer-subject-for-{op.proposal_id}",
                    )
                except LifecycleError:
                    # Already exists from a prior proposal — re-use.
                    pass
                from_status = SkillStatus.CANDIDATE.value
            else:
                subject = _make_hypothesis_subject(op)
                try:
                    lifecycle.ingest_draft(subject)
                    lifecycle.transition(
                        subject.skill_id,
                        to_status=SkillStatus.CANDIDATE,
                        rationale=f"hypothesis-subject-for-{op.proposal_id}",
                    )
                except LifecycleError:
                    pass
                from_status = SkillStatus.CANDIDATE.value

            # ── verdict ──
            #
            # Three-way dispatch matching ``--gate-mode``:
            #   external  →  consume verbatim from gate_verdicts.jsonl
            #   live      →  call GateService.evaluate(...)
            #   offline-  →  rule-based Stage 0 + LIMITED_PASS
            #   synthetic    placeholders for Stages 1–4 (default,
            #                Phase-1 documented behaviour).
            ev: Optional[SkillEvaluationRecord] = external_verdicts.get(op.proposal_id)
            if ev is not None:
                ev.judge_model = ev.judge_model or judge_model
            elif gate_mode == GATE_MODE_LIVE and gate is not None:
                ev = _evaluate_proposal_live(
                    gate, proposal=live_proposal, skill=subject,
                )
                ev.judge_model = judge_model
            elif gate_mode == GATE_MODE_OFFLINE_LLM_JUDGE:
                ev = _evaluate_proposal_offline_with_llm_judge(
                    proposal=live_proposal,
                    skill=subject,
                    judge_model=judge_model,
                    corpus_hint=corpus,
                    source_hint=source,
                    enable_thinking=enable_thinking,
                    max_tokens=judge_max_tokens,
                )
            elif gate_mode == GATE_MODE_PERMISSIVE:
                # Block B3 — w/o lifecycle gating ablation.
                ev = _build_permissive_evaluation(
                    proposal=live_proposal, skill=subject,
                    judge_model=judge_model,
                )
            else:
                ev = _build_synthetic_evaluation(
                    proposal=live_proposal, skill=subject,
                    judge_model=judge_model,
                )
            artifacts.put_evaluation(ev)
            res.evaluations.append(ev)
            res.by_verdict[ev.verdict.final_verdict.value] += 1

            # ── retire path: ALWAYS DEPRECATE → ROLLED_BACK regardless of verdict ──
            #
            # ``PromotionOrchestrator.rollback`` requires the subject to
            # already be in {ACTIVE, PROVISIONAL, SHADOW, DEPRECATED}.
            # Cold-start retires target CANDIDATE skills (the bank is
            # never promoted offline) so we drive the transition
            # CANDIDATE → DEPRECATED → ROLLED_BACK manually through the
            # lifecycle manager and emit the equivalent audit event.
            # Both transitions are explicitly allowed by
            # ``skill_bank.lifecycle._ALLOWED``.
            if isinstance(live_proposal, RetireProposal):
                retire_reason = (
                    f"retire_proposal:{live_proposal.reason or 'evidence-starved'} "
                    f"({op.proposal_id})"
                )
                final_status = SkillStatus.ROLLED_BACK
                try:
                    if subject.status not in {
                        SkillStatus.ROLLED_BACK,
                        SkillStatus.DEPRECATED,
                    }:
                        if subject.status != SkillStatus.DEPRECATED:
                            lifecycle.transition(
                                subject.skill_id,
                                to_status=SkillStatus.DEPRECATED,
                                rationale=f"pre-rollback: {retire_reason}",
                            )
                        lifecycle.transition(
                            subject.skill_id,
                            to_status=SkillStatus.ROLLED_BACK,
                            rationale=retire_reason,
                        )
                    artifacts.append_audit({
                        "kind":          "rollback",
                        "skill_id":      subject.skill_id,
                        "reason":        retire_reason,
                        "evaluation_id": ev.evaluation_id,
                        "proposal_id":   op.proposal_id,
                        "triggered_by":  "retire_proposal",
                    })
                except LifecycleError as exc:
                    logger.debug(
                        "retire path could not transition %s: %s",
                        subject.skill_id, exc,
                    )
                    final_status = subject.status
                side_decisions.append(_PromotionDecision(
                    proposal_id=op.proposal_id,
                    subject_skill_id=subject.skill_id,
                    subject_skill_content_hash=subject.content_hash(),
                    evaluation_id=ev.evaluation_id,
                    final_verdict=ev.verdict.final_verdict.value,
                    decision="ROLLBACK",
                    from_status=from_status,
                    target_status=final_status.value,
                    release_id=None,
                    from_snapshot=None,
                    to_snapshot=None,
                    rationale=live_proposal.reason or "retire-proposal",
                    audit_event_kind="rollback",
                    triggered_by="retire_proposal",
                    linked_eval_record=ev.evaluation_id,
                ))
                res.n_rollbacks += 1
                continue

            verdict = ev.verdict.final_verdict
            if verdict == GateVerdict.FAIL:
                # Direct REJECT path — bypass promote() (which refuses
                # FAIL).  Live equivalent is "skip promotion, archive
                # the proposal", which we model by transitioning the
                # candidate to REJECTED.
                try:
                    lifecycle.transition(
                        subject.skill_id,
                        to_status=SkillStatus.REJECTED,
                        rationale=f"gate-fail:{ev.evaluation_id}",
                    )
                except LifecycleError as exc:
                    logger.debug(
                        "could not REJECT %s: %s — recording decision regardless",
                        subject.skill_id, exc,
                    )
                artifacts.append_audit({
                    "kind": "rejection",
                    "skill_id": subject.skill_id,
                    "evaluation_id": ev.evaluation_id,
                    "proposal_id": op.proposal_id,
                    "triggered_by": "gate_fail",
                })
                side_decisions.append(_PromotionDecision(
                    proposal_id=op.proposal_id,
                    subject_skill_id=subject.skill_id,
                    subject_skill_content_hash=subject.content_hash(),
                    evaluation_id=ev.evaluation_id,
                    final_verdict=verdict.value,
                    decision="REJECT",
                    from_status=from_status,
                    target_status=SkillStatus.REJECTED.value,
                    release_id=None,
                    from_snapshot=None,
                    to_snapshot=None,
                    rationale=ev.verdict.rationale or "gate FAIL",
                    audit_event_kind="rejection",
                    triggered_by="gate_fail",
                    linked_eval_record=ev.evaluation_id,
                ))
                continue

            # PASS / LIMITED_PASS → enter the promote() batch.
            target_status = (
                SkillStatus.ACTIVE if verdict == GateVerdict.PASS
                else SkillStatus.PROVISIONAL
            )
            plan.add(
                skill=subject,
                target_status=target_status,
                evaluation=ev,
                rationale=ev.verdict.rationale or "gate-pass",
            )
            plan_meta.append(
                (op.proposal_id, subject.skill_id, target_status, ev, op.proposal_kind),
            )

        # ── batch promote ──
        promotion_result = None
        if len(plan) > 0:
            try:
                promotion_result = promoter.promote(
                    plan,
                    adapter_signature=[],
                    config_payload={
                        "teacher_model": teacher_model,
                        "judge_model":   judge_model,
                        "driver":        "labeling_supplement.decide_promotion_gpt54",
                    },
                    notes=f"offline-promotion: {corpus}/{source}",
                )
                if promotion_result.release is not None:
                    res.release_id = promotion_result.release.release_id
                    res.release = promotion_result.release
                else:
                    res.release_id = None
                    res.release = None
            except LifecycleError as exc:
                # If the live promote() refuses (e.g., LIMITED_PASS →
                # ACTIVE invariant), fall back to per-subject best-effort:
                # promote PASS only, downgrade LIMITED_PASS subjects to
                # PROVISIONAL via lifecycle.transition.
                logger.warning("batch promote failed: %s — retrying per-subject", exc)
                for prop_id, sid, target_status_meta, ev, _kind in plan_meta:
                    try:
                        lifecycle.transition(
                            sid, to_status=target_status_meta,
                            rationale=f"per-subject-fallback:{ev.evaluation_id}",
                        )
                    except LifecycleError as ex2:
                        logger.warning("  subject %s could not transition: %s", sid, ex2)

        # Map plan_meta entries to decisions (independent of which
        # promotion path actually executed — we record what *would* have
        # happened either way; the audit.jsonl reflects the actual
        # state).
        for prop_id, sid, target_status_meta, ev, _kind in plan_meta:
            cur = lifecycle.get(sid)
            if cur is None:
                continue
            res.decisions.append(_PromotionDecision(
                proposal_id=prop_id,
                subject_skill_id=sid,
                subject_skill_content_hash=cur.content_hash(),
                evaluation_id=ev.evaluation_id,
                final_verdict=ev.verdict.final_verdict.value,
                decision="PROMOTE",
                from_status=SkillStatus.CANDIDATE.value,
                target_status=cur.status.value,
                release_id=res.release_id,
                from_snapshot=None,
                to_snapshot=res.release_id,
                rationale=ev.verdict.rationale or "gate-pass",
                audit_event_kind="release",
                triggered_by="gate_pass",
                linked_eval_record=ev.evaluation_id,
            ))

        # Append the side decisions (REJECTs and retire-rollbacks).
        res.decisions.extend(side_decisions)

        # ── post-promotion regression check (rollback signal) ──
        #
        # The Plan §6.2 rollback signal is computed from the cold-start
        # actor batch summary (``_skill_actions_summary.json``).  In
        # Phase-1 promotions land at ``PROVISIONAL`` (not ACTIVE) and
        # ``PromotionOrchestrator.rollback`` accepts both — we mirror
        # that here so the path actually fires offline.  We index
        # ``actor_metrics`` by both the synthesised subject id (for
        # newly-crafted skills) and the original base/component skill
        # ids (for patches/composes whose selections live in the
        # bank-name histogram).
        if actor_metrics:
            base_index: Dict[str, str] = {}
            for prop_id, sid, _ts, _ev, _kind in plan_meta:
                cur = lifecycle.get(sid)
                if cur is None:
                    continue
                for parent_id in cur.parent_skill_ids or ():
                    base_index.setdefault(_safe_skill_id(parent_id), sid)
                base_index.setdefault(_safe_skill_id(cur.skill_id), sid)
                base_index.setdefault(_safe_skill_id(cur.name or ""), sid)

            rolled_back_subjects: set[str] = set()
            for prop_id, sid, _target_status_meta, ev, _kind in plan_meta:
                cur = lifecycle.get(sid)
                if cur is None or cur.status not in {
                    SkillStatus.ACTIVE,
                    SkillStatus.PROVISIONAL,
                }:
                    continue
                m: Dict[str, float] = dict(actor_metrics.get(sid) or {})
                for parent_id in (cur.parent_skill_ids or ()):
                    parent_metrics = actor_metrics.get(_safe_skill_id(parent_id)) or {}
                    for k, v in parent_metrics.items():
                        m.setdefault(k, v)
                if not m:
                    name_metrics = (
                        actor_metrics.get(_safe_skill_id(cur.name or "")) or {}
                    )
                    for k, v in name_metrics.items():
                        m.setdefault(k, v)
                if not m:
                    continue
                regress = (
                    m.get("selections", 0.0) < cfg["rollback_min_selections"]
                    or m.get("pass_rate", 1.0) < cfg["rollback_min_pass_rate"]
                )
                if not regress or sid in rolled_back_subjects:
                    continue
                from_status_value = cur.status.value
                try:
                    promoter.rollback(
                        skill_id=sid,
                        reason=(
                            f"non_regression_fail: selections={m.get('selections')!r}, "
                            f"pass_rate={m.get('pass_rate')!r}"
                        ),
                    )
                except LifecycleError as exc:
                    logger.debug("regression-rollback skipped for %s: %s", sid, exc)
                    continue
                rolled_back_subjects.add(sid)
                cur_after = lifecycle.get(sid)
                content_hash = (
                    cur_after.content_hash() if cur_after is not None else cur.content_hash()
                )
                res.decisions.append(_PromotionDecision(
                    proposal_id=prop_id,
                    subject_skill_id=sid,
                    subject_skill_content_hash=content_hash,
                    evaluation_id=ev.evaluation_id,
                    final_verdict=ev.verdict.final_verdict.value,
                    decision="ROLLBACK",
                    from_status=from_status_value,
                    target_status=SkillStatus.ROLLED_BACK.value,
                    release_id=None,
                    from_snapshot=None,
                    to_snapshot=None,
                    rationale=(
                        f"actor_batch_metrics regression "
                        f"(n_selections={m.get('selections')}, "
                        f"pass_rate={m.get('pass_rate')})"
                    ),
                    audit_event_kind="rollback",
                    triggered_by="non_regression_fail",
                    linked_eval_record=ev.evaluation_id,
                ))
                res.n_rollbacks += 1

        # ── per-source decision counters / final stats ──
        #
        # ``decisions`` may contain multiple rows per ``proposal_id`` —
        # specifically, a regression-rollback row appended after a
        # successful PROMOTE.  For the summary counters we collapse to
        # the *final* outcome per proposal (last row wins) so totals
        # stay 1:1 with the proposal stream while ``promotion_decisions
        # .jsonl`` retains the full audit trail.
        final_by_proposal: Dict[str, _PromotionDecision] = {}
        for d in res.decisions:
            final_by_proposal[d.proposal_id] = d
        for d in final_by_proposal.values():
            res.by_decision[d.decision] += 1
            res.by_target_status[d.target_status] += 1

        res.snapshot_ids = sorted(
            p.stem for p in (Path(temp_root) / "artifacts" / "snapshots").glob("*.json")
        )

        # ── copy the live ArtifactStore audit.jsonl + snapshots out ──
        live_audit = Path(temp_root) / "artifacts" / "audit.jsonl"
        if live_audit.exists():
            (out_src / "audit.jsonl").write_bytes(live_audit.read_bytes())
        else:
            (out_src / "audit.jsonl").write_text("")

        live_snaps = Path(temp_root) / "artifacts" / "snapshots"
        for sp in live_snaps.glob("*.json"):
            shutil.copy2(sp, snap_dir / sp.name)

        # ── write structured outputs ──
        res.elapsed_sec = round(time.time() - t0, 3)
        _write_per_source(res, out_src, cfg, teacher_model, judge_model)
        return res
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _write_per_source(
    res: _SourceRunResult,
    out_src: Path,
    cfg: Dict[str, Any],
    teacher_model: str,
    judge_model: str,
) -> None:
    out_src.mkdir(parents=True, exist_ok=True)

    # promotion_decisions.jsonl
    with (out_src / "promotion_decisions.jsonl").open("w") as f:
        for d in res.decisions:
            f.write(json.dumps(d.to_json(), ensure_ascii=False, sort_keys=True) + "\n")

    # gate_verdicts.jsonl  (live SkillEvaluationRecord shape — same one
    # the future Harness GateRunner mirror will produce)
    with (out_src / "gate_verdicts.jsonl").open("w") as f:
        for ev in res.evaluations:
            f.write(
                json.dumps(ev.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
            )

    # release.json — emit the *live* `RunRelease.to_json()` shape so it
    # round-trips through `data_structure.extensions.run_release.RunRelease`
    # without field drift (PLAN-PIPELINE-ORCHESTRATOR §7).  Null on
    # all-FAIL / all-REJECT runs (correct nullable artifact).
    rel: Optional[Dict[str, Any]] = None
    if res.release is not None:
        rel = res.release.to_json()
        rel["snapshot_ids"] = list(res.snapshot_ids)
    (out_src / "release.json").write_text(
        json.dumps(rel, indent=2) if rel is not None else "null"
    )

    # defer_followups.jsonl — empty in Phase 1; reserved per
    # implementation_notes/legacy/crafter-harness-orchestrator-roles.md §6.2 #4.
    (out_src / "defer_followups.jsonl").write_text("")

    # _promotion_summary.json
    #
    # ``n_decisions`` reports the count of unique proposal ids — i.e.,
    # the *final* decision per proposal — to stay 1:1 with the
    # proposal stream.  Follow-up audit rows (e.g., a regression
    # rollback appended after a PROMOTE) are still preserved in
    # ``promotion_decisions.jsonl`` and surfaced via ``n_audit_rows``.
    n_decisions_final = len({d.proposal_id for d in res.decisions})
    n_audit_rows = len(res.decisions)
    (out_src / "_promotion_summary.json").write_text(json.dumps({
        "corpus":               res.corpus,
        "source":               res.source,
        "status":               "ok" if res.error is None else "error",
        "error":                res.error,
        "n_proposals":          res.n_proposals,
        "n_decisions":          n_decisions_final,
        "n_audit_rows":         n_audit_rows,
        "n_skills_seeded":      res.n_skills_seeded,
        "n_rollbacks":          res.n_rollbacks,
        "n_defers":             res.n_defers,
        "by_kind":              dict(res.by_kind),
        "by_verdict":           dict(res.by_verdict),
        "by_decision":          dict(res.by_decision),
        "by_target_status":     dict(res.by_target_status),
        "release_id":           res.release_id,
        "snapshot_ids":         res.snapshot_ids,
        "thresholds":           cfg,
        "teacher_model":        teacher_model,
        "judge_model":          judge_model,
        "elapsed_sec":          res.elapsed_sec,
        "completed_at":         _utcnow_iso(),
    }, indent=2))


# ═══════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════

def _discover_pairs(
    proposals_run: Path,
    bank_run: Path,
    corpus_filter: Optional[str],
    source_filter: Optional[str],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for corpus in CORPORA:
        if corpus_filter and corpus != corpus_filter:
            continue
        cdir = proposals_run / corpus
        if not cdir.exists():
            continue
        for src_dir in sorted(cdir.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("_"):
                continue
            if source_filter and src_dir.name != source_filter:
                continue
            if not (src_dir / "proposals.jsonl").exists():
                continue
            if not (bank_run / corpus / src_dir.name / "skill_bank.jsonl").exists():
                continue
            out.append((corpus, src_dir.name))
    return out


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--proposals-run", type=Path, default=DEFAULT_PROPOSALS_RUN,
                   help="Directory produced by decide_skill_crafting_gpt54.py.")
    p.add_argument("--bank-run", type=Path, default=DEFAULT_BANK_RUN,
                   help="Original skill_bank_out/run_<ts> snapshot.")
    p.add_argument("--actions-run", type=Path, default=DEFAULT_ACTIONS_RUN,
                   help="Optional skill_actions_out/run_<ts> snapshot — used "
                        "for actor_batch_metrics ROLLBACK signal.")
    p.add_argument("--no-actions", action="store_true",
                   help="Disable rollback signal even if --actions-run exists.")
    p.add_argument("--gate-verdicts-run", type=Path, default=None,
                   help="Optional Harness-mirror output dir; if its "
                        "<corpus>/<source>/gate_verdicts.jsonl exists, "
                        "those verdicts are consumed instead of running "
                        "GateService inline.")
    p.add_argument("--gate-mode",
                   choices=[
                       GATE_MODE_OFFLINE_SYNTHETIC,
                       GATE_MODE_LIVE,
                       GATE_MODE_OFFLINE_LLM_JUDGE,
                       GATE_MODE_PERMISSIVE,
                   ],
                   default=GATE_MODE_OFFLINE_SYNTHETIC,
                   help="How to compute SkillEvaluationRecord when "
                        "--gate-verdicts-run is not provided: "
                        "'offline-synthetic' (DEFAULT) runs rule-based "
                        "Stage 0 inline + LIMITED_PASS placeholders for "
                        "Stages 1-4 (the documented Phase-1 behaviour); "
                        "'live' calls the LIVE GateService end-to-end "
                        "(currently FAILs Stage 3a when no target adapters "
                        "are registered, useful as a diagnostic); "
                        "'offline-with-llm-judge' extends offline-synthetic "
                        "with one 35B-A3B BACKBONE_JUDGE_MODEL call per "
                        "proposal (routed via VLLM_BASE_URL_MAP) and "
                        "appends an extra StageVerdict — an LLM 'fail' "
                        "verdict flips the aggregate to FAIL ⇒ REJECT.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output root; defaults to "
                        "labeling_supplement/promotion_decisions_out/run_<ts>.")
    p.add_argument("--corpus", choices=CORPORA, default=None)
    p.add_argument("--source", default=None)

    p.add_argument("--teacher-model", default=BACKBONE_TEACHER_MODEL,
                   help=f"Crafter / Harness / Orchestrator backbone identifier "
                        f"(logged into _run_meta.json; default "
                        f"{BACKBONE_TEACHER_MODEL!r}).")
    p.add_argument("--judge-model", default=BACKBONE_JUDGE_MODEL,
                   help=f"Eval-driver judge identifier "
                        f"(logged into SkillEvaluationRecord.judge_model; "
                        f"default {BACKBONE_JUDGE_MODEL!r}).")

    # ── Stage 2 cross-domain knobs (only meaningful with
    # ``--gate-mode offline-with-llm-judge``).  Defaults preserve
    # Stage-1 fast path: thinking off, 256 tokens / verdict.
    p.add_argument("--enable-thinking", action="store_true",
                   help="Forward enable_thinking=True into the LLM judge "
                        "(Qwen3-A3B <think> chain-of-thought).  Bumps "
                        "judge wall-time ~5-10x; combine with "
                        "--judge-max-tokens >= 2048.  Stage-1 default OFF.")
    p.add_argument("--judge-max-tokens", type=int, default=256,
                   help="Token budget per llm-judge response.  Stage-1 "
                        "default 256 (judge emits a tight JSON verdict).  "
                        "Stage-2 with --enable-thinking should set this "
                        "to 2048+ so the <think> block has room before "
                        "the verdict tokens.")

    p.add_argument("--rollback-min-selections", type=int,
                   default=DEFAULTS["rollback_min_selections"])
    p.add_argument("--rollback-min-pass-rate", type=float,
                   default=DEFAULTS["rollback_min_pass_rate"])

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    proposals_run: Path = args.proposals_run.resolve()
    bank_run: Path = args.bank_run.resolve()
    actions_run: Optional[Path]
    if args.no_actions:
        actions_run = None
    else:
        ar = args.actions_run.resolve() if args.actions_run else None
        actions_run = ar if (ar and ar.exists()) else None
        if ar is not None and not ar.exists():
            logger.warning(
                "actions-run does not exist, continuing without rollback "
                "signal: %s", ar,
            )

    gate_run: Optional[Path] = (
        args.gate_verdicts_run.resolve() if args.gate_verdicts_run else None
    )

    if not proposals_run.exists():
        logger.error("proposals-run does not exist: %s", proposals_run)
        return 2
    if not bank_run.exists():
        logger.error("bank-run does not exist: %s", bank_run)
        return 2

    output_dir: Path = (
        args.output_dir.resolve() if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / f"run_{_utc_run_stamp()}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(
        proposals_run, bank_run,
        corpus_filter=args.corpus, source_filter=args.source,
    )
    if not pairs:
        logger.error("no (corpus, source) pairs discovered under %s", proposals_run)
        return 2

    cfg = {
        "rollback_min_selections": args.rollback_min_selections,
        "rollback_min_pass_rate":  args.rollback_min_pass_rate,
    }

    started_at = _utcnow_iso()

    effective_gate_mode = (
        GATE_MODE_EXTERNAL if gate_run is not None else args.gate_mode
    )

    logger.info("decide_promotion: %d pair(s) under %s", len(pairs), proposals_run)
    logger.info("  bank_run        : %s", bank_run)
    logger.info("  actions_run     : %s", actions_run)
    logger.info("  gate_mode       : %s", effective_gate_mode)
    logger.info("  gate_verdicts   : %s", gate_run)
    logger.info("  output_dir      : %s", output_dir)
    logger.info("  teacher_model   : %s", args.teacher_model)
    logger.info("  judge_model     : %s", args.judge_model)
    logger.info("  thresholds      : %s",
                json.dumps(cfg, indent=None, sort_keys=True))

    if args.dry_run:
        for c, s in pairs:
            print(f"  {c} / {s}")
        return 0

    per_pair_summaries: List[Dict[str, Any]] = []
    by_decision_total: Counter = Counter()
    by_verdict_total:  Counter = Counter()
    by_kind_total:     Counter = Counter()
    by_target_status_total: Counter = Counter()
    n_proposals_total = 0
    n_rollbacks_total = 0

    for corpus, source in pairs:
        logger.info("processing %s / %s", corpus, source)
        res = _decide_per_source(
            corpus=corpus, source=source,
            proposals_run=proposals_run, bank_run=bank_run,
            actions_run=actions_run, gate_run=gate_run,
            gate_mode=(
                GATE_MODE_EXTERNAL if gate_run is not None else args.gate_mode
            ),
            output_root=output_dir, cfg=cfg,
            teacher_model=args.teacher_model,
            judge_model=args.judge_model,
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
            judge_max_tokens=int(getattr(args, "judge_max_tokens", 256)),
        )
        per_pair_summaries.append({
            "corpus":          res.corpus,
            "source":          res.source,
            "status":          "ok" if res.error is None else "error",
            "error":           res.error,
            "n_proposals":     res.n_proposals,
            "n_decisions":     len(res.decisions),
            "n_skills_seeded": res.n_skills_seeded,
            "n_rollbacks":     res.n_rollbacks,
            "by_kind":         dict(res.by_kind),
            "by_verdict":      dict(res.by_verdict),
            "by_decision":     dict(res.by_decision),
            "by_target_status": dict(res.by_target_status),
            "release_id":      res.release_id,
            "elapsed_sec":     res.elapsed_sec,
        })
        by_decision_total.update(res.by_decision)
        by_verdict_total.update(res.by_verdict)
        by_kind_total.update(res.by_kind)
        by_target_status_total.update(res.by_target_status)
        n_proposals_total += res.n_proposals
        n_rollbacks_total += res.n_rollbacks
        if res.error:
            logger.warning("  %s/%s -> %s", corpus, source, res.error)
        else:
            logger.info(
                "  %s/%s -> %d proposal(s), decisions=%s, verdicts=%s, "
                "rollbacks=%d, release=%s",
                corpus, source, res.n_proposals,
                ", ".join(f"{k}={v}" for k, v in res.by_decision.most_common()) or "-",
                ", ".join(f"{k}={v}" for k, v in res.by_verdict.most_common()) or "-",
                res.n_rollbacks, res.release_id,
            )

    (output_dir / "_run_meta.json").write_text(json.dumps({
        "proposals_run":   str(proposals_run),
        "bank_run":        str(bank_run),
        "actions_run":     str(actions_run) if actions_run else None,
        "gate_verdicts_run": str(gate_run) if gate_run else None,
        "gate_mode":       effective_gate_mode,
        "output_root":     str(output_dir),
        "teacher_model":   args.teacher_model,
        "judge_model":     args.judge_model,
        "thresholds":      cfg,
        "pairs":           [{"corpus": c, "source": s} for c, s in pairs],
        "argv":            [str(a) for a in (argv or sys.argv)],
        "started_at":      started_at,
    }, indent=2))

    (output_dir / "_run_summary.json").write_text(json.dumps({
        "proposals_run":      str(proposals_run),
        "bank_run":           str(bank_run),
        "actions_run":        str(actions_run) if actions_run else None,
        "gate_verdicts_run":  str(gate_run) if gate_run else None,
        "gate_mode":          effective_gate_mode,
        "output_root":        str(output_dir),
        "n_pairs":            len(pairs),
        "n_pairs_ok":         sum(1 for r in per_pair_summaries if r["status"] == "ok"),
        "n_proposals":        n_proposals_total,
        "n_rollbacks":        n_rollbacks_total,
        "by_kind":            dict(by_kind_total),
        "by_verdict":         dict(by_verdict_total),
        "by_decision":        dict(by_decision_total),
        "by_target_status":   dict(by_target_status_total),
        "per_pair":           per_pair_summaries,
        "started_at":         started_at,
        "completed_at":       _utcnow_iso(),
    }, indent=2))

    logger.info("DONE: %d pair(s), %d proposal(s)", len(pairs), n_proposals_total)
    logger.info("  by decision : %s",
                ", ".join(f"{k}={v}" for k, v in by_decision_total.most_common()) or "-")
    logger.info("  by verdict  : %s",
                ", ".join(f"{k}={v}" for k, v in by_verdict_total.most_common()) or "-")
    logger.info("  rollbacks   : %d", n_rollbacks_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
