#!/usr/bin/env python
"""
Per-episode Skill Crafter driver — invokes the LIVE
`SkillCrafterService.reflect_on_episode` against each episode in a
``labeling/skill_actions_out`` run, paired with that episode's
``per_episode_bank_management/bank_management_io.json``.

This is the cold-start *mirror* of the live two-tier trigger model
(see ``implementation_notes/legacy/crafter-harness-orchestrator-roles.md``
§"Two-tier trigger"): the live runtime calls
``SkillCrafterService.reflect_on_episode(EpisodeReflection)`` immediately
after the Skill Bank Agent finishes one episode. This script does the
same thing offline, so the cold-start corpus carries genuine
``BankMutationProposal`` records emitted by the *real* Crafter code,
not just rule-based stand-ins.

Per-episode flow
----------------
For each ``(corpus, source)`` pair:

  1. Load the FROZEN per-source skill bank
     (``skill_bank_run/<corpus>/<source>/skill_bank.jsonl``) and seed it
     into a temp ``SkillCrafterService`` with each skill at status
     ``CANDIDATE``.  ACTIVE promotion is intentionally skipped — most
     cold-start skills declare a single ``feasible_domain`` and the
     lifecycle invariant requires ≥2 for ACTIVE; the Failure-Reflector
     dispatch path uses ``repository.get`` which queries every store, so
     CANDIDATE seeding is sufficient for everything *except* the
     subsumption-retire heuristic (which compares CANDIDATE→ACTIVE and
     therefore stays dormant in this mirror — documented limitation,
     unblocked once Phase-2 multi-domain skills land).
  2. Walk every ``episode_*.json`` under
     ``skill_actions_run/<corpus>/<source>/`` in order.
  3. For each episode, locate the matching
     ``skill_bank_run/<corpus>/<source>/per_episode_bank_management/episode_<i>/bank_management_io.json``
     (if present) and use its ``stage_4_bank_maintenance`` outputs to
     populate ``new_candidate_skill_ids`` + ``bank_agent_actions`` on
     the :class:`EpisodeReflection`.
  4. Synthesize :class:`FailureTrace`s from on-disk per-step signals
     (see :func:`_synthesize_failures` for the rules).
  5. Build the :class:`EpisodeReflection` and call
     ``crafter.reflect_on_episode(reflection)``.
  6. Serialize the resulting ``CrafterCycleResult`` (proposals +
     subsumption count + bank-view summary) to disk.

Failure synthesis (offline approximation)
-----------------------------------------
The cold-start traces in ``labeling/skill_actions_out`` are mostly
healthy (the bank fits the rollouts cleanly, applicability ≈ 0.5 by
construction); to give the Crafter something meaningful to act on, we
synthesize :class:`FailureTrace`s from the following per-episode
signals:

* **OUTCOME_FAILURE** — ``episode["outcome"] != True``: one synthetic
  trace tagged to the most-selected skill of the episode
  (``failure_class = "INVARIANT_VIOLATION"``).
* **EMPTY_QUERY** — ``step.skill_query.empty == True``: one trace per
  occurrence (``failure_class = "MISSING_ADAPTER"``) signalling a
  bank gap that the Hypothesizer should pick up.
* **LOW_APPLICABILITY** — ``step.skills.applicability < threshold``
  (default 0.4): one trace per occurrence
  (``failure_class = "PRECONDITION_VIOLATION"``) — indicates the
  bound skill's preconditions don't hold strongly enough.
* **MISSING_EFFECTS** — ``step.skills.missing_effects`` non-empty: one
  trace per occurrence (``failure_class = "INVARIANT_VIOLATION"``) —
  the bound skill claimed effects that didn't materialize.

A per-episode cap (``--max-failures-per-episode``, default 8) keeps the
synthesized batch size bounded so a 200-step rollout doesn't blow up
``FailureMemory``.

Outputs
-------
``<output_dir>/<corpus>/<source>/episode_<NNN>/proposals.jsonl`` —
one JSON object per line, the typed proposals
``crafter.reflect_on_episode`` emitted (PLAN-SKILL-CRAFTER §2.5).

``<output_dir>/<corpus>/<source>/episode_<NNN>/reflection.json`` — the
``EpisodeReflection`` that was passed in (full audit input).

``<output_dir>/<corpus>/<source>/episode_<NNN>/result.json`` — the
``CrafterCycleResult`` minus the proposals (kind / proposer /
subsumption / bank-view-size summary).

``<output_dir>/<corpus>/<source>/_source_summary.json`` — per-source
totals.

``<output_dir>/_run_meta.json`` and ``_run_summary.json`` — run-level.

Usage
-----

    # Process every (corpus, source) pair, every episode, default thresholds.
    python labeling_supplement/reflect_per_episode_gpt54.py \
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \
        --output-dir   labeling_supplement/episode_reflections_out/run_<ts>

    # Smoke: one source, two episodes.
    python labeling_supplement/reflect_per_episode_gpt54.py \
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \
        --corpus       env_wrappers --source twenty_forty_eight \
        --max-episodes 2 -v

The companion bash dispatcher ``run_reflect_per_episode.sh`` fans this
out one worker per ``(corpus, source)``.
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
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
# Project imports — these load the LIVE Crafter code.
# ---------------------------------------------------------------------------
from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from crafter import SkillCrafterService
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    proposal_to_json,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from orchestrator import ArtifactStore
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName

logger = logging.getLogger("labeling_supplement.reflect_per_episode")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BANK_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_bank_out" / "run_20260430_030637"
)
DEFAULT_ACTIONS_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_actions_out" / "run_20260430_064325"
)
DEFAULT_OUTPUT_ROOT = (
    CODEBASE_ROOT / "labeling_supplement" / "episode_reflections_out"
)

CORPORA = ("gym_v", "env_wrappers")

# Failure-synthesis thresholds (rule-of-thumb for the cold-start corpus).
# Override via CLI.
DEFAULT_LOW_APPLICABILITY = 0.4
DEFAULT_MAX_FAILURES_PER_EPISODE = 8


# ═══════════════════════════════════════════════════════════════════════
# Bank seeding — turn on-disk skill_bank.jsonl into live SkillRecords
# ═══════════════════════════════════════════════════════════════════════

# Map between the on-disk evidence_role string and a SkillType + canonical
# expected_evidence_roles value. The bank emits roles in {GATHER, VERIFY,
# REASON, COMMIT}; the live SkillType vocabulary has different
# granularity (REASONING / ACTION / GROUNDING / MIXED). We map
# defensively and let the gate sort out semantics later.
_ROLE_TO_SKILL_TYPE = {
    "GATHER": SkillType.GROUNDING,
    "VERIFY": SkillType.REASONING,
    "REASON": SkillType.REASONING,
    "COMMIT": SkillType.ACTION,
}


def _safe_skill_id(skill_id: str) -> str:
    """Cold-start labels use ``OPERATOR/SUBGOAL`` (e.g. ``COMMIT/ATTACK``)
    which the on-disk ``SkillStore`` flat-filename layout rejects (it
    builds ``<root>/<skill_id>.json`` so ``/`` would force a nested
    path that the store doesn't pre-create). Map ``/`` → ``__`` and
    apply the same mapping to ``parent_skill_ids`` so subsumption /
    failure-resolution links keep resolving."""
    return skill_id.replace("/", "__")


def _wrap_protocol_steps(raw_steps: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert on-disk ``protocol.steps`` (often natural-language strings
    in cold-start banks) into the live ``protocol: List[Dict[...]]``
    shape that ``Repairer._rule_repair`` and friends expect."""
    out: List[Dict[str, Any]] = []
    for s in raw_steps or []:
        if isinstance(s, dict):
            out.append(dict(s))
        elif isinstance(s, str):
            out.append({"action": "EXEC", "payload": {}, "notes": s})
        else:
            out.append({"action": "EXEC", "payload": {}, "notes": str(s)})
    return out


def _record_from_bank_entry(entry: Dict[str, Any], default_domain: str) -> SkillRecord:
    """Hydrate a `SkillRecord` from one ``skill_bank.jsonl`` line (the
    ``{"skill": ..., "report": ...}`` envelope)."""
    skill = entry.get("skill") or {}
    contract = skill.get("contract") or {}
    role = (skill.get("evidence_role") or "COMMIT").upper()
    skill_type = _ROLE_TO_SKILL_TYPE.get(role, SkillType.MIXED)

    feasible = list(skill.get("applicable_domains") or []) or [default_domain]
    protocol_blob = skill.get("protocol") or {}

    sk = SkillRecord.new(
        name=skill.get("name", skill.get("skill_id", "_unknown")),
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=feasible,
        protocol=_wrap_protocol_steps(protocol_blob.get("steps") or []),
        contract=SkillContract(
            preconditions=list(protocol_blob.get("preconditions") or []),
            effects_add=list(contract.get("eff_add") or []),
            effects_del=list(contract.get("eff_del") or []),
            expected_evidence_roles=[role] if role else [],
            success_criteria=list(protocol_blob.get("success_criteria") or []),
            abort_criteria=list(protocol_blob.get("abort_criteria") or []),
        ),
    )
    # Force the bank-given skill_id (overrides the freshly-minted UUID)
    # so that downstream `parent_skill_ids` references resolve. We must
    # bypass `SkillRecord.__setattr__` (which guards status mutations) by
    # going through `object.__setattr__`.
    raw_id = skill.get("skill_id") or sk.skill_id
    object.__setattr__(sk, "skill_id", _safe_skill_id(raw_id))
    return sk


def _seed_bank(
    lifecycle: SkillLifecycleManager,
    bank_path: Path,
    default_domain: str,
) -> Tuple[int, int]:
    """Seed the temp bank as CANDIDATE from a ``skill_bank.jsonl`` file.

    Returns ``(n_seeded, n_skipped)`` for the run summary.
    """
    if not bank_path.exists():
        return 0, 0
    n = 0
    skipped = 0
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
                rec = _record_from_bank_entry(entry, default_domain)
                lifecycle.ingest_draft(rec)
                lifecycle.transition(
                    rec.skill_id,
                    to_status=SkillStatus.CANDIDATE,
                    rationale="seed-from-bank-snapshot",
                )
                n += 1
            except Exception as exc:                                # noqa: BLE001
                logger.debug("skip bank seed %s: %s", entry.get("skill", {}).get("skill_id"), exc)
                skipped += 1
    return n, skipped


def _seed_refines_as_candidates(
    lifecycle: SkillLifecycleManager,
    bank_mgmt: Dict[str, Any],
    default_domain: str,
) -> List[str]:
    """Seed the per-episode bank-mgmt refine_details as fresh CANDIDATE
    skills.

    These mimic the candidates the live Bank Agent would have minted in
    this episode, giving the Crafter's per-episode pass real
    ``new_candidate_skill_ids`` to chew on. The new skills carry a
    ``parent_skill_ids`` link to the original (when the refine has a
    ``skill_id`` pointer), which is what the subsumption heuristic
    actually checks.

    Returns the list of newly-seeded skill_ids (so the caller can plumb
    them into ``EpisodeReflection.new_candidate_skill_ids``).
    """
    out: List[str] = []
    s4 = (bank_mgmt.get("stage_4_bank_maintenance") or {}).get("outputs") or {}
    for r in (s4.get("refine_details") or []):
        fields = r.get("_fields") or {}
        parent_id_raw = fields.get("skill_id")
        new_contract = fields.get("new_contract") or {}
        if not parent_id_raw or not new_contract:
            continue
        # The refine's `new_contract` lacks `applicable_domains`; mirror
        # the parent. Also sanitise `parent_skill_ids` to match the
        # filename-safe form we used when seeding the bank, so the
        # subsumption / repair lookups can find the parent.
        parent_id = _safe_skill_id(parent_id_raw)
        cand = SkillRecord.new(
            name=f"{new_contract.get('name', parent_id_raw)}__refined_v{new_contract.get('version', 0)}",
            skill_type=SkillType.MIXED,
            source_type=SkillSourceType.MINED,
            feasible_domains=[default_domain],
            protocol=[],                                             # refines target the contract, not protocol
            contract=SkillContract(
                preconditions=[],
                effects_add=_clean_str_list(new_contract.get("eff_add") or []),
                effects_del=_clean_str_list(new_contract.get("eff_del") or []),
                expected_evidence_roles=[],
                success_criteria=[],
                abort_criteria=[],
            ),
            parent_skill_ids=[parent_id],
        )
        try:
            lifecycle.ingest_draft(cand)
            lifecycle.transition(
                cand.skill_id,
                to_status=SkillStatus.CANDIDATE,
                rationale=f"bank-mgmt refine for {parent_id}",
            )
            out.append(cand.skill_id)
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("skip refine seed %s: %s", parent_id, exc)
    return out


def _clean_str_list(items: Iterable[Any]) -> List[str]:
    """Filter out ``<depth-limit: ...>`` placeholders that the bank-mgmt
    JSONs use when a list was truncated by the on-disk serialiser."""
    out: List[str] = []
    for it in items:
        if isinstance(it, str) and not it.startswith("<depth-limit:"):
            out.append(it)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Failure synthesis — turn step-level signals into FailureTraces
# ═══════════════════════════════════════════════════════════════════════

def _synthesize_failures(
    *,
    episode: Dict[str, Any],
    episode_id: str,
    domain: str,
    low_applicability: float,
    max_failures: int,
) -> List[FailureTrace]:
    """Walk per-step signals and emit a bounded list of FailureTraces.

    See module docstring for the four signals (OUTCOME_FAILURE,
    EMPTY_QUERY, LOW_APPLICABILITY, MISSING_EFFECTS). The traces are
    ordered by signal severity so a low ``max_failures`` cap retains
    the most informative ones first.
    """
    out: List[FailureTrace] = []
    exps = episode.get("experiences") or episode.get("steps") or []

    # NB: we route all per-step skill_ids through `_safe_skill_id` so
    # the FailureMemory keys agree with the IDs we used when seeding
    # the bank — otherwise `_resolve_base("COMMIT/ATTACK")` misses the
    # seeded `COMMIT__ATTACK` record and the dispatch silently drops
    # the patch.
    skill_select_counter: Counter[str] = Counter()
    for exp in exps:
        sk = exp.get("skills") or {}
        sid = sk.get("skill_id")
        if sid:
            skill_select_counter[_safe_skill_id(sid)] += 1

    # ── 1. OUTCOME_FAILURE — single-shot, episode-level ───────────────
    # `outcome=True` is a strong success signal and short-circuits the
    # status fallback (some rollouts have outcome=True with
    # status="running" because the trace was sliced mid-success).
    outcome = episode.get("outcome")
    status = (episode.get("episode_status") or "").lower()
    is_outcome_failure = (
        outcome is False
        or (outcome is None and status in ("failed", "timeout"))
    )
    if is_outcome_failure:
        rep_skill = (
            skill_select_counter.most_common(1)[0][0]
            if skill_select_counter
            else ""
        )
        out.append(FailureTrace(
            skill_id=rep_skill,
            skill_episode_id=f"{episode_id}#outcome",
            domain=domain,
            failed_step_index=len(exps) - 1 if exps else None,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=f"episode_outcome={outcome}, status={status}",
            extra={
                "synthesis_signal": "OUTCOME_FAILURE",
                "episode_id": episode_id,
                "n_steps": len(exps),
            },
        ))

    # ── 2. EMPTY_QUERY — bank had no match for this step ─────────────
    for i, exp in enumerate(exps):
        sq = exp.get("skill_query") or {}
        if sq.get("empty") and not sq.get("error"):
            out.append(FailureTrace(
                skill_id="",
                skill_episode_id=f"{episode_id}#empty_query@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="MISSING_ADAPTER",
                abort_reason="skill_query.empty",
                extra={"synthesis_signal": "EMPTY_QUERY", "step_index": i},
            ))

    # ── 3. LOW_APPLICABILITY — bound skill has weak fit ──────────────
    for i, exp in enumerate(exps):
        sk = exp.get("skills") or {}
        if not sk:
            continue
        app = sk.get("applicability")
        if app is None or not isinstance(app, (int, float)):
            continue
        if app < low_applicability:
            out.append(FailureTrace(
                skill_id=_safe_skill_id(sk.get("skill_id", "") or ""),
                skill_episode_id=f"{episode_id}#low_app@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="PRECONDITION_VIOLATION",
                abort_reason=f"applicability={app:.3f} < {low_applicability}",
                extra={
                    "synthesis_signal": "LOW_APPLICABILITY",
                    "step_index": i,
                    "applicability": float(app),
                    "confidence": sk.get("confidence"),
                },
            ))

    # ── 4. MISSING_EFFECTS — bound skill claimed effects that
    #      didn't materialize.
    for i, exp in enumerate(exps):
        sk = exp.get("skills") or {}
        miss = sk.get("missing_effects")
        if miss:
            out.append(FailureTrace(
                skill_id=_safe_skill_id(sk.get("skill_id", "") or ""),
                skill_episode_id=f"{episode_id}#miss_eff@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="INVARIANT_VIOLATION",
                abort_reason=f"missing_effects={miss}",
                extra={
                    "synthesis_signal": "MISSING_EFFECTS",
                    "step_index": i,
                    "missing_effects": list(miss),
                },
            ))

    # Cap: severity ordering matches list order (outcome > empty_query >
    # low_app > missing_effects), so taking the head retains the most
    # informative signals.
    if len(out) > max_failures:
        out = out[:max_failures]
    return out


# ═══════════════════════════════════════════════════════════════════════
# Per-episode driver
# ═══════════════════════════════════════════════════════════════════════

def _bank_mgmt_path(bank_run: Path, corpus: str, source: str, ep_idx: int) -> Path:
    return (
        bank_run / corpus / source / "per_episode_bank_management"
        / f"episode_{ep_idx}" / "bank_management_io.json"
    )


def _process_episode(
    *,
    crafter: SkillCrafterService,
    lifecycle: SkillLifecycleManager,
    episode_path: Path,
    bank_mgmt_path: Path,
    out_dir: Path,
    corpus: str,
    source: str,
    domain: str,
    low_app: float,
    max_failures: int,
) -> Dict[str, Any]:
    """Process one episode end-to-end. Returns a row for the per-source summary."""
    episode = json.loads(episode_path.read_text())
    episode_id = (
        episode.get("episode_id")
        or f"{corpus}/{source}/{episode_path.stem}"
    )

    # ── new_candidate_skill_ids + bank_agent_actions from bank-mgmt ──
    new_cand_ids: List[str] = []
    bank_agent_actions: Dict[str, Any] = {}
    if bank_mgmt_path.exists():
        try:
            bank_mgmt = json.loads(bank_mgmt_path.read_text())
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("could not parse %s: %s", bank_mgmt_path, exc)
            bank_mgmt = {}
        if bank_mgmt:
            new_cand_ids = _seed_refines_as_candidates(
                lifecycle, bank_mgmt, default_domain=domain
            )
            s4 = (bank_mgmt.get("stage_4_bank_maintenance") or {}).get("outputs") or {}
            bank_agent_actions = {
                "n_splits": int(s4.get("n_splits") or 0),
                "n_merges": int(s4.get("n_merges") or 0),
                "n_refines": int(s4.get("n_refines") or 0),
                "alias_map": s4.get("alias_map") or {},
            }

    # ── failure traces from per-step signals ─────────────────────────
    failure_traces = _synthesize_failures(
        episode=episode,
        episode_id=episode_id,
        domain=domain,
        low_applicability=low_app,
        max_failures=max_failures,
    )

    # ── outcome summary ──────────────────────────────────────────────
    outcome_summary = {
        "outcome": episode.get("outcome"),
        "episode_status": episode.get("episode_status"),
        "n_steps": len(episode.get("experiences") or episode.get("steps") or []),
        "total_reward": _sum_reward(episode),
    }

    reflection = EpisodeReflection(
        episode_id=episode_id,
        domain=domain,
        parent_run_id=str(episode_path.parent),
        failure_traces=failure_traces,
        skill_episodes=[],                                          # offline mirror has no SkillEpisode log
        new_candidate_skill_ids=new_cand_ids,
        bank_agent_actions=bank_agent_actions,
        outcome_summary=outcome_summary,
    )

    # ── invoke the LIVE Crafter ──────────────────────────────────────
    t0 = time.time()
    result = crafter.reflect_on_episode(reflection)
    elapsed = time.time() - t0

    # ── serialise per-episode artefacts ──────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals_path = out_dir / "proposals.jsonl"
    with proposals_path.open("w") as f:
        for p in result.proposals:
            f.write(json.dumps(proposal_to_json(p), ensure_ascii=False, sort_keys=True) + "\n")

    (out_dir / "reflection.json").write_text(
        json.dumps(reflection.to_json(), indent=2, ensure_ascii=False, sort_keys=True)
    )

    by_kind: Counter[str] = Counter(type(p).__name__ for p in result.proposals)
    by_proposer: Counter[str] = Counter()
    for p in result.proposals:
        # The proposer string is encoded in `source_type` for the live
        # proposals; mirror the offline driver's `proposer` label by
        # mapping back.
        by_proposer[_infer_proposer(p)] += 1

    (out_dir / "result.json").write_text(json.dumps({
        "trigger": result.trigger,
        "episode_id": result.episode_id,
        "n_failures_ingested": result.n_failures_ingested,
        "n_patterns_examined": result.n_patterns_examined,
        "n_proposals": len(result.proposals),
        "n_subsumption_retires": result.n_subsumption_retires,
        "n_patches_coalesced": result.n_patches_coalesced,
        "n_patches_skipped_cooldown": result.n_patches_skipped_cooldown,
        "by_kind": dict(by_kind),
        "by_proposer": dict(by_proposer),
        "bank_view_summary": dict(result.bank_view_summary),
        "elapsed_sec": round(elapsed, 4),
        "completed_at": _utcnow_iso(),
    }, indent=2))

    return {
        "episode_path": str(episode_path),
        "episode_id": episode_id,
        "n_failures_synthesized": len(failure_traces),
        "n_new_candidates": len(new_cand_ids),
        "bank_agent_actions": bank_agent_actions,
        "n_proposals": len(result.proposals),
        "n_subsumption_retires": result.n_subsumption_retires,
        "n_patches_coalesced": result.n_patches_coalesced,
        "n_patches_skipped_cooldown": result.n_patches_skipped_cooldown,
        "by_kind": dict(by_kind),
        "by_proposer": dict(by_proposer),
        "bank_view_summary": dict(result.bank_view_summary),
        "elapsed_sec": round(elapsed, 4),
    }


def _sum_reward(episode: Dict[str, Any]) -> float:
    exps = episode.get("experiences") or episode.get("steps") or []
    total = 0.0
    for e in exps:
        r = e.get("reward")
        if isinstance(r, (int, float)):
            total += float(r)
    return total


def _infer_proposer(p: BankMutationProposal) -> str:
    """Mirror the offline labels (composer / generalizer / hypothesizer /
    reflector) that the labeling_supplement summary tooling expects."""
    cls = type(p).__name__
    if cls == "ComposeProposal":
        return "composer"
    if cls == "GeneralizeProposal":
        return "generalizer"
    if cls == "HypothesisProposal":
        return "hypothesizer"
    # Patches and retires both come from the failure-reflection chain.
    return "reflector"


# ═══════════════════════════════════════════════════════════════════════
# Per-source driver
# ═══════════════════════════════════════════════════════════════════════

def _process_source(
    *,
    corpus: str,
    source: str,
    bank_run: Path,
    actions_run: Path,
    output_root: Path,
    max_episodes: Optional[int],
    low_app: float,
    max_failures: int,
) -> Dict[str, Any]:
    """Run reflect_on_episode across every episode of one (corpus, source)."""
    t0 = time.time()
    src_actions = actions_run / corpus / source
    src_bank = bank_run / corpus / source
    bank_path = src_bank / "skill_bank.jsonl"

    out_src = output_root / corpus / source
    out_src.mkdir(parents=True, exist_ok=True)

    eps = sorted(src_actions.glob("episode_*.json"))
    if max_episodes is not None:
        eps = eps[:max_episodes]
    if not eps:
        logger.warning("%s/%s: no episode_*.json under %s", corpus, source, src_actions)
        (out_src / "_source_summary.json").write_text(json.dumps({
            "corpus": corpus, "source": source,
            "status": "no_episodes",
            "actions_dir": str(src_actions),
        }, indent=2))
        return {
            "corpus": corpus, "source": source,
            "status": "no_episodes", "n_episodes": 0,
            "n_proposals": 0, "elapsed_sec": 0.0,
        }

    domain = _infer_domain(corpus, source)

    # ── temp bank, fresh per source ──────────────────────────────────
    temp_root = Path(tempfile.mkdtemp(prefix=f"crafter_mirror_{corpus}_{source}_"))
    try:
        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(temp_root / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(temp_root / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(temp_root / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(temp_root / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(temp_root / "artifacts"))
        # Offline diagnostic driver — keep the Repairer path live so the
        # JSONL captures every proposal kind for inspection, even though
        # the live trainer's default lane-(a) flag is ``False`` (T1.3a /
        # ``implementation_notes/legacy/skill-lane-decision.md``).
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            enable_protocol_patching=True,
        )

        n_seeded, n_seed_skipped = _seed_bank(lifecycle, bank_path, domain)

        # ── per-episode loop ────────────────────────────────────────
        rows: List[Dict[str, Any]] = []
        total_props = 0
        total_subsumes = 0
        total_coalesced = 0
        total_cooldown = 0
        by_kind: Counter[str] = Counter()
        by_proposer: Counter[str] = Counter()

        for ep_path in eps:
            # Resolve the matching bank-mgmt JSON. Naming is
            # ``per_episode_bank_management/episode_<i>/bank_management_io.json``
            # where <i> is the *integer* episode index — note that
            # skill_actions_out filenames are zero-padded
            # ``episode_NNN.json`` while the bank-mgmt directory names
            # are unpadded ``episode_<i>``. Strip leading zeros.
            try:
                ep_idx = int(ep_path.stem.split("_")[-1])
            except ValueError:
                ep_idx = -1
            bank_mgmt_path = _bank_mgmt_path(bank_run, corpus, source, ep_idx)

            ep_out = out_src / ep_path.stem
            row = _process_episode(
                crafter=crafter,
                lifecycle=lifecycle,
                episode_path=ep_path,
                bank_mgmt_path=bank_mgmt_path,
                out_dir=ep_out,
                corpus=corpus,
                source=source,
                domain=domain,
                low_app=low_app,
                max_failures=max_failures,
            )
            rows.append(row)
            total_props += row["n_proposals"]
            total_subsumes += row["n_subsumption_retires"]
            total_coalesced += row.get("n_patches_coalesced", 0)
            total_cooldown += row.get("n_patches_skipped_cooldown", 0)
            by_kind.update(row["by_kind"])
            by_proposer.update(row["by_proposer"])

        elapsed = time.time() - t0
        (out_src / "_source_summary.json").write_text(json.dumps({
            "corpus": corpus,
            "source": source,
            "domain": domain,
            "bank_path": str(bank_path),
            "actions_dir": str(src_actions),
            "status": "ok",
            "n_skills_seeded": n_seeded,
            "n_skills_skipped_in_seed": n_seed_skipped,
            "n_episodes": len(eps),
            "n_proposals": total_props,
            "n_subsumption_retires": total_subsumes,
            "n_patches_coalesced": total_coalesced,
            "n_patches_skipped_cooldown": total_cooldown,
            "by_kind": dict(by_kind),
            "by_proposer": dict(by_proposer),
            "elapsed_sec": round(elapsed, 3),
            "thresholds": {
                "low_applicability": low_app,
                "max_failures_per_episode": max_failures,
            },
            "completed_at": _utcnow_iso(),
            "per_episode": rows,
        }, indent=2))

        return {
            "corpus": corpus,
            "source": source,
            "domain": domain,
            "status": "ok",
            "n_skills_seeded": n_seeded,
            "n_episodes": len(eps),
            "n_proposals": total_props,
            "n_subsumption_retires": total_subsumes,
            "n_patches_coalesced": total_coalesced,
            "n_patches_skipped_cooldown": total_cooldown,
            "by_kind": dict(by_kind),
            "by_proposer": dict(by_proposer),
            "elapsed_sec": round(elapsed, 3),
        }
    finally:
        # Tear down the temp bank — proposals are persisted to
        # ``out_src/`` so dropping the temp dir doesn't lose data.
        shutil.rmtree(temp_root, ignore_errors=True)


def _infer_domain(corpus: str, source: str) -> str:
    """Best-effort domain id for the live Crafter / lifecycle invariants.

    The on-disk bank uses ``"gymv"`` for both gym_v envs and
    env_wrappers games (Phase-1 single-domain). We honour that.
    """
    return "gymv"


# ═══════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════

def _discover_pairs(
    bank_run: Path,
    actions_run: Path,
    corpus_filter: Optional[str],
    source_filter: Optional[str],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for corpus in CORPORA:
        if corpus_filter and corpus != corpus_filter:
            continue
        cdir_b = bank_run / corpus
        cdir_a = actions_run / corpus
        if not cdir_b.exists() or not cdir_a.exists():
            continue
        for src_dir in sorted(cdir_b.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("_"):
                continue
            if source_filter and src_dir.name != source_filter:
                continue
            if not (src_dir / "skill_bank.jsonl").exists():
                continue
            actions_dir = cdir_a / src_dir.name
            if not actions_dir.exists():
                continue
            if not list(actions_dir.glob("episode_*.json")):
                continue
            out.append((corpus, src_dir.name))
    return out


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bank-run", type=Path, default=DEFAULT_BANK_RUN,
                   help="Skill-bank snapshot directory.")
    p.add_argument("--actions-run", type=Path, default=DEFAULT_ACTIONS_RUN,
                   help="Skill-actions snapshot directory (per-episode JSONs).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output root; defaults to "
                        "labeling_supplement/episode_reflections_out/run_<ts>.")
    p.add_argument("--corpus", choices=CORPORA, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Per-source cap on episodes processed (smoke testing).")
    p.add_argument("--low-applicability", type=float,
                   default=DEFAULT_LOW_APPLICABILITY,
                   help="Threshold below which a per-step applicability "
                        "is treated as a PRECONDITION_VIOLATION.")
    p.add_argument("--max-failures-per-episode", type=int,
                   default=DEFAULT_MAX_FAILURES_PER_EPISODE,
                   help="Hard cap on synthesised FailureTraces per episode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print discovered pairs and exit without invoking the Crafter.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    bank_run: Path = args.bank_run.resolve()
    actions_run: Path = args.actions_run.resolve()
    if not bank_run.exists():
        logger.error("bank-run does not exist: %s", bank_run)
        return 2
    if not actions_run.exists():
        logger.error("actions-run does not exist: %s", actions_run)
        return 2

    output_root: Path = (
        args.output_dir.resolve() if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / f"run_{_utc_run_stamp()}").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(
        bank_run, actions_run,
        corpus_filter=args.corpus,
        source_filter=args.source,
    )
    if not pairs:
        logger.error("no (corpus, source) pairs discovered")
        return 2

    logger.info("reflect_per_episode: %d pair(s) under bank=%s actions=%s",
                len(pairs), bank_run, actions_run)
    logger.info("  output_root : %s", output_root)
    logger.info("  thresholds  : low_app=%s, max_failures=%s",
                args.low_applicability, args.max_failures_per_episode)
    if args.dry_run:
        for c, s in pairs:
            print(f"  {c} / {s}")
        return 0

    started_at = _utcnow_iso()

    per_pair_summaries: List[Dict[str, Any]] = []
    total_props = 0
    total_subsumes = 0
    total_coalesced = 0
    total_cooldown = 0
    total_eps = 0
    by_kind_total: Counter[str] = Counter()
    by_proposer_total: Counter[str] = Counter()

    for corpus, source in pairs:
        logger.info("processing %s / %s", corpus, source)
        row = _process_source(
            corpus=corpus, source=source,
            bank_run=bank_run, actions_run=actions_run,
            output_root=output_root,
            max_episodes=args.max_episodes,
            low_app=args.low_applicability,
            max_failures=args.max_failures_per_episode,
        )
        per_pair_summaries.append(row)
        total_props += row.get("n_proposals", 0)
        total_subsumes += row.get("n_subsumption_retires", 0)
        total_coalesced += row.get("n_patches_coalesced", 0)
        total_cooldown += row.get("n_patches_skipped_cooldown", 0)
        total_eps += row.get("n_episodes", 0)
        by_kind_total.update(row.get("by_kind") or {})
        by_proposer_total.update(row.get("by_proposer") or {})
        logger.info("  %s/%s -> %d ep(s), %d proposal(s), subsumes=%d (%s)",
                    corpus, source,
                    row.get("n_episodes", 0),
                    row.get("n_proposals", 0),
                    row.get("n_subsumption_retires", 0),
                    ", ".join(f"{k}={v}" for k, v in (row.get("by_kind") or {}).items()) or "-")

    (output_root / "_run_meta.json").write_text(json.dumps({
        "bank_run":      str(bank_run),
        "actions_run":   str(actions_run),
        "output_root":   str(output_root),
        "thresholds": {
            "low_applicability": args.low_applicability,
            "max_failures_per_episode": args.max_failures_per_episode,
        },
        "max_episodes":  args.max_episodes,
        "pairs":         [{"corpus": c, "source": s} for c, s in pairs],
        "started_at":    started_at,
        "argv":          [str(a) for a in (argv or sys.argv)],
    }, indent=2))

    (output_root / "_run_summary.json").write_text(json.dumps({
        "bank_run":                  str(bank_run),
        "actions_run":               str(actions_run),
        "output_root":               str(output_root),
        "n_pairs":                   len(pairs),
        "n_pairs_ok":                sum(1 for r in per_pair_summaries if r.get("status") == "ok"),
        "n_episodes":                total_eps,
        "n_proposals":               total_props,
        "n_subsumption_retires":     total_subsumes,
        "n_patches_coalesced":       total_coalesced,
        "n_patches_skipped_cooldown":total_cooldown,
        "by_kind":                   dict(by_kind_total),
        "by_proposer":               dict(by_proposer_total),
        "per_pair":                  per_pair_summaries,
        "started_at":                started_at,
        "completed_at":              _utcnow_iso(),
    }, indent=2))

    logger.info(
        "DONE: %d pair(s), %d episode(s), %d proposal(s), "
        "%d subsumption-retire(s), %d coalesced, %d cooldown-skipped",
        len(pairs), total_eps, total_props, total_subsumes,
        total_coalesced, total_cooldown,
    )
    logger.info("  by kind     : %s",
                ", ".join(f"{k}={v}" for k, v in by_kind_total.most_common()) or "-")
    logger.info("  by proposer : %s",
                ", ".join(f"{k}={v}" for k, v in by_proposer_total.most_common()) or "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
