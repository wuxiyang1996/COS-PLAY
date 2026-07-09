#!/usr/bin/env python
"""
Skill Crafter — decide whether to craft / patch / transfer / retire skills.

This script is the offline "between-batch" decision step described in
``plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`` (Composer + Generalizer
+ Hypothesizer + Failure-Reflector roles) and ``plans/06-orchestrator``
§3a (the orchestrator hands the resulting proposals to the Harness
``GateRunner`` for G0–G5 evaluation).

What it does
------------
For each ``(corpus, source_name)`` in a skill-bank snapshot, the script:

  1. Loads the skill bank (``skill_bank.jsonl`` + ``skill_catalog.json``
     + the per-episode ``bank_management_io.json`` update history);
  2. Optionally enriches every skill with usage statistics from a
     skill-actions snapshot (selection histogram, mean confidence,
     mean applicability, adjacent co-occurrence transitions);
  3. Applies five rule-based decision heuristics (R1–R5 below) to
     decide which typed ``BankMutationProposal`` (if any) to emit;
  4. Writes one ``proposals.jsonl`` per source with the typed proposals
     plus a ``decision_trace.json`` that records, for every skill, which
     rules fired with what numbers — so a downstream reviewer (or the
     Harness GateRunner replay path) can audit the decision.

Why no Harness / no Orchestrator wiring?
---------------------------------------
This script is **strictly content production**: it produces the
``BankMutationProposal`` artifacts that the bank lifecycle
(``draft_store`` → ``candidate_store``) expects, but never moves bank
pointers, never calls the gate stages, never writes
``verified_domains``. Those are the responsibilities of the
``GateService`` / ``PromotionOrchestrator``. Concretely:

  * proposals here are **proposals only** — no skill is promoted or
    retired by running this script;
  * proposals respect the typed schema in PLAN-SKILL-CRAFTER §2.5
    (each proposal carries ``evidence_role``, ``evidence_interface``,
    ``target_domains`` covering all five, ``adapter_plan``,
    ``replay_slice_ids``, and a free-form ``rationale``);
  * one proposer string is recorded per proposal (``"composer"``,
    ``"generalizer"``, ``"hypothesizer"``, ``"reflector"``) so the
    downstream gate dashboard can break failures down by source.

Decision rules (configurable thresholds)
----------------------------------------
R1 — **Retire (evidence-starved)** ::

    (eff_add_n + eff_del_n + eff_event_n) == 0 AND n_instances < min_inst_for_keep
    OR usage_pct < retire_usage_pct_min  (when skill_actions is provided)
    -> RetireProposal{retire_reason: "evidence-starved"}

R2 — **Patch (warrant-strengthen)** ::

    evidence_role == COMMIT AND (eff_add_n + eff_event_n) == 0
    AND n_instances >= min_inst_for_keep
    -> PatchProposal{patch_kind: "warrant-strengthen"}

R3 — **Patch (precondition tightening)** ::

    pass_rate == 1.0 AND mean_applicability looks saturated
    (default 0.5 ± 0.02 across >= min_usage_for_signal selections)
    -> PatchProposal{patch_kind: "precondition"}

R4 — **Compose (sequence)** ::

    co_occurrence(A, B) >= compose_threshold * total_transitions
    AND neither A nor B already retired
    AND a similar pair has not already been proposed
    -> ComposeProposal{components: [A, B], compose_op: "sequence"}

R5 — **Transfer** ::

    len(applicable_domains) == 1 AND n_instances >= transfer_min_instances
    AND pass_rate >= transfer_min_pass_rate
    -> TransferProposal -> per-target slot_remap stub for browser /
       osworld / video / visual_reasoning

These thresholds are conservative-on-purpose: in Phase-1 of the Crafter
plan we want a high-precision "low false-positive" pipeline so the
downstream Harness gate stack (G0–G5) is not flooded with junk. All
thresholds are CLI-overridable (see ``--help``).

Inputs
------
* ``--bank-run`` (required) — a ``labeling/skill_bank_out/run_<ts>``
  directory laid out as ::

      <run>/<corpus>/<source>/skill_bank.jsonl
      <run>/<corpus>/<source>/skill_catalog.json
      <run>/<corpus>/<source>/_lifecycle_meta.json
      <run>/<corpus>/<source>/per_episode_bank_management/episode_<i>/bank_management_io.json

* ``--actions-run`` (optional) — a ``labeling/skill_actions_out/run_<ts>``
  directory used only for *usage statistics* (selection histogram,
  adjacent co-occurrence). When omitted, R1's usage-pct branch and R4
  (compose) are skipped — the bank-only heuristics still fire.

Outputs
-------
``<output_dir>/<corpus>/<source>/proposals.jsonl`` — one JSON object
per line, conforming to PLAN-SKILL-CRAFTER §2.5.

``<output_dir>/<corpus>/<source>/decision_trace.json`` — per-skill
diagnostics (which rules fired, with what numeric inputs).

``<output_dir>/<corpus>/<source>/_crafter_summary.json`` — per-source
stats (n_skills_in, n_proposals_out, by-rule histogram).

``<output_dir>/_run_meta.json`` and ``_run_summary.json`` — run-level.

Usage
-----

    # All (corpus, source) pairs in the latest bank, with action stats.
    python labeling_supplement/decide_skill_crafting_gpt54.py \
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \
        --output-dir   labeling_supplement/crafter_proposals_out/run_<ts>

    # Smoke test — one source, no action stats.
    python labeling_supplement/decide_skill_crafting_gpt54.py \
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \
        --corpus       env_wrappers --source twenty_forty_eight \
        --output-dir   labeling_supplement/crafter_proposals_out/_smoke -v

    # Override decision thresholds.
    python labeling_supplement/decide_skill_crafting_gpt54.py \
        --bank-run ... \
        --retire-usage-pct-min 0.01 \
        --compose-threshold 0.10 \
        --transfer-min-instances 50

The companion dispatcher ``run_decide_skill_crafting.sh`` fans this
out one worker per ``(corpus, source)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utcnow_iso() -> str:
    """Timezone-aware UTC ISO-8601 stamp (avoids the deprecated
    ``datetime.utcnow()``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_run_stamp() -> str:
    """Compact UTC stamp used in run-directory names."""
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

logger = logging.getLogger("labeling_supplement.decide_skill_crafting")


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
    CODEBASE_ROOT / "labeling_supplement" / "crafter_proposals_out"
)

CORPORA = ("gym_v", "env_wrappers")

# Plan §0.1 / §2.5 — every proposal must declare all five domains as
# its target set so the gate stack can verify general-protocol
# feasibility. The Crafter never narrows this list.
ALL_FIVE_DOMAINS: Tuple[str, ...] = (
    "gymv", "browser", "osworld", "video", "visual_reasoning",
)

# The Skill Crafter is the "synthesis-reflection agent" running on the
# control-plane teacher backbone. Phase 1 is rule-based + frozen teacher
# only (no trainable updates) — we still log the teacher identifier in
# the run meta for downstream auditing. Source of truth:
# ``common.models.BACKBONE_TEACHER_MODEL`` (set by the 2026-04-28
# model-stack migration to ``Qwen/Qwen3.5-35B-A3B``); the historical
# ``gpt-5.4`` literal that lived here is stale.
try:
    from common.models import BACKBONE_TEACHER_MODEL as _CRAFTER_TEACHER_MODEL
except Exception:  # pragma: no cover — fallback for very old checkouts
    _CRAFTER_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"

DEFAULT_TEACHER_MODEL = _CRAFTER_TEACHER_MODEL

# Rule thresholds (Phase-1 conservative defaults — see PLAN-SKILL-CRAFTER §10).
DEFAULTS = dict(
    min_inst_for_keep=5,           # < this and no effects -> retire
    retire_usage_pct_min=0.005,    # < 0.5 % of selections -> retire
    min_usage_for_signal=10,       # ignore stats below this many selections
    saturated_app_centre=0.5,      # applicability cluster centre
    saturated_app_tol=0.02,        # +/- this counts as saturated
    compose_threshold=0.05,        # 5 % of transitions -> compose
    transfer_min_instances=20,
    transfer_min_pass_rate=0.7,
    protocol_min_steps=3,
)


# ═══════════════════════════════════════════════════════════════════════
# Typed proposals — schema mirrored from PLAN-SKILL-CRAFTER §2.5
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceInterfaceDecl:
    """Per PLAN-SKILL-CRAFTER §2.5 — declares how a skill reads / writes
    addressable evidence. The `evidence_outputs_or_warrant_spec` shape
    is role-dependent (see PLAN-SKILL-BANK §4.2)."""

    evidence_inputs_spec: List[str] = field(default_factory=list)
    evidence_outputs_or_warrant_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseProposal:
    """Common fields enforced at gate-time (PLAN-SKILL-CRAFTER §2.5)."""

    proposal_id: str
    proposer: str                                 # composer | generalizer | hypothesizer | reflector
    evidence_role: str                            # GATHER | VERIFY | REASON | COMMIT
    evidence_interface: EvidenceInterfaceDecl
    target_domains: List[str]                     # MUST cover all five
    adapter_plan: Dict[str, Any]                  # per-domain adapter strategy stub
    replay_slice_ids: List[str]                   # for G3 replay
    rationale: str
    proposal_kind: str = ""                       # set by subclass: patch | compose | transfer | retire

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_interface"] = asdict(self.evidence_interface)
        return d


@dataclass
class PatchProposal(BaseProposal):
    target_skill_id: str = ""
    target_skill_version: int = 0
    patch_kind: str = ""              # precondition | protocol | contract | warrant-strengthen
    patch_body: Dict[str, Any] = field(default_factory=dict)
    proposal_kind: str = "patch"


@dataclass
class ComposeProposal(BaseProposal):
    components: List[str] = field(default_factory=list)        # ordered sub-skill_ids
    compose_op: str = "sequence"                                # sequence | branch | loop | while-insufficient
    component_evidence_roles: List[str] = field(default_factory=list)
    co_occurrence_count: int = 0
    co_occurrence_pct: float = 0.0
    proposal_kind: str = "compose"


@dataclass
class TransferProposal(BaseProposal):
    source_skill_id: str = ""
    source_skill_version: int = 0
    source_domain: str = ""                                     # currently always "gymv"
    new_adapter_per_target: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence_interface_remap: Dict[str, Dict[str, str]] = field(default_factory=dict)
    slot_remap_per_target: Dict[str, Dict[str, str]] = field(default_factory=dict)
    proposal_kind: str = "transfer"


@dataclass
class RetireProposal(BaseProposal):
    target_skill_id: str = ""
    target_skill_version: int = 0
    retire_reason: str = ""           # opaque | evidence-starved | subsumed | unsafe | regressing | superseded
    evidence_stats: Dict[str, Any] = field(default_factory=dict)
    proposal_kind: str = "retire"


# ═══════════════════════════════════════════════════════════════════════
# Bank loading
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LoadedSkill:
    """One skill from a per-source bank, normalized for decision rules."""

    skill_id: str
    name: str
    version: int
    summary: str
    description: str
    evidence_role: str
    applicable_domains: List[str]
    verified_domains: List[str]
    status: str
    n_instances: int
    pass_rate: float
    eff_add: List[str]
    eff_del: List[str]
    eff_event: List[str]
    sub_episodes: List[Dict[str, Any]]
    protocol: Dict[str, Any]
    contract: Dict[str, Any]
    raw: Dict[str, Any]                            # original record for traceability


def _load_skill_bank_jsonl(path: Path) -> List[LoadedSkill]:
    """Parse ``skill_bank.jsonl`` into a list of ``LoadedSkill``.

    The on-disk schema is one ``{"skill": {...}, "report": {...}}`` per
    line (see PLAN-SKILL-BANK / labeling/readme.md). We extract the
    fields the decision rules need and keep the raw record for the
    decision-trace audit.
    """
    out: List[LoadedSkill] = []
    if not path.exists():
        logger.warning("skill_bank.jsonl missing: %s", path)
        return out

    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping malformed jsonl line %d in %s: %s",
                               line_no, path, exc)
                continue

            skill = rec.get("skill") or {}
            report = rec.get("report") or {}
            contract = skill.get("contract") or {}

            out.append(LoadedSkill(
                skill_id=skill.get("skill_id", f"_unknown_{line_no}"),
                name=skill.get("name", ""),
                version=int(skill.get("version", 0) or 0),
                summary=contract.get("description", "") or skill.get("strategic_description", ""),
                description=skill.get("strategic_description", ""),
                evidence_role=skill.get("evidence_role", ""),
                applicable_domains=list(skill.get("applicable_domains") or []),
                verified_domains=list(skill.get("verified_domains") or []),
                status=skill.get("status", "draft"),
                n_instances=int(skill.get("n_instances", 0) or 0),
                pass_rate=float(report.get("overall_pass_rate", 0.0) or 0.0),
                eff_add=list(contract.get("eff_add") or []),
                eff_del=list(contract.get("eff_del") or []),
                eff_event=list(contract.get("eff_event") or []),
                sub_episodes=list(skill.get("sub_episodes") or []),
                protocol=skill.get("protocol") or {},
                contract=contract,
                raw=skill,
            ))
    return out


def _load_lifecycle_meta(source_dir: Path) -> Dict[str, Any]:
    p = source_dir / "_lifecycle_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _load_recent_bank_updates(source_dir: Path, max_episodes: int = 5) -> Dict[str, Any]:
    """Summarise the most recent N per-episode bank-management actions.

    The full per-episode I/O lives at
    ``per_episode_bank_management/episode_<i>/bank_management_io.json``.
    We surface only the splits / merges / refines counts so a follow-on
    PatchProposal can cite them without re-loading the (large) JSONs.
    """
    summary: Dict[str, Any] = {
        "episodes_inspected": 0,
        "n_splits": 0, "n_merges": 0, "n_refines": 0,
        "skills_refined": [],
    }
    base = source_dir / "per_episode_bank_management"
    if not base.exists():
        return summary

    eps = sorted(base.iterdir(), key=lambda p: p.name)
    if len(eps) > max_episodes:
        eps = eps[-max_episodes:]

    refined: Counter = Counter()
    for ep_dir in eps:
        f = ep_dir / "bank_management_io.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        summary["episodes_inspected"] += 1
        s4 = (data.get("stage_4_bank_maintenance") or {}).get("outputs") or {}
        summary["n_splits"]  += int(s4.get("n_splits", 0) or 0)
        summary["n_merges"]  += int(s4.get("n_merges", 0) or 0)
        summary["n_refines"] += int(s4.get("n_refines", 0) or 0)
        for r in (s4.get("refine_details") or []):
            sid = ((r.get("_fields") or {}).get("skill_id"))
            if sid:
                refined[sid] += 1

    summary["skills_refined"] = [
        {"skill_id": sid, "n_refines": n} for sid, n in refined.most_common()
    ]
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Skill-actions usage statistics (optional input)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UsageStats:
    """Per-source aggregate usage stats from a ``skill_actions_out`` run."""

    n_steps: int
    n_with_skill: int
    selection_histogram: Dict[str, int]
    mean_confidence_per_skill: Dict[str, float]
    mean_applicability_per_skill: Dict[str, float]
    n_selections_per_skill: Dict[str, int]
    transitions: Dict[Tuple[str, str], int]                # adjacent (A, B) pairs
    n_transitions: int


def _zero_usage_stats() -> UsageStats:
    return UsageStats(0, 0, {}, {}, {}, {}, {}, 0)


def _aggregate_usage_for_source(actions_run: Path, corpus: str, source: str,
                                ) -> UsageStats:
    """Walk every ``episode_*.json`` under
    ``<actions_run>/<corpus>/<source>/`` and collect (a) per-skill
    selection counts and means, plus (b) adjacent-step transition
    counts for compose proposals.

    Falls back gracefully if the folder or summary is missing — the
    caller treats that as "no usage stats available" and skips R1's
    usage-pct branch and R4 entirely.
    """
    src_dir = actions_run / corpus / source
    if not src_dir.exists():
        return _zero_usage_stats()

    sel_hist: Counter = Counter()
    n_sel: Counter = Counter()
    confs: Dict[str, List[float]] = defaultdict(list)
    apps:  Dict[str, List[float]] = defaultdict(list)
    transitions: Counter = Counter()
    n_steps = 0
    n_with = 0
    n_trans = 0

    eps = sorted(src_dir.glob("episode_*.json"))
    for ep in eps:
        try:
            data = json.loads(ep.read_text())
        except Exception as exc:
            logger.warning("skip unparseable %s: %s", ep, exc)
            continue
        exps = data.get("experiences") or data.get("steps") or []
        prev_id: Optional[str] = None
        for exp in exps:
            n_steps += 1
            sk = exp.get("skills") or {}
            sid = sk.get("skill_id")
            if sid:
                n_with += 1
                sel_hist[sid] += 1
                n_sel[sid] += 1
                conf = sk.get("confidence")
                if isinstance(conf, (int, float)):
                    confs[sid].append(float(conf))
                app = sk.get("applicability")
                if isinstance(app, (int, float)):
                    apps[sid].append(float(app))
                if prev_id is not None and prev_id != sid:
                    transitions[(prev_id, sid)] += 1
                    n_trans += 1
                prev_id = sid
            else:
                prev_id = None

    return UsageStats(
        n_steps=n_steps,
        n_with_skill=n_with,
        selection_histogram=dict(sel_hist),
        mean_confidence_per_skill={k: sum(v) / len(v) for k, v in confs.items()},
        mean_applicability_per_skill={k: sum(v) / len(v) for k, v in apps.items()},
        n_selections_per_skill=dict(n_sel),
        transitions=dict(transitions),
        n_transitions=n_trans,
    )


# ═══════════════════════════════════════════════════════════════════════
# Helpers shared across rules
# ═══════════════════════════════════════════════════════════════════════

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def _stable_proposal_id(corpus: str, source: str, skill_id: str,
                        rule: str, idx: int = 0) -> str:
    return f"prop_{_slug(corpus)}_{_slug(source)}_{_slug(skill_id)}_{rule}_{idx}"


def _evidence_interface_from_skill(sk: LoadedSkill) -> EvidenceInterfaceDecl:
    """Reflect the bank skill's contract back into the proposal as the
    EvidenceInterface declaration. The Crafter does NOT decide the
    interface — it inherits from the source skill and patches narrowly.
    """
    out_kinds = sorted({s for s in (sk.eff_add + sk.eff_event) if s})
    in_kinds  = sorted({s for s in sk.eff_del if s})
    spec: Dict[str, Any] = {}

    role = (sk.evidence_role or "").upper()
    if role == "GATHER":
        spec = {"evidence_out_kinds": out_kinds or ["world_predicate"]}
    elif role == "VERIFY":
        spec = {
            "verdict_domain": ["PASS", "FAIL", "INSUFFICIENT"],
            "referenced_evidence_kinds": in_kinds or ["world_predicate"],
        }
    elif role == "REASON":
        spec = {
            "hypothesis_schema": {"hypothesis": "str", "warrant": "list[evidence_ref]"},
            "warrant_kinds": in_kinds or ["world_predicate"],
        }
    elif role == "COMMIT":
        spec = {
            "decision_schema": {"action": "str"},
            "evidence_warrant_kinds": (in_kinds + out_kinds) or ["world_predicate"],
        }

    return EvidenceInterfaceDecl(
        evidence_inputs_spec=in_kinds or ["world_predicate"],
        evidence_outputs_or_warrant_spec=spec,
    )


def _replay_slice_ids_for(sk: LoadedSkill, max_slices: int = 8) -> List[str]:
    """Pick representative ``sub_episodes`` segments to feed Gate G3.

    We pick the first / median / last few segments by ``cumulative_reward``
    so the replay validator sees both typical and outlier instances.
    """
    sub = sk.sub_episodes
    if not sub:
        return []
    ranked = sorted(
        sub, key=lambda s: float(s.get("cumulative_reward", 0.0) or 0.0)
    )
    if len(ranked) <= max_slices:
        picked = ranked
    else:
        picks: List[Dict[str, Any]] = []
        n = len(ranked)
        picks.extend(ranked[:2])
        picks.append(ranked[n // 2])
        picks.extend(ranked[-2:])
        seen = set()
        picked = []
        for r in picks:
            key = (r.get("seg_start"), r.get("seg_end"), r.get("episode_id"))
            if key in seen:
                continue
            seen.add(key)
            picked.append(r)

    return [
        f"{sk.skill_id}@ep:{r.get('episode_id') or '_'}#{r.get('seg_start')}-{r.get('seg_end')}"
        for r in picked
    ]


def _adapter_plan_stub(sk: LoadedSkill) -> Dict[str, Any]:
    """Stub adapter plan covering all five domains.

    Concrete adapter synthesis happens later in the Harness
    (``AdapterRegistry.request_synthesis``); here we just commit to
    *which* of the five target domains we expect to bind without
    rework, and leave the synthesis hook flagged for the rest.
    """
    src = (sk.applicable_domains or ["gymv"])[0]
    out: Dict[str, Any] = {}
    for d in ALL_FIVE_DOMAINS:
        out[d] = {
            "strategy": "reuse" if d == src else "synthesize_from_slot_ontology",
            "needs_72b_synthesis": d != src,
            "source_domain": src,
        }
    return out


# Mapping between domain "shared schema slots" (PLAN-VISUAL-GROUNDING §3a)
# used by the TransferProposal slot_remap stub. Edit cautiously — the
# Generalizer plan §4 gives the canonical alphabet.
_DEFAULT_SLOT_REMAP: Dict[str, Dict[str, str]] = {
    "browser":          {"target": "ui_element", "blocker": "modal_overlay",
                         "candidate_set": "clickable_set", "constraint": "form_validation"},
    "osworld":          {"target": "desktop_object", "blocker": "modal_window",
                         "candidate_set": "window_set",   "constraint": "permission_constraint"},
    "video":            {"target": "tracked_entity", "blocker": "occlusion",
                         "candidate_set": "frame_candidate_set", "constraint": "temporal_ordering"},
    "visual_reasoning": {"target": "image_entity",   "blocker": "ambiguity",
                         "candidate_set": "candidate_pool", "constraint": "answer_consistency"},
}


# ═══════════════════════════════════════════════════════════════════════
# Decision rules (R1 – R5)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DecisionContext:
    corpus: str
    source: str
    bank: List[LoadedSkill]
    bank_by_id: Dict[str, LoadedSkill]
    usage: UsageStats
    has_usage: bool
    bank_updates: Dict[str, Any]
    cfg: Dict[str, Any]                          # threshold dict

    def usage_pct(self, sid: str) -> float:
        if not self.has_usage or self.usage.n_with_skill == 0:
            return -1.0
        return self.usage.n_selections_per_skill.get(sid, 0) / float(self.usage.n_with_skill)


def rule_R1_retire(sk: LoadedSkill, ctx: DecisionContext) -> Tuple[bool, Dict[str, Any]]:
    """Evidence-starved retire candidate (PLAN-SKILL-CRAFTER §6.5
    Recovery 'skill retirement' + PLAN-HARNESS §10a 'evidence_starved')."""
    n_eff = len(sk.eff_add) + len(sk.eff_del) + len(sk.eff_event)
    fired = False
    diag: Dict[str, Any] = {"n_eff": n_eff, "n_instances": sk.n_instances}

    # Branch A — bank-only: empty contract + low support.
    if n_eff == 0 and sk.n_instances < ctx.cfg["min_inst_for_keep"]:
        fired = True
        diag["branch"] = "empty-contract+low-support"

    # Branch B — usage-based: skill exists but is rarely selected.
    if ctx.has_usage:
        pct = ctx.usage_pct(sk.skill_id)
        diag["usage_pct"] = pct
        if 0.0 <= pct < ctx.cfg["retire_usage_pct_min"]:
            fired = True
            diag.setdefault("branch", "low-usage-pct")

    return fired, diag


def rule_R2_warrant(sk: LoadedSkill, ctx: DecisionContext) -> Tuple[bool, Dict[str, Any]]:
    """COMMIT-role skill that produces no tracked add / event effects
    is "opaque" — its commits aren't backed by a measurable warrant."""
    diag: Dict[str, Any] = {
        "evidence_role": sk.evidence_role,
        "n_eff_add": len(sk.eff_add),
        "n_eff_event": len(sk.eff_event),
        "n_instances": sk.n_instances,
    }
    fired = (
        (sk.evidence_role or "").upper() == "COMMIT"
        and (len(sk.eff_add) + len(sk.eff_event)) == 0
        and sk.n_instances >= ctx.cfg["min_inst_for_keep"]
    )
    return fired, diag


def rule_R3_precondition(sk: LoadedSkill, ctx: DecisionContext) -> Tuple[bool, Dict[str, Any]]:
    """Saturated applicability + ``pass_rate==1.0`` is a sign the
    discriminator is degenerate — a tighter precondition is warranted."""
    if not ctx.has_usage:
        return False, {"reason": "no-usage-stats"}
    n_sel = ctx.usage.n_selections_per_skill.get(sk.skill_id, 0)
    diag: Dict[str, Any] = {
        "n_selections": n_sel, "pass_rate": sk.pass_rate,
        "min_usage_for_signal": ctx.cfg["min_usage_for_signal"],
    }
    if n_sel < ctx.cfg["min_usage_for_signal"]:
        return False, {**diag, "reason": "low-signal"}
    mean_app = ctx.usage.mean_applicability_per_skill.get(sk.skill_id)
    diag["mean_applicability"] = mean_app
    if mean_app is None or sk.pass_rate < 0.999:
        return False, {**diag, "reason": "no-saturation"}
    centre = ctx.cfg["saturated_app_centre"]
    tol    = ctx.cfg["saturated_app_tol"]
    saturated = abs(mean_app - centre) <= tol
    diag["saturated"] = saturated
    return saturated, diag


def rule_R4_compose(ctx: DecisionContext) -> List[Tuple[str, str, int, float]]:
    """Return ranked ``(skill_a, skill_b, count, pct)`` co-occurrences
    that pass the compose threshold. We yield at most one pair per
    ``skill_a`` so the Stage-4 verification queue isn't flooded by a
    single dominant predecessor."""
    if not ctx.has_usage or ctx.usage.n_transitions == 0:
        return []
    threshold = ctx.cfg["compose_threshold"]
    by_a: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    for (a, b), c in ctx.usage.transitions.items():
        pct = c / float(ctx.usage.n_transitions)
        if pct >= threshold:
            by_a[a].append((b, c, pct))
    out: List[Tuple[str, str, int, float]] = []
    for a, candidates in by_a.items():
        candidates.sort(key=lambda x: -x[1])
        b, c, pct = candidates[0]
        if a == b:
            continue
        if a not in ctx.bank_by_id or b not in ctx.bank_by_id:
            continue
        out.append((a, b, c, pct))
    out.sort(key=lambda x: -x[3])
    return out


def rule_R5_transfer(sk: LoadedSkill, ctx: DecisionContext) -> Tuple[bool, Dict[str, Any]]:
    """Mature, reliable, single-domain skill — propose K-shot transfer."""
    diag: Dict[str, Any] = {
        "applicable_domains": sk.applicable_domains,
        "n_instances": sk.n_instances,
        "pass_rate": sk.pass_rate,
        "verified_domains": sk.verified_domains,
    }
    fired = (
        len(sk.applicable_domains) == 1
        and sk.n_instances >= ctx.cfg["transfer_min_instances"]
        and sk.pass_rate >= ctx.cfg["transfer_min_pass_rate"]
        and len(sk.verified_domains) == 0
    )
    return fired, diag


def rule_R3b_protocol(sk: LoadedSkill, ctx: DecisionContext) -> Tuple[bool, Dict[str, Any]]:
    """Well-supported skill with a thin / empty protocol -- emit a
    PatchProposal{patch_kind=protocol} so the Crafter's protocol-rewrite
    pass picks it up.
    """
    steps = list(sk.protocol.get("steps") or [])
    diag: Dict[str, Any] = {"n_protocol_steps": len(steps),
                            "n_instances": sk.n_instances}
    fired = (
        len(steps) < ctx.cfg["protocol_min_steps"]
        and sk.n_instances >= ctx.cfg["min_inst_for_keep"] * 2
    )
    return fired, diag


# ═══════════════════════════════════════════════════════════════════════
# Proposal builders
# ═══════════════════════════════════════════════════════════════════════

def _build_retire(sk: LoadedSkill, diag: Dict[str, Any], ctx: DecisionContext,
                  ) -> RetireProposal:
    return RetireProposal(
        proposal_id=_stable_proposal_id(ctx.corpus, ctx.source, sk.skill_id, "retire"),
        proposer="reflector",
        evidence_role=sk.evidence_role or "COMMIT",
        evidence_interface=_evidence_interface_from_skill(sk),
        target_domains=list(ALL_FIVE_DOMAINS),
        adapter_plan=_adapter_plan_stub(sk),
        replay_slice_ids=_replay_slice_ids_for(sk),
        rationale=(
            f"R1 evidence-starved: contract is empty (eff_add+del+event=0) "
            f"and either support is below the keep threshold "
            f"(n_instances={sk.n_instances} < {ctx.cfg['min_inst_for_keep']}) "
            f"or selection rate is below the floor "
            f"(usage_pct={diag.get('usage_pct')!r}). "
            f"This skill no longer assists reasoning."
        ),
        target_skill_id=sk.skill_id,
        target_skill_version=sk.version,
        retire_reason="evidence-starved",
        evidence_stats={
            "n_instances": sk.n_instances,
            "n_eff_add": len(sk.eff_add),
            "n_eff_del": len(sk.eff_del),
            "n_eff_event": len(sk.eff_event),
            "usage_pct": diag.get("usage_pct"),
            "pass_rate": sk.pass_rate,
            "diag_branch": diag.get("branch"),
        },
    )


def _build_patch_warrant(sk: LoadedSkill, diag: Dict[str, Any],
                         ctx: DecisionContext) -> PatchProposal:
    return PatchProposal(
        proposal_id=_stable_proposal_id(ctx.corpus, ctx.source, sk.skill_id, "warrant"),
        proposer="reflector",
        evidence_role=sk.evidence_role or "COMMIT",
        evidence_interface=_evidence_interface_from_skill(sk),
        target_domains=list(ALL_FIVE_DOMAINS),
        adapter_plan=_adapter_plan_stub(sk),
        replay_slice_ids=_replay_slice_ids_for(sk),
        rationale=(
            f"R2 warrant-strengthen: COMMIT-role skill with empty add/event "
            f"effects but {sk.n_instances} instances — commits cite no "
            f"tracked predicate change and would fail Gate G0 "
            f"(opaque_skill_violation) on every shadow run. Require "
            f"`evidence_warrant` to cite at least one of the segment's "
            f"observed predicate flips."
        ),
        target_skill_id=sk.skill_id,
        target_skill_version=sk.version,
        patch_kind="warrant-strengthen",
        patch_body={
            "require_evidence_warrant_kinds": ["world_predicate", "tile_state", "score_delta"],
            "min_warrant_count": 1,
            "rejection_label": "opaque_skill_violation",
        },
    )


def _build_patch_precondition(sk: LoadedSkill, diag: Dict[str, Any],
                              ctx: DecisionContext) -> PatchProposal:
    return PatchProposal(
        proposal_id=_stable_proposal_id(ctx.corpus, ctx.source, sk.skill_id, "precond"),
        proposer="reflector",
        evidence_role=sk.evidence_role or "COMMIT",
        evidence_interface=_evidence_interface_from_skill(sk),
        target_domains=list(ALL_FIVE_DOMAINS),
        adapter_plan=_adapter_plan_stub(sk),
        replay_slice_ids=_replay_slice_ids_for(sk),
        rationale=(
            f"R3 precondition tightening: pass_rate={sk.pass_rate:.3f} and "
            f"mean_applicability={diag.get('mean_applicability'):.3f} "
            f"saturated near {ctx.cfg['saturated_app_centre']:.2f} across "
            f"{diag.get('n_selections')} selections. Discriminator is "
            f"degenerate — emit a tighter precondition predicate so the "
            f"skill triggers only when it strictly applies."
        ),
        target_skill_id=sk.skill_id,
        target_skill_version=sk.version,
        patch_kind="precondition",
        patch_body={
            "add_predicate": "state_changes_after_action(target=$target)",
            "current_preconditions": list(sk.protocol.get("preconditions") or []),
            "saturation_signal": {
                "mean_applicability": diag.get("mean_applicability"),
                "pass_rate": sk.pass_rate,
                "n_selections": diag.get("n_selections"),
            },
        },
    )


def _build_patch_protocol(sk: LoadedSkill, diag: Dict[str, Any],
                          ctx: DecisionContext) -> PatchProposal:
    return PatchProposal(
        proposal_id=_stable_proposal_id(ctx.corpus, ctx.source, sk.skill_id, "protocol"),
        proposer="hypothesizer",
        evidence_role=sk.evidence_role or "COMMIT",
        evidence_interface=_evidence_interface_from_skill(sk),
        target_domains=list(ALL_FIVE_DOMAINS),
        adapter_plan=_adapter_plan_stub(sk),
        replay_slice_ids=_replay_slice_ids_for(sk),
        rationale=(
            f"R3b protocol-rewrite: skill is well supported "
            f"(n_instances={sk.n_instances}) but its protocol has only "
            f"{diag.get('n_protocol_steps')} step(s) — too thin for the "
            f"actor to follow. Ask the frozen 32B/72B teacher to expand "
            f"the protocol while preserving contract."
        ),
        target_skill_id=sk.skill_id,
        target_skill_version=sk.version,
        patch_kind="protocol",
        patch_body={
            "current_steps": list(sk.protocol.get("steps") or []),
            "min_steps_target": ctx.cfg["protocol_min_steps"],
            "preserve_fields": ["preconditions", "success_criteria",
                                "abort_criteria", "expected_duration"],
        },
    )


def _build_compose(a_id: str, b_id: str, count: int, pct: float,
                   ctx: DecisionContext) -> ComposeProposal:
    a = ctx.bank_by_id[a_id]
    b = ctx.bank_by_id[b_id]
    # Compose evidence-role: per §2.5, a ComposeProposal whose components
    # are both evidence-driven inherits the role of its terminal step
    # for COMMIT/REASON behaviour. We pick `b`'s role as the macro role.
    role = (b.evidence_role or a.evidence_role or "COMMIT").upper()
    iface = EvidenceInterfaceDecl(
        evidence_inputs_spec=sorted(set(
            (_evidence_interface_from_skill(a).evidence_inputs_spec or [])
            + (_evidence_interface_from_skill(b).evidence_inputs_spec or [])
        )),
        evidence_outputs_or_warrant_spec=_evidence_interface_from_skill(b)
            .evidence_outputs_or_warrant_spec,
    )
    return ComposeProposal(
        proposal_id=_stable_proposal_id(
            ctx.corpus, ctx.source, f"{a_id}__then__{b_id}", "compose"),
        proposer="composer",
        evidence_role=role,
        evidence_interface=iface,
        target_domains=list(ALL_FIVE_DOMAINS),
        adapter_plan={
            "components_share_adapter": (
                set(a.applicable_domains or []) == set(b.applicable_domains or [])
            ),
            "macro_strategy": "sequence_via_postcondition_chaining",
        },
        replay_slice_ids=(_replay_slice_ids_for(a, max_slices=4)
                          + _replay_slice_ids_for(b, max_slices=4)),
        rationale=(
            f"R4 compose: ({a_id} -> {b_id}) co-occurs {count} times "
            f"({pct*100:.1f}% of {ctx.usage.n_transitions} adjacent "
            f"transitions). Promote the chain to a named compound skill "
            f"so the actor can retrieve it as one unit."
        ),
        components=[a_id, b_id],
        compose_op="sequence",
        component_evidence_roles=[a.evidence_role or "COMMIT",
                                  b.evidence_role or "COMMIT"],
        co_occurrence_count=count,
        co_occurrence_pct=pct,
    )


def _build_transfer(sk: LoadedSkill, diag: Dict[str, Any],
                    ctx: DecisionContext) -> TransferProposal:
    src_dom = (sk.applicable_domains or ["gymv"])[0]
    targets = [d for d in ALL_FIVE_DOMAINS if d != src_dom]
    return TransferProposal(
        proposal_id=_stable_proposal_id(ctx.corpus, ctx.source, sk.skill_id, "transfer"),
        proposer="generalizer",
        evidence_role=sk.evidence_role or "COMMIT",
        evidence_interface=_evidence_interface_from_skill(sk),
        target_domains=list(ALL_FIVE_DOMAINS),
        adapter_plan=_adapter_plan_stub(sk),
        replay_slice_ids=_replay_slice_ids_for(sk),
        rationale=(
            f"R5 transfer: mature single-domain skill "
            f"(applicable_domains={sk.applicable_domains}, "
            f"n_instances={sk.n_instances}, pass_rate={sk.pass_rate:.3f}). "
            f"Hand to FewShotAdapter (Stage 3a) for K-shot probes against "
            f"the four other target domains; populate `verified_domains` "
            f"only after the gate passes."
        ),
        source_skill_id=sk.skill_id,
        source_skill_version=sk.version,
        source_domain=src_dom,
        new_adapter_per_target={
            t: {"strategy": "synthesize_from_slot_ontology",
                "needs_72b_synthesis": True,
                "fallback": "skip_target_with_diagnostic"}
            for t in targets
        },
        evidence_interface_remap={
            t: {"world_predicate": "domain_specific_predicate"}
            for t in targets
        },
        slot_remap_per_target={
            t: dict(_DEFAULT_SLOT_REMAP.get(t, {}))
            for t in targets
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Per-source orchestration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CrafterRunResult:
    corpus: str
    source: str
    bank_path: Path
    n_skills_in: int
    proposals: List[BaseProposal]
    decision_trace: List[Dict[str, Any]]
    by_kind: Counter
    by_proposer: Counter
    elapsed_sec: float
    error: Optional[str] = None


def decide_for_source(
    corpus: str,
    source: str,
    bank_run: Path,
    actions_run: Optional[Path],
    cfg: Dict[str, Any],
) -> CrafterRunResult:
    """Run all five rules for one (corpus, source) and return the
    typed proposal list + per-skill decision trace."""

    t0 = time.time()
    src_dir = bank_run / corpus / source
    bank_path = src_dir / "skill_bank.jsonl"
    by_kind: Counter = Counter()
    by_proposer: Counter = Counter()
    proposals: List[BaseProposal] = []
    trace: List[Dict[str, Any]] = []

    try:
        bank = _load_skill_bank_jsonl(bank_path)
    except Exception as exc:
        return CrafterRunResult(
            corpus=corpus, source=source, bank_path=bank_path,
            n_skills_in=0, proposals=[], decision_trace=[],
            by_kind=by_kind, by_proposer=by_proposer,
            elapsed_sec=time.time() - t0,
            error=f"bank-load-failed: {type(exc).__name__}: {exc}",
        )

    if not bank:
        return CrafterRunResult(
            corpus=corpus, source=source, bank_path=bank_path,
            n_skills_in=0, proposals=[], decision_trace=[],
            by_kind=by_kind, by_proposer=by_proposer,
            elapsed_sec=time.time() - t0,
            error="empty-bank",
        )

    if actions_run is not None:
        usage = _aggregate_usage_for_source(actions_run, corpus, source)
    else:
        usage = _zero_usage_stats()
    has_usage = usage.n_with_skill > 0

    bank_updates = _load_recent_bank_updates(src_dir)

    ctx = DecisionContext(
        corpus=corpus, source=source,
        bank=bank,
        bank_by_id={s.skill_id: s for s in bank},
        usage=usage, has_usage=has_usage,
        bank_updates=bank_updates,
        cfg=cfg,
    )

    # Per-skill rules R1, R2, R3, R3b, R5.
    for sk in bank:
        skill_trace: Dict[str, Any] = {
            "skill_id": sk.skill_id,
            "version": sk.version,
            "evidence_role": sk.evidence_role,
            "n_instances": sk.n_instances,
            "pass_rate": sk.pass_rate,
            "n_eff_add": len(sk.eff_add),
            "n_eff_del": len(sk.eff_del),
            "n_eff_event": len(sk.eff_event),
            "applicable_domains": sk.applicable_domains,
            "verified_domains": sk.verified_domains,
            "rules_fired": [],
            "rules": {},
        }

        r1, d1 = rule_R1_retire(sk, ctx)
        skill_trace["rules"]["R1_retire"] = {"fired": r1, **d1}
        if r1:
            p = _build_retire(sk, d1, ctx)
            proposals.append(p)
            by_kind[p.proposal_kind] += 1
            by_proposer[p.proposer] += 1
            skill_trace["rules_fired"].append("R1_retire")
            # If a skill is being retired we skip the patch / transfer rules
            # for it — those would be wasted work for the gate stack.
            trace.append(skill_trace)
            continue

        r2, d2 = rule_R2_warrant(sk, ctx)
        skill_trace["rules"]["R2_warrant"] = {"fired": r2, **d2}
        if r2:
            p = _build_patch_warrant(sk, d2, ctx)
            proposals.append(p)
            by_kind[p.proposal_kind] += 1
            by_proposer[p.proposer] += 1
            skill_trace["rules_fired"].append("R2_warrant")

        r3, d3 = rule_R3_precondition(sk, ctx)
        skill_trace["rules"]["R3_precondition"] = {"fired": r3, **d3}
        if r3:
            p = _build_patch_precondition(sk, d3, ctx)
            proposals.append(p)
            by_kind[p.proposal_kind] += 1
            by_proposer[p.proposer] += 1
            skill_trace["rules_fired"].append("R3_precondition")

        r3b, d3b = rule_R3b_protocol(sk, ctx)
        skill_trace["rules"]["R3b_protocol"] = {"fired": r3b, **d3b}
        if r3b:
            p = _build_patch_protocol(sk, d3b, ctx)
            proposals.append(p)
            by_kind[p.proposal_kind] += 1
            by_proposer[p.proposer] += 1
            skill_trace["rules_fired"].append("R3b_protocol")

        r5, d5 = rule_R5_transfer(sk, ctx)
        skill_trace["rules"]["R5_transfer"] = {"fired": r5, **d5}
        if r5:
            p = _build_transfer(sk, d5, ctx)
            proposals.append(p)
            by_kind[p.proposal_kind] += 1
            by_proposer[p.proposer] += 1
            skill_trace["rules_fired"].append("R5_transfer")

        trace.append(skill_trace)

    # Source-level rule R4 (compose).
    composes = rule_R4_compose(ctx)
    for a_id, b_id, count, pct in composes:
        p = _build_compose(a_id, b_id, count, pct, ctx)
        proposals.append(p)
        by_kind[p.proposal_kind] += 1
        by_proposer[p.proposer] += 1

    return CrafterRunResult(
        corpus=corpus, source=source, bank_path=bank_path,
        n_skills_in=len(bank), proposals=proposals,
        decision_trace=trace,
        by_kind=by_kind, by_proposer=by_proposer,
        elapsed_sec=time.time() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════
# Output writing
# ═══════════════════════════════════════════════════════════════════════

def _write_results_for_source(out_root: Path, res: CrafterRunResult,
                              cfg: Dict[str, Any]) -> Path:
    out_dir = out_root / res.corpus / res.source
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals_path = out_dir / "proposals.jsonl"
    with proposals_path.open("w") as f:
        for p in res.proposals:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    trace_path = out_dir / "decision_trace.json"
    trace_path.write_text(json.dumps({
        "corpus": res.corpus,
        "source": res.source,
        "bank_path": str(res.bank_path),
        "thresholds": cfg,
        "skills": res.decision_trace,
    }, indent=2))

    summary_path = out_dir / "_crafter_summary.json"
    summary_path.write_text(json.dumps({
        "corpus": res.corpus,
        "source": res.source,
        "bank_path": str(res.bank_path),
        "status": "ok" if res.error is None else "error",
        "error": res.error,
        "n_skills_in": res.n_skills_in,
        "n_proposals_out": len(res.proposals),
        "by_kind": dict(res.by_kind),
        "by_proposer": dict(res.by_proposer),
        "elapsed_sec": round(res.elapsed_sec, 3),
        "completed_at": _utcnow_iso(),
    }, indent=2))

    return out_dir


# ═══════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════

def _discover_pairs(bank_run: Path,
                    corpus_filter: Optional[str],
                    source_filter: Optional[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for corpus in CORPORA:
        if corpus_filter and corpus != corpus_filter:
            continue
        cdir = bank_run / corpus
        if not cdir.exists():
            continue
        for src_dir in sorted(cdir.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("_"):
                continue
            if source_filter and src_dir.name != source_filter:
                continue
            if (src_dir / "skill_bank.jsonl").exists():
                out.append((corpus, src_dir.name))
    return out


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank-run", type=Path, default=DEFAULT_BANK_RUN,
                   help="Skill-bank snapshot directory (skill_bank_out/run_<ts>).")
    p.add_argument("--actions-run", type=Path, default=DEFAULT_ACTIONS_RUN,
                   help="Optional skill-actions snapshot for usage statistics.")
    p.add_argument("--no-actions", action="store_true",
                   help="Disable usage-stats enrichment (R1-usage / R3 / R4 still defined "
                        "but won't fire from data).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output root; defaults to "
                        "labeling_supplement/crafter_proposals_out/run_<ts>.")
    p.add_argument("--corpus", choices=CORPORA, default=None,
                   help="Restrict to one corpus.")
    p.add_argument("--source", default=None,
                   help="Restrict to one source (game / Temporal_*-v0 env).")
    p.add_argument("--all", action="store_true",
                   help="Process every (corpus, source) pair.")

    p.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL,
                   help="Frozen teacher identifier (logged into _run_meta.json).")

    # Threshold overrides.
    p.add_argument("--min-inst-for-keep", type=int,
                   default=DEFAULTS["min_inst_for_keep"])
    p.add_argument("--retire-usage-pct-min", type=float,
                   default=DEFAULTS["retire_usage_pct_min"])
    p.add_argument("--min-usage-for-signal", type=int,
                   default=DEFAULTS["min_usage_for_signal"])
    p.add_argument("--saturated-app-centre", type=float,
                   default=DEFAULTS["saturated_app_centre"])
    p.add_argument("--saturated-app-tol", type=float,
                   default=DEFAULTS["saturated_app_tol"])
    p.add_argument("--compose-threshold", type=float,
                   default=DEFAULTS["compose_threshold"])
    p.add_argument("--transfer-min-instances", type=int,
                   default=DEFAULTS["transfer_min_instances"])
    p.add_argument("--transfer-min-pass-rate", type=float,
                   default=DEFAULTS["transfer_min_pass_rate"])
    p.add_argument("--protocol-min-steps", type=int,
                   default=DEFAULTS["protocol_min_steps"])

    p.add_argument("--dry-run", action="store_true",
                   help="Just print discovered pairs and thresholds; no writes.")
    p.add_argument("-v", "--verbose", action="store_true")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    bank_run: Path = args.bank_run.resolve()
    if not bank_run.exists():
        logger.error("bank-run does not exist: %s", bank_run)
        return 2

    actions_run: Optional[Path]
    if args.no_actions:
        actions_run = None
    else:
        ar = args.actions_run.resolve() if args.actions_run else None
        actions_run = ar if (ar and ar.exists()) else None
        if ar is not None and not ar.exists():
            logger.warning("actions-run does not exist, continuing without "
                           "usage stats: %s", ar)

    output_dir: Path = (
        args.output_dir.resolve() if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / f"run_{_utc_run_stamp()}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(
        bank_run,
        corpus_filter=args.corpus,
        source_filter=args.source,
    )
    if not pairs:
        logger.error("no (corpus, source) pairs discovered under %s", bank_run)
        return 2

    cfg = {
        "min_inst_for_keep":     args.min_inst_for_keep,
        "retire_usage_pct_min":  args.retire_usage_pct_min,
        "min_usage_for_signal":  args.min_usage_for_signal,
        "saturated_app_centre":  args.saturated_app_centre,
        "saturated_app_tol":     args.saturated_app_tol,
        "compose_threshold":     args.compose_threshold,
        "transfer_min_instances":args.transfer_min_instances,
        "transfer_min_pass_rate":args.transfer_min_pass_rate,
        "protocol_min_steps":    args.protocol_min_steps,
    }

    logger.info("decide_skill_crafting: %d pair(s) under %s",
                len(pairs), bank_run)
    logger.info("  output_dir   : %s", output_dir)
    logger.info("  actions_run  : %s", actions_run)
    logger.info("  thresholds   : %s",
                json.dumps(cfg, indent=None, sort_keys=True))

    if args.dry_run:
        for c, s in pairs:
            print(f"  {c} / {s}")
        return 0

    started_at = _utcnow_iso()

    # Run.
    per_pair_summaries: List[Dict[str, Any]] = []
    total_props = 0
    total_skills = 0
    by_kind_total: Counter = Counter()
    by_proposer_total: Counter = Counter()

    for corpus, source in pairs:
        logger.info("processing %s / %s", corpus, source)
        res = decide_for_source(
            corpus=corpus, source=source,
            bank_run=bank_run, actions_run=actions_run,
            cfg=cfg,
        )
        _write_results_for_source(output_dir, res, cfg)
        per_pair_summaries.append({
            "corpus": res.corpus,
            "source": res.source,
            "status": "ok" if res.error is None else "error",
            "error": res.error,
            "n_skills_in": res.n_skills_in,
            "n_proposals_out": len(res.proposals),
            "by_kind": dict(res.by_kind),
            "by_proposer": dict(res.by_proposer),
            "elapsed_sec": round(res.elapsed_sec, 3),
        })
        total_props += len(res.proposals)
        total_skills += res.n_skills_in
        by_kind_total.update(res.by_kind)
        by_proposer_total.update(res.by_proposer)
        if res.error:
            logger.warning("  %s/%s -> %s", corpus, source, res.error)
        else:
            logger.info("  %s/%s -> %d skill(s) in, %d proposal(s) out (%s)",
                        corpus, source,
                        res.n_skills_in, len(res.proposals),
                        ", ".join(f"{k}={v}" for k, v in res.by_kind.most_common()) or "-")

    # Run-level metadata.
    (output_dir / "_run_meta.json").write_text(json.dumps({
        "bank_run":     str(bank_run),
        "actions_run":  str(actions_run) if actions_run else None,
        "output_root":  str(output_dir),
        "teacher_model": args.teacher_model,
        "thresholds":   cfg,
        "pairs":        [{"corpus": c, "source": s} for c, s in pairs],
        "started_at":   started_at,
        "argv":         [str(a) for a in (argv or sys.argv)],
        "no_actions":   args.no_actions,
    }, indent=2))

    (output_dir / "_run_summary.json").write_text(json.dumps({
        "bank_run":             str(bank_run),
        "actions_run":          str(actions_run) if actions_run else None,
        "output_root":          str(output_dir),
        "n_pairs":              len(pairs),
        "n_pairs_ok":           sum(1 for r in per_pair_summaries if r["status"] == "ok"),
        "n_skills_in":          total_skills,
        "n_proposals_out":      total_props,
        "by_kind":              dict(by_kind_total),
        "by_proposer":          dict(by_proposer_total),
        "per_pair":             per_pair_summaries,
        "started_at":           started_at,
        "completed_at":         _utcnow_iso(),
    }, indent=2))

    logger.info("DONE: %d pair(s), %d skill(s) in, %d proposal(s) out",
                len(pairs), total_skills, total_props)
    logger.info("  by kind     : %s",
                ", ".join(f"{k}={v}" for k, v in by_kind_total.most_common()) or "-")
    logger.info("  by proposer : %s",
                ", ".join(f"{k}={v}" for k, v in by_proposer_total.most_common()) or "-")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
