"""Shared helpers for ``dump_harness_io_gpt54.py``.

This module owns the *pure data-transformation* layer that turns the
on-disk cold-start corpus (`labeling/skill_bank_out`,
`labeling/skill_actions_out`, `labeling_supplement/crafter_proposals_out`,
`labeling_supplement/episode_reflections_out`) into the typed inputs the
live Harness expects:

  * `SkillRecord` (from `skill_bank.jsonl`)
  * `StateSchema` (from `metadata.schema_canonical` + per-step `state`)
  * `BankMutationProposal` (from either the rule-based or the
    reflect-per-episode proposals.jsonl shape)
  * `SkillEpisode` seeds (replay) — synthesised from
    `sub_episodes.json`
  * `RewardLogger` (shadow) — synthesised from skill_actions episodes
  * `(baseline, post)` scalars — read from `_skill_actions_summary.json`

It is deliberately import-side-effect free: the dump driver may import
selectively.

PLAN refs: PLAN-UNIFIED-SKILL-GATE §§3, 7; PLAN-HARNESS §§5, 6;
implementation_notes/legacy/crafter-harness-orchestrator-roles.md.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common.enums import (
    DOMAINS,
    SOURCE_DOMAINS,
    TRANSFER_TARGET_DOMAINS,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
)
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from skill_bank import SkillLifecycleManager

logger = logging.getLogger("labeling_supplement.dump_harness_io.helpers")


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

CORPORA: Tuple[str, ...] = ("gym_v", "env_wrappers")

# Cold-start banks use evidence_role ∈ {GATHER, VERIFY, REASON, COMMIT}.
# Map to live SkillType for adapter dispatch.
_ROLE_TO_SKILL_TYPE = {
    "GATHER": SkillType.GROUNDING,
    "VERIFY": SkillType.REASONING,
    "REASON": SkillType.REASONING,
    "COMMIT": SkillType.ACTION,
}


# ─────────────────────────────────────────────────────────────────────────
# 1. Bank loading
#    Mirror of `reflect_per_episode_gpt54._record_from_bank_entry` but
#    factored out so the dump driver can reuse without circular import.
# ─────────────────────────────────────────────────────────────────────────


def safe_skill_id(raw: str) -> str:
    """Cold-start labels use ``OPERATOR/SUBGOAL`` (e.g. ``COMMIT/MERGE``)
    which the flat-file ``SkillStore`` rejects. Map ``/`` → ``__``."""
    return (raw or "").replace("/", "__")


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


def record_from_bank_entry(entry: Dict[str, Any], default_domain: str) -> SkillRecord:
    """Hydrate a `SkillRecord` from one ``skill_bank.jsonl`` line.

    Task-axis seeding (harness/README §22): cold-start banks decorated by
    `labeling/_decorate_skill_records.py` carry `feasible_tasks` /
    `verified_tasks` directly. Older banks (decorator_version <= v1) only
    carry `provenance.source_name` (the env / game name), which we use
    as a fallback seed for `feasible_tasks=[source_name]`. Both shapes
    end up at the same `SkillRecord.feasible_tasks` after this call.

    Protocol shape (harness/README §21):

    * **v2 lifted bank** — `skill["protocol"]` is `List[Dict]` (typed
      hops). The decorator preserves the original prose body under
      `skill["protocol_raw"]: Dict` for diffing / re-lifting. Contract
      `preconditions / success_criteria / abort_criteria` are read from
      `protocol_raw` if present.
    * **v1 / pre-lift bank** — `skill["protocol"]` is `Dict` with prose
      `steps`, `preconditions`, etc. We `_wrap_protocol_steps` to get
      shape-only typed hops (every op is `"EXEC"`).
    * **dump-driver shape-lift output** — already `List[Dict]` but every
      hop is `{"action": "EXEC", "notes": "<prose>"}`. Pass through.

    All three load to a usable `SkillRecord`; the F2′ task-axis filter
    only depends on the new `feasible_tasks` field, not the protocol.
    """
    skill = entry.get("skill") or {}
    contract = skill.get("contract") or {}
    role = (skill.get("evidence_role") or "COMMIT").upper()
    skill_type = _ROLE_TO_SKILL_TYPE.get(role, SkillType.MIXED)

    feasible = list(skill.get("applicable_domains") or []) or [default_domain]

    # Prefer explicit `feasible_tasks` from the decorator; fall back to
    # `provenance.source_name` (always set by `_decorate_skill_records`).
    raw_feasible_tasks = skill.get("feasible_tasks")
    if isinstance(raw_feasible_tasks, list) and raw_feasible_tasks:
        feasible_tasks = [str(t) for t in raw_feasible_tasks if t]
    else:
        provenance = skill.get("provenance") or {}
        src_name = provenance.get("source_name") if isinstance(provenance, dict) else None
        feasible_tasks = [str(src_name)] if src_name else []
    raw_verified_tasks = skill.get("verified_tasks")
    verified_tasks = (
        [str(t) for t in raw_verified_tasks if t]
        if isinstance(raw_verified_tasks, list)
        else []
    )

    # Protocol body — accept all three on-disk shapes.
    proto_field = skill.get("protocol")
    proto_raw_field = skill.get("protocol_raw")
    proto_meta: Dict[str, Any] = {}
    if isinstance(proto_field, list):
        # Already typed (v2 lifted, or dump-driver shape-lifted).
        protocol_hops: List[Dict[str, Any]] = [dict(h) for h in proto_field if isinstance(h, dict)]
        if isinstance(proto_raw_field, dict):
            proto_meta = proto_raw_field
    elif isinstance(proto_field, dict):
        # Pre-lift prose dict — shape-lift via `_wrap_protocol_steps`.
        protocol_hops = _wrap_protocol_steps(proto_field.get("steps") or [])
        proto_meta = proto_field
    else:
        protocol_hops = []

    # Effect contracts: legacy game banks emit `eff_add` / `eff_del`;
    # cross-domain banks (skill_transfer_test/skill_bank_local/full_v5/)
    # emit `effects_add` / `effects_del`. Read both, preferring the
    # cross-domain spelling when present.
    eff_add_raw = contract.get("effects_add")
    if eff_add_raw is None:
        eff_add_raw = contract.get("eff_add")
    eff_del_raw = contract.get("effects_del")
    if eff_del_raw is None:
        eff_del_raw = contract.get("eff_del")

    sk = SkillRecord.new(
        name=skill.get("name", skill.get("skill_id", "_unknown")),
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=feasible,
        feasible_tasks=feasible_tasks,
        verified_tasks=verified_tasks,
        protocol=protocol_hops,
        contract=SkillContract(
            preconditions=list(proto_meta.get("preconditions") or []),
            effects_add=list(eff_add_raw or []),
            effects_del=list(eff_del_raw or []),
            expected_evidence_roles=[role] if role else [],
            success_criteria=list(proto_meta.get("success_criteria") or []),
            abort_criteria=list(proto_meta.get("abort_criteria") or []),
        ),
    )
    raw_id = skill.get("skill_id") or sk.skill_id
    object.__setattr__(sk, "skill_id", safe_skill_id(raw_id))
    return sk


def load_bank_records(bank_jsonl: Path, default_domain: str = "gymv") -> List[SkillRecord]:
    """Load every line of ``skill_bank.jsonl`` into typed `SkillRecord`s."""
    records: List[SkillRecord] = []
    if not bank_jsonl.exists():
        return records
    with bank_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                records.append(record_from_bank_entry(entry, default_domain))
            except Exception as exc:                                # noqa: BLE001
                logger.debug("skip bank entry: %s", exc)
    return records


def seed_lifecycle(
    lifecycle: SkillLifecycleManager,
    records: Iterable[SkillRecord],
    *,
    promote_to: SkillStatus = SkillStatus.CANDIDATE,
) -> Tuple[int, int]:
    """Seed `lifecycle`'s repository with the given records, transitioning
    each to ``promote_to``.

    The dump driver typically promotes to ``PROVISIONAL`` so the
    eligibility filter (which gates on ``status ∈ {ACTIVE, SHADOW,
    PROVISIONAL}``) actually returns them. ``PROVISIONAL`` is preferred
    over ``ACTIVE`` because it skips the ``≥2 feasible_domains``
    invariant (most cold-start skills declare a single domain).

    Returns ``(n_seeded, n_skipped)``.
    """
    n = 0
    skipped = 0
    for rec in records:
        try:
            lifecycle.ingest_draft(rec)
            lifecycle.transition(
                rec.skill_id,
                to_status=SkillStatus.CANDIDATE,
                rationale="dump-harness-io: seed-from-bank-snapshot",
            )
            if promote_to is SkillStatus.PROVISIONAL:
                lifecycle.transition(
                    rec.skill_id,
                    to_status=SkillStatus.PROVISIONAL,
                    rationale=(
                        "dump-harness-io: force-runnable for eligibility "
                        "exercise (single-domain cold-start skill)"
                    ),
                )
            n += 1
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("skip seed %s: %s", rec.skill_id, exc)
            skipped += 1
    return n, skipped


# ─────────────────────────────────────────────────────────────────────────
# 2. State parsing
#    The on-disk `metadata.schema_canonical` is a small DSL; we extract
#    the fields the harness adapter dispatch + eligibility filter
#    actually look at (domain, task, elements, facts).
#
#    Day-3 (harness/README §22): we additionally fold the `<attributes>`
#    and `<state_flags>` blocks into `state.facts` so the gymv
#    success_fn can evaluate per-hop effect predicates
#    (entity_value_increased, cumulative_reward_increased,
#    phase_transitioned, …) against pre/post state without re-parsing
#    the raw schema text. Hot-path keys:
#      facts["score"]              — score entity value (numeric if parseable)
#      facts["highest_tile"]       — highest-tile entity value
#      facts["lines_cleared"]      — lines-cleared entity value (tetris)
#      facts["phase"]              — <state_flags> phase (e.g. "gameover")
#      facts["progress"]           — <state_flags> progress (numeric ∈ [0,1])
#      facts["entity_attrs"]       — {label → {field → value}}
#      facts["entity_label_count"] — {label → count}  (entity_count_changed)
#      facts["goal"]               — preserved from pre-Day-3 contract
# ─────────────────────────────────────────────────────────────────────────


_DOMAIN_LINE = re.compile(r"^domain=(\S+)\s*$", re.M)
_TASK_LINE = re.compile(r"^task=(.*)$", re.M)
_GOAL_LINE = re.compile(r"^goal=(.*)$", re.M)
# Entity declarations: e0[type=tile, label=2 ...]
_ENTITY_LINE = re.compile(
    r"^([a-zA-Z_][\w]*)\[type=([\w\-]+)(?:,\s*([^\]]+))?\]\s*$", re.M
)
_SECTION_BLOCK = re.compile(
    r"<(?P<name>entities|attributes|affordances|relations|state_flags|"
    r"targets|uncertainty|actions|evidence|answer)>"
    r"(?P<body>.*?)(?=<\w+>|</state>|\Z)",
    re.DOTALL,
)
_ATTR_LINE = re.compile(r"^([a-zA-Z_]\w*)\.(\w+)\s*=\s*(.+?)\s*$", re.M)
_STATE_FLAG_LINE = re.compile(r"^(\w+)\s*=\s*(.+?)\s*$", re.M)


def _maybe_number(value: str) -> Any:
    """Parse a stringified attribute value into the cheapest numeric form.

    Used by the gymv success_fn so predicate evaluation can compare
    `score` / `highest_tile` numerically without each caller redoing
    the int/float dance. Returns the original string when neither int
    nor float parse succeeds.
    """
    s = (value or "").strip()
    if not s or s.lower() in {"null", "none"}:
        return None
    try:
        if "." not in s and "e" not in s.lower():
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _split_canonical_sections(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    haystack = text if "</state>" in text else text + "</state>"
    for m in _SECTION_BLOCK.finditer(haystack):
        out[m.group("name")] = m.group("body").strip()
    return out


def parse_schema_canonical(
    text: str, *, default_domain: str = "gymv"
) -> StateSchema:
    """Best-effort parser for the typed ``<state>`` block emitted by the
    actor.

    Always returns a `StateSchema` (never raises) — parse failures
    degrade to a minimal schema with what we could recover. Sufficient
    for the eligibility filter (which only needs `state.domain` and
    `state.elements` for slot-binding sanity checks) AND for the
    gymv success_fn (which reads `state.facts["score"]` /
    `state.facts["highest_tile"]` / `state.facts["entity_attrs"]` /
    `state.facts["entity_label_count"]` / `state.facts["phase"]`).
    """
    text = text or ""
    domain = default_domain
    if m := _DOMAIN_LINE.search(text):
        d = m.group(1).strip()
        if d in DOMAINS:
            domain = d

    task = ""
    if m := _TASK_LINE.search(text):
        task = m.group(1).strip()

    goal = ""
    if m := _GOAL_LINE.search(text):
        goal = m.group(1).strip()

    sections = _split_canonical_sections(text)

    # ── entities ────────────────────────────────────────────────────────
    elements: List[Dict[str, Any]] = []
    eid_to_index: Dict[str, int] = {}
    for m in _ENTITY_LINE.finditer(sections.get("entities", "") or text):
        eid, etype = m.group(1), m.group(2)
        if eid in eid_to_index:
            continue
        attrs_blob = (m.group(3) or "").strip()
        elem: Dict[str, Any] = {"id": eid, "type": etype}
        for kv in attrs_blob.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                elem[k.strip()] = v.strip()
        eid_to_index[eid] = len(elements)
        elements.append(elem)

    # ── attributes (e1.value=2 / e2.state=visible / …) ──────────────────
    # Decorate each element with its attribute key/value pairs and build
    # a label-keyed roll-up so the success_fn can read
    # `facts["entity_attrs"]["highest_tile"]["value"]` regardless of
    # which `e<N>` slot the schema happened to assign.
    entity_attrs: Dict[str, Dict[str, Any]] = {}
    for m in _ATTR_LINE.finditer(sections.get("attributes", "")):
        eid, field_name, value = m.group(1), m.group(2), m.group(3).strip()
        idx = eid_to_index.get(eid)
        if idx is None:
            continue
        elem = elements[idx]
        elem[field_name] = value
        label = elem.get("label")
        if label:
            entity_attrs.setdefault(label, {})[field_name] = _maybe_number(value)

    # ── label counts (entity_count_changed predicate) ───────────────────
    label_count: Dict[str, int] = {}
    for elem in elements:
        lbl = elem.get("label")
        if lbl:
            label_count[lbl] = label_count.get(lbl, 0) + 1

    # ── <state_flags> block (phase / progress / …) ──────────────────────
    state_flags: Dict[str, Any] = {}
    for m in _STATE_FLAG_LINE.finditer(sections.get("state_flags", "")):
        key, raw = m.group(1).strip(), m.group(2).strip()
        # Skip stray tag remnants that the regex picked up before the
        # next section opener (defensive — `<state_flags>\nphase=play`
        # is fine; we only ever land here on actual `key=value` rows).
        if key in {"state", "entities", "attributes", "affordances",
                   "relations", "targets", "uncertainty",
                   "actions", "evidence", "answer"}:
            continue
        state_flags[key] = _maybe_number(raw) if raw.lower() not in {
            "null", "none", ""
        } else None

    facts: Dict[str, Any] = {}
    if goal:
        facts["goal"] = goal
    if entity_attrs:
        facts["entity_attrs"] = entity_attrs
    if label_count:
        facts["entity_label_count"] = label_count
    # Hot-path scalars used by the gymv success_fn (kept as top-level
    # `facts` keys so callers don't have to re-traverse `entity_attrs`).
    for hot_label in ("score", "highest_tile", "lines_cleared",
                      "tetris_score", "moves_remaining"):
        rec = entity_attrs.get(hot_label)
        if rec is None:
            continue
        if "value" in rec:
            facts[hot_label] = rec["value"]
        elif "state" in rec:
            facts[hot_label] = rec["state"]
    if state_flags:
        # Promote the two flags the gymv predicates care about.
        if "phase" in state_flags and state_flags["phase"] is not None:
            facts["phase"] = state_flags["phase"]
        if "progress" in state_flags and state_flags["progress"] is not None:
            facts["progress"] = state_flags["progress"]
        facts["state_flags"] = state_flags

    return StateSchema(
        task=task or goal,
        domain=domain,
        elements=elements,
        facts=facts,
        extra={},
    )


def parse_step_state(
    step: Dict[str, Any], *, fallback_domain: str = "gymv"
) -> StateSchema:
    """Build a `StateSchema` for one rollout step.

    Prefers ``step.metadata.schema_canonical`` when present (typed),
    falls back to the natural-language ``step.state`` string otherwise.
    """
    md = (step.get("metadata") or {})
    canonical = md.get("schema_canonical") or ""
    if canonical:
        return parse_schema_canonical(canonical, default_domain=fallback_domain)

    raw_state = step.get("state") or ""
    if isinstance(raw_state, dict):
        raw_state = json.dumps(raw_state)
    return StateSchema(
        task="",
        domain=fallback_domain,
        elements=[],
        facts={"raw_state": str(raw_state)[:512]},
        extra={"parser": "fallback_no_schema_canonical"},
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. Proposal loading
#    Two on-disk shapes:
#      (a) reflect-per-episode → uses `proposal_to_json` (typed `type`).
#      (b) decide_skill_crafting → flat custom shape with `proposal_kind`.
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _ProposalLoadResult:
    proposals: List[BankMutationProposal]
    n_skipped: int
    by_kind: Dict[str, int]
    by_source_file: Dict[str, int]


def proposal_from_json(d: Dict[str, Any]) -> Optional[BankMutationProposal]:
    """Inverse of ``proposal_to_json`` that ALSO accepts the rule-based
    decide_skill_crafting record shape.

    Returns ``None`` if the record can't be classified (logged at
    DEBUG so the run summary tracks the skip).
    """
    t = d.get("type")
    if t in {
        "PatchProposal",
        "ComposeProposal",
        "GeneralizeProposal",
        "HypothesisProposal",
        "RetireProposal",
    }:
        return _typed_from_json(t, d)

    kind = d.get("proposal_kind")
    if kind:
        return _rule_from_json(str(kind), d)

    return None


def _contract_from_json(d: Optional[Dict[str, Any]]) -> Optional[SkillContract]:
    if not d:
        return None
    try:
        return SkillContract(
            preconditions=list(d.get("preconditions", [])),
            effects_add=list(d.get("effects_add", [])),
            effects_del=list(d.get("effects_del", [])),
            expected_evidence_roles=list(d.get("expected_evidence_roles", [])),
            success_criteria=list(d.get("success_criteria", [])),
            abort_criteria=list(d.get("abort_criteria", [])),
        )
    except Exception:                                              # noqa: BLE001
        return None


def _common_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        proposal_id=d.get("proposal_id") or "",
        rationale=d.get("rationale") or "",
        parent_skill_ids=[safe_skill_id(p) for p in d.get("parent_skill_ids") or []],
        seed_failure_ids=list(d.get("seed_failure_ids") or []),
        target_domains=list(d.get("target_domains") or []),
        teacher_model=d.get("teacher_model"),
        proposed_at=d.get("proposed_at"),
    )


def _typed_from_json(t: str, d: Dict[str, Any]) -> Optional[BankMutationProposal]:
    kw = _common_kwargs(d)
    if t == "PatchProposal":
        return PatchProposal(
            **kw,
            base_skill_id=safe_skill_id(d.get("base_skill_id") or ""),
            patched_protocol=list(d.get("patched_protocol") or []),
            patched_contract=_contract_from_json(d.get("patched_contract")),
            recovery_strategy=d.get("recovery_strategy") or "",
        )
    if t == "RetireProposal":
        return RetireProposal(
            **kw,
            target_skill_id=safe_skill_id(d.get("target_skill_id") or ""),
            reason=d.get("reason") or "",
        )
    if t == "ComposeProposal":
        return ComposeProposal(
            **kw,
            name=d.get("name") or "",
            component_skill_ids=[safe_skill_id(c) for c in d.get("component_skill_ids") or []],
            composed_protocol=list(d.get("composed_protocol") or []),
            contract=_contract_from_json(d.get("contract")) or SkillContract(),
        )
    if t == "GeneralizeProposal":
        return GeneralizeProposal(
            **kw,
            name=d.get("name") or "",
            base_skill_id=safe_skill_id(d.get("base_skill_id") or ""),
            abstracted_protocol=list(d.get("abstracted_protocol") or []),
            contract=_contract_from_json(d.get("contract")) or SkillContract(),
            source_domain=d.get("source_domain") or "",
            target_domain=d.get("target_domain") or "",
            slot_remap=dict(d.get("slot_remap") or {}),
            demo_selection=dict(d.get("demo_selection") or {}),
            demo_episode_ids=list(d.get("demo_episode_ids") or []),
            k_shot_budget=int(d.get("k_shot_budget") or 5),
        )
    if t == "HypothesisProposal":
        return HypothesisProposal(
            **kw,
            name=d.get("name") or "",
            novel_protocol=list(d.get("novel_protocol") or []),
            contract=_contract_from_json(d.get("contract")) or SkillContract(),
            source_failure_pattern_ids=list(d.get("source_failure_pattern_ids") or []),
        )
    return None


def _rule_from_json(kind: str, d: Dict[str, Any]) -> Optional[BankMutationProposal]:
    """Best-effort coercion of a rule-based crafter_proposals_out record
    into a typed `BankMutationProposal`.

    The rule-based shape has rich `replay_slice_ids`,
    `evidence_interface`, and `adapter_plan` payloads that the typed
    schema doesn't carry; we drop them here (they're still on disk if a
    downstream consumer wants them). What matters for the gate is the
    `(proposal_kind, target_skill_id, target_domains, source_type)`
    quadruple — which we faithfully reconstruct.

    Note on the rule-based shape's skill-id field naming: a "patch"
    record uses ``target_skill_id`` (the skill being patched), but a
    "transfer" record uses ``source_skill_id`` (the skill being
    generalized) because its ``target_skill_id`` slot is reserved for
    the *new* generalized skill (which doesn't exist yet). We accept
    either field as the bank-lookup key.
    """
    target = safe_skill_id(
        d.get("target_skill_id")
        or d.get("source_skill_id")
        or d.get("base_skill_id")
        or ""
    )
    base = dict(
        proposal_id=d.get("proposal_id") or "",
        rationale=d.get("rationale") or "",
        parent_skill_ids=[target] if target else [],
        seed_failure_ids=[],
        target_domains=list(d.get("target_domains") or []),
        teacher_model=None,
        proposed_at=None,
    )

    if kind == "patch":
        return PatchProposal(
            **base,
            base_skill_id=target,
            patched_protocol=[],
            patched_contract=None,
            recovery_strategy=d.get("patch_kind") or "patch",
        )
    if kind == "retire":
        return RetireProposal(
            **base,
            target_skill_id=target,
            reason=(d.get("reason") or d.get("rationale") or "rule-based-retire"),
        )
    if kind == "compose":
        comp_ids = [
            safe_skill_id(c) for c in (
                d.get("component_skill_ids")
                or d.get("components")
                or []
            )
        ]
        return ComposeProposal(
            **base,
            name=d.get("name") or target or "compose",
            component_skill_ids=comp_ids,
            composed_protocol=[],
            contract=SkillContract(),
        )
    if kind in ("generalize", "transfer"):
        # Pick a real transfer target: filter the proposal's
        # `target_domains` against the canonical TRANSFER_TARGET_DOMAINS
        # (excludes source-only domains like `gymv`). Fall back to
        # `browser` if nothing matches — the dump's gate.evaluate path
        # will then exercise a meaningful Stage 3a.
        candidate_targets = [
            d for d in (d.get("target_domains") or [])
            if d in TRANSFER_TARGET_DOMAINS
        ]
        # Honour `target_domain` if explicitly set AND legal.
        explicit_target = d.get("target_domain")
        if explicit_target and explicit_target in TRANSFER_TARGET_DOMAINS:
            chosen_target = explicit_target
        elif candidate_targets:
            chosen_target = candidate_targets[0]
        else:
            chosen_target = "browser"

        chosen_source = d.get("source_domain") or "gymv"
        if chosen_source not in SOURCE_DOMAINS:
            chosen_source = "gymv"

        return GeneralizeProposal(
            **base,
            name=d.get("name") or target or "transfer",
            base_skill_id=target,
            abstracted_protocol=[],
            contract=SkillContract(),
            source_domain=chosen_source,
            target_domain=chosen_target,
            slot_remap=dict(d.get("slot_remap") or {}),
            demo_selection=dict(d.get("demo_selection") or {}),
            demo_episode_ids=list(d.get("demo_episode_ids") or []),
            k_shot_budget=int(d.get("k_shot_budget") or 5),
        )
    if kind in ("hypothesis", "hypothesize", "net_new"):
        return HypothesisProposal(
            **base,
            name=d.get("name") or "hypothesis",
            novel_protocol=list(d.get("novel_protocol") or []),
            contract=SkillContract(),
            source_failure_pattern_ids=list(d.get("source_failure_pattern_ids") or []),
        )
    return None


def load_proposals(
    crafter_proposals_run: Optional[Path],
    reflections_run: Optional[Path],
    corpus: str,
    source: str,
    *,
    max_per_source: Optional[int] = None,
) -> _ProposalLoadResult:
    """Load proposals from BOTH the rule-based crafter_proposals_out and
    the reflect_per_episode episode_reflections_out trees.

    De-dupes by ``proposal_id``. Returns proposals in deterministic
    order (rule-based first, then reflect — file-sorted within each).
    """
    out: List[BankMutationProposal] = []
    seen: set = set()
    skipped = 0
    by_kind: Dict[str, int] = {}
    by_src: Dict[str, int] = {}

    def _ingest(d: Dict[str, Any], where: str) -> None:
        nonlocal skipped
        try:
            prop = proposal_from_json(d)
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("[%s] proposal coerce failed: %s", where, exc)
            prop = None
        if prop is None:
            skipped += 1
            return
        if prop.proposal_id and prop.proposal_id in seen:
            return
        if prop.proposal_id:
            seen.add(prop.proposal_id)
        out.append(prop)
        kc = type(prop).__name__
        by_kind[kc] = by_kind.get(kc, 0) + 1
        by_src[where] = by_src.get(where, 0) + 1

    if crafter_proposals_run is not None:
        rb = crafter_proposals_run / corpus / source / "proposals.jsonl"
        if rb.exists():
            with rb.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    _ingest(d, "crafter_proposals_out")
                    if max_per_source is not None and len(out) >= max_per_source:
                        break

    if reflections_run is not None:
        src_dir = reflections_run / corpus / source
        if src_dir.exists():
            for ep_dir in sorted(src_dir.iterdir()):
                if not ep_dir.is_dir() or ep_dir.name.startswith("_"):
                    continue
                pp = ep_dir / "proposals.jsonl"
                if not pp.exists():
                    continue
                if max_per_source is not None and len(out) >= max_per_source:
                    break
                with pp.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            skipped += 1
                            continue
                        _ingest(d, f"episode_reflections_out/{ep_dir.name}")
                        if max_per_source is not None and len(out) >= max_per_source:
                            break

    return _ProposalLoadResult(
        proposals=out,
        n_skipped=skipped,
        by_kind=by_kind,
        by_source_file=by_src,
    )


# ─────────────────────────────────────────────────────────────────────────
# 4. Replay seed synthesis
#    `sub_episodes.json` segments each rollout into per-skill spans;
#    we wrap each span as a minimal `SkillEpisode` keyed to the target
#    skill so `ReplayValidator.validate(...)` has something to dispatch.
# ─────────────────────────────────────────────────────────────────────────


def synthesize_replay_seeds(
    sub_episodes_json: Path,
    skill_id: str,
    domain: str,
    *,
    n_max: int = 4,
) -> List[SkillEpisode]:
    """Build `SkillEpisode` seeds for ``skill_id`` from
    ``sub_episodes.json``.

    Note: we synthesise the *outcome* but not the per-step adapter
    trace — that's exactly what `ReplayValidator` recomputes in
    ``dry_run=True`` mode (PLAN-UNIFIED-SKILL-GATE Stage 1).
    """
    if not sub_episodes_json.exists():
        return []
    try:
        data = json.loads(sub_episodes_json.read_text())
    except Exception:                                              # noqa: BLE001
        return []

    target_label = skill_id.replace("__", "/").upper()
    seeds: List[SkillEpisode] = []
    for sub in (data.get("sub_episodes") or []):
        sub_task = (sub.get("sub_task") or "").upper()
        if sub_task != target_label:
            continue
        ep = SkillEpisode.begin(
            skill_id=skill_id,
            skill_version="v1",
            skill_type=SkillType.MIXED,
            domain=domain,
            parent_run_id=str(sub_episodes_json.parent),
            initial_state=StateSchema(task="", domain=domain),
        )
        success = (sub.get("outcome") == "success")
        # `SkillEpisode.finalize` enforces G0 (evidence-driven) for
        # non-ACTION skills; we declare a synthetic GATHER warrant
        # rather than bypassing the invariant because gate Stage 1
        # checks the same invariant downstream.
        try:
            ep.finalize(
                outcome=SkillEpisodeOutcome(
                    success=success,
                    contract_satisfied=success,
                    evidence_role=["GATHER"] if success else [],
                    score=float(sub.get("cumulative_reward") or 0.0),
                    extra={
                        "synthetic": True,
                        "source": "sub_episodes.json",
                        "seg_start": sub.get("seg_start"),
                        "seg_end": sub.get("seg_end"),
                    },
                ),
            )
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("skip replay seed (G0 violation): %s", exc)
            continue
        seeds.append(ep)
        if len(seeds) >= n_max:
            break
    return seeds


# ─────────────────────────────────────────────────────────────────────────
# 5. Shadow log synthesis (for Stage 2)
#    Wraps a slice of skill_actions episodes as synthetic `SkillEpisode`s
#    in a fresh `RewardLogger`. Each step that has a bound skill becomes
#    one entry, success-flagged from the reward signal.
# ─────────────────────────────────────────────────────────────────────────


def synthesize_shadow_log(
    actions_dir: Path,
    *,
    max_episodes: int = 5,
    domain: str = "gymv",
):
    """Returns a fresh `harness.reward_logger.RewardLogger`."""
    from harness.reward_logger import RewardLogger

    log = RewardLogger()
    eps = sorted(actions_dir.glob("episode_*.json"))[:max_episodes]
    for ep_path in eps:
        try:
            data = json.loads(ep_path.read_text())
        except Exception:                                          # noqa: BLE001
            continue
        for idx, step in enumerate(data.get("experiences") or []):
            sk = step.get("skills") or {}
            sid = sk.get("skill_id")
            if not sid:
                continue
            r = step.get("reward")
            success = isinstance(r, (int, float)) and float(r) > 0
            ep = SkillEpisode.begin(
                skill_id=safe_skill_id(sid),
                skill_version="v1",
                skill_type=SkillType.ACTION,
                domain=domain,
                parent_run_id=str(actions_dir),
                initial_state=StateSchema(task="", domain=domain),
            )
            try:
                ep.finalize(
                    outcome=SkillEpisodeOutcome(
                        success=success,
                        contract_satisfied=success,
                        evidence_role=[],
                        score=float(r) if isinstance(r, (int, float)) else None,
                        extra={
                            "synthetic": True,
                            "source": "skill_actions.experiences",
                            "step_idx": idx,
                            "episode_file": ep_path.name,
                        },
                    ),
                )
            except Exception as exc:                               # noqa: BLE001
                logger.debug("skip shadow ep (G0): %s", exc)
                continue
            log.log_episode(ep)
    return log


# ─────────────────────────────────────────────────────────────────────────
# 6. Non-regression baseline + post score
# ─────────────────────────────────────────────────────────────────────────


def baseline_post_from_summary(
    summary_json: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Read `_skill_actions_summary.json` and return ``(baseline, post)``.

    Until a frozen pre-/post-promotion eval suite exists, we use
    ``mean_confidence_per_episode`` as a proxy: ``post = baseline``
    (null hypothesis ⇒ Stage 4 ``LIMITED_PASS`` with delta=0). This is
    a documented placeholder until §11 / §12 lands.
    """
    if not summary_json.exists():
        return None, None
    try:
        data = json.loads(summary_json.read_text())
        b = float(data.get("mean_confidence_per_episode") or 0.0)
        return b, b
    except Exception:                                              # noqa: BLE001
        return None, None


# ─────────────────────────────────────────────────────────────────────────
# 7. Online surface helpers — turn one rollout episode into a stream of
#    (step_state, intention, retrieved_skill_ids, actor_choice, reward).
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _OnlineStepInputs:
    step_idx: int
    state: StateSchema
    intention: str
    retrieved_skill_ids: List[str]
    selected_skill_id: Optional[str]
    actor_action: Any
    reward: Optional[float]
    raw_step_keys: List[str]


def parse_online_step(
    step: Dict[str, Any], *, fallback_domain: str = "gymv"
) -> _OnlineStepInputs:
    """Turn one ``experiences[i]`` record into the inputs the online
    Harness expects.

    Plumbed fields:
      * ``state``                   → ``StateSchema``
      * ``intentions``              → ``intention`` (raw NL)
      * ``skill_query.candidates``  → ``retrieved_skill_ids``
                                       (full record lookup happens upstream)
      * ``skill_query.selected_skill_id`` → ``selected_skill_id``
      * ``action``                  → ``actor_action``
      * ``reward``                  → ``reward``
    """
    state = parse_step_state(step, fallback_domain=fallback_domain)
    intention = str(step.get("intentions") or "")
    sq = step.get("skill_query") or {}
    cands = sq.get("candidates") or []
    retrieved = [safe_skill_id(c.get("skill_id") or "") for c in cands if isinstance(c, dict)]
    selected = sq.get("selected_skill_id")
    if selected:
        selected = safe_skill_id(selected)
    return _OnlineStepInputs(
        step_idx=int(step.get("idx") or 0),
        state=state,
        intention=intention,
        retrieved_skill_ids=[r for r in retrieved if r],
        selected_skill_id=selected,
        actor_action=step.get("action"),
        reward=step.get("reward"),
        raw_step_keys=sorted(step.keys()),
    )


# ─────────────────────────────────────────────────────────────────────────
# 8. Diagnostic: turn an EligibleSkill list + actor's selection into a
#    "harness-vs-actor agreement" record.
# ─────────────────────────────────────────────────────────────────────────


def diagnose_agreement(
    eligible_skill_ids: List[str],
    selected_skill_id: Optional[str],
) -> Dict[str, Any]:
    """Compute the {actor's pick} ∩ {harness eligible} relation."""
    if selected_skill_id is None:
        return {
            "actor_chose": None,
            "harness_eligible_count": len(eligible_skill_ids),
            "agreement": "no_actor_selection",
        }
    in_set = selected_skill_id in eligible_skill_ids
    return {
        "actor_chose": selected_skill_id,
        "harness_eligible_count": len(eligible_skill_ids),
        "agreement": "actor_pick_eligible" if in_set else "actor_pick_vetoed_or_unranked",
        "actor_pick_in_eligible_set": in_set,
    }


# ─────────────────────────────────────────────────────────────────────────
# 9. Synthesise a *target* SkillRecord for non-Patch / non-Retire proposals
#
# `Generalize` / `Compose` / `Hypothesis` proposals produce a *new* skill
# that isn't in the bank yet. The orchestrator's promotion path
# materialises a draft `SkillRecord` from the proposal *before* the gate
# sees it (PLAN-PIPELINE-ORCHESTRATOR §3). The dump driver mirrors that
# step here so Stage 0's `source_type` / `feasible_domains` /
# `expected_evidence_roles` invariants get to evaluate against the *right*
# skill — passing the source skill instead would produce false-positive
# `source_type mismatch` and `feasible_domains < 2` failures (the source
# was never claimed to be transferable).
# ─────────────────────────────────────────────────────────────────────────


def synthesize_target_skill_for_proposal(
    proposal: BankMutationProposal,
    *,
    source_skill: Optional[SkillRecord] = None,
    default_domain: str = "gymv",
) -> Tuple[Optional[SkillRecord], Dict[str, Any]]:
    """Build a draft `SkillRecord` matching ``proposal``'s implied target.

    Returns ``(skill_or_None, debug_payload)``. ``debug_payload`` records
    the synthesis decision so the dump artefact can flag it (e.g.
    ``target_skill_synthetic: true``).

    Only Generalize/Compose/Hypothesis are synthesised. Patch and
    Retire return ``source_skill`` unchanged because their target IS
    the source.
    """
    debug: Dict[str, Any] = {"synthesised": False, "kind": type(proposal).__name__}

    if isinstance(proposal, RetireProposal):
        # Retire mutates status, not body — pass source unchanged.
        debug["reason"] = "target_is_source_skill_retire"
        return source_skill, debug

    if isinstance(proposal, PatchProposal):
        if source_skill is None:
            debug["reason"] = "patch_no_source_skill_to_clone"
            return None, debug
        # The orchestrator would mint a NEW skill version with the patch
        # applied and `source_type=REPAIRED`. The dump replays this so
        # Stage 0's `proposal.source_type == skill.source_type` check
        # doesn't false-positive on the unpatched skill in the bank.
        protocol = (
            list(proposal.patched_protocol)
            if proposal.patched_protocol
            else list(source_skill.protocol)
        )
        contract = (
            proposal.patched_contract
            if (proposal.patched_contract and
                proposal.patched_contract.expected_evidence_roles)
            else source_skill.contract
        )
        sk = SkillRecord.new(
            name=f"{source_skill.name}__patched",
            skill_type=source_skill.skill_type,
            source_type=proposal.source_type,        # REPAIRED
            feasible_domains=list(source_skill.feasible_domains),
            protocol=protocol,
            contract=contract,
            proposal_id=proposal.proposal_id,
            parent_skill_ids=[source_skill.skill_id],
            source_domains=list(source_skill.source_domains),
            transfer_target_domains=list(source_skill.transfer_target_domains),
        )
        # In the live flow a patch is a *new version* of the same
        # skill_id (orchestrator/lifecycle bumps `version`, not the id).
        # Preserve the source id so shadow-log lookups
        # (`log.filter(skill_id=skill.skill_id)` in gate_service) hit
        # the rollout entries the actor keyed under the original id.
        object.__setattr__(sk, "skill_id", source_skill.skill_id)
        debug.update(
            synthesised=True,
            reason="patch_target_synthesised_preserve_skill_id",
            patch_protocol_applied=bool(proposal.patched_protocol),
            patch_contract_applied=bool(
                proposal.patched_contract and
                proposal.patched_contract.expected_evidence_roles
            ),
        )
        return sk, debug

    if isinstance(proposal, GeneralizeProposal):
        if source_skill is None:
            debug["reason"] = "no_source_skill_to_clone"
            return None, debug
        # Target = source body, but with the proposal's source_type,
        # source/transfer-target lineage, and an extended feasible_domains
        # that includes the new target domain (so Stage 0's ≥2 invariant
        # is satisfied — that's exactly what generalisation promises).
        feas = list(set(list(source_skill.feasible_domains)
                        + ([proposal.target_domain] if proposal.target_domain else [])))
        protocol = (
            list(proposal.abstracted_protocol)
            if proposal.abstracted_protocol
            else list(source_skill.protocol)
        )
        contract = (
            proposal.contract if proposal.contract.expected_evidence_roles
            else source_skill.contract
        )
        sk = SkillRecord.new(
            name=proposal.name or f"{source_skill.name}__transferred",
            skill_type=source_skill.skill_type,
            source_type=proposal.source_type,
            feasible_domains=feas,
            protocol=protocol,
            contract=contract,
            proposal_id=proposal.proposal_id,
            parent_skill_ids=[source_skill.skill_id],
            source_domains=[proposal.source_domain] if proposal.source_domain else [],
            transfer_target_domains=[proposal.target_domain] if proposal.target_domain else [],
        )
        debug.update(
            synthesised=True,
            reason="generalize_target_synthesised_from_source_plus_proposal",
            feasible_domains=feas,
        )
        return sk, debug

    if isinstance(proposal, ComposeProposal):
        # Compose: the new skill's body is `composed_protocol` (if non-empty)
        # plus any contract from the proposal. Without those, we can't
        # produce a meaningful target — flag as "compose_no_body".
        if not proposal.composed_protocol and not proposal.contract.expected_evidence_roles:
            debug["reason"] = "compose_no_body_in_rule_based_record"
            return None, debug
        sk = SkillRecord.new(
            name=proposal.name or "composed",
            skill_type=SkillType.MIXED,
            source_type=proposal.source_type,
            feasible_domains=[default_domain],
            protocol=list(proposal.composed_protocol or []),
            contract=proposal.contract,
            proposal_id=proposal.proposal_id,
            parent_skill_ids=list(proposal.component_skill_ids),
        )
        debug.update(synthesised=True, reason="compose_target_synthesised")
        return sk, debug

    if isinstance(proposal, HypothesisProposal):
        if not proposal.novel_protocol and not proposal.contract.expected_evidence_roles:
            debug["reason"] = "hypothesis_no_body"
            return None, debug
        sk = SkillRecord.new(
            name=proposal.name or "hypothesis",
            skill_type=SkillType.MIXED,
            source_type=proposal.source_type,
            feasible_domains=[default_domain],
            protocol=list(proposal.novel_protocol or []),
            contract=proposal.contract,
            proposal_id=proposal.proposal_id,
            parent_skill_ids=list(proposal.parent_skill_ids),
        )
        debug.update(synthesised=True, reason="hypothesis_target_synthesised")
        return sk, debug

    debug["reason"] = "unknown_proposal_type"
    return None, debug


__all__ = [
    "CORPORA",
    "safe_skill_id",
    "record_from_bank_entry",
    "load_bank_records",
    "seed_lifecycle",
    "parse_schema_canonical",
    "parse_step_state",
    "parse_online_step",
    "proposal_from_json",
    "load_proposals",
    "synthesize_replay_seeds",
    "synthesize_shadow_log",
    "baseline_post_from_summary",
    "diagnose_agreement",
    "synthesize_target_skill_for_proposal",
]
