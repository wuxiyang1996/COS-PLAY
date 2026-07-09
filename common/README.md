# `common/` — Canonical types, IDs, schema, and backbone-model registry

Single import point for every cross-cutting type used by `harness/`, `orchestrator/`, `crafter/`, and `skill_bank/`. Every concrete definition lives in a sub-module so import graphs stay readable; this package only re-exports.

```python
from common import (
    # enums (state machines + categorical fields)
    SkillStatus, SkillType, SkillSourceType,
    GateStage, GateVerdict,
    InnerAction, RecoveryStrategy,
    DOMAINS, EVIDENCE_ROLES,
    # ID factories (UUID4 + 8-hex content hash)
    new_skill_id, new_episode_id, new_run_id,
    new_proposal_id, new_snapshot_id, new_span_id,
    schema_hash,
    # typed <state> schema (PLAN-SKILL-BANK §3)
    StateSchema, StateTargets, EvidenceRef,
    # backbone-model pins (root README §"Backbone models — three-tier stack")
    BACKBONE_MODEL,                # "Qwen/Qwen3.5-9B"        — actor + skill-bank
    BACKBONE_TEACHER_MODEL,        # "Qwen/Qwen3.5-35B-A3B"   — crafter / harness / orchestrator
    BACKBONE_JUDGE_MODEL,          # "Qwen/Qwen3.5-35B-A3B"   — eval-driver / skill-eval judge
                                   #   (same weights as TEACHER, different role)
    BACKBONE_SFT_TEACHER_MODEL,    # "gpt-5.5"                — SFT cold-start data (frontier)
    DEFERRED_MODELS, assert_default_backbone, is_deferred,
    # type aliases
    JSONDict,
)
```

---

## Module map

| File | What lives here |
|---|---|
| `enums.py` | All `Enum` types: `SkillStatus` (DRAFT → CANDIDATE → SHADOW → PROVISIONAL → ACTIVE → DEPRECATED → ROLLED_BACK), `SkillType`, `SkillSourceType`, `GateStage` (the 7 canonical stages 0–6), `GateVerdict` (PASS / LIMITED_PASS / FAIL), `InnerAction` (`GROUND` / `CHECK` / `RETRIEVE` / `COMMIT` / `EXECUTE`), `RecoveryStrategy`. Also the frozen sets `DOMAINS`, `SOURCE_DOMAINS = {"gymv"}`, `TRANSFER_TARGET_DOMAINS = {browser, osworld, video, visual_reasoning}`, and `EVIDENCE_ROLES = {GATHER, VERIFY, REASON, COMMIT}` |
| `ids.py` | UUID4-based ID factories with stable prefixes (`skill_…`, `ep_…`, `run_…`, `prop_…`, `snap_…`, `span_…`) and `schema_hash(record)` for content-hash-bound promotion (invariant 5) |
| `state_schema.py` | The typed `<state>` schema produced by `vlm_wrapper`'s grounding cascade and consumed by every adapter: `StateSchema(domain, task, goal, step, entities, attributes, relations, state_flags, targets, uncertainty, evidence, actions/answer)` plus the `EvidenceRef` value type |
| `models.py` | Backbone-model registry — single source of truth for every API call.  The current phase ships a **three-tier stack**: `BACKBONE_MODEL = "Qwen/Qwen3.5-9B"` (actor + skill-bank, LoRA-trained), `BACKBONE_TEACHER_MODEL = BACKBONE_JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"` (frozen control-plane backbone shared by crafter / harness / orchestrator AND the skill-evaluation / promotion-gate judge — same weights, two roles, served by a single `inference/serve_qwen35_35b_a3b.sh` instance to keep the GPU footprint flat), and `BACKBONE_SFT_TEACHER_MODEL = "gpt-5.5"` (SFT cold-start data teacher; stays on the frontier model because cold-start labels are baked once into the SFT adapters and never re-run during training). The deferred Qwen tracks (8B / 32B / 72B / VL-32B / VL-235B-A22B) appear in `DEFERRED_MODELS`; `is_deferred(name)` is the predicate Qwen-specific entrypoints check before booting. `assert_default_backbone()` is exercised by `tests/test_backbone_model.py` to fail loudly if a future edit silently changes the actor backbone. Override at process start with `VLM_AGENT_BACKBONE_MODEL` / `_TEACHER_MODEL` / `_JUDGE_MODEL` / `_SFT_TEACHER_MODEL` (e.g. `VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5` to swap the judge to an off-distribution oracle for paper / formal eval runs) |
| `typing.py` | Type aliases (`JSONDict = dict[str, Any]`, …) shared by serialisation paths |

---

## Invariants this package owns

1. **Single backbone source.** Every component reads `BACKBONE_MODEL` (actor / skill-bank), `BACKBONE_TEACHER_MODEL` (crafter / harness / orchestrator), `BACKBONE_JUDGE_MODEL` (validation), or `BACKBONE_SFT_TEACHER_MODEL` (cold-start data) from `common.models`; no module hardcodes a runtime default model name. Enforced by `tests/test_backbone_model.py`.
2. **No ad-hoc IDs.** Components must use `common.ids` factories so artefact provenance and content-hash drift detection (invariant 5 in the root README) keep working.
3. **One `<state>` schema, repo-wide.** `StateSchema` is imported by `vlm_wrapper`, `harness`, `crafter`, and `orchestrator`. Adding a field requires touching `state_schema.py` only — every consumer is automatically forward-compatible because `to_dict()` round-trips unknown keys.

---

## Cross-references

- Root [`readme.md`](../readme.md) — "Backbone model" and "Mechanically-enforced invariants".
- [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](../plans/03-skill-bank/PLAN-SKILL-BANK.md) §3 — `<state>` schema spec.
- [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §3 — `SkillStatus` / `GateStage` / `GateVerdict` state machine.
- [`vlm_wrapper/schema.py`](../vlm_wrapper/schema.py) — the parser-side counterpart that produces `<state>` blocks for `StateSchema.from_state_block(...)`.
