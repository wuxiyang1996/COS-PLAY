# `crafter/` — Slow-timescale typed proposal layer

Spec: [`PLAN-SKILL-CRAFTER`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase C).

The crafter is the **creative** layer of the system: it proposes new skills (compositions, generalisations, novel hypotheses) and patches to existing ones (failure repairs, retirements). Every proposal is typed (`BankMutationProposal`) and is *only* a proposal — it lands in `draft_store/` and must pass the unified gate to be promoted (invariant 6 — crafter scope).

```python
from crafter import (
    SkillCrafterService,
    Composer, Generalizer, Hypothesizer,
    FailureDiagnoser,
    FailureMemory, FailurePattern,
)
```

---

## Architectural rules (mechanically enforced)

These are tested by `tests/test_invariants.py` and a static dependency rule in the package's `__init__.py`:

1. The crafter never imports `skill_bank.stores` directly.
2. The crafter never holds a `SkillLifecycleManager` reference.
3. The crafter writes proposals via `ArtifactStore.put_proposal` and ingests draft records via `SkillLifecycleManager.ingest_draft` **only** through `SkillCrafterService` — no other entry point exists.

The point: the crafter can dream up anything, but the lifecycle manager + the gate are the only path to runtime ACTIVE state.

---

## Module map

| File | Role |
|---|---|
| `failure_memory.py` | `FailureMemory` — append-only `FailureTrace[]` indexed by `(skill_id, failure_mode)`. Mines `FailurePattern`s (recurring (skill, mode) clusters with shared evidence signatures). The single source for "why is X failing in the wild?" |
| `failure_diagnoser.py` | `FailureDiagnoser` — given a `FailurePattern`, decides which `BankMutationProposal` family is appropriate. Outputs `FailureDiagnosis(family, reasoning, target_skills)`. The teacher (currently `gpt-4o`) lives here |
| `composer.py` | `Composer` — builds `ComposeProposal`s by chaining N existing ACTIVE skills whose evidence outputs match the next skill's evidence inputs. The chain itself becomes a typed inner-MDP protocol |
| `generalizer.py` | `Generalizer` — builds `GeneralizeProposal`s by lifting concrete slot bindings to slot variables. Often emitted when `FailureDiagnoser` flags `slot_binding_failed` patterns across multiple domains |
| `hypothesizer.py` | `Hypothesizer` — builds `HypothesisProposal`s for novel skills suggested by the teacher. Used when no compose / generalise route closes the failure pattern |
| `repairer.py` | `Repairer` — builds `PatchProposal`s for **existing** bank skills. Maps each `RecoveryStrategy` (HOP_INSERTION, PRECONDITION_STRENGTHENING, FALLBACK_INJECTION, REGROUNDING_TRIGGER, PROTOCOL_PATCH, SKILL_DECOMPOSITION) to a deterministic protocol/contract edit; teacher-LLM hook (`set_llm_repairer`) replaces the rule path when present |
| `service.py` | `SkillCrafterService` — the **only** public entry point. Two cadenced entry points: `reflect_on_episode(EpisodeReflection)` (per-episode reactive) and `cycle()` (per-batch reflective). Both pull failures from `FailureMemory`, run `FailureDiagnoser`, dispatch to the right proposer, persist via `ArtifactStore.put_proposal`, and ingest the resulting draft `SkillRecord`s via `SkillLifecycleManager.ingest_draft` |
| `_bank_view.py` | `BankView` — frozen, read-only multi-store snapshot (active ∪ candidate ∪ draft) built only by `SkillCrafterService._take_bank_view`. Exposes `subsumed_pairs(candidate_ids=...)` so the per-episode pass can detect when a freshly-minted candidate strictly covers an existing active skill (subsumption-retire path) |

---

## Proposal taxonomy

Every proposer emits one of five typed `BankMutationProposal` subclasses (defined in [`data_structure/extensions/`](../data_structure/extensions/)):

| Family | When emitted | Touches |
|---|---|---|
| `ComposeProposal` | Two or more ACTIVE skills repeatedly co-occur in successful episodes with matching evidence handoff | New DRAFT chain skill |
| `GeneralizeProposal` | A skill succeeds across multiple domains with isomorphic slot bindings | New DRAFT generalised skill, parent-id pointer |
| `HypothesisProposal` | A failure pattern has no compose / generalise route; teacher proposes a novel skill | New DRAFT skill, marked `source_type=HYPOTHESIS` |
| `PatchProposal` | A known failure pattern has a clear recovery; produces an adapter / contract patch | Same skill_id, new `content_hash` (gate revalidates) |
| `RetireProposal` | A skill's success rate drops below `GateThresholds.retire_floor` over the eval window | DEPRECATED status (gate-bound) |

The crafter never decides the *outcome* — it only proposes. The orchestrator's `GateService` evaluates, and `PromotionOrchestrator` (with `SkillLifecycleManager`) applies.

---

## Failure-driven loop (two-tier trigger)

The crafter is invoked at two cadences. See `implementation_notes/legacy/crafter-harness-orchestrator-roles.md` §"Two-tier trigger" for the rationale.

```
                    PER-EPISODE REACTIVE                       PER-BATCH REFLECTIVE
                    (every episode end)                        (every K episodes)

runtime episode ───┬──► FailureTrace[]  (this episode)         FailureTrace[]    (last K eps)
                   │           │                                       │
                   ▼           ▼                                       ▼
   bank-mgmt agent ──► new candidate skill_ids                FailureMemory.hot_patterns
              │              + bank_agent_actions                       │
              │                                                         │ (count ≥ hot_pattern_threshold)
              ▼                                                         ▼
     EpisodeReflection                                       repair > retire > hypothesise
              │                                                         │
              ▼                                                         │
  SkillCrafterService.reflect_on_episode                                │
   ├── ingest_failures(reflection.failure_traces)                       │
   ├── _run_failure_dispatch(min_count=1)  ◄────── shared dispatch ────►├── _run_failure_dispatch(min_count=hot_pattern_threshold)
   ├── _take_bank_view()                                                │
   └── subsumed_pairs(new_candidate_skill_ids) → RetireProposal[]       │
              │                                                         │
              ▼                                                         ▼
                       BankMutationProposal[]
                                │
                                ▼
                  SkillCrafterService persistence
                   ├── ArtifactStore.put_proposal
                   └── SkillLifecycleManager.ingest_draft
                                │
                                ▼
                          draft_store/  (DRAFT)
                                │
                                ▼ (offline gate)
                       GateService → PromotionOrchestrator
                                │
                                ▼
                       new ACTIVE skill enters runtime
```

Threshold split: per-episode pass uses `min_count=1` so a single failure or a single fresh candidate is enough to act on; per-batch pass keeps the original `hot_pattern_threshold` (default 3) so cross-episode noise has to repeat before the gate stack sees it. The dispatch chain itself (repair → retire → hypothesise) is identical on both paths — the only difference is which failure patterns reach it.

`Composer` and `Generalizer` belong on the per-batch path (they need multi-episode statistics); the per-episode path deliberately runs neither.

---

## Phase boundaries

| Phase | What this package contains | Status |
|---|---|---|
| C (MVP) | `FailureMemory`, `FailureDiagnoser`, `Composer`, `Generalizer`, `Hypothesizer`, `SkillCrafterService` | **Delivered** — covered by `tests/test_smoke.py` (failure → DRAFT cycle) |
| D | `Repairer` + `PatchProposal` repair plumbing exposed via `SkillCrafterService.propose_repair`; the failure-driven `cycle()` now dispatches **repair > retire > hypothesize** for failures whose `skill_id` resolves to an existing bank skill | **Delivered** — covered by `tests/test_crafter_repair.py::TestPhaseDRepair` |
| F | Frozen Qwen3-VL-32B / 235B-A22B teacher backbones registered in `common.models.QWEN3_VL_TEACHERS`; activate via `SkillCrafterService.with_qwen3_vl_teacher(...)`, `SkillCrafterService.set_teacher_model(...)`, or the `VLM_AGENT_PHASE_F_TEACHER` env switch read by `SkillCrafterService.from_env(...)` | **Wiring delivered** — covered by `tests/test_crafter_repair.py::TestPhaseFFrozenTeacher`. The project-wide `BACKBONE_TEACHER_MODEL` default is `Qwen/Qwen3.5-35B-A3B`; the Qwen3-VL Phase-F teachers are an opt-in upgrade path |

### Phase-D dispatch order (failure-driven `cycle()`)

```
hot FailurePattern
        │
        ▼
   FailureDiagnoser.diagnose(representative trace)
        │
        ├── pattern.skill_id ∈ bank? ──── yes ──┐
        │                                       ▼
        │                               strategy = SKILL_RETIREMENT?
        │                                       │
        │                                yes ───┼─── propose_retirement → RetireProposal
        │                                       │
        │                                no ────┴─── propose_repair → PatchProposal
        │
        └── unknown skill_id ──────────────── Hypothesizer.propose → HypothesisProposal
```

### Phase-F teacher swap

```python
from common.models import qwen3_vl_teacher
from crafter import SkillCrafterService

# 1) Construct with the frozen teacher.
crafter = SkillCrafterService.with_qwen3_vl_teacher(
    lifecycle=lifecycle, artifact_store=artifacts, size="32b",
)

# 2) Or swap mid-run.
crafter.set_teacher_model(qwen3_vl_teacher("235b-a22b"))

# 3) Or flip via env-var, no code edits.
#    VLM_AGENT_PHASE_F_TEACHER=qwen3-vl-32b
crafter = SkillCrafterService.from_env(lifecycle=lifecycle, artifact_store=artifacts)
```

---

## Teacher-LLM integration — state and roadmap

### Current state (the dormant teacher)

The teacher backbone has migrated to the project-wide control-plane model `BACKBONE_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"` (`common/models.py`).  It is stamped on every proposal as provenance metadata, but the LLM-call hooks are still dormant today (the rule path is the tested baseline; the hooks are an integration point):

- `FailureDiagnoser._llm`, `Hypothesizer._llm`, and `Repairer._llm` default to `None`.
- The `set_llm_diagnoser` / `set_llm_proposer` / `set_llm_repairer` setters are exercised only in `tests/test_crafter_repair.py` (lambda mocks).
- Every `cycle()` invocation runs through the deterministic rule path; on any LLM exception or `None` return the hooks silently fall through to the same rule path (see `failure_diagnoser.diagnose`, `hypothesizer.propose`, `repairer.repair`).

This is intentional — the rule path is the tested baseline (60 passing tests, Phases C/D/F green) and forms the safety net beneath any future teacher. The dormant hooks are an integration point, not a half-built feature.

### Capability assessment for a 32–35B-class frozen teacher

`Qwen3-VL-32B` is the registered Phase-F candidate (`common.models.qwen3_vl_teacher("32b")`). Per-hook complexity for a 32B-class VL teacher:

| Hook | Effective LLM task | Vocabulary scope | Single-pass 32B fit |
|---|---|---|---|
| `FailureDiagnoser` | 7-way `RecoveryStrategy` classification + free-text `root_cause` over a compact `FailureTrace` | 7 enum values | Strong — well below the model's ceiling |
| `Repairer` | Edit `protocol` / `SkillContract` per a strategy already chosen by the diagnoser | 6 actions × 4 evidence roles × 5 domains | Strong — templated code-edit task |
| `Composer` | Verify evidence handoff (skill A's outputs ↔ skill B's `expected_evidence_roles`) | Pairwise compatibility check | Strong |
| `Generalizer` | Pick a `slot_remap` over `_DOMAIN_TOKENS` (`bbox|dom_id|css_selector|xpath|cell|grid_xy|tile_id|frame_index`) | Constrained mapping | Strong with VL evidence; weaker text-only |
| `Hypothesizer` | Generate a novel 2–10-hop protocol + 8-field `SkillContract` for an unmatched failure pattern | Constrained generative synthesis | Adequate single-pass, **strong with Best-of-N=4** |

[`PLAN-SKILL-CRAFTER`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) §2 ("Multi-run reasoning requirement") explicitly assumes a 32B/72B teacher and budgets 3–6× tokens for proposal-then-verify, Best-of-N, and counterfactual passes. The unified gate ([`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) eats teacher noise: a noisier teacher means more proposals are rejected at Stage 0/1, not that bad skills land in `active_store/`.

### Prospective failure modes (only bite once a hook is live)

1. **Counterfactual synthesis** ([PLAN-SKILL-CRAFTER §6.9](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md)) — the Hypothesizer / Failure-Reflector chain. The most reasoning-heavy call; a 32B single-pass lags GPT-4o by ~15–25%, single-digit gap with the spec's Pass 1–4 scaffolding.
2. **Cross-domain analogy in `Generalizer`** — text-only 32B is the configuration that degrades meaningfully on `osworld` / `browser` / `video` evidence. Mitigation: use a VL variant (`Qwen3-VL-32B` is already registered).
3. **Strict typed JSON** — bare 32B emits ~5–15% malformed JSON against the typed `BankMutationProposal` schema; constrained decoding (vLLM + xgrammar / outlines) drops this to ~0.1%. Pure engineering, model-agnostic.

### Integration roadmap (gated by telemetry between steps)

| Step | What | Why this slot | Effort |
|---|---|---|---|
| **0 (prerequisite)** | Wire one concrete hook (start with `Repairer`) calling `API_func.ask_model` on GPT-4o through a new `crafter/_llm_runtime.py` | Failure modes 1–3 are moot until a hook actually runs. Smallest unit that activates the integration point and produces telemetry. | 1–2 days |
| **1 (now)** | Failure mode 3 — constrained-JSON decoding + retry + audit telemetry (`crafter.llm.calls`, `crafter.llm.parse_failures`, `crafter.llm.fallthrough_to_rule`, `crafter.llm.exceptions`, fed through `ArtifactStore.append_audit`) | Zero design risk, prerequisite for any teacher swap. Without it, JSON parse failures dominate the noise floor and mask everything else. | 3–5 days |
| **2 (after Step 1 telemetry)** | Failure mode 1 — multi-pass scaffolding for `Hypothesizer` + `FailureDiagnoser` per [PLAN-SKILL-CRAFTER §2](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md). Ship Step 1's hook as `_call_once(prompt)` so it's trivial to swap for `_call_with_passes([symptom, root_cause, counterfactual])` later. | Defer until telemetry shows single-pass quality is the bottleneck — `(LLM-proposed valid skill rate)` vs `(rule-path fallthrough rate)` vs `(gate Stage-0 rejection rate)`. | 1–2 weeks |
| **3 (parallel track)** | Failure mode 2 — flip teacher to `Qwen3-VL-32B` via `SkillCrafterService.with_qwen3_vl_teacher(...)` once `FailureTrace.pre_state` carries visual evidence from `osworld_wrapper`, `browser_adapter`, etc. | Switching the teacher is one line; making the *evidence* visual is a system-wide change cutting across `harness/`, `env_wrappers/`, and `orchestrator/artifact_store.py`. | Cross-cutting; tracked in [`PLAN-HARNESS`](../plans/05-harness/PLAN-HARNESS.md) / [`PLAN-FAILURE-ROUTING`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md) |

### Explicitly deferred

- `CounterfactualTrace` accumulation + regret-driven Hypothesizer source 4 ([`PLAN-SKILL-CRAFTER`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) §6.9). Requires both a working teacher *and* a measurable-regret policy; revisit after Step 2.
- Per-hook Best-of-N tuning beyond the spec's N=4 default. Add as a knob in `LLMHookConfig`; do not over-engineer the selection function before Step 1 telemetry exists.
- Phase-F frozen-teacher inference plumbing for the 235B-A22B variant. The constants are registered (`QWEN3_VL_TEACHERS["235b-a22b"]`); the serving stack lands later.

### Anti-pattern to avoid

Do not optimize the multi-pass / counterfactual / VL paths *speculatively* before Step 0 ships. The gate-bound architecture means the rule path is always a safe fallback, so the sequence "wire one hook → measure → improve the failing slot" dominates "design the perfect teacher up front." A noisy GPT-4o hook with telemetry beats a perfectly designed Qwen3-VL-32B hook that hasn't been measured against real failure traces.

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the crafter sits in the four-stage pipeline.
- [`../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) — full proposal-family spec.
- [`../skill_bank/README.md`](../skill_bank/README.md) — the lifecycle authority that the crafter writes through (never around).
- [`../data_structure/extensions/README.md`](../data_structure/extensions/README.md) — the typed proposal records.
