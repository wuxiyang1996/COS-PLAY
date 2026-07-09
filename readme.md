# Multi-hop Reasoning VLM Agent

**A skill-centric, evidence-driven, gate-bound visual agent.** This repository builds a Visual Language Model (VLM) agent that converts pixels into a structured `<state>` schema and acts on it through a four-stage pipeline of *Visual Grounding → Action → Skill Bank → Skill Crafter*, governed by three operational components (*Skill Harness*, *Pipeline Orchestrator*, *Unified Skill Gate*).

The system learns **transferable reasoning, grounding, and control skills as general protocols feasible across game, webagent, os-agent, video-understanding, and visual reasoning tasks**. The first concrete arena is **short-video evidence-grounded reasoning** (Video-Holmes-style); cross-domain generalization is a hard, mechanically-enforced invariant of the skill bank, not an aspiration.

This repo supersedes the COS-PLAY codebase that lives alongside it under `decision_agents/`, `skill_agents/`, `vlm_wrapper/`, and `data_structure/legacy/`. Those modules remain importable as a reference for the legacy single-domain GRPO loop; the new build under `common/`, `harness/`, `orchestrator/`, `crafter/`, `skill_bank/`, and `data_structure/extensions/` implements the canonical plan from [`plans/`](plans/README.md).

---

## Table of Contents

- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Mechanically-enforced invariants](#mechanically-enforced-invariants)
- [Skill transfer layer](#skill-transfer-layer)
- [Trainer integration — co-evolution loop wires the harness](#trainer-integration--co-evolution-loop-wires-the-harness)
- [Backbone models — three-tier stack](#backbone-models--three-tier-stack)
- [Cold-start data generation — lean plan + `reasoning_effort` policy](#cold-start-data-generation--lean-plan--reasoning_effort-policy)
- [Running experiments — instrumentation, ablations, cross-domain eval, analysis](#running-experiments--instrumentation-ablations-cross-domain-eval-analysis)
- [Repository layout](#repository-layout)
- [Implementation status](#implementation-status)
- [Quick start](#quick-start)
- [Plans index](#plans-index)
- [Legacy COS-PLAY notes](#legacy-cos-play-notes)
- [Citation and license](#citation-and-license)

---

## Why this project

Modern VLMs reason well on a single image but break down on multi-hop visual tasks where evidence must be gathered, verified, chained, and committed across hops. Existing skill-based agents either (a) bind their skills to one domain (game-only or browser-only) or (b) treat skills as opaque function calls with no evidence interface. Both choices block transfer and hide failures.

**Core thesis — games as the skill foundry, other domains as few-shot transfer targets.** Games are uniquely well-suited to *learn* multi-hop reasoning, grounding, and control skills: they expose dense, cheap, verifiable rewards; deterministic resets; and rich multi-hop structure (gather → verify → chain → commit) over a controllable visual state. This project uses games as the **source domain** in which skills are discovered, gated, and hardened, and then **transfers those same skills to webagent, os-agent, short-video understanding, and visual reasoning via few-shot adaptation** — a handful of target-domain episodes are enough to bind a game-learned protocol to a new adapter, because the skill is a typed protocol over evidence, not a domain-specific policy.

This project takes the opposite stance to opaque, single-domain skill agents:

1. **Every skill is a general protocol, learned in games and few-shot adapted elsewhere.** A skill must declare adapter bindings to all five domains (game, webagent, os-agent, video, visual reasoning); the *game* binding is where it is first discovered and stress-tested, and the remaining bindings are earned via few-shot transfer trials at the gate. Single-domain skills are rejected at promotion time. See [Skill Bank §0.1](plans/03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families).
2. **Every skill is evidence-driven** — it must declare a role from `{GATHER, VERIFY, REASON, COMMIT}` and record a non-empty evidence interface on every successful episode. Opaque skills are rejected at Gate G0. The evidence interface is what makes few-shot transfer tractable: only the adapter bindings change across domains, while the typed evidence contract stays fixed. See [Skill Bank §0.3](plans/03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills).
3. **Every promotion is gate-bound, with a dedicated transfer gate.** No proposal reaches `ACTIVE` without passing the canonical gate stack (`static → replay → shadow → transfer → non-regression`). The `transfer` stage is the few-shot adaptation check: a game-learned skill must succeed on a small budget of episodes in at least one non-game domain before its `verified_domains` entry is granted. See [Unified Skill Gate](plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md).
4. **The Actor is the policy, the Harness is a frozen verifier.** The Skill Bank provides candidates, the Harness narrows + may veto, the Actor decides, the Orchestrator handles offline promotion. The frozen large model never silently becomes the policy. See [Pipeline Orchestrator §0a](plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a-actor-harness-skill-bank-orchestrator-boundary).

The first benchmark target outside the game foundry is short-video multi-hop reasoning, but the skill ontology is fixed across phases — short-video is the first **transfer arena** in which game-learned general protocols (e.g. `collect_evidence_chain`, `disambiguate_target`, `locate_filter_select`, `actor_action_binding`, `verify_constraint`) earn their `verified_domains` entry through few-shot adaptation. Webagent, os-agent, and visual reasoning follow as additional transfer arenas under the same gate.

---

## Architecture

### Pipeline (canonical four stages)

```
Pixels (game frame / screenshot / video / image)
    ↓
(1) Visual Grounding   — VLM parser → structured <state> schema
    ↓
(2) Action Agent       — two-level MDP: inner reasoning hops → environment actions
    ↓                       ↑
(3) Skill Bank         — segmentation → contracts → cross-domain retrieval
    ↓                       ↑
(4) Skill Crafter      — compose / generalize / hypothesize / repair
    ↓
    └──→ proposals enter the Unified Gate → bank as DRAFT → CANDIDATE → ACTIVE
```

### Operational components

| Component | Role | Module |
| --- | --- | --- |
| **Skill Harness** | Per-invocation runtime: eligibility filter → adapter run → tracing → `SkillEpisode` → replay validation | [`harness/`](harness/) |
| **Pipeline Orchestrator** | System control plane: `EpisodeRunner`, `ArtifactStore`, `BudgetController`, `GateService`, `PromotionOrchestrator`, `SnapshotManager` | [`orchestrator/`](orchestrator/) |
| **Unified Skill Gate** | Canonical `SkillStatus` / `SkillSourceType` / `SkillRecord` / `GateVerdict` lifecycle, split-storage (`draft / candidate / active / archive`), `SkillLifecycleManager` as the *only* writer | [`skill_bank/`](skill_bank/) |
| **Skill Crafter** | Slow-timescale typed proposal layer (composition, generalization, hypothesis, repair, retire) — outputs are *candidates only* | [`crafter/`](crafter/) |

### Two-level MDP

The agent runs an **outer** environment loop with a **lightweight typed inner loop** of inner actions `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`, capped at 0–3 hops. Skills capture *how to think* (typed hop chains) — not just *what to do*. Heavy reasoning (failure diagnosis, composition, transfer, hypothesis generation) is strictly offline. See [Action Agent §5](plans/02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control).

### Three-agent role split

| Agent | Default model (current phase) | Role | Update timescale |
| --- | --- | --- | --- |
| **Actor / Decision Agent** | `gpt-4o` | Online action execution, hop selection, skill selection, protocol following | Fast |
| **Skill-Use / Operational** | `gpt-4o` | Skill retrieval, segmentation, contract learning, curation | Medium |
| **Synthesis-Reflection (Teacher)** | `gpt-4o` (will swap to frozen 32B/72B later) | Failure reflection, composition, hypothesis, transfer, judging | Slow, gated |

See [Action Agent §2](plans/02-action-agent/PLAN-ACTION-AGENT.md#three-agent-role-split) for the canonical specification.

---

## Mechanically-enforced invariants

The plan calls out six invariants that must hold across *every* phase. Each is enforced by code, not by reviewer discipline, and is exercised by `tests/test_invariants.py`.

| # | Invariant | Enforcement point |
| --- | --- | --- |
| 1 | **G0 — evidence-driven**: every successful non-`ACTION` skill records a non-empty `evidence_in` / `evidence_out` / `evidence_warrant`. | `SkillEpisode.finalize` raises on empty evidence; `SkillLifecycleManager` rejects ACTIVE promotion when `expected_evidence_roles` is empty. |
| 2 | **No-memory**: no cross-episode storage layer. | `SkillEpisodeStep.__post_init__` rejects any action type starting with `QUERY_MEM` / `WRITE_MEM`. |
| 3 | **General-protocol**: every active skill must be feasible in ≥ 2 domains. | `SkillLifecycleManager` rejects ACTIVE promotion when `feasible_domains < 2`; `GateService.Stage 0` flags it during evaluation. |
| 4 | **Bank-write isolation**: only `SkillLifecycleManager` may mutate any skill store. | `SkillStore.put` / `remove` raise `StoreLockedError` unless called via the lifecycle manager. |
| 5 | **Gate-bound promotion**: no skill reaches `ACTIVE` without a passing `GateVerdictPayload` and stable content hash. | `PromotionOrchestrator.promote` rejects `FAIL` verdicts and content-hash drift; refuses ACTIVE on `LIMITED_PASS`. |
| 6 | **Crafter scope**: the crafter materialises only `DRAFT` records and never touches `active_store`. | `SkillCrafterService._persist` calls `SkillLifecycleManager.ingest_draft`; static dependency rule keeps `crafter/` from importing `skill_bank/stores`. |
| 7 | **Source-domain / transfer-target asymmetry**: every `ACTIVE` skill must declare at least one `source_domains` entry from `SOURCE_DOMAINS` (currently `{"gymv"}`) **and** at least one `verified_domains` entry from `TRANSFER_TARGET_DOMAINS` (`{browser, osworld, video, visual_reasoning}`). Game-only or transfer-only skills cannot be active. | `SkillLifecycleManager._validate_invariants` raises on ACTIVE promotion when either side is empty (additive to invariant 3, back-compat for legacy records). |
| 8 | **`verified_domains` is gate-owned**: the `verified_domains` field (and the matching `adapter_history` entries) reflect Stage 3a outcomes only. The pipeline is: `GateService._run_transfer` runs `FewShotAdapter.adapt()` on K demos, the verified targets are emitted in `GateVerdictPayload.eligible_domains`, and `PromotionOrchestrator.promote(...)` mirrors them into the `SkillRecord` via `SkillLifecycleManager.record_transfer_verification(...)` *before* the status transition. The lifecycle manager is the only writer; any other component is forbidden. | `SkillLifecycleManager.record_transfer_verification` is the sole mutator of `verified_domains` / `adapter_history`; `PromotionOrchestrator._record_transfer_verifications` calls it from the pre-transition step. |

---

## Skill transfer layer

The "games as foundry, other domains as transfer targets" thesis from [§Why this project](#why-this-project) is implemented end-to-end as a stack of seven concrete components. None of them is a placeholder — they all run today as part of the gate stack and the test suite (`tests/test_few_shot_transfer.py`).

### Source / target asymmetry — the substrate

Two hard-coded tuples in [`common/enums.py`](common/enums.py) encode the asymmetry. They are imported everywhere a domain string is interpreted, so "is this a foundry domain?" / "is this a transfer arena?" is one constant lookup, never a free string:

```30:36:common/enums.py
SOURCE_DOMAINS: Tuple[str, ...] = ("gymv",)
TRANSFER_TARGET_DOMAINS: Tuple[str, ...] = (
    "browser",
    "osworld",
    "video",
    "visual_reasoning",
)
```

A skill that only ever ran in `gymv` carries `source_domains={"gymv"}` and an empty `verified_domains`; it cannot reach `ACTIVE` (invariants 7 & 8). A skill that earned a transfer-gate pass on, say, `video` gets `"video"` appended to `verified_domains` *and* a `TransferAdapterEvent` appended to `adapter_history` — both writes go through `SkillLifecycleManager.record_transfer_verification(...)` and nowhere else.

### Generalizer — proposing transferable templates

[`crafter/generalizer.py`](crafter/generalizer.py) consumes per-domain skills and emits `GeneralizeProposal` records that strip away domain-specific predicates, replacing them with the canonical inner-MDP vocabulary (`GROUND / CHECK / RETRIEVE / COMMIT / EXECUTE`) and shared schema slots. The output is always a `DRAFT` record with `source_domains={"gymv"}` and `verified_domains=()` — the gate stack is the only path that can extend either field.

The bridging types `TransferableSkill`, `SlotBinding`, `ReasoningProtocol`, `HopStep`, `AbstractPredicate`, and the `FAMILY_PROTOCOLS` registry live in [`skill_agents/skill_template.py`](skill_agents/skill_template.py); the discovery pipeline that produces them from legacy episodes is [`skill_agents/extract_transferable.py`](skill_agents/extract_transferable.py). See [`skill_agents/README.md`](skill_agents/README.md) for the legacy-bridge details.

### Few-Shot Adapter — the transferability probe

[`harness/few_shot_adapter.py`](harness/few_shot_adapter.py) is the single place where "does this skill bind to a new domain?" is decided. It is stateless: given a `SkillRecord`, a target domain, and a sequence of `FewShotDemo`s, it

1. Validates eligibility (`SkillType` ∈ adapter's supported types, source-domain check),
2. Coerces each demo's `state` into a target-domain `StateSchema`,
3. Runs the skill via `SkillHarness.run_skill(...)` against the registered adapter for the target domain,
4. Scores each shot with a pluggable `success_fn` (default: episode outcome + contract satisfaction),
5. Returns an `AdaptResult{success_rate, n_success, n_total, episodes, …}`.

Critically it never mutates the bank — it is a read-only probe used by the gate stack and by ad-hoc transferability tests.

### Stub adapters — common scaffolding for the four target domains

[`harness/adapters/_stub_base.py`](harness/adapters/_stub_base.py) defines `StubTransferTargetAdapter`, the shared hop-loop scaffolding for `browser`, `osworld`, `video`, and `visual_reasoning`. Each concrete adapter (e.g. [`harness/adapters/browser.py`](harness/adapters/browser.py)) is just a name + a default `HopExecutor`. The executor is pluggable (`adapter.set_executor(real_executor)`), so today's deterministic stubs let the gate stack be exercised end-to-end while the real `vlm_wrapper/<domain>_adapter.py` executors are wired up independently.

### Gate stage 3a — wiring the probe into promotion

[`orchestrator/gate_service.py`](orchestrator/gate_service.py) composes the canonical gate stack `static → replay → shadow → transfer → non-regression`. Stage 3a (`_run_transfer`) is the only stage that calls `FewShotAdapter.adapt(...)`:

```219:316:orchestrator/gate_service.py
def _run_transfer(
    self,
    skill: SkillRecord,
    *,
    few_shot_demos: Optional[Mapping[str, Sequence[FewShotDemo]]] = None,
) -> tuple[StageVerdict, List[str]]:
    # ... infer targets from TRANSFER_TARGET_DOMAINS ∩ adapter registry ...
    adapt_results: List[AdaptResult] = []
    for tgt in targets:
        shots = (few_shot_demos or {}).get(tgt, ())
        adapt_results.append(
            self._few_shot.adapt(skill=skill, target_domain=tgt, demos=shots)
        )
    # ...
    if len(verified) >= thresholds.transfer_min_target_domains_verified:
        verdict = GateVerdict.PASS
    elif len(verified) >= 1:
        verdict = GateVerdict.LIMITED_PASS
    else:
        verdict = GateVerdict.FAIL
```

The verified target-domain list is emitted as `GateVerdictPayload.eligible_domains` and passed back to the orchestrator. The verdict is `PASS` if ≥ `transfer_min_target_domains_verified` targets succeed, `LIMITED_PASS` if at least one does, otherwise `FAIL`.

### Promotion Orchestrator — mirroring results into the bank

[`orchestrator/promotion_orchestrator.py`](orchestrator/promotion_orchestrator.py) extracts the Stage-3a verdict from the `SkillEvaluationRecord`, converts it into per-target metrics, and calls the lifecycle manager **before** the status transition:

```208:254:orchestrator/promotion_orchestrator.py
self._lifecycle.record_transfer_verification(
    skill.skill_id,
    verified_targets=verified,
    evaluation_id=evaluation.evaluation_id,
    per_target_metrics=per_target_metrics,
    rationale=f"stage_3a:{rationale}" if rationale else "stage_3a",
)
```

This is the only call site outside the lifecycle manager itself that touches the transfer-verification path.

### Lifecycle Manager — the only writer + the ACTIVE-promotion invariant

[`skill_bank/lifecycle.py`](skill_bank/lifecycle.py) is the sanctioned writer. `record_transfer_verification(...)` extends `verified_domains` and appends a `TransferAdapterEvent` to `adapter_history` atomically; `_validate_invariants(...)` then refuses any `ACTIVE` promotion of a source-tagged skill that has not yet earned at least one verified target domain:

```272:283:skill_bank/lifecycle.py
if to_status == SkillStatus.ACTIVE and record.source_domains:
    if not any(d in SOURCE_DOMAINS for d in record.source_domains):
        raise LifecycleError(
            f"Cannot promote {record.skill_id!r} to ACTIVE: "
            f"source-domain (game-foundry) lineage required, …"
        )
    if not any(d in TRANSFER_TARGET_DOMAINS for d in record.verified_domains):
        raise LifecycleError(
            f"Cannot promote {record.skill_id!r} to ACTIVE: "
            f"few-shot transfer gate (G3a) requires ≥1 verified target "
            f"domain, …"
        )
```

This is the mechanical realisation of invariants 7 and 8: source-domain lineage and gate-owned `verified_domains` are not conventions — they are preconditions on the only function that can mark a skill `ACTIVE`.

### TL;DR call graph

```
crafter/generalizer.py        → DRAFT SkillRecord (source_domains={"gymv"}, verified_domains=())
        ↓
orchestrator/gate_service._run_transfer
        ↓ for each target ∈ TRANSFER_TARGET_DOMAINS ∩ AdapterRegistry
harness/few_shot_adapter.adapt
        ↓ runs skill via SkillHarness.run_skill against StubTransferTargetAdapter
        ↓ scores K demos → AdaptResult
        ↓
orchestrator/gate_service       → GateVerdictPayload(eligible_domains=verified[, …])
        ↓
orchestrator/promotion_orchestrator._record_transfer_verifications
        ↓
skill_bank/lifecycle.record_transfer_verification
        ↓ extends verified_domains + appends adapter_history
skill_bank/lifecycle._validate_invariants
        ↓ allows ACTIVE iff source ∩ SOURCE_DOMAINS ≠ ∅ and verified ∩ TRANSFER_TARGET_DOMAINS ≠ ∅
ACTIVE SkillRecord (transferable, evidence-driven, gate-bound)
```

End-to-end coverage lives in [`tests/test_few_shot_transfer.py`](tests/test_few_shot_transfer.py): it exercises the `SkillRecord` persistence path, the lifecycle invariants, the `FewShotAdapter` execution loop, and the `GateService` verdict shape in a single fixture set.

### What's still pending (transfer-side)

| Item | Owning module |
| --- | --- |
| Two-phase **shadow → active** transfer protocol on top of Stage 3a | `harness/transfer_manager.py` |
| Unified six-gate runner (`G0–G5`) consuming `gate_service` stages | `harness/gate_runner.py` |
| Real (non-stub) executors for each transfer-target adapter | `vlm_wrapper/<domain>_adapter.py` |
| Held-out replay seeds for Stage 1 | `harness/replay_validator.py` |

These are tracked under Phase D in [`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md) and in the [Pending](#pending-next-sessions) row above.

---

## Trainer integration — co-evolution loop wires the harness

The runtime in [`orchestrator/runner.py`](orchestrator/runner.py) is one of two
consumers of the harness; the other is the **co-evolution training loop** in
[`trainer/coevolution/orchestrator.py`](trainer/coevolution/orchestrator.py).
The trainer already runs the **Crafter** + **Promotion Orchestrator** in a
Phase B′ that fires once per training step (after rollout collection, before
GRPO). The Day-10 wire-up additionally plugs the harness's two **LLM-free,
deterministic** surfaces — `select_eligible_skills` and
`validate_invocation` — into Phase A's live rollout, and feeds the resulting
rejection signal back into the existing Crafter pipeline.

### Per-step phase map

| Phase | What runs | Harness role |
|---|---|---|
| **A** — Rollout collection ([`rollout_collector.py`](trainer/coevolution/rollout_collector.py) → [`episode_runner.run_episode_async`](trainer/coevolution/episode_runner.py)) | Per env step: cold-start RAG → `skill_selection` LoRA → `action_taking` LoRA → `env.step()` | **Pre-LLM eligibility filter + post-LLM `validate_invocation` veto** (both LLM-free; opt-in via `--harness-enabled`). |
| **B** — Legacy 4-stage skill mining | `sb_manager.finalize_all()` writes per-game `skill_bank.jsonl` | n/a (orthogonal). |
| **B′** — Crafter + Promotion ([`_crafter_hook.py`](trainer/coevolution/_crafter_hook.py) + [`_promotion_hook.py`](trainer/coevolution/_promotion_hook.py)) | Seeds ephemeral `SkillRepository` from each `skill_bank.jsonl`, synthesizes `FailureTrace`, calls `SkillCrafterService.reflect_on_episode`, subprocesses `decide_promotion_gpt54.py`, writes promoted skills back via `skill_bank.legacy_writeback` | **Drains the per-game `RejectedSkillSink`** → `SkillLifecycleManager.record_false_binding_pattern` so the Repairer sees live veto evidence on `SkillRecord.false_binding_patterns` (PLAN-SKILL-BANK §4.3b). |
| **C** — GRPO | `run_grpo_training` over rollout records + skill-bank GRPO data | n/a. |

### How the harness rides Phase A

```
Phase A (live rollout)
  ┌─────────────────────────────────────────────────────────────────┐
  │  episode_runner.run_episode_async(... harness_hook=hook)        │
  │     │                                                           │
  │     ├─ get_top_k_skill_candidates(...)              (RAG)       │
  │     │           │                                               │
  │     ├─ hook.filter_candidates(records, state)                   │
  │     │     └── EligibilityFilter.filter_with_rejections          │
  │     │             ├── admitted ─▶ skill_selection LoRA picks    │
  │     │             └── rejected ─▶ RejectedSkillSink.observe()   │
  │     │                                                           │
  │     ├─ hook.validate_choice(skill_id, state)                    │
  │     │     └── SkillHarness.validate_invocation                  │
  │     │             ├── ok=True   ─▶ proceed                      │
  │     │             └── ok=False  ─▶ fall back to next eligible   │
  │     │                                                           │
  │     └─ env.step(...)                                            │
  └─────────────────────────────────────────────────────────────────┘
                  │
                  ▼
Phase B′ (Crafter)
  ┌─────────────────────────────────────────────────────────────────┐
  │  _crafter_hook.run_crafter_step(... harness_hooks=hooks)        │
  │     │                                                           │
  │     ├─ _seed_repo_from_legacy_jsonl  →  ephemeral lifecycle     │
  │     │                                                           │
  │     ├─ harness_hook.flush_to_lifecycle(lifecycle)               │
  │     │     └── RejectedSkillSink.flush_to                        │
  │     │             └── lifecycle.record_false_binding_pattern    │
  │     │                     ↳ writes SkillRecord.false_binding_   │
  │     │                       patterns                            │
  │     │                                                           │
  │     ├─ SkillCrafterService.reflect_on_episode (existing)        │
  │     │     └── Repairer reads false_binding_patterns ──▶ emits   │
  │     │         PatchProposal records that previously had no      │
  │     │         live signal to fire on                            │
  │     │                                                           │
  │     └─ writes proposals.jsonl ─▶ Phase B′ Promotion subprocess  │
  └─────────────────────────────────────────────────────────────────┘
```

### CLI surface

```bash
python scripts/run_coevolution.py \
    --crafter-promotion-enabled \
    --harness-enabled \
    [--no-harness-allow-shadow]
```

`--harness-enabled` is **off** by default, so existing runs are byte-identical.
`--no-harness-allow-shadow` forces the eligibility filter to refuse SHADOW
skills (matches `HarnessConfig.allow_shadow=False`).

### Cost — zero added LLM calls

Both `EligibilityFilter` and `validate_invocation` are deterministic CPU
paths (microseconds per step). The trainer's existing **3 LLM calls per env
step** (intention + skill-selection + action) are unchanged.

### What the wire-up does *not* do (deliberate)

1. **No `harness.run_skill(...)` from the trainer.** The episode runner still
   drives the env directly via primitive actions through the `action_taking`
   LoRA. Plumbing `run_skill` requires a real `gymv` `set_executor(env_step_fn)`
   plus an `EnvLike` shim per env wrapper — multi-day env-binding work tracked
   in [`harness/README.md`](harness/README.md) §16.1–§16.5.
2. **No status persistence.** Skills hydrated from the live `skill_bank.jsonl`
   are mounted as a *runtime view* with `status=PROVISIONAL` (so the F1 status
   check admits them); the `SkillLifecycleManager` remains the only authority
   that may write status to disk (PLAN-SKILL-BANK §0.5). Only
   `false_binding_patterns` flows back, via the lifecycle's existing write
   surface inside the Crafter hook.
3. **No new training signal for GRPO.** Phase C consumes the same rollout
   records + skill-bank GRPO data as before; the eligibility veto only
   reshapes which skills the `skill_selection` LoRA sees as candidates.

### Spec gaps this closes

| Gap (see [`harness/README.md`](harness/README.md)) | Trainer status |
| --- | --- |
| §9.1 second-pass `validate_invocation` | **closed** in trainer's live rollout (per-step) |
| §9.3 per-check booleans on `EligibleSkill` | **closed** (logged into `experiences[].harness.filter[].rejected[]`) |
| §22 task-axis F2′ | **closed** for trainer Phase A (`state.task = game name`) |
| PLAN-SKILL-BANK §4.3b `false_binding_patterns` from live signal | **closed** via `RejectedSkillSink → record_false_binding_pattern` in Phase B′ |

### Module map for the wire-up

| File | Role |
|---|---|
| [`trainer/coevolution/_harness_hook.py`](trainer/coevolution/_harness_hook.py) | `SkillHarnessHook` — per-game façade exposing `filter_candidates`, `validate_choice`, `flush_to_lifecycle`. Hydrates `skill_bank.jsonl` → `SkillRecord` cache via [`_record_from_bank_entry`](trainer/coevolution/_crafter_hook.py). |
| [`trainer/coevolution/episode_runner.py`](trainer/coevolution/episode_runner.py) | Calls the hook before / after the `skill_selection` LLM; logs `experiences[].harness = {filter, validate}`. |
| [`trainer/coevolution/rollout_collector.py`](trainer/coevolution/rollout_collector.py) | Threads the per-game `harness_hooks` dict through to each episode. |
| [`trainer/coevolution/orchestrator.py`](trainer/coevolution/orchestrator.py) | Builds one hook per game per step (gated by `config.harness_enabled`) and passes the same dict into the Phase B′ Crafter hook. |
| [`trainer/coevolution/_crafter_hook.py`](trainer/coevolution/_crafter_hook.py) | After seeding, calls `hook.flush_to_lifecycle(lifecycle)` so the Repairer's `false_binding_patterns` signal is non-empty. |
| [`trainer/coevolution/config.py`](trainer/coevolution/config.py) | Adds `harness_enabled: bool = False` and `harness_allow_shadow: bool = True`. |
| [`scripts/run_coevolution.py`](scripts/run_coevolution.py) | `--harness-enabled` and `--no-harness-allow-shadow` CLI flags. |
| [`tests/test_trainer_harness_hook.py`](tests/test_trainer_harness_hook.py) | 21 unit tests: filter admit/veto by status / domain / task, sink → lifecycle drainage, bank hydration, graceful degradation, factory, stats. |

For the full topology diagram + closed/open-gap table, see
[`harness/README.md`](harness/README.md) §22.

---

## Backbone models — three-tier stack

The single source of truth is [`common/models.py`](common/models.py).  The
project ships a **three-tier backbone stack**, one model per tier:

```python
from common.models import (
    BACKBONE_MODEL,               # "Qwen/Qwen3.5-9B"        — actor + skill-bank (trained)
    BACKBONE_TEACHER_MODEL,       # "Qwen/Qwen3.5-35B-A3B"   — crafter / harness / orchestrator
    BACKBONE_JUDGE_MODEL,         # "Qwen/Qwen3.5-35B-A3B"   — eval driver / skill-eval judge
                                  #                            (same weights as TEACHER, different role)
    BACKBONE_SFT_TEACHER_MODEL,   # "gpt-5.5"                — SFT cold-start data generation only
)
```

| Tier | Model | Used by | Trained? |
|---|---|---|---|
| Actor + Skill-Bank | `Qwen/Qwen3.5-9B` | `decision_agents/`, `skill_agents/`, `trainer/` | LoRA-trained (5 adapters: `skill_selection`, `action_taking`, `segment`, `contract`, `curator`) |
| Control plane + LLM-as-judge | `Qwen/Qwen3.5-35B-A3B` (35B-total / 3B-active MoE) | `crafter/`, `harness/`, `orchestrator/`, `skill_agents/skill_evaluation/`, `orchestrator.JudgeConfig` | Frozen — served via [`inference/serve_qwen35_35b_a3b.sh`](inference/serve_qwen35_35b_a3b.sh). One vLLM instance services both the control-plane teacher role and the eval-driver judge role; per-model dispatch is wired by `API_func._candidate_vllm_urls` + `VLLM_BASE_URL_MAP`. |
| SFT cold-start teacher | `gpt-5.5` | `cold_start/`, `labeling/` | External frontier model — kept on the frontier because cold-start labels are baked once into SFT adapters and never re-run during training |

The **Qwen3-VL Phase-F teachers** (`Qwen/Qwen3-VL-32B`, `Qwen/Qwen3-VL-235B-A22B`) and the older 8B / 32B / 72B Qwen tracks remain reachable through dedicated entrypoints — `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`, `inference/run_academic_benchmarks.py`, `skill_agents/lora/`, and `SkillCrafterService.with_qwen3_vl_teacher(...)` — but no library default points at them. Override at process start with one of:

```bash
export VLM_AGENT_BACKBONE_MODEL=...                # actor / skill-bank policy
export VLM_AGENT_BACKBONE_TEACHER_MODEL=...        # crafter / harness / orchestrator
export VLM_AGENT_BACKBONE_JUDGE_MODEL=...          # eval-driver judge
export VLM_AGENT_BACKBONE_SFT_TEACHER_MODEL=...    # SFT cold-start data
```

Test coverage for the three-tier pin lives in [`tests/test_backbone_model.py`](tests/test_backbone_model.py).

---

## Cold-start data generation — lean plan + `reasoning_effort` policy

The SFT cold-start corpus is built by `cold_start/` (`gpt-5.5` teacher → trajectories
fine-tuning `Qwen/Qwen3.5-9B`). Two design choices drive both cost and correctness;
the full per-benchmark sizing tables live in [`cold_start/readme.md`](cold_start/readme.md#multi-domain-cold-start-lean-plan).

### Asymmetric volume — source vs. transfer targets

The cold-start runner respects the source-/transfer-target asymmetry from
[`common/enums.py`](common/enums.py): `gymv` is the **foundry** (volume should be
heaviest), while `browser` / `osworld` / `video` / `visual_reasoning` are
**transfer probes** consumed at gate Stage 3a (`harness/few_shot_adapter.py`,
`k_shot_default=5`, `k_shot_max=16` per skill per domain). Volumes are therefore
sized for *diverse pool* coverage, not full-benchmark sweeps:

| Domain (role) | Bucket | Total in benchmark | Pool (this run) | Holdout (frozen for E0/E1/E2 eval) | Sampler |
|---|---|---:|---:|---:|---|
| `gymv` (**source**) | 13 retro envs (Temporal/Airstriker, Columns, …) | 13 envs | ~130 ep (~10 ep / env × 20 steps) | n/a | `run_coldstart_actor_gymv_all.sh --episodes 10` |
| `browser` | AssistantBench (open web, no infra) | 215 tasks | **180 stratified** | + 30 | `cold_start/task_samples/build_browsergym_diverse_200.py` |
| `browser` | MiniWoB++ (atomic primitives) | 125 tasks | **125 (full)** | + 25 | same |
| `browser` | WebArena | 812 tasks | *deferred* — overlaps AssistantBench coverage at much higher infra cost | — | — |
| `browser` | WebShop (princeton-nlp) | 12,087 goals | **50 sampled** — `browsergym/webshop.0..49`, configurable via `WEBSHOP_NUM_GOALS` | — | `webshop_wrapper.register_webshop_tasks` (in-tree bridge; see [`webshop_wrapper/README.md`](webshop_wrapper/README.md)) |
| `browser` | ~~VisualWebArena~~ | 910 tasks | *dropped 2026-05-03* — see `legacy/visualwebarena/README.md` for the 10 infra bugs that motivated the cut | — | — |
| `osworld` | OSWorld desktop tasks | 369 tasks | **250 stratified** | + 50 | `cold_start/evaluation_dataset/build_pool_and_holdout.py` |
| `visual_reasoning` | VisualToolBench (image, single-turn) | 603 samples | **300 stratified** | + 100 | same |
| `visual_reasoning` | TIR-Bench (image, tool-use) | 1,215 samples | **300 stratified** | + 100 | same |
| `video` (**headline**) | Video-Holmes | 1,837 questions | **1,000** | + 200 | same |
| `video` | SIV-Bench | 8,728 questions | **400 stratified** | + 100 | same |

The pool/holdout split is critical: few-shot demos at Stage 3a must be
**disjoint** from the eval slice or the E0 scoreboard is contaminated. The
samplers in `cold_start/task_samples/build_*.py` (BrowserGym row) and
`cold_start/evaluation_dataset/build_pool_and_holdout.py` (rows 4–8 above)
emit both files in one pass.

#### How IDs are chosen and stored — `cold_start/evaluation_dataset/`

For each transfer-target benchmark we ship a deterministic
(seed = 0) **stratified** sample organized into two sibling subdirs so
the pool and the eval holdout never cross-contaminate at run time:

```text
cold_start/evaluation_dataset/
├── build_pool_and_holdout.py        # one-shot regenerator (seed=0, byte-stable)
├── load_manifests.py                # Python API + integrity check (consumers go here)
├── manifest.json                    # locked snapshot: per-file SHA-256, sizes, build provenance
├── _axis_distribution.json          # measured per-axis counts (diagnostic)
├── pool/                            # 2,250 ids — used by cold-start actor
│   ├── osworld.txt                  # 250 UUIDs
│   ├── osworld_catalog.json         # OSWorld --task_catalog format
│   ├── visual_toolbench.txt         # 300 HF row ids (single-turn only)
│   ├── tir_bench.txt                # 300 HF row ids
│   ├── video_holmes.txt             # 1000 "{video_id}.Q{qid}"
│   └── siv_bench.txt                # 400 "{video_id}.Q{tsv_row_index}"
└── holdout/                         # 550 ids — frozen for E0/E1/E2 eval
    ├── osworld.txt                  # 50 UUIDs
    ├── osworld_catalog.json         # same format as pool/
    ├── visual_toolbench.txt         # 100
    ├── tir_bench.txt                # 100
    ├── video_holmes.txt             # 200
    └── siv_bench.txt                # 100
```

One sample id per line; comment header records `count`, `seed`, and the
per-axis distribution. Re-run `python
cold_start/evaluation_dataset/build_pool_and_holdout.py` (or pass
`--<bench>_pool / --<bench>_holdout N` to override) to regenerate all
12 manifest files in one pass.

Sampling axis, ID format, and exact bucket distribution per file
(measured, seed = 0):

| Benchmark | Sizes (pool / holdout) | Stratification axis | Pool axis distribution | Holdout axis distribution | ID format |
|---|---:|---|---|---|---|
| `osworld` | 250 / 50 | `(snapshot × possibility_of_env_change)`; report per `snapshot` (11 normalized apps) | base_setup=9, chrome=36, gimp=27, libreoffice_calc=36, libreoffice_impress=34, libreoffice_writer=26, multi_apps=11, os=26, thunderbird=17, vlc=3, vs_code=25 | base_setup=2, chrome=7, gimp=5, libreoffice_calc=7, libreoffice_impress=8, libreoffice_writer=5, multi_apps=2, os=5, thunderbird=3, vlc=1, vs_code=5 | task UUID (matches `OSWorld/evaluation_examples/examples/<app>/<uuid>.json`) |
| `visual_toolbench` | 300 / 100 (single-turn only, 603 of 1,204 raw rows) | `prompt_category` (9 STEM / business categories) | biology=18, chemistry=16, engineering=18, finance=55, generalist=53, maths=18, medical=53, physics=16, sports=53 | biology=6, chemistry=6, engineering=6, finance=16, generalist=18, maths=6, medical=18, physics=6, sports=18 | HF row id (`row['id']`) |
| `tir_bench` | 300 / 100 | `task` (13 tool-use families) | each of {color, contrast, instrument, jigsaw, math, maze, ocr, refcoco, rotation_game, spot_difference, symbolic, visual_search, word_search} ≈ 22–24 | each family ≈ 7–8 | HF row id (`row['id']`) |
| `video_holmes` | 1000 / 200 | `Question Type` (7 reasoning skills: CTI / IMC / MHR / PAR / SR / TA / TCI) | each ≈ 142–144 | each ≈ 27–29 | `"{video_id}.Q{question_id}"` |
| `siv_bench` | 400 / 100 | `category` (10 social-intelligence dimensions) | each of {Action Recognition, Attitude Inference, Counterfactual Prediction, Emotion Inference, Environment Perception, Facial Expression Recognition, Factual Prediction, Human Attribute Identification, Intent Inference, Relation Inference} = 40 | each category = 10 | `"{video_id}.Q{tsv_row_index}"` |

Diversity guarantees:

1. **Every category that appears in the pool also appears in the holdout**
   (proportional within-bucket split at ratio `pool / (pool + holdout)`,
   tie-broken so any bucket with ≥ 2 sampled items contributes to both
   halves). For OSWorld this means all 11 normalized snapshots and all
   3 `possibility_of_env_change` tiers (low / medium / high) reach the
   eval slice; vlc-only contributes 1 holdout item because the bucket
   itself has only 4 sampled tasks total.
2. **Round-robin equal-bucket sampling** over-samples rare categories
   relative to natural frequency — exactly what we want for a few-shot
   probe that aims to characterize per-category transfer rather than
   estimate population accuracy.
3. **Disjoint by construction** — `pool ∩ holdout = ∅` is asserted at
   build time for every benchmark; `_axis_distribution.json` records
   the per-axis counts as a checked-in diagnostic.
4. **Seed = 0**, deterministic across machines and re-runs (per-benchmark
   sub-seeding so adding/removing one benchmark does not reshuffle the
   others).

Wire-up at run time:

```bash
# BrowserGym: pool manifests already in cold_start/task_samples/
python cold_start/generate_cold_start_actor_browsergym.py \
  --tasks_file cold_start/task_samples/browsergym_assistantbench_200.txt \
  --reasoning_effort minimal ...

# OSWorld: --task_catalog reads the JSON catalog directly
python cold_start/generate_cold_start_actor_osworld.py \
  --task_catalog cold_start/evaluation_dataset/pool/osworld_catalog.json \
  --reasoning_effort minimal ...

# Visual reasoning (image + video): point --sample_ids_dir at the pool
# subdir; the launcher autoglobs <benchmark>.txt for each enabled bench.
python cold_start/generate_cold_start_actor_visual_reasoning.py \
  --benchmarks visual_toolbench tir_bench video_holmes siv_bench \
  --sample_ids_dir cold_start/evaluation_dataset/pool \
  --reasoning_effort medium ...
```

The `holdout/` mirror is consumed only by the gate / few-shot adapter
(`harness/few_shot_adapter.py` at gate Stage 3a, plus the E0/E1/E2
benches) and **never** loaded during cold-start data generation —
keeping the eval scoreboard honest.

**Reusing the IDs from Python.** Every consumer in the project (gate,
few-shot adapter, baselines, eval harnesses) should read these ID lists
through `cold_start/evaluation_dataset/load_manifests.py` rather than
re-parsing the text files inline:

```python
from cold_start.evaluation_dataset.load_manifests import (
    load_ids, load_osworld_catalog, verify_integrity, load_manifest,
)
pool_ids = load_ids("video_holmes", split="pool")     # list[str], 1000 ids
held_ids = load_ids("osworld",       split="holdout") #            50 ids
catalog  = load_osworld_catalog("pool")               # {domain: [uuid, ...]}
verify_integrity()                                    # raises on hash drift
load_manifest()                                       # full provenance dict
```

`manifest.json` records seed, build timestamp, Python version, per-file
SHA-256, and per-benchmark size. `verify_integrity()` re-hashes every
file and fails loudly if any consumer (or a stray `sed`) silently edited
a manifest. The build script is byte-stable: re-running it on the same
dataset versions reproduces the same bytes (verified at build time).

### Per-pipeline runtime settings

Per-step token budgets are calibrated for `gpt-5.x` reasoning models (which
charge hidden thinking against the same `max_completion_tokens` cap as the
visible response); non-reasoning fallbacks use the smaller `_SCHEMA_MAX_TOKENS`
budget. Defaults below apply unless overridden via CLI flag.

| Pipeline | Episodes / task | Max steps / episode | Schema cap (non-reasoning / reasoning) | Action cap | Vision input | Parallelism unit |
|---|---:|---:|---|---:|---|---|
| `gymv` (`generate_cold_start_actor_gymv.py`) | 1 | 60 (cap; natural end usually earlier) | 4 k / 12 k | 128 | rendered frame | one process per env (13×) |
| BrowserGym (`generate_cold_start_actor_browsergym.py`) | 1 | 30 (default) — covers MiniWoB / WebArena / AssistantBench | 4 k / 12 k | 400 | screenshot + AXTree | 4–8 headless Chromium / Playwright |
| OSWorld (`generate_cold_start_actor_osworld.py`) | 1 | 50 (cap) — recommend 30 for cold-start | 4 k / 12 k | 500 | screenshot + AT-SPI tree | 1–8 KVM guests (dominant wall-clock lever) |
| Visual reasoning (`generate_cold_start_actor_visual_reasoning.py`) | 1 sample / call (no env) | 1 | 4 k / 12 k | 350 | image OR 6 sampled frames per video | 16+ pure API workers |

Other invariants of the actor pipeline:

- **Headless by default** for BrowserGym (Xvfb-backed Chromium) and OSWorld (KVM guest); pass `--no_headless` to render visibly when debugging.
- **Frames** are NOT saved by default; pass `--save_frames` to persist the PNGs sent to the VLM under `<run>/<task>/frames/ep_NNN/step_NNN.png`.
- **API keys** are auto-loaded from `<workspace>/api_keys.py` on import (no `export` needed).
- **Self-hosted site env files** (`webarena_env.sh`) are auto-sourced by the BrowserGym launcher when the relevant tasks are in `--tasks`. (`visualwebarena_env.sh` was dropped 2026-05-03 — see `legacy/visualwebarena/README.md`.)
- **`gpt-5.x` detection** is regex-based (`_is_reasoning_model`); matches route to `max_completion_tokens` automatically and accept `--reasoning_effort`.
- **Resume** is on by default (skip episodes that already have an `episode_NNN.json` on disk); pass `--no_resume` (or omit `--resume` on launchers that opt-in) to overwrite.

### `reasoning_effort` policy

A `--reasoning_effort` CLI flag (lives on `cold_start/generate_cold_start_actor_*.py`,
threaded through to the OpenAI client; one of `{minimal, low, medium, high}`)
governs the teacher's hidden-thinking budget. Without it, OpenAI defaults
to `medium` for `gpt-5.x` reasoning models, silently billing 1–4 k thinking
tokens per call. **For cold-start data generation that's pure waste** —
`Qwen/Qwen3.5-9B` only learns from the visible `<state>` and action JSON;
hidden tokens never reach the trained policy.

| Pipeline | Recommended effort | Rationale |
|---|---|---|
| `gymv` source-domain trajectories | `minimal` | Structured extraction; student can't use thinking |
| BrowserGym / OSWorld trajectories | `minimal` | Schema + constrained action; the schema *is* the planning surface |
| Visual reasoning MCQ (image + video) | `medium` | Teacher answer correctness is the bottleneck on multi-hop QA |

A paired smoke test (`gpt-5.4`, 5 mid-difficulty MiniWoB tasks,
`Cold-start-out-smoke-effort/`) confirmed this empirically: `minimal`
hit **5/5 task success** vs. `medium`'s 4/5, with **2.1× lower wall-clock
per step** (14.8 s vs. 31.5 s). On `guess-number`, `medium` chose a
linear-search policy (0, 1, 5, 6) and exhausted the step budget while
`minimal` chose binary search (5, 7, 8, 9) and won — a concrete case of
more hidden thinking *worsening* policy on a structured-action task.
Full numbers + reproduction recipe in
[`cold_start/readme.md#smoke-test-calibration`](cold_start/readme.md#smoke-test-calibration-gpt-54-n--5-paired-miniwob-tasks).

Cost & wall-clock impact for one full cold-start pass on the lean plan above
(GPT-5 reasoning class pricing, $1.25 / M input, $10 / M output). All four
launchers now accept `--reasoning_effort {minimal,low,medium,high}` (added
in this revision); the column "API spend" reflects the cheapest **safe**
policy for each row:

| Setting | API spend | Wall-clock @ realistic per-bucket parallelism |
|---|---:|---:|
| Original full sweep, default `medium` everywhere | ~$1,500 – $1,800 | ~12 – 15 h |
| Lean plan, `minimal` for env / `medium` for visual reasoning | ~$260 – $280 | ~3 – 6 h |
| **Lean plan, `minimal` everywhere** ← cheapest safe default | **~$200 – $220** | **~3 – 5 h** (set by OSWorld KVM concurrency) |
| Lean plan, `gpt-5.4-mini` everywhere except Video-Holmes | ~$70 – $100 | ~3 – 5 h |

#### Per-domain parallelism — what each launcher actually does

Three of the four Python launchers are single-process loops; parallelism
comes from one of three layered mechanisms:

| Domain | Concurrency primitive | How to scale | Hard ceiling |
|---|---|---|---|
| `gymv` | shell wrapper, one process / env | `run_coldstart_actor_gymv_all.sh --parallel` (already default) | 13 envs (= one process / env, retro emulator binds the process) |
| BrowserGym | shell-level **shard wrapper** (NEW) | `run_coldstart_actor_browsergym_shard.sh --num_shards N` | RAM (Chromium ≈ 500 MB / shard) + WebArena self-host QPS — practical sweet spot **8–12** |
| OSWorld | shell wrapper, domain-level dispatch | `run_coldstart_actor_osworld_all.sh --parallel --max_parallel N` (default **8**, was 3) | KVM RAM (≈ 6 GB / guest) — **8** on 64 GB host, **10+** on ≥ 96 GB |
| Visual reasoning | Python **`--num_workers N`** (ThreadPoolExecutor, NEW) | `--num_workers 32` on the launcher | OpenAI tier RPM — **16–32** on tier 4 (10 k RPM), **32–64** on tier 5 (30 k RPM) |

Per-domain wall-clock at the recommended parallelism (`gpt-5.4`,
`reasoning_effort=minimal` for env / `medium` for visual MCQ, source
video frames pre-extracted):

| Domain | Volume | Per-unit | Parallelism | Wall-clock |
|---|---|---|---:|---:|
| `gymv` (source) | 13 envs × 10 ep × ~20 steps ≈ 2.6 k steps | ~10 s / step | 13 (one process / env) | **~30–40 min** |
| BrowserGym | 306 tasks (125 MiniWoB + 181 AssistantBench) | ~70 s / task | 8 shards | **~45–60 min** (was ~6 h serial) |
| OSWorld ⬅ critical path | 250 tasks × 30 steps | ~10 s / step | 8 KVM guests | **~1.6–2.5 h** (was ~12 h @ 1 KVM) |
| Visual reasoning (image) | 600 (VTB 300 + TIR 300) | ~6 s / sample | 32 workers | **~2 min** (was ~1 h serial) |
| Visual reasoning (video) | 1,400 (VH 1,000 + SIV 400) | ~10 s / sample | 32 workers | **~8 min** (was ~4 h serial) |

**End-to-end: ~1.6–2.5 h** at recommended parallelism (set by OSWorld KVM
count + BrowserGym Chromium count). Two dominant levers stay the same:
`reasoning_effort` for cost and OSWorld KVM concurrency for wall-clock.

#### Quick start — run on all tasks with `gpt-5.4`

The four pipelines are independent (retro-emulator vs. Chromium vs.
KVM vs. pure-API), so you can launch them in parallel terminals.
Manifests live in [`cold_start/task_samples/`](cold_start/task_samples/)
(BrowserGym) and
[`cold_start/evaluation_dataset/pool/`](cold_start/evaluation_dataset/)
(OSWorld + visual reasoning). Output goes to `Cold-start-out-<domain>/`.

```bash
cd /workspace/Multi-hop-Reasoning-VLM-Agent
export OPENAI_API_KEY=...   # or set in /workspace/api_keys.py (auto-loaded)

# 1. gymv (~30-40 min) — wrapper dispatches one process per env, parallel default
bash cold_start/run_coldstart_actor_gymv_all.sh --parallel \
  -- --episodes 10 --max_steps 60 \
     --model gpt-5.4 --reasoning_effort minimal -v

# 2. BrowserGym (~45-60 min @ 8 shards) — auto-loads the lean-plan task
#    pools (MiniWoB + AssistantBench) and auto-sources webarena_env.sh
#    when relevant.
bash cold_start/run_coldstart_actor_browsergym_shard.sh \
  --num_shards 8 \
  --model gpt-5.4 --reasoning_effort minimal \
  -- --episodes 1 --max_steps 12 -v

# 3. OSWorld (~1.6-2.5 h @ 8 KVMs) — defaults to --max_parallel 8;
#    drop to 3-4 on hosts with < 64 GB RAM.
bash cold_start/run_coldstart_actor_osworld_all.sh --parallel --max_parallel 8 \
  -- --task_catalog cold_start/evaluation_dataset/pool/osworld_catalog.json \
     --episodes 1 --max_steps 30 \
     --model gpt-5.4 --reasoning_effort minimal -v

# 4. Visual reasoning (~10 min @ 32 workers) — pure-API ThreadPoolExecutor.
#    --reasoning_effort medium is the SAFER default for visual MCQ
#    (multi-hop CoT helps); flip to minimal only after a paired smoke test
#    confirms no accuracy regression.
python cold_start/generate_cold_start_actor_visual_reasoning.py \
  --benchmarks visual_toolbench tir_bench video_holmes siv_bench \
  --sample_ids_dir cold_start/evaluation_dataset/pool \
  --model gpt-5.4 --reasoning_effort medium \
  --num_workers 32 \
  --output_dir Cold-start-out-visual_reasoning -v
```

If you'd rather run BrowserGym serial (single Chromium, no shard log
churn), the original `python cold_start/generate_cold_start_actor_browsergym.py
--tasks $(grep -hv '^#' cold_start/task_samples/browsergym_*.txt | sort -u) ...`
form still works — it just takes ~10 h instead of ~1.5 h.

Resume is on by default for `gymv` / BrowserGym (skip episodes that
already have an `episode_NNN.json` on disk). OSWorld + visual reasoning
write per-task / per-sample summaries; re-running with the same
`--output_dir` skips finished work. The shard wrapper writes per-shard
logs and task-list audits to `Cold-start-out-browsergym/_shard_logs/`.

See
[`cold_start/readme.md#multi-domain-cold-start-lean-plan`](cold_start/readme.md#multi-domain-cold-start-lean-plan)
for sampler scripts, alternate model knobs (mini vs. full), and host-sizing
guidance for OSWorld KVM counts.

#### AssistantBench full-eval workflow

AssistantBench is the only browser-based benchmark in the lean plan with
gradable rewards out-of-the-box (DROP F1 against shipped gold answers on
the validation split). Treat it as the headline browser number; it
exercises *open-web multi-hop research* in a way MiniWoB cannot.

The full-eval pipeline has four pieces, each runnable in isolation:

1.  **`search_web("…")` synthetic action**
    ([`cold_start/search_backends.py`](cold_start/search_backends.py),
     wired into the actor at
     [`cold_start/generate_cold_start_actor_browsergym.py`](cold_start/generate_cold_start_actor_browsergym.py))

    Search engines TLS-fingerprint Playwright and serve CAPTCHAs. The
    harness intercepts `search_web(query)`, performs a server-side HTTP
    fetch through a tiered backend chain
    (Tavily / Serper / Brave → DDG-HTML → DDG-Lite → Yahoo →
    Wikipedia), renders the results into a self-contained HTML page,
    and injects them into the live page via a `data:text/html` URL.
    The agent sees results without ever hitting the rate-limited TLS
    surface. Paid-API keys are auto-detected from env (`TAVILY_API_KEY`,
    `SERPER_API_KEY`, `BRAVE_API_KEY`); missing keys silently fall
    through to the free chain.

2.  **Feasibility filter**
    ([`cold_start/filter_assistantbench_feasibility.py`](cold_start/filter_assistantbench_feasibility.py))

    20 of the 181 AB test tasks are systematically out of reach for a
    public-web agent (require login, require purchases, ask for
    real-time data). A 10-second `gpt-4o-mini` pre-screen labels each
    task `FEASIBLE / REQUIRES_LOGIN / REAL_TIME / TRANSACTIONAL /
    OPEN_ENDED` and writes a filtered task list, dropping the
    obvious-fail buckets so the eval doesn't waste 5 min × 20 tasks
    chasing impossible answers.

    ```bash
    python cold_start/filter_assistantbench_feasibility.py \
      --split test --classifier_model gpt-4o-mini
    # writes:
    #   cold_start/task_samples/assistantbench_feasibility_test.json
    #   cold_start/task_samples/browsergym_assistantbench_test_feasible.txt
    ```

3.  **Sharded eval launch** (validation 33 + feasible test 161 = 194
    tasks; ~5 h wall on 4 shards with `gpt-5.4 medium`):

    ```bash
    bash cold_start/run_coldstart_actor_browsergym_shard.sh \
      --num_shards 4 \
      --tasks_file cold_start/task_samples/browsergym_assistantbench_validation_all.txt \
      --tasks_file cold_start/task_samples/browsergym_assistantbench_test_feasible.txt \
      --output_dir Cold-start-out-browsergym/ab_full_eval_v1 \
      --model gpt-5.4 --reasoning_effort medium \
      -- --max_steps 16 -v
    ```

    Notes:
    - 4 shards is the sweet spot — 8 shards halves wall time but
      doubles DDG rate-limit pressure on the free search chain.
    - Validation has gold answers (gradable locally); test has none
      (predictions must be uploaded to AB's server for scoring).
    - Resume is on by default — re-running with the same `--output_dir`
      skips finished tasks.

4.  **Grade + AB-server submission**
    ([`cold_start/grade_assistantbench_eval.py`](cold_start/grade_assistantbench_eval.py))

    Walks the rollout summaries, aggregates DROP F1 on validation, and
    emits a JSONL keyed by AB's canonical `id` — exactly the format
    the [AB leaderboard space](https://huggingface.co/spaces/AssistantBench/leaderboard)
    accepts. Safe to run mid-eval (skips tasks that haven't completed).

    ```bash
    python cold_start/grade_assistantbench_eval.py \
      --run_dir Cold-start-out-browsergym/ab_full_eval_v1
    # writes:
    #   grading_summary.json / .csv               (per-task table)
    #   assistantbench_validation_score.json      (headline val number)
    #   assistantbench_test_predictions.jsonl     (AB-server upload)
    #   assistantbench_test_predictions_human.json (with task text)
    ```

    Reported numbers:
    `mean_reward` (DROP F1 on val) · `perfect_rate` (=1.0) ·
    `nonzero_rate` (>0) · `answered_rate` (emitted `send_msg_to_user`
    vs. truncated/infeasible) · `mean_steps` · `search_web_calls/task`.

Reproducing the v6 baseline (3-task smoke, gpt-5.4 medium, +0.370
mean_reward — a 9× lift over the no-search v4 baseline) is documented
in
[`implementation_notes/assistantbench-search-web-baseline.md`](implementation_notes/assistantbench-search-web-baseline.md)
once the full-eval results land.

#### WebShop bridge — frontier-model 4-way comparison

WebShop is the second graded browser benchmark in the lean plan and
the simplest to spin up: a single Flask server with rule-based
rewards in [0, 1], no Docker fleet, no LLM judge. The
[`webshop_wrapper/`](webshop_wrapper/README.md) module fronts
`princeton-nlp/WebShop` as `browsergym/webshop.<idx>` envs so the
existing BrowserGym driver, schema generator, and tool registry all
work unmodified.

Three install levels (stub → lite → full); see
[`webshop_wrapper/README.md`](webshop_wrapper/README.md) for the
trade-offs and
[`install/install_webshop.sh`](install/install_webshop.sh) for the
automated lite installer (BM25 via `rank_bm25`, no Java / pyserini
/ Lucene). The driver auto-discovers WebShop tasks because
`webshop_wrapper` is registered in
[`cold_start/generate_cold_start_actor_browsergym.py`](cold_start/generate_cold_start_actor_browsergym.py)'s
`_OPTIONAL_TASK_SUITE_MODULES`.

```bash
# 0. one-shot install of the WebShop conda env + dataset (~10 min, lite mode)
bash install/install_webshop.sh

# 1. boot the WebShop server (separate terminal, separate conda env)
conda activate webshop
cd $WEBSHOP_DIR && python -m web_agent_site.app  # ⇒ http://127.0.0.1:3000

# 2. from the agent env, run a 50-task eval with any OpenRouter slug
conda activate browsergym
cd /workspace/Multi-hop-Reasoning-VLM-Agent
export WEBSHOP_BASE_URL=http://127.0.0.1:3000
export WEBSHOP_NUM_GOALS=50
TASKS=$(for i in $(seq 0 49); do echo -n "browsergym/webshop.$i "; done)

python cold_start/generate_cold_start_actor_browsergym.py \
  --tasks $TASKS --episodes 1 --max_steps 20 \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --output_dir Cold-start-out-browsergym/webshop_50task_qwen -v
```

**Validated on 4 frontier models (50 tasks each, 2026-05-04;** full
report in
[`Cold-start-out-browsergym/REPORT_4way_comparison.md`](Cold-start-out-browsergym/REPORT_4way_comparison.md)**):**

| Model | Mean reward (95% CI) | SR pass (r≥0.5) | sec/task |
|---|---|---|---:|
| `qwen/qwen3-vl-235b-a22b-instruct` | **0.559** [0.483, 0.635] | **74%** | 319 |
| `openai/gpt-5.4` (effort=low)      | 0.377 [0.272, 0.482]    | 48%     | 226 |
| `anthropic/claude-sonnet-4.5`      | 0.330 [0.227, 0.433]    | 42%     | 335 |
| `google/gemini-3.1-pro-preview`    | 0.289 [0.174, 0.404]    | 32%     | 559 |

For reference: human expert ≈ 0.604, ReAct + GPT-4 ≈ 0.455, IL+RL ≈
0.300 (Yao et al. 2022). Qwen3-VL-235B-instruct's lead over the other
three frontier models is statistically significant (95% CIs do not
overlap with any of GPT-5.4-low / Claude / Gemini).

---

## Running experiments — instrumentation, ablations, cross-domain eval, analysis

This section is the one-stop guide for reproducing every figure /
table the NeurIPS 2026 reviewers asked for.  It assumes the cold-start
SFT corpus from the previous section is on disk and that one or more
co-evolution training runs have produced a `run_dir` of the form

```
runs/skillbridge_<tag>/
├── lora_adapters/                # per-LoRA adapter checkpoints
├── skillbank/<game>/skill_bank.jsonl
├── checkpoints/step_*/
├── phase_snapshots/phase_<k>_<game>/   # per-phase frozen state (see §Trainer integration)
├── reward_log.jsonl              # legacy per-step reward log
├── audit.jsonl                   # crafter / promotion audit
├── promotion_decisions_out/*_run_summary.json
├── reward_shaping_log/ratio.jsonl  # intrinsic vs raw_env shaping ratio
└── (instrumentation streams — see §1 below)
```

### 1. Reviewer instrumentation streams (Block A)

The trainer's `_run_loggers.py` facade owns five append-only JSONL
streams. They are **lazy-opened** on the first emit and disabled
entirely when `config.reviewer_instrumentation_enabled = False`. All
five live alongside the legacy `reward_log.jsonl` / `audit.jsonl`
artifacts:

| Stream                                             | Path                                          | Drives                                          |
| -------------------------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| Per-event harness eligibility rejection            | `harness_log/rejections.jsonl`                | §5.2 failure-mode pie chart (E4)                |
| Per-event `validate_invocation` diagnostic         | `harness_log/validate.jsonl`                  | §5.2 retrieval long tail (E2) + repair traces   |
| Skill lifecycle transitions                        | `lifecycle_log/transitions.jsonl`             | §5.2 promotion / lifetime curves (E1, E3, E6)   |
| Per-step intention switch (`z_t` updates)          | `intention_log/switches.jsonl`                | §4.1 intention-trigger ablation (B4)            |
| Per-trainer-step component runtime                 | `runtime_log/component_timings.jsonl`         | §5.6 token / wall breakdown (E5)                |
| Per-step shaping-ratio diagnostics                 | `reward_shaping_log/ratio.jsonl`              | Imbalance check between intrinsic + survival shaping vs raw env reward; emits a WARN at >5x |

Schema and field-by-field meaning lives in
[`trainer/coevolution/_run_loggers.py`](trainer/coevolution/_run_loggers.py).

#### Resume safety: bank restoration

Resuming from a checkpoint (`--resume`, `--resume-from-step`, or auto-
resume) now eagerly initializes every per-game `SkillBankAgent` *before*
the checkpoint loader runs.  Without this the lazy pipelines hand
`load_checkpoint` a `{game: None}` dict, the loader's `if agent is
None: continue` clause silently no-ops the bank restore, and the next
outer step reads `bank=0` → flips into spurious cold-start mode.
Confirmed via the new `tests/test_resume_bank_restore.py` regression
suite which pins both the post-fix behavior and the pre-fix
silent-no-op as a negative-test guard.

#### High-variance gymv games default to 16 episodes/step

`trainer.coevolution.config.HIGH_VARIANCE_GYMV_EPISODES` bumps the
default `episodes_per_game` from 8 to 16 for the gymv shooters /
brawlers (TF3, Altered Beast, Streets of Rage 2, Strider, Space
Harrier II, Airstriker, Dynamite Headdy).  Bootstrap from the
empirical TF3 episode-reward distribution shows the per-step
mean-reward sampling-noise floor drops from ~22 % (P(zero-mean | n=8))
to ~4 % (n=16); see the post-mortem in `tests/test_episodes_per_game_overrides.py`
for the rationale.  Override the dict with
`--episodes-per-game-overrides '{"gymv_thunder_force_iii": 24}'` to
go further (or pass `--episodes-per-game-overrides '{}'` to disable
the high-variance defaults).

#### Game-specific critical actions (action prior)

`trainer.coevolution.config.GAME_CRITICAL_ACTIONS` declares per-game
"must use" actions that are surfaced two ways:

  1. As a one-line in-context hint in the action-selection prompt
     ("Critical actions for this game (use frequently when scoring): B.").
  2. As an anti-stagnation substitution in `_apply_anti_repetition` —
     when the policy is stuck on a single non-scoring action OR runs
     8 consecutive zero-reward decisions without picking a critical
     action, the shim force-substitutes the critical action.

This is a **hard escape valve**, not a reward signal — it costs zero
GRPO advantage budget but ensures the action-vocab knowledge isn't
solely a function of GRPO convergence.  See
`tests/test_critical_action_prior.py` for the full behavior matrix.

### 2. Ablation flags (Block B)

Five CLI flags on [`scripts/run_coevolution.py`](scripts/run_coevolution.py)
turn off / re-route individual SkillBridge components.  All defaults
preserve the historical co-evolution behaviour, so adding the flag
package is byte-identical to the previous full system unless an
ablation flag is explicitly set.

| Flag                                              | Default      | Effect                                                                                          |
| ------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `--harness-mode {full, plain-text-skills, off}`   | `full`       | `plain-text-skills` strips bindings (LLM sees skills as text only); `off` bypasses the harness  |
| `--no-crafter`                                    | crafter on   | Disables the LLM crafter step (deterministic crafter still runs); promotion + lifecycle stay on |
| `--promotion-bypass-mode {gated, permissive}`     | `gated`      | `permissive` accepts every proposal (every gate stage forced to `PASS`)                         |
| `--intention-trigger {sharp-shift, every-step, disabled}` | `every-step` | `sharp-shift` only re-picks `z_t` on a real subgoal switch; `disabled` freezes the initial intention |
| `--actor-bank-cap-K K`                            | `0` (uncapped) | Cap how many top-K skills the `skill_selection` LoRA sees per step                              |

Smoke-running each ablation:

```bash
# B1 — runtime layer ablation
python scripts/run_coevolution.py --harness-mode plain-text-skills ...
python scripts/run_coevolution.py --harness-mode off ...

# B2 — disable LLM crafter
python scripts/run_coevolution.py --no-crafter ...

# B3 — strip the promotion gate
python scripts/run_coevolution.py --promotion-bypass-mode permissive ...

# B4 — intention switching
python scripts/run_coevolution.py --intention-trigger sharp-shift ...
python scripts/run_coevolution.py --intention-trigger disabled ...

# B5 — actor bank-size sweep
python scripts/run_coevolution.py --actor-bank-cap-K 8 ...
```

Each ablation run writes its own `run_dir` so the cross-domain eval
in §3 can later compare them pairwise.  See
[`tests/test_block_b_ablation_flags.py`](tests/test_block_b_ablation_flags.py)
for unit coverage of every flag.

### 3. Cross-domain evaluation pipeline (Block C)

Five domain-specific drivers + an aggregator + a one-shot launcher
live under [`scripts/skillbridge_eval/`](scripts/skillbridge_eval/).
All five drivers reuse the existing `cold_start/generate_cold_start_actor_<domain>.py`
scripts under the hood, just pointed at a vLLM endpoint where the
trained LoRA adapters have been loaded.

| Driver                                                     | Domain            | Underlying engine                                       |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------------- |
| `python -m scripts.skillbridge_eval.eval_browsergym`       | BrowserGym        | `cold_start/generate_cold_start_actor_browsergym.py`    |
| `python -m scripts.skillbridge_eval.eval_osworld`          | OSWorld           | `cold_start/generate_cold_start_actor_osworld.py`       |
| `python -m scripts.skillbridge_eval.eval_visual_reasoning` | Visual reasoning  | `cold_start/generate_cold_start_actor_visual_reasoning.py` (image benches) |
| `python -m scripts.skillbridge_eval.eval_video`            | Video             | same script restricted to `video_holmes` + `siv_bench`  |
| `python -m scripts.skillbridge_eval.eval_gymv`             | GymV              | `trainer.coevolution.episode_runner.run_episode_async`  |
| `python -m scripts.skillbridge_eval.eval_aggregator`       | Aggregator        | scans `<run-dir>/eval/*_result_*.json` → CSV + Markdown |

A common actor wrapper [`scripts/skillbridge_eval/eval_actor.py`](scripts/skillbridge_eval/eval_actor.py)
encapsulates LoRA / skill-bank / harness loading and is reused by the
GymV driver and the transfer-matrix runner (block D1).

#### One-shot end-to-end eval

```bash
bash scripts/run_skillbridge_eval.sh \
    --run-dir runs/skillbridge_v12 \
    --vllm-base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3.5-9B \
    --label skillbridge_full \
    --episodes-per-task 1 --max-steps 50 \
    [--skip osworld]              # comma-separated list of domains to skip
    [--judge]                     # enable LLM-as-judge for VR / video
    [--gymv-games crafter,procgen]
```

Each driver writes a uniform `<run-dir>/eval/<domain>_result_<ts>.json`
with `domain`, `label`, `model`, `overall.{success_rate_macro|accuracy_micro|mean_reward_macro}`
and a per-task / per-benchmark breakdown.  The aggregator picks the
most-recent file per domain and emits:

```
<run-dir>/eval/aggregate.json    # full payload incl. per-row primary metric
<run-dir>/eval/aggregate.csv     # one row per domain
<run-dir>/eval/aggregate.md      # ready-to-paste paper table
```

#### Per-driver direct invocation

If you only want one domain (e.g. for an ablation × domain cell) the
driver CLIs are stand-alone:

```bash
python -m scripts.skillbridge_eval.eval_browsergym \
    --run-dir runs/skillbridge_v12 \
    --tasks-file cold_start/task_samples/browsergym_assistantbench_test_feasible.txt \
    --episodes-per-task 1 --max-steps 30 \
    --model Qwen/Qwen3.5-9B \
    --vllm-base-url http://localhost:8000/v1 \
    --label skillbridge_full \
    --output runs/skillbridge_v12/eval/browsergym_result.json

python -m scripts.skillbridge_eval.eval_visual_reasoning \
    --run-dir runs/skillbridge_v12 \
    --benchmarks visual_toolbench tir_bench \
    --num-test-cases 200 --num-workers 32 \
    --model Qwen/Qwen3.5-9B \
    --vllm-base-url http://localhost:8000/v1 \
    --judge --judge-model Qwen/Qwen3.5-35B-A3B
```

### 4. Cross-domain transfer matrix + few-shot scaling (Block D)

Built on top of the per-domain drivers in §3:

```bash
# D1 — every phase snapshot × every held-out domain
python -m scripts.skillbridge_eval.run_transfer_matrix \
    --run-dir runs/skillbridge_v12 \
    --domains visual_reasoning video gymv \
    --vllm-base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3.5-9B \
    --snapshot-loader 'curl -X POST http://localhost:8000/v1/load_lora -d {snapshot}/lora_adapters' \
    --output runs/skillbridge_v12/eval/transfer_matrix.json

# D2 — k ∈ {0, 1, 4, 16, 64} target-domain demonstrations
python -m scripts.skillbridge_eval.run_few_shot_sweep \
    --run-dir runs/skillbridge_v12 \
    --domain visual_reasoning \
    --ks 0 1 4 16 64 \
    --vllm-base-url http://localhost:8000/v1 \
    --enable-warmup \
    --warmup-cmd-template 'python scripts/run_coevolution.py --target-domain {domain} --max-warmup-eps {k} --run-dir {run_dir}'
```

`run_transfer_matrix.py` automatically masks the snapshot's own
training-domain cell (use `--include-self` to disable).  Both drivers
expose hook flags (`--snapshot-loader`, `--warmup-cmd-template`) so
the user can wire LoRA hot-swapping / warm-up to whatever vLLM
deployment they run.

### 5. Analysis + plotting (Block E)

[`scripts/skillbridge_analysis/`](scripts/skillbridge_analysis/) ships
eight stand-alone CLIs that consume the JSONL streams in §1.  Each
writes both a JSON summary (paper-quotable numbers) and a PNG figure;
pass `--no-plot` to skip the figure when matplotlib is unavailable.

| Script                                                          | Source                                                      | Output                                                          |
| --------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- |
| `plot_skill_dynamics.py`                                        | `lifecycle_log/transitions.jsonl` + `audit.jsonl`           | promotion / rejection / deprecation curves + mutation pie       |
| `plot_skill_long_tail.py`                                       | `harness_log/validate.jsonl` (falls back to `reward_log.jsonl`) | per-skill retrieval frequency (linear + log-log)                |
| `plot_skill_lifetime.py`                                        | `lifecycle_log/transitions.jsonl`                           | skill lifetime histogram + percentiles (right-censored)         |
| `plot_failure_modes.py`                                         | `harness_log/{rejections,validate}.jsonl`                   | 8-class veto pie + 4-axis validate failure pie                  |
| `plot_runtime_overhead.py`                                      | `runtime_log/component_timings.jsonl`                       | per-component wall + token bar chart                            |
| `plot_skill_flow_map.py`                                        | `lifecycle_log/transitions.jsonl`                           | DRAFT → … → RETIRED flow diagram (Figure 2 candidate)           |
| `compute_significance.py`                                       | two `*_result.json` files (paired tasks)                    | bootstrap CI95 + p-value (one- and two-sided)                   |
| `case_study_skill_trace.py`                                     | every stream + `audit.jsonl`                                | Markdown provenance trail for a single `skill_id`               |

Typical full analysis pass:

```bash
RUN=runs/skillbridge_v12

for s in plot_skill_dynamics plot_skill_long_tail plot_skill_lifetime \
         plot_failure_modes plot_runtime_overhead plot_skill_flow_map; do
    python -m scripts.skillbridge_analysis.$s --run-dir $RUN
done

# pairwise significance (baseline vs. SkillBridge full) for each domain
python -m scripts.skillbridge_analysis.compute_significance \
    --domain browsergym \
    --baseline runs/baseline/eval/browsergym_result_*.json \
    --treatment $RUN/eval/browsergym_result_*.json \
    --n-bootstrap 5000 \
    --output $RUN/analysis/significance_browsergym.json

# zoom in on a particular skill — provenance Markdown
python -m scripts.skillbridge_analysis.case_study_skill_trace \
    --run-dir $RUN --skill-id sk_collect_evidence_chain_v3 \
    --out-path $RUN/analysis/case_study_sk_collect_evidence_chain_v3.md
```

All summary JSONs and PNGs land under `<run-dir>/analysis/` by
default; pass `--out-dir` to redirect.

### 6. Writing stubs (Block F)

[`paper/stubs/`](paper/stubs/) maps each generator above to the paper
section that consumes it.  When a number changes, re-run the matching
script and re-paste the JSON — the stubs include the exact CLI for
every figure so the paper provenance is reproducible.

| Stub                                          | Paper section                                  | Generator |
| --------------------------------------------- | ---------------------------------------------- | --------- |
| [`paper/stubs/05.2_skill_dynamics.md`](paper/stubs/05.2_skill_dynamics.md)             | §5.2 Skill-bank dynamics                       | E1 + E2 + E3 + E4 |
| [`paper/stubs/05.3_lifecycle_gating_ablation.md`](paper/stubs/05.3_lifecycle_gating_ablation.md) | §5.3 Lifecycle / intention ablations           | B1–B5 + C7 |
| [`paper/stubs/05.5_cross_domain_transfer.md`](paper/stubs/05.5_cross_domain_transfer.md) | §5.5 Cross-domain transfer matrix              | D1 + D2 |
| [`paper/stubs/05.6_runtime_overhead.md`](paper/stubs/05.6_runtime_overhead.md)         | §5.6 Runtime / token overhead                  | E5 |
| [`paper/stubs/06_limitations.md`](paper/stubs/06_limitations.md)                       | §6 Limitations                                 | _(prose)_ |
| [`paper/stubs/algorithms.md`](paper/stubs/algorithms.md)                               | Alg. 1 (episode step) + Alg. 2 (bank update)   | _(prose, source-cross-referenced)_ |
| [`paper/stubs/_consistency_sweep.md`](paper/stubs/_consistency_sweep.md)               | F2 — paper cleanup checklist                   | `rg`-driven |

### 7. End-to-end recipe — from a fresh `run_dir` to a NeurIPS table

```bash
# 0. Train. Produces runs/skillbridge_v12/{lora_adapters, skillbank, phase_snapshots, *.jsonl}.
bash scripts/run_phase1_curriculum.sh   # 6 phases, with reviewer instrumentation on by default

# 1. Cross-domain eval (block C).
bash scripts/run_skillbridge_eval.sh \
    --run-dir runs/skillbridge_v12 \
    --vllm-base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3.5-9B \
    --label skillbridge_full

# 2. Transfer matrix + few-shot (block D).
python -m scripts.skillbridge_eval.run_transfer_matrix \
    --run-dir runs/skillbridge_v12 --domains visual_reasoning video gymv
python -m scripts.skillbridge_eval.run_few_shot_sweep \
    --run-dir runs/skillbridge_v12 --domain visual_reasoning --ks 0 1 4 16 64

# 3. Analysis + plots (block E).
for s in plot_skill_dynamics plot_skill_long_tail plot_skill_lifetime \
         plot_failure_modes plot_runtime_overhead plot_skill_flow_map; do
    python -m scripts.skillbridge_analysis.$s --run-dir runs/skillbridge_v12
done

# 4. Drop the resulting JSON / PNG / aggregate.md numbers into the
#    matching paper/stubs/*.md stubs (block F).
```

The aggregate Markdown table at
`runs/skillbridge_v12/eval/aggregate.md` is the headline cross-domain
results table; the per-component runtime bar chart at
`runs/skillbridge_v12/analysis/runtime_overhead_bar.png` is the
NeurIPS Q8 compute disclosure.

---

## Repository layout

### New modules (canonical build, this plan)

```
common/                   # canonical enums, ID helpers, <state> schema, BACKBONE_MODEL
data_structure/extensions/   # SkillEpisode, SkillRecord, GateVerdict, SkillEvaluationRecord,
                          # BankMutationProposal (Compose/Generalize/Hypothesis/Patch/Retire),
                          # FailureTrace, FailureDiagnosis, RunRelease
skill_bank/               # split-storage (draft/candidate/active/archive) + SkillLifecycleManager
                          # + SkillRepository — bank-write isolation invariant
harness/                  # SkillHarness, AdapterRegistry, EligibilityFilter, ReplayValidator,
                          # RewardLogger, SkillAdapter base + adapters/{gymv,browser}
orchestrator/             # EpisodeRunner, ArtifactStore, BudgetController, GateService,
                          # PromotionOrchestrator, SnapshotManager, OrchestratorConfig
                          # (TeacherConfig + JudgeConfig + backbone_model)
crafter/                  # FailureMemory, FailureDiagnoser, Composer, Generalizer,
                          # Hypothesizer, SkillCrafterService — proposals only
tests/                    # test_invariants.py (14), test_smoke.py (2),
                          # test_backbone_model.py (13)
```

### Legacy modules (COS-PLAY, kept for reference and incremental migration)

```
decision_agents/          # legacy COS-PLAY decision agent, intention/skill/reward
skill_agents/             # legacy skill-bank pipeline + GRPO training (LoRA, segmentation)
vlm_wrapper/              # legacy visual-grounding parsers and benchmark loaders
data_structure/legacy/    # legacy Episode / Experience records (extensions/ supersedes)
inference/                # legacy inference scripts (incl. Qwen3-8B / vLLM entrypoints)
trainer/                  # legacy SFT / GRPO / FSDP training infrastructure
env_wrappers/             # NL wrappers, Gymnasium adapters, game configs
labeling/, cold_start/    # cold-start labeling and seed-trajectory generation
```

### Plans

```
plans/
├── 00-system/        north-star scoreboard, eval-first target, role walkthrough
├── 01-visual-grounding/ Stage 1 — VLM parser + milestones
├── 02-action-agent/  Stage 2 — decision agent (single-MDP shipped; PLAN = inner-hop design)
├── 03-skill-bank/    Stage 3 — cross-task skill bank, retrieval, contracts
├── 04-skill-crafter/ Stage 4 — compose / generalize / hypothesize
├── 05-harness/       per-invocation runtime + gate stack
├── 06-orchestrator/  system control plane (DAG, promotion / rollback)
├── 07-skill-gate/    canonical lifecycle and gate spec
├── 08-cross-cutting/ failure routing, uncertainty calibration, experience ext.
├── 09-implementation/ Cursor-ready build sheet — Phase A → F + invariants
└── legacy/           DONE edit plans (`legacy/10-edits/`) + archive (`legacy/99-archive/`)
```

See [`plans/README.md`](plans/README.md) for the full index.

---

## Implementation status

A more detailed view lives in [`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md). Headline:

### Delivered (29 tests passing)

- **Common** — enums, IDs, `<state>` schema, backbone-model registry.
- **P0** — extension records (7 dataclasses) under `data_structure/extensions/`.
- **Skill bank** — split-storage with `SkillLifecycleManager` enforcing the unified gate.
- **Phase A — Harness MVP** — `SkillHarness`, eligibility filter, adapters (`gymv`, `browser`), reward log, replay validator stub.
- **Phase B — Orchestrator MVP** — `EpisodeRunner`, atomic `ArtifactStore`, `BudgetController`, `GateService` (stages 0–4), `PromotionOrchestrator`, `SnapshotManager`.
- **Phase C — Crafter MVP** — failure memory + diagnoser, composer, generalizer, hypothesizer, `SkillCrafterService`.
- **Backbone model** — GPT-4o pinned across actor / teacher / judge with env-var override path.
- **Invariants** — six invariants mechanically enforced and tested.

### Pending (next sessions)

| Track | Item | Owning module |
| --- | --- | --- |
| **P1 — Visual Grounding** | Lightweight grounding stabilisation; routing policy A/B/C | `vlm_wrapper/grounding.py` |
| **P2 — Eval E0 driver** | JSONL driver, MCQ answer evaluator, LLM judge, easy/medium/hard slices, headline triple report | `evaluation/{driver,answer_evaluator,llm_judge,slices,report}.py` |
| **Phase D — Transfer + Replay** | Two-phase shadow → active transfer; full six-gate `GateRunner` (G0–G5); held-out replay; adapters for `osworld`, `video`, `visual_reasoning` | `harness/{transfer_manager,gate_runner,replay_validator}.py` + adapters |
| **Phase E — Eval suite + dashboards** | Frozen eval suite for non-regression; slice / label dashboards; `eval_suite_id` wiring | `orchestrator/eval_suite.py` |
| **Phase F — Trainable extensions** | LoRA heads `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` | TBD |
| **Actor rewire** | `HarnessSkillProvider` so the Actor consumes `SkillHarness.select_eligible_skills` instead of querying the bank directly | `decision_agents/skill_interface.py` |
| **Legacy bridge** | One-way migration of `skill_agents/skill_bank` Stage-3 records into `SkillRecord` | `skill_bank/legacy_bridge.py` |
| **Repair plumbing** | `SkillCrafterService.propose_repair` exposing the existing `PatchProposal` type | `crafter/service.py` |

Phases A → B → C → D → E → F are strict; do not start a phase before its predecessor's acceptance criteria are green.

---

## Quick start

### Install

```bash
cd Multi-hop-Reasoning-VLM-Agent
conda create -n vlm-agent python=3.11 -y
conda activate vlm-agent
pip install -e .
```

For full setup (CUDA toolchain, vLLM for the deferred Qwen tracks, game environments) see [`install/README.md`](install/README.md). The new build (`harness/`, `orchestrator/`, `crafter/`, `skill_bank/`, `tests/`) only requires the standard scientific Python stack and an OpenAI / OpenRouter API key for GPT-4o.

### Configure the backbone

```bash
cp .env.example .env
# minimum: OPENAI_API_KEY=...  (or OPENROUTER_API_KEY=...)
set -a && source .env && set +a
```

The default backbone is `gpt-4o`; nothing else needs to be set.

### Run the test suite

```bash
python -m pytest tests/ -v
```

Expected: `29 passed`. The suite covers the six invariants, an end-to-end `EpisodeRunner` smoke run, the crafter's failure → DRAFT proposal cycle, and the GPT-4o backbone pin.

### Smoke-run the orchestrator

```python
from common import StateSchema
from harness import AdapterRegistry, HarnessConfig, SkillHarness
from harness.adapters import GymvAdapter
from orchestrator import (
    ArtifactStore,
    BudgetController,
    EpisodeRunner,
    OrchestratorConfig,
)
from skill_bank import SkillLifecycleManager, SkillRepository
from skill_bank.stores import SkillStore, StoreName

# wire the bank
repo = SkillRepository(
    draft_store=SkillStore(StoreName.DRAFT, "_bank/draft"),
    candidate_store=SkillStore(StoreName.CANDIDATE, "_bank/candidate"),
    active_store=SkillStore(StoreName.ACTIVE, "_bank/active"),
    archive_store=SkillStore(StoreName.ARCHIVE, "_bank/archive"),
)
lifecycle = SkillLifecycleManager(repo)

# wire the harness
registry = AdapterRegistry()
registry.register(GymvAdapter())
harness = SkillHarness(adapter_registry=registry, config=HarnessConfig())

# run an episode (fake env / actor — see tests/test_smoke.py for a full example)
artifacts = ArtifactStore("_artifacts")
runner = EpisodeRunner(
    config=OrchestratorConfig(),
    harness=harness,
    repository=repo,
    artifact_store=artifacts,
)
result = runner.run(
    budget=BudgetController(),
    env=...,        # any object exposing reset() / step()
    actor=...,      # any object exposing choose(state, eligible) -> action
    initial_state=StateSchema(domain="gymv", task="demo", goal="demo", step=0),
)
print(result.outcome)
```

See [`tests/test_smoke.py::test_smoke_end_to_end`](tests/test_smoke.py) for the full runnable example.

---

## Plans index

The full plan corpus is in [`plans/`](plans/README.md). Recommended reading order:

1. [`plans/00-system/PLAN-SYSTEM-NORTHSTAR.md`](plans/00-system/PLAN-SYSTEM-NORTHSTAR.md) — single canonical scoreboard and stop/go rules.
2. [`plans/00-system/PLAN-EVAL-FIRST-TARGET.md`](plans/00-system/PLAN-EVAL-FIRST-TARGET.md) — Joint Success Rate contract and the `E0 → E1 → E2` rollout.
3. [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`](plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md) and [milestones](plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md).
4. [`plans/02-action-agent/PLAN-ACTION-AGENT.md`](plans/02-action-agent/PLAN-ACTION-AGENT.md) — two-level MDP, three-agent role split.
5. [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](plans/03-skill-bank/PLAN-SKILL-BANK.md) — invariants §0.1 / §0.3, cross-domain retrieval.
6. [`plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) — frozen-teacher proposal layer.
7. [`plans/05-harness/PLAN-HARNESS.md`](plans/05-harness/PLAN-HARNESS.md) — per-invocation runtime, gates G0–G5.
8. [`plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) — control plane, episode-local evidence contract.
9. [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — canonical lifecycle.
10. [`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) — Cursor build sheet, Phase A → F.

Cross-cutting (read on demand): [`plans/08-cross-cutting/`](plans/08-cross-cutting/) — failure routing, uncertainty calibration, experience extension.

---

## Legacy COS-PLAY notes

The `decision_agents/`, `skill_agents/`, `vlm_wrapper/`, `trainer/`, `inference/`, and `env_wrappers/` directories carry the COS-PLAY codebase that this build supersedes. COS-PLAY is the **co-evolution framework over Qwen3-8B for game agents** described in:

> *COS-PLAY: Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Game Play.*

The COS-PLAY entrypoints (`scripts/run_coevolution.py`, `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`, `bash scripts/run_2048.sh`, etc.) and the corresponding documentation in their per-module READMEs remain valid — they are retained as the **deferred 8B/32B/72B Qwen tracks**. They do not run by default; the new build under `common/`, `harness/`, `orchestrator/`, `crafter/`, `skill_bank/` defaults to GPT-4o end-to-end.

### Schema → predicate bridge ([`skill_agents/schema_predicates.py`](skill_agents/schema_predicates.py))

The unified visual grounding head ([`visual_grounding_tests/generate_gymv_image_schema.py`](visual_grounding_tests/generate_gymv_image_schema.py)) emits `<state>...</state>` blocks built by [`vlm_wrapper.schema`](vlm_wrapper/schema.py). The legacy Stage-3 contract learner ([`skill_agents/stage3_mvp/`](skill_agents/stage3_mvp)) consumes a flat predicate-probability map (`Dict[str, float]`). [`skill_agents/schema_predicates.py`](skill_agents/schema_predicates.py) is the pure-Python adapter between them — it lets game and `env_wrappers` rollouts produced by the unified grounding head feed the legacy COS-PLAY skill-mining pipeline (boundary proposal → segmentation → contract learning → bank maintenance) without re-implementing per-domain extractors.

Predicate keys are flat strings encoding the source section, so the cross-domain skill templates in [`skill_agents/skill_template.py`](skill_agents/skill_template.py) match them directly:

| Section | Predicate key shape |
| --- | --- |
| `<entities>` | `entity:<eid>:exists`, `entity:<eid>:type:<t>`, `entity:<eid>:ontology:<role>` |
| `<attributes>` | `attr:<eid>:<key>=<value>` |
| `<affordances>` | `afford:<eid>:<verb>` |
| `<relations>` | `rel:<verb>:<eA>:<eB>[:<eC>...]` |
| `<state_flags>` | `flag:<name>` (boolean) / `flag:<name>=<bucket>` (categorical) |
| `<targets>` | `target:eid=<eid>`, `target:blocker=<eid>`, `target:candidate:<eid>`, `target:history_anchor=<eid>` |

The `<uncertainty>` block attenuates per-(eid, field) probabilities to `0.4` for `high`, `0.6` for `medium`, `1.0` for `low`. Stage 3 booleanises predicates at ~0.5, so a `high`-uncertainty entity flips to *absent* in the contract learner — exactly the desired behaviour.

Plug it in as a source on the existing predicate extractor:

```python
from skill_agents.stage3_mvp.extract_predicates import CompositePredicateExtractor
from skill_agents.stage3_mvp.predicate_vocab import PredicateVocab
from skill_agents.schema_predicates import schema_to_predicates

vocab = PredicateVocab()
extractor = CompositePredicateExtractor(vocab)
extractor.add_source(schema_to_predicates)

preds = extractor(experience.summary_state)  # accepts str / dict / Experience-like
```

Coverage lives in [`tests/test_schema_predicates.py`](tests/test_schema_predicates.py) (entity / attribute / affordance / relation / target / state-flag / uncertainty / input-coercion / robustness — including a real `CompositePredicateExtractor` integration check).

### Other glue scheduled to land in the next sessions

- A `decision_agents.skill_interface.HarnessSkillProvider` so the COS-PLAY Actor can consume `SkillHarness.select_eligible_skills` instead of querying the legacy bank directly.
- A one-way `skill_bank/legacy_bridge.py` that migrates Stage-3 `skill_agents/skill_bank` records into the new `SkillRecord` format.

---

## Citation and license

If you use this codebase, please cite both the multi-hop reasoning agent (this build) and the COS-PLAY paper:

```bibtex
@misc{multihop-vlm-agent,
  title={Multi-hop Reasoning VLM Agent: A Skill-Centric, Evidence-Driven, Gate-Bound Visual Agent},
  author={Wu, Xiyang and others},
  year={2026},
  note={Codebase: \url{https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent}}
}

@inproceedings{wu2026cosplay,
  title={Co-Evolving {LLM} Decision and Skill Bank Agents for Long-Horizon Game Play},
  author={Wu, Xiyang and others},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
