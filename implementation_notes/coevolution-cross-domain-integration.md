# Co-evolution × cross-domain transfer integration

> **Status (2026-05-02 PM):** 🟢 **CAD LANDED — Layers C / A / D implemented.**
> All four cross-domain real-env executors landed in the harness on
> 2026-05-02 AM (Phase-5 §12 Tier 1 items 1-4 closed; see commits
> `7cc83bd` → `868f137`). On 2026-05-02 PM, Layers **C** (predicate
> translator splice into `SkillHarnessHook`), **A** (cross-domain
> admit-rate as promotion gate), and **D** (periodic dashboard) of
> this memo were implemented and tested:
>
> | Layer | Module | Tests | Commit |
> |---|---|---|---|
> | C | `trainer/coevolution/_harness_hook.py` (modified) | `tests/test_trainer_harness_hook.py` (5 new + 21 prior, 26 total) | `bc07599` |
> | A | `trainer/coevolution/_transfer_hook.py` (new, 798 LOC) | `tests/test_transfer_hook.py` (28 tests) | `10b23b3` |
> | D | `trainer/coevolution/_dashboard_hook.py` (new, 649 LOC) | `tests/test_dashboard_hook.py` (39 tests) | `bf83fec` |
>
> Layer **B** (cross-domain admit-rate as a GRPO reward channel)
> remains DESIGN-only — see §6 for the deferred plan.
>
> All three landed CAD layers are off by default; flip the master
> flags in `CoEvolutionConfig` (`crafter_transfer_gate_enabled`,
> `crafter_dashboard_enabled`) to opt in. The combined
> `tests/test_{trainer_harness_hook,transfer_hook,dashboard_hook}.py`
> suite has 93 tests, 0 skipped, ~1.5s wall-clock under pytest.

> **Cross-refs:**
> - [`implementation_notes/legacy/phase5-cross-domain-measurement.md`](legacy/phase5-cross-domain-measurement.md) §12 — full inventory of cross-domain code that landed
> - [`implementation_notes/cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md) — original Stage 0-6 measurement plan
> - [`implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](legacy/crafter-harness-orchestrator-roles.md) §6.3 — "no driver imports another driver's code" rule
> - [`harness/README.md`](../harness/README.md) §22 — trainer integration block (existing within-domain `SkillHarnessHook`)
> - [`trainer/coevolution/orchestrator.py`](../trainer/coevolution/orchestrator.py) lines 612-660 (harness hook construction) and 825-880 (Phase B′ Crafter + Promotion) — the two extension points this memo plugs into
> - [`trainer/coevolution/_harness_hook.py`](../trainer/coevolution/_harness_hook.py), [`_promotion_hook.py`](../trainer/coevolution/_promotion_hook.py), [`_crafter_hook.py`](../trainer/coevolution/_crafter_hook.py) — the per-step trainer hooks already wired into the loop

---

## 1. Current state — what IS and is NOT wired

### 1.1 What IS wired

The harness has been integrated into the live co-evolution loop via
`trainer/coevolution/_harness_hook.py:SkillHarnessHook`. Per-game
`SkillHarnessHook` instances are built in
[`orchestrator.py` lines 612-660](../trainer/coevolution/orchestrator.py)
and threaded into the episode runner. Per-step contract:

1. **Pre-LLM filter** — `SkillHarnessHook.filter_candidates` calls
   `SkillHarness.select_eligible_skills()` over the actor's RAG
   candidate set. Skills the harness vetoes (status / domain / task /
   adapter / can_handle) are dropped before the LLM sees them.
2. **Post-LLM validation** — `SkillHarnessHook.validate_choice` calls
   `SkillHarness.validate_invocation()` on the LLM's chosen skill.
   Vetoed → fall through to next surviving candidate.
3. **Rejection sink** → `SkillLifecycleManager` via the Crafter hook
   (`_crafter_hook.py` consumes `RejectedSkillSink.flush_to_lifecycle`).

This is real and already running. **But:** it operates only on the
*within-domain* surface. The adapter the hook registers is a
deterministic `GymvAdapter` stub — by docstring contract, *"No live
LLM call; no env binding... Real `set_executor(env_step_fn)` wiring
stays out of scope"*. It does NOT touch any cross-domain executor.

### 1.2 What IS NOT wired

Verified by grep against `trainer/coevolution/` and
`scripts/run_coevolution.py` on 2026-05-02:

| Module | Imported by `trainer/coevolution/`? |
|---|---|
| `labeling_supplement/_phase4_target_dispatch` | ❌ No |
| `labeling_supplement/_phase4_transfer_matrix` (Stage 6 NxN driver) | ❌ No |
| `labeling_supplement/_phase4_transfer_report` (G1-G6 generator) | ❌ No |
| `harness/_vr_per_sample_executor:TaskAwareVisualReasoningExecutor` | ❌ No |
| `harness/_video_per_sample_executor:TaskAwareVideoReasoningExecutor` | ❌ No |
| `harness/_osworld_per_sample_executor:TaskAwareOsworldExecutor` | ❌ No |
| `harness/_browser_per_sample_executor:TaskAwareBrowserExecutor` | ❌ No |
| `harness/_executor_helpers/{osworld_client,browser_helper}` | ❌ No |
| `harness/predicate_translator:translate_skill_contract / with_predicate_translation` | ❌ No |
| `cross_domain_results/` (Stage 6 reports) | ❌ Not consumed by trainer |

The cross-domain stack was built as **offline measurement
infrastructure** that emits
`cross_domain_results/_final/run_*Z/_report.md` for the G1-G6
acceptance gates. Live training-loop integration was always going to
be a separate cut.

### 1.3 Why this matters

Without integration:
* The trainer's promotion gate (`_promotion_hook.py`) admits skills
  based on within-domain LLM judging only. A skill that passes
  within-game gating but completely fails cross-domain transfer
  is indistinguishable from one that transfers cleanly — both reach
  PROMOTED status.
* The Repairer (`_crafter_hook.py`) sees only same-domain failure
  classes (`BANK_GAP` / `KNOWN_SKILL_FAIL` / etc.; see
  `configs/failure_routing.yaml`). Cross-domain failure modes
  ("predicate vocabulary mismatch", "slot-binding feasibility = 0",
  "post-translation admit-rate below band") have no entry in the
  taxonomy — they cannot route to `generalizer` / `composer` /
  `retirer` even when the offline NxN driver flagged them.
* The `feasible_domains` slot on `SkillRecord` cannot be widened
  by training signal — only by manual offline curation.

Merging the two pipelines is the load-bearing piece that turns the
Stage 6 NxN report into a *signal* the live loop responds to.

---

## 2. Architecture overview — four-layer stack

The user-selected integration plan is **all four layers** in
dependency order:

```
        ┌───────────────────────────────────────────────────────┐
        │ Layer C — Predicate translator in SkillHarnessHook    │  smallest
        │   Runtime: per-step (microseconds)                    │  cheapest
        │   Hot path: yes (inside actor's skill-selection)      │  ↓
        └───────────────────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────────────────────┐
        │ Layer A — Cross-domain admit-rate as promotion gate   │
        │   Runtime: per Phase B′ promotion step (~1-5 min)     │
        │   Hot path: no (offline-style, gated on               │
        │   ``crafter_promotion_enabled``)                      │
        └───────────────────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────────────────────┐
        │ Layer D — Periodic offline pass (orchestrator-driven) │
        │   Runtime: every N training steps (~30-60 min)        │
        │   Hot path: no (separate process tree)                │
        └───────────────────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────────────────────┐
        │ Layer B — Cross-domain admit-rate as reward channel   │  largest
        │   Runtime: amortised per-rollout (TBD)                │  riskiest
        │   Hot path: yes (GRPO reward computation)             │  ↓
        └───────────────────────────────────────────────────────┘
```

**Why this ordering:** Layer C is a pure-function rebinding step (no
new dependencies, no subprocess, no infra). It unblocks the harness's
per-step skill-selection from rejecting cross-domain skills purely on
predicate-vocabulary mismatch. Layer A is the smallest "consumes the
Stage 6 driver" wiring — it lifts an existing offline subprocess
pattern (`_promotion_hook.py` already shells out to
`decide_promotion_gpt54.py`) and reuses it for the transfer driver.
Layer D generalises Layer A's pattern to a periodic dashboard. Layer
B is the riskiest because it puts cross-domain measurement inside
the GRPO reward path — it should land last, after A and D have
demonstrated the per-skill admit-rate signal is stable.

---

## 3. Layer C — Predicate translator in `SkillHarnessHook` 🟢 IMPLEMENTED 2026-05-02 PM (commit `bc07599`)

> **Implementation summary.** `harness.predicate_translator.translate_skill_contract`
> is now threaded into `SkillHarnessHook.filter_candidates` between
> the candidate→record mapping and the eligibility filter. Diagonal
> cells short-circuit the deep-copy. Cross-domain cells rewrite
> `contract.effects_{add,del}` so the eligibility filter — and the
> downstream success_fn — see predicates the target adapter can
> actually ground. Translator crashes degrade to identity (the
> original record) and bump a separate `n_predicate_translations_failed`
> counter. The cached `SkillRecord` is never mutated (deep-copy
> semantics enforced by the translator).
>
> New diagnostic counters surfaced both on the per-call `diag` dict
> and the per-step `HarnessStepStats.to_json()` payload:
>
> * `n_predicate_translations_applied` — how many candidate
>   contracts had their effects rewritten this step.
> * `n_predicate_translations_failed` — translator-crash count
>   (always identity-fallback; the rollout never breaks).
>
> Tests: 5 new cases in `tests/test_trainer_harness_hook.py`
> covering diagonal no-op, cross-domain rebind (gymv→visual_reasoning
> canonical cell), translator-crash fallback, eligibility-crash
> counter preservation, and `to_json()` exposure.

### 3.1 Goal

When the actor on game `2048` is offered a candidate skill that lifted
from `tir_bench` (image-VR), the contract's `effects_add` may carry
predicates like `answer_emitted` / `answer_matches_gold` that the
gymv `success_fn` cannot evaluate. Today, F4 (`can_handle`) vetoes
the skill on this mismatch, even though
`harness/predicate_translator.py` ships a target-vocab-validated
translation table that would rewrite the contract to use predicates
the gymv schema *can* evaluate.

The integration is small: have `SkillHarnessHook.filter_candidates`
translate each candidate's contract to the target-domain vocabulary
*before* passing it to `SkillHarness.select_eligible_skills()`.

### 3.2 Plumbing

Single-file change to
[`trainer/coevolution/_harness_hook.py`](../trainer/coevolution/_harness_hook.py).

```python
# new import
from harness.predicate_translator import translate_skill_contract

# inside filter_candidates(), right before select_eligible_skills:
def filter_candidates(self, candidates: List[SkillRecord], state: StateSchema, ...):
    target_domain = state.domain  # "gymv" / "browser" / "osworld" / ...
    translated_candidates: List[SkillRecord] = []
    for skill in candidates:
        # Diagonal cells (source == target) get the identity translation
        # via the predicate_translator's default-pass behaviour, so the
        # within-domain path remains mechanism-equivalent.
        try:
            translated = translate_skill_contract(skill, target_domain=target_domain)
            translated_candidates.append(translated)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "predicate_translator failed for skill=%s target=%s: %r",
                skill.skill_id, target_domain, exc,
            )
            # Fall through to the original record so a translator bug
            # never breaks training.
            translated_candidates.append(skill)
    # rest of existing logic, but using translated_candidates
    return self._harness.select_eligible_skills(
        translated_candidates, state, ...
    )
```

### 3.3 Contract changes

* `SkillRecord.contract.effects_{add,del}` predicate strings may
  differ between the in-memory record the actor sees and the on-disk
  record the bank persists. **The hook must NOT write the translated
  record back to disk** — translation is per-evaluation only. The
  bank's source of truth is the source-domain vocabulary.
* `SkillRecord.notes` gets a `"predicate_translation: <source>->target"`
  tag (the existing `translate_skill_contract` already tags this).
  That tag must NOT trip the lifecycle's mutation detector, so verify
  the lifecycle's hash key over `(skill_id, version, contract)` does
  not include `notes`.

### 3.4 LOC + risk

| Item | Estimate |
|---|---|
| Code change | ~30 LOC in `_harness_hook.py` |
| New tests | ~80 LOC under `tests/test_trainer_harness_hook.py` (a new test class `TestPredicateTranslationInHook`) |
| Risk | **Low.** Pure function. Existing fallback (`try/except`) ensures translator bugs don't break rollouts. |
| Hot-path cost | ~1-5 µs per skill (table lookup + list comprehension) |

### 3.5 Acceptance criteria

* On a fresh trainer step with a candidate set containing a
  cross-domain skill (e.g. an image-VR skill in a gymv rollout), the
  hook's `n_candidates_after_filter` should be `>=` what it would be
  without translation. (Translation can only relax F4, never
  tighten it.)
* The `RejectedSkillSink` reason distribution should show fewer
  `EligibilityFailure.reason == 'F4 can_handle: predicate not in target vocab'`
  rejections after translation. Add a new diagnostic counter
  `n_predicate_translations_applied` to the per-step hook record.
* `test_trainer_harness_hook.py::TestPredicateTranslationInHook`
  passes hermetically (no on-disk dependencies; uses the
  table-sanity checks already in `test_predicate_translator.py`).

---

## 4. Layer A — Cross-domain admit-rate as promotion gate 🟢 IMPLEMENTED 2026-05-02 PM (commit `10b23b3`)

> **Implementation summary.** `trainer/coevolution/_transfer_hook.py`
> (new, 798 LOC) and the orchestrator wire-up at the Phase B′ block
> implement the gate. The hook runs **after** `run_promotion_step`
> finishes its writeback, re-evaluates each just-promoted skill's
> cross-domain admit rate against the configured target corpora,
> and rolls back promotions that fall below
> `crafter_transfer_admit_band[0]` on every requested target by
> dropping the skill_id rows from the per-game `skill_bank.jsonl`
> via atomic `<path>.tmp` + `os.replace`.
>
> *Architecture deviation from the design:* §4.2 sketched the gate
> running **between** the promotion driver's decisions and the
> writeback. The shipped version runs **after** writeback as a
> rollback step. This avoids refactoring `_promotion_hook.py` and
> makes the gate composable / opt-in. Functionally identical:
> rolled-back skills are out of the trainer's bank either way.
>
> *Failure-routing taxonomy* (§4.5) shipped: `configs/failure_routing.yaml`
> gained a `cross_domain_taxonomy:` block with
> `CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION` (→ generalizer / retirer)
> and `CROSS_DOMAIN_PREDICATE_VOCAB_MISMATCH` (→ repairer / drop).
>
> *Config* (§4.4) shipped verbatim plus
> `crafter_transfer_max_skills_per_cell` (default 5) for the
> `--max-skills` flag.
>
> Tests (`tests/test_transfer_hook.py`, 28 cases): pure band
> scoring, per-skill JSONL parsing (mean collapse, success-fallback,
> malformed-line tolerance), synthetic bank-run materialisation,
> atomic in-place demotion, end-to-end run with a `python -c` mock
> subprocess, dry-run mode (`apply_demotions=False`),
> subprocess-failure / timeout conservatism, summary file emission.

### 4.1 Goal

A skill whose offline cross-domain admit rate falls below a
configurable band (e.g. <15% on every cross-cluster cell) should
not graduate to PROMOTED status, regardless of within-domain
gating. Conversely, a skill that meets the §11.5.4 band on
multiple targets should bypass the LLM judge's promotion gate.

### 4.2 Extension point

[`orchestrator.py` lines 825-880](../trainer/coevolution/orchestrator.py)
— the existing Phase B′ block:

```
if config.crafter_promotion_enabled:
    crafter_step = run_crafter_step(...)
    if crafter_step.n_proposals > 0:
        promotion_step = run_promotion_step(...)   # ← existing
        # ↓ new layer-A insertion
        transfer_step = run_transfer_gate_step(
            step=step,
            run_dir=Path(config.run_dir),
            promotion_decisions=promotion_step.decisions_path,
            bank_paths=bank_paths,
            transfer_targets=config.crafter_transfer_targets,  # new
            transfer_admit_band=config.crafter_transfer_admit_band,  # new
            transfer_driver_timeout_s=config.crafter_transfer_timeout_s,  # new
        )
        # writeback_promotion now reads from transfer_step.decisions_path
        # rather than promotion_step.decisions_path directly.
```

### 4.3 New module: `trainer/coevolution/_transfer_hook.py`

Mirrors `_promotion_hook.py`'s shape (subprocess, no Python coupling):

```python
def run_transfer_gate_step(
    *, step, run_dir, promotion_decisions, bank_paths,
    transfer_targets: Sequence[str],         # e.g. ("video", "browser")
    transfer_admit_band: Tuple[float, float], # e.g. (0.15, 0.60)
    transfer_driver_timeout_s: float = 1800.0,
) -> TransferGateReport:
    # 1. Read promotion_decisions JSONL (which skills are about to promote)
    # 2. For each skill, build a synthetic <bank-run>/<corpus>/<source>
    #    skill_bank.jsonl containing JUST that skill (symlink-tree
    #    pattern from _promotion_hook.py)
    # 3. Subprocess-invoke labeling_supplement/_phase4_transfer_matrix.py
    #    --sources <synthetic bank> --targets video,browser --output <step_run_dir>
    # 4. Parse cross_domain_results/_final/<step_run_id>/_report.md
    #    OR the JSONL artefact, whichever is more stable
    # 5. For each (skill, target) cell, compare admit_rate to the band:
    #    - rate < band[0] on EVERY target -> DEMOTE (annotate decision)
    #    - rate in band on >= K targets   -> KEEP/UPGRADE
    #    - missing data (cell N-A)        -> KEEP (don't punish for
    #                                        non-applicable target)
    # 6. Write modified decisions JSONL to <step_run_dir>/transfer_decisions.jsonl
    # 7. Return TransferGateReport with per-skill verdict + run_dir
```

The cross-domain measurement subprocess runs **outside** the trainer's
Python process, exactly like `decide_promotion_gpt54.py` does today.
This preserves the "no driver imports another driver's code" rule
from `crafter-harness-orchestrator-roles.md` §6.3.

### 4.4 Config knobs (new; in `trainer/coevolution/config.py:CoevolutionConfig`)

```python
crafter_transfer_gate_enabled: bool = False      # opt-in
crafter_transfer_targets: Tuple[str, ...] = ("video", "visual_reasoning")
crafter_transfer_admit_band: Tuple[float, float] = (0.15, 0.60)  # §11.5.4 band
crafter_transfer_timeout_s: float = 1800.0
crafter_transfer_min_targets_in_band: int = 1   # K -- how many targets
                                                  # must hit the band to KEEP
```

### 4.5 Failure-routing extension

`configs/failure_routing.yaml` gains new entries under a new
`cross_domain_taxonomy:` block:

```yaml
cross_domain_taxonomy:
  CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION:
    description: |
      Skill's offline cross-domain admit rate fell below
      crafter_transfer_admit_band[0] on every target. Indicates
      the skill is over-fit to a single domain's vocabulary
      and should be either generalised or retired.
    crafter_mode: generalizer    # widen feasible_domains, retry transfer
    fallback_mode: retirer       # if generalizer fails twice, retire
  CROSS_DOMAIN_PREDICATE_VOCAB_MISMATCH:
    description: |
      Skill's contract uses predicates not in the predicate_translator
      table for the active target. Either the skill needs a contract
      patch or the table needs an entry.
    crafter_mode: repairer       # contract patch (lane-b)
    fallback_mode: drop          # if repair gated off, just record
```

### 4.6 LOC + risk

| Item | Estimate |
|---|---|
| `_transfer_hook.py` (new) | ~250 LOC mirroring `_promotion_hook.py` |
| `config.py` knobs | ~20 LOC |
| `orchestrator.py` insertion | ~30 LOC |
| `failure_routing.yaml` taxonomy entries | ~30 LOC |
| New tests (`tests/test_transfer_hook.py`) | ~200 LOC (mock subprocess + verify decision rewrite) |
| Risk | **Medium.** The Stage 6 driver's full run is slow (~1-30 min depending on bank size + N targets). Per-step gating is only viable when `crafter_promotion_enabled` already runs at low cadence (every K steps). Conda-env split: the driver subprocess will spawn helper subprocesses for browser; trainer must tolerate the helper boot cost. |

### 4.7 Acceptance criteria

* On a fresh trainer step with `crafter_transfer_gate_enabled=True`,
  `transfer_step.decisions_path` exists and is a strict subset of
  `promotion_step.decisions_path`.
* `legacy_writeback.writeback_promotion` reads from
  `transfer_step.decisions_path` rather than the upstream — verified
  by injecting a synthetic skill that fails the cross-domain band and
  confirming it is absent from the post-Phase-B' on-disk
  `skill_bank.jsonl`.
* The transfer gate is correctly skipped when
  `crafter_transfer_gate_enabled=False` (default off; opt-in).

---

## 5. Layer D — Periodic offline pass (dashboard) 🟢 IMPLEMENTED 2026-05-02 PM (commit `bf83fec`)

> **Implementation summary.** `trainer/coevolution/_dashboard_hook.py`
> (new, 649 LOC) and the orchestrator end-of-step wire-up implement
> the dashboard. Every `crafter_dashboard_every_k_steps` trainer
> steps the hook snapshots the trainer's per-game `skill_bank.jsonl`
> files into a synthetic `--game-bank-root`, subprocess-invokes
> `_phase4_transfer_matrix.py` against the configured target set,
> parses `cells.json`, computes the G1-G5 acceptance gates inline
> (memo §11.5.6), and emits a structured metrics dict the trainer
> sinks alongside its existing wandb / TB log under the
> `cross_domain/...` namespace.
>
> *G6 omission:* upper-bound conformance requires the offline
> Stage-0 `upper_bounds.csv` artefact, which is not always available
> during a live trainer run. Computing G6 in the dashboard is left
> for a follow-up that pre-stages the CSV during run setup.
>
> *Snapshot semantics:* copies, not symlinks — the trainer's banks
> are never mutated. The dashboard is a measurement layer, not a
> gate.
>
> *Cadence isolation:* a separate cadence knob from Layer A. The
> gate (Layer A) can run every K steps at high frequency while the
> dashboard runs every M*K steps at low frequency, avoiding
> subprocess load doubling.
>
> *Metrics shape:* `DashboardReport.to_metrics()` returns a flat
> `{str: float}` dict with G1-G5 verdicts encoded as
> `1.0=PASS / 0.5=soft-FAIL / 0.0=FAIL / -1.0=N-A` so wandb /
> TensorBoard don't have to special-case strings. Empty dict on
> skipped / disabled / failed runs ⇒ no metrics emitted that step.
>
> Tests (`tests/test_dashboard_hook.py`, 39 cases): cadence (8
> parametrised), verdict-to-scalar mapping (7), pure summary
> helpers, G1-G5 verdict computation across all PASS/FAIL/N-A/
> soft-FAIL paths (10), `to_metrics()` shape + empty-on-skipped
> behaviour, and end-to-end `run_dashboard_step` with mocked
> subprocess covering happy-path, missing-banks, no-targets,
> missing-cells.json, subprocess failure / timeout, and
> no-mutation-of-trainer-banks invariant.

### 5.1 Goal

Independent of per-step promotion gating, run the full Stage 6 NxN
matrix on a configurable cadence (e.g. every 100 training steps) so
the operator gets a dashboard view of how cross-domain admit rates
evolve over training. Surface the G1-G6 gate verdicts to wandb /
TensorBoard.

This is the closest analog to the existing offline-mirror pattern:
the trainer triggers the same `_phase4_transfer_matrix.py` driver
that the offline workflow already runs, but against the live bank
snapshot rather than a corpus on disk.

### 5.2 Extension point

`trainer/coevolution/orchestrator.py` end-of-step block.
A new `_dashboard_hook.py` module that:

1. Triggers iff `step % config.cross_domain_dashboard_every_k_steps == 0`
   AND `config.cross_domain_dashboard_enabled`.
2. Snapshots all per-game `skill_bank.jsonl` to a dedicated dashboard
   run dir.
3. Subprocess-invokes `_phase4_transfer_matrix.py` with the full
   N-target spread (video, visual_reasoning, osworld, browser, gymv).
4. Subprocess-invokes `_phase4_transfer_report.py` to compute G1-G6
   verdicts.
5. Reads the report's per-cell admit rates + the G1-G6 verdicts and
   emits them to `wandb` via the trainer's existing wandb client
   (`trainer/coevolution/orchestrator.py` already wires this).

The actual cross-domain measurement subprocess can run on a
**different process** (or even a different machine) than the trainer
— it just needs read access to the bank snapshot dir. This decouples
the dashboard from the training-step latency budget entirely.

### 5.3 LOC + risk

| Item | Estimate |
|---|---|
| `_dashboard_hook.py` (new) | ~200 LOC |
| `config.py` knobs | ~15 LOC |
| `orchestrator.py` insertion | ~20 LOC |
| Risk | **Low.** Decoupled from training-step latency; dashboard failures don't break rollouts. |
| Compute cost | The Stage 6 driver itself is cheap-per-cell (5-30 sec per (source, target) pair); ~1-30 min depending on N and bank size. With the dashboard cadence at every 100 steps, this is amortised. |

### 5.4 Acceptance criteria

* `wandb` (or stdout if wandb disabled) shows new metrics per
  dashboard step:
  - `cross_domain/G1_pass`, `G2_pass`, ..., `G6_pass` (booleans)
  - `cross_domain/admit_rate_<source>_<target>` (per cell)
  - `cross_domain/median_off_diagonal_admit_rate`
  - `cross_domain/n_violators_above_upper_bound`
* Dashboard step does NOT block the next training step (runs
  asynchronously via subprocess; trainer continues immediately).

---

## 6. Layer B — Cross-domain admit-rate as reward channel 🟡 DEFERRED (out of CAD scope)

> **Status (2026-05-02 PM):** Not implemented. Layers C, A, D were
> the user-requested CAD slice; Layer B remains the most ambitious
> and risk-laden of the four (reward-shaping a live RL signal off
> a measurement that cannot run on the GRPO hot path), and is
> deferred to a follow-up session. The design below is unchanged.

### 6.1 Goal

The most ambitious layer: make the GRPO buffer's per-rollout reward
include a cross-domain transferability term. Skills that the actor
exercises during a rollout should receive shaped reward not only on
within-game success but also on offline-measured cross-domain admit
rate.

This turns "transferability" from a passive promotion signal into an
active training objective.

### 6.2 Why this is risky

* **Hot-path cost.** Even amortised (e.g. once per N rollouts), the
  Stage 6 cell's 5-30s per-(source,target) cost is real. A 16-rollout
  GRPO batch with N=4 sampled cells per rollout = 16×4×5 = ~5 min
  per gradient update of pure cross-domain measurement. That's a
  serious GPU-utilisation regression.
* **Reward-hacking surface.** A skill can game cross-domain admit by
  emitting trivially-true predicates (e.g. always emitting
  `answer_emitted` regardless of whether it actually answered). The
  predicate_translator's contract is that target-domain success_fns
  evaluate against typed evidence, but verifying that property
  empirically per-rollout is non-trivial.
* **Variance.** Cross-domain admit rate over 4-cell samples is
  noisy. Combining it with GRPO's already-noisy advantage estimator
  may inflate variance enough to slow learning.

### 6.3 Layer-B can be deferred indefinitely

A reasonable position is: **ship A + C + D and stop there.**
Layers A and D give the bank-curation loop the cross-domain signal
it needs without putting cross-domain measurement inside the
training-step latency budget. Layer B is a research direction, not
a deployment requirement.

### 6.4 If pursued, the spec is

* **Where:** `trainer/coevolution/rollout_collector.py` or
  `trainer/coevolution/grpo_training.py` (TBD which is cleanest).
* **How:** A new `cross_domain_reward_signal` channel that the GRPO
  loss combines with the within-domain reward via a configurable
  weight (default 0; enable via `config.cross_domain_reward_weight`).
* **Amortisation:** Per-skill admit rates are cached in a TTL'd
  dict keyed by `(skill_id, contract_hash, target_domain)`. The
  cache is refreshed by Layer D's periodic dashboard pass (so the
  reward-channel signal is always at-most `dashboard_every_k_steps`
  steps stale).
* **Fallback:** When `cross_domain_reward_weight=0` (default), this
  layer is a no-op. Production trainer runs should keep it 0 until
  the variance question is resolved by ablation.

### 6.5 LOC + risk

| Item | Estimate |
|---|---|
| `rollout_collector.py` / `grpo_training.py` patch | ~150 LOC |
| New cache module `trainer/coevolution/_transfer_cache.py` | ~120 LOC |
| `config.py` knobs (weight, TTL, sample-K) | ~15 LOC |
| Ablation harness | ~200 LOC (sweep `cross_domain_reward_weight ∈ {0, 0.1, 0.3, 1.0}`) |
| Risk | **High.** As above. Default `weight=0` so the layer is opt-in. |

---

## 7. Sequencing rationale

| Order | Layer | Why this slot | Demo-able artefact |
|---|---|---|---|
| 1 | C | Pure function. No new deps. Unblocks the within-domain F4 from rejecting cross-domain skills on vocabulary mismatch. Lowest LOC; smoke-testable in <1h. | New rejection-distribution diagnostic showing fewer F4 rejections after translation. |
| 2 | A | Reuses the `_promotion_hook.py` subprocess pattern verbatim. The Stage 6 driver subprocess is the same one A and D both use; landing A makes B and D incremental. | Synthetic skill failing the cross-domain band is absent from on-disk `skill_bank.jsonl` after Phase B′. |
| 3 | D | Generalises A's pattern to a periodic decoupled pass. No coupling to per-step training latency. Dashboard metrics let the team observe whether A's gate is doing useful work before pursuing B. | wandb dashboard showing G1-G6 verdicts and per-cell admit rates over training. |
| 4 | B | Most ambitious; depends on D's cache being stable. Default-off knob means landing it does not commit to using it in production until ablation says yes. | Ablation showing `cross_domain_reward_weight=0.3` improves end-of-training cross-domain admit rate by ≥5pp without harming within-domain reward. |

---

## 8. Open design questions (for the implementer)

These need resolution but should not block writing the layer-C patch:

1. **Layer A: cell sampling strategy.** Per-skill, do we run all (N
   sources × M targets) cells, or sample? The bank has 13
   game-source corpora × 4 cross-domain targets = 52 cells per
   skill. At ~10 sec/cell that's ~9 min per skill. Sampling K=3
   targets per skill cuts to ~30 sec/skill. **Proposal:** start with
   K=3 random-sample (seeded by `skill_id`), revisit when
   measurement noise is characterised.

2. **Layer A: how to surface the gate's verdict to the operator.**
   The `_promotion_hook.py` writes to
   `<run_dir>/promotion_decisions_out/step_<N>/`. Layer A should
   write to a sibling `<run_dir>/transfer_decisions_out/step_<N>/`
   with the same JSONL shape extended by per-skill admit-rate fields.

3. **Layer A: interaction with `feasible_domains`.** When the
   transfer gate DEMOTES a skill on cross-domain admit-floor
   violation, should `feasible_domains` shrink? Or should we leave
   the bank record alone and only reflect this in the lifecycle's
   demotion log? **Recommendation:** the Repairer (`generalizer`
   mode) should be the only writer of `feasible_domains`, so the
   gate just signals via `failure_class=CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION`
   and the Repairer decides.

4. **Layer C: identity-translation no-op semantics.** The
   `predicate_translator` table uses `(source_domain, target_domain)`
   as the key. When `source == target`, the table has no entry and
   the function passes the contract through unchanged. Confirm this
   is the desired behaviour — diagonal cells should be
   mechanism-equivalent to the un-translated path. (A regression
   test `test_predicate_translator_identity_diagonal` already
   exists; ensure it stays green.)

5. **Layer D: dashboard run vs concurrent training.** Can the
   dashboard subprocess share the same `cross_domain_results/`
   output dir as the offline-mirror's runs? Easiest answer is no —
   write to `<trainer_run_dir>/dashboard/_phase4/<step>/` so the
   two paths never collide.

6. **Layer B: how to expose the per-rollout reward signal to the
   actor.** Two options: (a) a separate scalar in the
   `(prompt, response, reward, advantage)` GRPO tuple; (b) folded
   into the existing `reward` scalar with a configurable weight.
   Option (a) is cleaner for ablation but requires GRPO-loss-side
   changes. Defer until Layer A + D are landed and we know what we
   want.

7. **Conda-env complexity in Layer A.** The Stage 6 driver subprocess
   itself has to spawn helper subprocesses (in `osworld` and
   `browsergym` envs) when the run includes those targets. The
   trainer's process tree depth grows accordingly. Verify
   `subprocess.Popen(...)` works correctly when the parent is itself
   a `conda run` invocation (it should — `_promotion_hook.py`
   already does this with `decide_promotion_gpt54.py`).

---

## 9. Acceptance criteria — overall

For the four-layer integration to be considered complete:

* [ ] Layer C: a trainer run with cross-domain candidate skills
      shows fewer F4 rejections after translation; the new
      `n_predicate_translations_applied` counter is non-zero.
* [ ] Layer A: a synthetic skill known to fail the cross-domain
      band is absent from the post-Phase-B' on-disk
      `skill_bank.jsonl`; a synthetic skill known to pass is
      present with a `transfer_admit_rate=...` annotation.
* [ ] Layer D: at training step 100 (or whatever cadence is
      configured), wandb shows `cross_domain/G1_pass` etc.
      metrics; the metrics are within ±1pp of what the offline
      `_phase4_transfer_report.py` driver computes against the
      same bank snapshot.
* [ ] Layer B (if pursued): an ablation sweep on a 10k-step
      training run shows `cross_domain_reward_weight=0.3` improves
      end-of-training cross-domain median admit rate by ≥5pp
      without reducing within-domain `episode_reward_mean` by
      more than 2%.

---

## 10. Risk register

| Risk | Layer | Severity | Mitigation |
|---|---|---|---|
| Predicate translator bug breaks within-domain path | C | Medium | `try/except` fallback in `filter_candidates`; identity diagonal regression test |
| Cross-domain measurement subprocess is slow / unstable | A, B, D | Medium | All three layers run as subprocesses with timeouts (`crafter_transfer_timeout_s`); on timeout the trainer logs a warning and proceeds with the un-gated decisions |
| Conda-env subprocess depth issues | A, D | Low | Already exercised by `decide_promotion_gpt54.py` in `_promotion_hook.py`; pattern is proven |
| Reward-channel variance hurts learning | B | High | Default `cross_domain_reward_weight=0`; opt-in only; ablation gates the default flip |
| Bank-snapshot disk pressure (every dashboard step writes a copy) | D | Low | Snapshots are small (one `.jsonl` per game, ~MB each); rotate every 100 dashboard steps |
| Stage 6 driver schema drift breaks Layer A's parser | A | Medium | Layer A subscribes to a stable JSONL artefact (NOT the markdown report); add a `_phase4_transfer_matrix.py --emit-jsonl` flag if not already present |

---

## 11. Appendix — what I shipped today (and what it doesn't reach)

| Module shipped 2026-05-02 | Reached by trainer? |
|---|---|
| `harness/_executor_helpers/_proto.py` (JSON-RPC framing) | No |
| `harness/_executor_helpers/osworld_client.py` (HTTP client + container pool) | No |
| `harness/_executor_helpers/browser_helper.py` (Playwright subprocess) | No |
| `harness/_osworld_per_sample_executor.py:TaskAwareOsworldExecutor` | No |
| `harness/_browser_per_sample_executor.py:TaskAwareBrowserExecutor` | No |
| `harness/_video_per_sample_executor.py:TaskAwareVideoReasoningExecutor` (earlier 2026-05-02) | No |
| `harness/_vr_per_sample_executor.py:TaskAwareVisualReasoningExecutor` (earlier 2026-05-02) | No |
| `harness/predicate_translator.py` + `with_predicate_translation` | No |
| `labeling_supplement/_phase4_target_dispatch.py` rewires | No |

All seven of the per-sample executors and helpers are imported only
by `labeling_supplement/_phase4_target_dispatch.py` and the
`tests/` suite. Layer C reaches the predicate translator. Layer A
reaches the Stage 6 driver (which transitively imports everything
above). Layer D and B reach the dashboard / cache layers
respectively.

After all four layers land:

| Layer | New trainer-side file count | New trainer→harness imports |
|---|---|---|
| C | 0 (modify `_harness_hook.py`) | +1 (`predicate_translator`) |
| A | 1 (`_transfer_hook.py`) | +0 (subprocess only; no Python coupling) |
| D | 1 (`_dashboard_hook.py`) | +0 (subprocess only) |
| B | 1-2 (`_transfer_cache.py` + reward-channel patch) | +1 (cache reads `harness/predicate_translator` indirectly) |
| **Total** | **3-4 new files** | **2 new harness imports from trainer** |

Layers A, D, B all preserve the "no driver imports another driver's
code" rule by going through subprocesses; Layer C is the only
in-process Python coupling, and it's a single function import
(`translate_skill_contract`).
