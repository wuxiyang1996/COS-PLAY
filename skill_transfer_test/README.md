# `skill_transfer_test/` — harness usability test, intra-gymv ablation runner, cross-corpus skill bank lift

> **Status:** mixed.
>
> - **`extract/`** (this folder's `extract/` subpackage): full v5 run
>   shipped 2026-05-01 covering **all 2334 GPT 5.4 cold-start tasks**
>   across 6 corpora (LLM-free path; canonical `SkillBankAgent` path
>   stubbed). **1,083 skill records emitted** (46% yield); **885
>   verified** (with `verified_domains` populated). Three audit rounds,
>   18 issues triaged: 16 fixed, 1 confirmed not-a-bug (PENALIZE
>   residuals), 1 documented v0 limit (`visual_toolbench` actor
>   accuracy = lift yield). v5 ships populated effects contracts,
>   action-hops carrying granular `click` / `hotkey` / `typewrite`
>   verbs (not just `pyautogui`), label-fallback bindings for `"any"`
>   slots, TIR-Bench cluster keys, de-duped skill names AND skill_ids,
>   plus a clean final probe (zero short-protocols / type-pollution /
>   verified-domains contradictions).
>   See [`extract/README.md`](./extract/README.md).
>
> ### Combined GPT 5.4 cold-start skill coverage (across all three folders)
>
> | Folder | Path | What | Skills | Coverage |
> |---|---|---|---:|---|
> | `labeling/` | `skill_bank_out/run_20260430_030637/` | LLM-driven `SkillBankAgent` lift | 489 | 4 `env_wrappers` ROMs + 12 `gym_v` ROMs |
> | `labeling_supplement/` | (ablation pipeline; not a skill-bank emitter) | Crafter proposals / harness IO / per-episode reflections / promotion decisions | n/a | gym_v ablation experiments |
> | `skill_transfer_test/extract/` | `skill_bank_local/full_v5/` | LLM-free per-corpus lift | **1,083** (885 verified) | All 6 cross-domain corpora |
> | **TOTAL** | | | **1,572** | env_wrappers + gym_v + browsergym + osworld + 4 VR/video benchmarks |
>
> **Per-corpus full_v5 yield (lifted / cold-start):**
> browsergym 301/301, osworld 30/30, siv_bench 220/382, tir_bench
> 105/308, video_holmes 396/1000, visual_toolbench 31/313. Sequence
> corpora lift every episode; single-shot corpora lift only
> `correct=True` samples by default (use `--include-incorrect` to
> recover the rest as `verified_domains=[]` skills).
> - **`runner.py` / `cell_configs/` / harness ablation cells:** plan only,
>   no code yet. Phase 0 is reversible and can start once
>   [D1–D7](#8-decisions-locked-elsewhere) are confirmed.
>
> **Last reviewed:** 2026-05-02.
> **Cross-refs (design rationale lives there, not here):**
> [`implementation_notes/cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md) §5.5 (cross-corpus skill bank lift — Phase 1.5),
> [`implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md`](../implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md) (intra-gymv ablation roots),
> [`implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md) (Phase-5/6 measurement plan; Stages 0-6 shipped 2026-05-02),
> [`skill_transfer_test/TODO.md`](./TODO.md) (Phase-1.5b status tracker),
> [`skill_transfer_test/extract/audits/`](./extract/audits/) (Stage 0 oracle subfolder — vocab Jaccard, predicate firing, slot binding feasibility),
> [`skill_transfer_test/extract/archetype_aggregator.py`](./extract/archetype_aggregator.py) (closes TODO-1; per-corpus archetype bank emitter),
> [`labeling_supplement/_phase4_target_dispatch.py`](../labeling_supplement/_phase4_target_dispatch.py) (central per-target dispatcher),
> [`labeling_supplement/_phase5_matrix.py`](../labeling_supplement/_phase5_matrix.py) + [`_phase4_transfer_matrix.py`](../labeling_supplement/_phase4_transfer_matrix.py) + [`_phase4_transfer_report.py`](../labeling_supplement/_phase4_transfer_report.py) (Stage 5/6 NxN matrix driver + report),
> [`plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) §20 (ablations),
> [`harness/README.md`](../harness/README.md) §16, §17, §21, §22 (audit + suggested work-order),
> [`labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py) (the driver this folder wraps),
> [`labeling_supplement/decide_promotion_gpt54.py`](../labeling_supplement/decide_promotion_gpt54.py) (the harness-free promotion path).

---

## 1. What this folder is

Two cooperating measurement layers:

1. **Extraction layer (`extract/`)** — lifts GPT-5.4 cold-start
   rollouts from browsergym / osworld / 4 visual-reasoning benchmarks
   into the canonical `{report, skill}` shape that
   [`labeling/extract_skillbank_gpt54.py`](../labeling/extract_skillbank_gpt54.py)
   produces for env_wrappers / gym_v. After this layer runs, all 6
   corpora share one disk format and feed the same Phase-6
   cross-domain transfer matrix. See
   [`extract/README.md`](./extract/README.md).

2. **Ablation runner (`runner.py` + `cell_configs/`, planned)** — wraps
   shipped drivers (`dump_harness_io_gpt54.py`,
   `decide_skill_crafting_gpt54.py`, `decide_promotion_gpt54.py`) and
   emits the three reports defined in
   [`PLAN-HARNESS.md` §20.6](../plans/05-harness/PLAN-HARNESS.md).
   Five ablation cells × {intra-gymv | cross-domain} probe × four
   research questions × three report templates — nothing more, nothing
   less.

Neither layer writes the audit-trail artefacts (`BankMutationProposal`,
`SkillEpisode`, `GateVerdictPayload`, `SkillEvaluationRecord`,
`AuditRecord`, `bank_snapshot_id`). Those are owned by Crafter /
Harness / Orchestrator per
[`crafter-harness-orchestrator-roles.md` §3](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md).
On-disk JSONL is the only API.

---

## 2. Folder layout

Two subtrees: **shipped today** (extract/) and **target** (cell_configs/, runner.py, ...).

```
skill_transfer_test/
├── README.md                          ← this file
├── conftest.py                        ← (planned) pytest discovery, shared fixtures
│
├── extract/                           ← SHIPPED 2026-05-01. Cross-corpus skill bank
│   │                                    lift across 6 corpora. See extract/README.md.
│   ├── README.md
│   ├── __init__.py
│   ├── _corpus_specs.py               ← CorpusSpec registry (6 entries)
│   ├── runner.py                      ← `python -m skill_transfer_test.extract.runner ...`
│   ├── single_shot_lift.py            ← VTB / TIR-Bench / Video-Holmes / SIV-Bench
│   ├── sequence_lift.py               ← browsergym / osworld
│   └── tests/                         ← (planned) golden-file lift tests
│
├── skill_bank_local/                  ← gitignored. Output of extract/runner.py.
│   └── <run_id>/                        Per-corpus per_sample/ or per_episode/
│       ├── rollup.json                  skill_bank.jsonl files. Layout matches
│       ├── <corpus>/extraction_summary.json
│       └── <corpus>/{per_sample,per_episode}/skill_bank.jsonl
│
├── cell_configs/                      ← (planned) one YAML per §20.3 cell. No new code paths.
│   ├── a0_no_harness.yaml             ← Actor + bank retrieval only
│   ├── a1_harness_lite.yaml           ← + EligibilityFilter (G1 binding only)
│   ├── a2_harness_core.yaml           ← + G0 evidence + G2 adapter + veto + scoring
│   ├── a3_harness_transfer.yaml       ← + G3 replay + G4 shadow + G3a few-shot (task axis on)
│   └── a4_full_system.yaml            ← + G5 non-regression + promotion / rollback
│
├── runner.py                          ← (planned) ablation-cell dispatcher CLI.
│                                        Distinct from extract/runner.py.
│                                        --cells {a0,a1,a2,a3,a4,all} --probe {intra_gymv,cross_domain}
│                                        --max-episodes N --max-steps M --sources <list>
│
├── slices.py                          ← (planned) §20.5 axis builders.
│                                        in_domain_reuse / cross_domain_transfer / before_promotion
│                                        / after_promotion / easy / hard / per-game.
│
├── metrics/                           ← (planned)
│   ├── __init__.py
│   ├── validity.py                    ← Q1: invalid_invocation_rate, slot_binding_pass_rate, ...
│   ├── veto.py                        ← Q3: veto precision / recall (where ground truth exists)
│   ├── transfer.py                    ← Q2: transfer_pass_rate, regression_rate_after_transfer, ...
│   └── actor_quality.py               ← Q4: actor top-1 / top-k accuracy on Harness-eligible set
│
├── reports/                           ← (planned)
│   ├── __init__.py
│   ├── report_a_actor_decision.py     ← §20.6(a)  — per-cell × per-slice numbers
│   ├── report_b_harness_filtering.py  ← §20.6(b)  — needs G0/G2 active (Phase 2+)
│   ├── report_c_system_outcome.py     ← §20.6(c)  — overall reward / pass-rate by cell
│   └── render_summary.py              ← markdown roll-up consumed by humans
│
├── runs/                              ← (planned) gitignored. One subdir per invocation.
│   └── <ts>/                            (DISTINCT from skill_bank_local/<run_id>/)
│       ├── _run_meta.json             ← argv + cell configs + input run paths
│       ├── <cell>/<corpus>/<source>/  ← per-cell harness IO dumps (forwarded from dump driver)
│       └── reports/{a,b,c}.md         ← rendered §20.6 reports
│
└── tests/                             ← (planned)
    ├── test_cell_configs_load.py      ← every YAML parses + validates against a schema
    ├── test_metric_q1_validity.py     ← golden-file test on a single source pair
    ├── test_actor_quality_q4.py       ← golden-file test on a single source pair
    └── test_smoke_a0_a4_one_source.py ← Airstriker only, --max-episodes 2 --max-steps 5, end-to-end
```

> **Two `runner.py` files.** `extract/runner.py` (shipped) drives the
> cross-corpus lift; the top-level `runner.py` (planned) drives the
> ablation cells. They are independent dispatchers and do not share
> code.

---

## 3. CLI — extraction layer (`extract/runner.py`, shipped)

```bash
# all six corpora, 100 samples / episodes each:
python -m skill_transfer_test.extract.runner \
    --corpora all --max-samples 100

# one corpus, custom run id:
python -m skill_transfer_test.extract.runner \
    --corpora siv_bench --max-samples 200 --run-id baseline_v1

# include incorrect single-shot samples (default: skip them):
python -m skill_transfer_test.extract.runner \
    --corpora visual_toolbench --include-incorrect
```

`extract/runner.py` picks the right driver per `CorpusSpec.lift_kind`
(`single_shot` → `single_shot_lift.lift_corpus`; `sequence` →
`sequence_lift.lift_corpus_per_episode`) and writes a `rollup.json`
summarising all corpora.

Output lands in `skill_transfer_test/skill_bank_local/<run_id>/` —
a layout that mirrors
`labeling/skill_bank_out/run_<ts>/<corpus>/<source>/skill_bank.jsonl`
so downstream consumers (Phase 6 transfer matrix, the unified skill
index) walk both roots interchangeably.

**Smoke v5 (2026-05-01) numbers:** fallback rates 0.0%-5.9% (env_wrappers
gold ≈ 3%, gym_v ≈ 45.8%); slot-binding 43-66% real-bound (osworld
66% via label-fallback); protocols 5-16 hops (sequence corpora
~16 hops to capture intent + action); 0/3671 hops with corrupted
notes; 100% of records ship populated `effects_add` / `effects_del`
contracts; 10 distinct OSWorld action verbs (`click`, `press`,
`hotkey`, `doubleClick`, `typewrite`, ...) instead of the single
`pyautogui` head from v4; all `skill_id` values unique. Three
audit rounds, 18 issues triaged, 16 fixed (1 confirmed
not-a-bug, 1 v0 limit). See
[`extract/README.md`](./extract/README.md) §5-§7 for the full breakdown.

---

## 4. CLI — ablation runner (`runner.py`, planned)

```bash
python -m skill_transfer_test.runner \
    --cells a0,a1 \
    --probe intra_gymv \
    --sources Airstriker-v0 \
    --bank-run    labeling/skill_bank_out/run_<ts> \
    --actions-run labeling/skill_actions_out/run_<ts> \
    --max-episodes 2 --max-steps 5 \
    --out-root runs/
```

What it does internally — every cell is a thin shell over **existing** drivers:

| Cell | Inner invocation chain |
|---|---|
| **A0** | read `<bank-run>/<corpus>/<source>/skill_bank.jsonl` directly + read `<actions-run>/.../episode_*.json` `skill_query.selected_skill_id`; compute Q1 / Q3 / Q4 from those. **No driver call.** |
| **A1** | `dump_harness_io_gpt54.py --surface online --disable-g0 --disable-g2 --disable-transfer` |
| **A2** | `dump_harness_io_gpt54.py --surface online` (default cell — all gates default-on except transfer) |
| **A3** | `dump_harness_io_gpt54.py --surface offline` over `crafter_proposals_out/`, with `--enable-g3a-task-axis` |
| **A4** | A3 + `decide_promotion_gpt54.py --gate-mode external --gate-verdicts-run <a3_out>` |

The **top-level runner.py is a dispatcher**. It does not contain harness logic.
Harness logic lives in `harness/` and `decide_promotion_gpt54.py`. See
[`crafter-harness-orchestrator-roles.md` §8](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md):
no driver under `skill_transfer_test/` may import another driver's code or write
into another driver's output directory.

> **Phase 1.5 + cross-domain coverage.** Once
> [`extract/`](./extract/) ships records for all 6 corpora, the
> ablation runner gains a new probe value `--probe cross_domain` per
> [`cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md)
> §10.2 (the 6×6 transfer matrix). Phases 2-5 in §6 below stage the
> per-domain executors that probe needs.

---

## 5. Cell configs — schema

Every `cell_configs/*.yaml` is a flat dict. The runner translates it to
CLI flags for the underlying driver. Keep it boring.

```yaml
# cell_configs/a2_harness_core.yaml
cell_id: a2
human_label: harness-core
driver: dump_harness_io_gpt54
surface: online
gates:
  g0_evidence: true
  g1_binding:  true
  g2_adapter:  true
  g3_replay:   false      # A3+
  g3a_transfer: false     # A3+
  g4_shadow:   false      # A3+
  g5_non_regression: false  # A4+
scoring:
  fit_score:   true       # PLAN-HARNESS §1a.5 — Actor must still do work
  risk_score:  true
veto:
  enable: true            # invocation-time veto
notes: |
  A1→A2 delta = G0 evidence + G2 adapter + veto + advisory scoring.
  Phase 2 prerequisite: harness audit §21 (protocol lift) must land first
  or G0 is degenerate (zero hops in iter_hops()).
```

Cell-pair deltas that matter (canon §20.3):
A0→A1 = structural validation alone; A1→A2 = G0+veto contribution;
A2→A3 = transfer-safety contribution; A3→A4 = promotion/rollback contribution.

---

## 6. Phased rollout

Verbatim from
[`harness-usability-and-intra-gymv-transfer.md` §7](../implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md)
plus the Phase-1.5 cross-corpus addition from
[`cross-domain-transfer-suite-rollout.md` §5.5](../implementation_notes/cross-domain-transfer-suite-rollout.md);
restated here as work items.

| Phase | Maps to | Deliverable here | Cells active | Blocking prereq |
|---|---|---|---|---|
| **0** Pre-investment check | Suggested-work-order #5 (shipped) | `runner.py --cells a0,a1` end-to-end on smoke slice (Airstriker, `--max-episodes 2 --max-steps 5`); reports §20.6(a) + (c) on `in_domain_reuse` | A0, A1 | none — runs today |
| **1** Protocol lift | Suggested-work-order #6 + harness/README §21 | (no work in this folder — landed in `labeling/_decorate_skill_records.py`-style transformer) | unblocks A2 | needs upstream lift |
| **1.5** Cross-corpus skill bank lift ✅ | [`cross-domain-transfer-suite-rollout.md` §5.5](../implementation_notes/cross-domain-transfer-suite-rollout.md) | **shipped 2026-05-01.** [`extract/`](./extract/) lifts browsergym + osworld + 4 visual benchmarks into the canonical `{report, skill}` shape. LLM-free path only; canonical `SkillBankAgent` path stubbed. | unblocks `--probe cross_domain` on Phase 6 | none — runs today |
| **2** Task axis | Suggested-work-order #7 + harness/README §22 | (no work in this folder — landed in `data_structure/extensions/skill_record.py` + `harness/eligibility.py` + `harness/few_shot_adapter.py`) | unblocks A3 | needs upstream additive contract change |
| **3** gymv real executor | Suggested-work-order #8 (first half) + harness/README §16.1 | smoke through `tests/test_smoke_a0_a4_one_source.py` once executor wired | A0, A1, A2 honest; A3 transferable | needs `GymvAdapter.set_executor` plumbed from `cold_start/generate_cold_start_actor_gymv.py` |
| **4** Stage 3a probe | Suggested-work-order #8 (second half) | gymv-shape `success_fn` + `FewShotDemo` builder over `labeling/skill_actions_out/.../episode_*.json` | A3 transfer cell active | depends on Phase 2 + Phase 3 |
| **5** Full sweep + reports | Suggested-work-order #5 (offline half) | `runner.py --cells all --sources <13 games>` + §20.6(a)(b)(c) reports. **This is the first offline promotion cycle** ([`harness/README.md` §17](../harness/README.md)). | A4 reference cell | depends on Phases 1–4 |
| **6** Cross-domain follow-up | Suggested-work-order #16 | swap `--probe intra_gymv` -> `--probe cross_domain`; consume Phase-1.5 records as transfer sources via `python -m labeling_supplement._phase4_transfer_matrix` (NxN matrix driver + G1-G6 acceptance gates) | **shipped 2026-05-02 (real-env)** — Stage 1-4 executors all bind real-env wrappers when cold-start data + runtime infra are present: image-VR + video drive real VLM tools (`harness/_{vr,video}_per_sample_executor.py`); osworld drives real `pyautogui` against the live `happysixd/osworld-docker` container fleet (`harness/_osworld_per_sample_executor.py`); browser drives a real Playwright `gym.Env` via JSON-RPC subprocess in the `browsergym` conda env (`harness/_browser_per_sample_executor.py`). Stub fallback only triggers on missing cold-start data or runtime errors. Predicate translator (`harness/predicate_translator.py`) bridges game-vocab -> target-vocab effects so non-zero game->cross-domain admit rates are achievable. Outstanding work: re-run Stage 6 NxN against the now-fully-wired pipeline and regenerate `cross_domain_results/_final/run_*Z/_report.md`. | depends on Phase 1.5 (closed) + reality-grounded executors (closed 2026-05-02) |

---

## 7. Acceptance gates (don't start phase N+1 until phase N passes)

| Phase | Gate | How to check |
|---|---|---|
| **0** | A0 vs. A1 numbers differ on `in_domain_reuse`; both reports render | `cat runs/<ts>/reports/a.md` shows non-zero `delta_a0_a1` row |
| **1.5** | All 6 corpora produce non-empty `skill_bank.jsonl`; per-corpus fallback rate ≤ 10%; per-corpus mean hops in [4, 16]; OSWorld success_source distribution shows DONE / FAIL / incomplete tri-state; 100% of records ship populated effects contracts; OSWorld `actor_used_action` distribution covers ≥ 5 distinct verb heads (not collapsed to `pyautogui`); all `skill_id` values unique | `python -m skill_transfer_test.extract.runner --corpora all --max-samples 100` then inspect `<run>/rollup.json`. Passed on `smoke_v5` 2026-05-01 — see [`extract/README.md` §5](./extract/README.md). |
| **3** | `tests/test_smoke_a0_a4_one_source.py` passes: 2 episodes × 5 steps × Airstriker, no crash, A2 produces non-stub `SkillEpisode`s | `pytest skill_transfer_test/tests/test_smoke_a0_a4_one_source.py -xvs` |
| **5** | All 5 cells × 13 games complete; `bank_snapshot/<id>/` for each `(corpus, source)` is non-empty after A4 | `runs/<ts>/_run_meta.json:n_promoted_skills > 0` |

---

## 8. Decisions locked elsewhere

These are **not re-litigated here**. The runner respects whatever was decided.

| ID | Decision | Pinned in |
|---|---|---|
| D1–D7 | bridge direction, status filter, K, N, compose-reject default, failure-synth home, reflection-builder home | [`implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md` §8](../implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md) |
| D8 | one-way `legacy_writeback.py` (Promotion → per-game `skill_bank.jsonl`) | trainer wire-up convo (next plan to write up) |
| D9 | Crafter alongside Stage 4 curator (not replacing yet) | trainer wire-up convo |

---

## 9. Limitations of this configuration (state in every report)

### 9.1 Ablation runner (top-level `runner.py`)

1. **Skills never reach `ACTIVE`** under `--gate-mode offline-synthetic`. Cap is `PROVISIONAL`.
2. **No invocation veto in real time.** Cells A2–A4 measure veto *as if* it had been live; they don't actually stop a bad call.
3. **No transfer probes** until Phase 2+3 land. Cells A3/A4 numbers are not meaningful before then.
4. **`EpisodeReflection.skill_episodes = []`** because no Harness emits them ([`§7.1` mismatch #1](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md)). Q1/Q4 are computed against the synthesized `FailureTrace`s, not observed `SkillEpisode`s.
5. **`ROLLBACK` cannot fire** without batch metrics from a real gate stack. A4 reports promotion-precision only, not rollback-reactivity.
6. **Compose / Transfer / Generalize proposals are auto-rejected** in Phase 0 per D5 (cold-start `feasible_domains=["gymv"]` ⇒ Stage-0 fails). Don't read the zero count as a regression.

When Harness lands, every limit lifts mechanically. No re-architecting in this folder.

### 9.2 Extraction layer (`extract/`)

Full breakdown in [`extract/README.md` §7](./extract/README.md). Headlines:

7. **`effects_add` / `effects_del` contracts always empty.** `mine_effects` has a gaming-centric trigger table; QA-style criteria don't fire it. Effects-aware transfer matching disabled until a single-shot effect miner ships in Phase 2.
8. **`"any"`-typed slots post-bound from notes, not schema.** EVALUATE / COMPARE slots get `e\d+` references mined from hop notes — good enough for transfer matching, not as principled as ontology-aware binding.
9. **Per-episode lift granularity for sequence corpora.** Without the canonical LLM-driven segmenter, every browsergym / osworld episode becomes ONE skill instead of N sub-skills. `lift_corpus_with_agent` is stubbed pending API budget.
10. **TIR-Bench has no archetype cluster_key.** All records ship with `cluster_key=None`; archetype-grouped transfer experiments need a Phase-2 manual taxonomy or LLM-driven clustering.
11. **OSWorld success heuristic is best-effort.** `last_action=DONE` doesn't *guarantee* the task succeeded — only that the agent declared completion. Real OSWorld evaluator is the only ground truth. `report.success_source` records the heuristic for audit.

---

## 9.3. Empirical transferability assessment (2026-05-01)

> **Will skills extracted from games (`labeling/skill_bank_out/`) transfer to the cross-domain corpora that `extract/` covers?** Honest answer, calibrated against the harness's actual mechanism + the 2026-05-01 vocabulary audit + the Phase-4 within-gymv empirical numbers. This section also lives at [`implementation_notes/cross-domain-transfer-suite-rollout.md` §11.5](../implementation_notes/cross-domain-transfer-suite-rollout.md#115-empirical-transferability-assessment-2026-05-01) — read either; they should stay in sync. **Revised 2026-05-02:** game→VR-corpora estimates flipped from 0-5% to 15-35% (image) / 15-30% (video) after a `visual_reasoning_wrapper` audit confirmed the tool registry is the env (real `VisualReasoningExecutor` already wired; only a small `VideoExecutor` port pending). See the row-level revisions in §9.3.4 / §9.3.5.

### 9.3.1 Vocabulary alignment (Jaccard-overlap audit)

Game banks (489 skills, env_wrappers + gym_v) vs cross-domain bank (1,083 skills, full_v5):

| Layer | Jaccard | Implication |
|---|---:|---|
| Protocol ops (verb taxonomy: `INSPECT`, `EVALUATE`, `COMPARE`, `MOVE`, `EXECUTE`, `VERIFY`, ...) | **0.82** | shape transfers — both pipelines lifted via `_protocol_lift.py` |
| **Slot-type ontology** (`tracked_entity`, `goal_indicator`, `container_entity`, `enum`, `effect_predicate`, `any`, ...) | **1.00** | **the universal interlingua** between game and cross banks |
| Hop-level `effects_add` predicate types | 0.00 surface | **operationally bridgeable** via per-domain schema producers — see §9.3.2 |
| Contract-level effect predicates | 0.00 surface | same — disjoint vocabularies are an artefact of two effect miners running over two corpora, not a transfer barrier |

> **Reproducible.** This table is regenerated programmatically via
> `python -m skill_transfer_test.extract.audits.vocab_jaccard` (shipped
> 2026-05-02); output lands at
> [`cross_domain_results/_phase0/phase0_canonical/vocab_jaccard.json`](../cross_domain_results/_phase0/phase0_canonical/vocab_jaccard.json)
> + `vocab_jaccard.md` (44 banks discovered: 38 game, 6 cross-domain;
> protocol_ops Jaccard=0.82, slot_types=1.00, predicates_combined=0.00 —
> matches the table above).

Naive read ("0 Jaccard on predicates → no transfer") is **wrong**. The harness does not match predicates by string equality; it calls `evaluate_predicate(predicate_type, pre_state, post_state)` against a per-domain `SchemaProducer` output. `entity_appeared{label='dialog'}` fires when a modal pops up in OSWorld iff the desktop schema producer surfaces a `dialog`-labelled entity attribute — independent of whether any game skill ever used that label.

### 9.3.2 The harness IS the predicate-translation layer

Four pluggable per-domain surfaces, all keyed off the universal slot-type ontology:

1. **Adapter** ([`harness/adapters/<domain>_adapter.py`](../harness/adapters/)) translates abstract typed hops to the target's action vocabulary. The same `MOVE(direction=left)` becomes `dpad_left` in Genesis, `pyautogui.press('left')` in OSWorld, `scroll(-300, 0)` in BrowserGym — different action spaces, **same hop encoding**.
2. **Schema producer** ([`harness/gym_schema_producer.py`](../harness/gym_schema_producer.py)) translates target env state to a `StateSchema` queryable by domain-agnostic predicate evaluators.
3. **`success_fn` registry** ([`harness/gymv_success.py::register_success_fn`](../harness/gymv_success.py)) decides predicates per-domain.
4. **`FewShotAdapter.adapt(skill, target_domain, demos, target_task)`** ([`harness/few_shot_adapter.py`](../harness/few_shot_adapter.py)) is where transfer actually happens: K target-domain demos rebind the skill's `${slot}` payloads to target entities and the success_fn scores per-shot. PASS appends the target to `SkillRecord.verified_tasks` (Day-7c writer at `record_task_verification`).

**Right model**: protocol shape transfers, slot-type ontology transfers, predicate types transfer; only the action vocabulary and the entity labels are domain-specific — and those are exactly what the adapter / schema producer / FewShotAdapter rebind.

### 9.3.3 What's been empirically validated

[`labeling_supplement/harness_io_out/_phase4_report.md`](../labeling_supplement/harness_io_out/_phase4_report.md):

| Probe | k | Result | Eligibility shift |
|---|---:|---|---|
| `2048 → 2048` (sanity) | 4 | 3/3 admitted; 1 skill 0.75 (rigor signal) | 3/3 → 3/3 |
| **`2048 → tetris`** | 4 | **2/3 admitted**, 1 correctly rejected | **0/3 → 2/3** |
| **`tetris → 2048`** | 4 | **4/6 admitted**, 2 correctly rejected | **0/6 → 4/6** |

Real cross-task transfers via `FewShotAdapter`, with both correct admits AND correct rejects (predicates require source-task surface absent in target). **Mechanism works at the within-source-domain task axis.**

### 9.3.4 What's measured today (with stub-tier caveat)

All three previously-listed Phase-6 prerequisites — real adapter,
per-domain schema producer, per-domain `success_fn` + `FewShotDemo`
loader — shipped 2026-05-02 for **all four target domains**
(`visual_reasoning`, `video`, `osworld`, `browser`); see §9.3.5 for
the per-target file inventory. **Update (2026-05-02 PM):** the four
Stage 1-4 executors that originally landed as deterministic stubs
have all been retired in favour of real-env per-sample wrappers
(`harness/_{vr,video,osworld,browser}_per_sample_executor.py`).
Image-VR + video drive real VLM tools (OmniParser-v2 / Florence-2 /
video frame decode); osworld drives real `pyautogui` against the
live `happysixd/osworld-docker` container fleet over HTTP
(`harness/_executor_helpers/osworld_client.py`); browser drives a
real Playwright `gym.Env` via JSON-RPC subprocess in the
`browsergym` conda env (`harness/_executor_helpers/browser_helper.py`).
Stub fallback only triggers on missing cold-start data or runtime
errors. The remaining open work is empirical re-measurement of the
G1-G6 gates against the now-fully-wired pipeline. Cross-link:
[`implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md)
§12 (full inventory of closed gaps) +
[`cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv`](../cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv)
(Stage 0 static-feasibility upper-bounds; G6 acceptance gate
evaluates `measured <= upper_bound + 0.10`).

Calibrated estimates (informed by the 67-83% within-gymv admit rate, with mechanism-level discounts):

| Source → Target | Plausible admit rate | Why |
|---|---|---|
| `gym_v` games → `osworld` | **30-50%** | `entity_appeared{dialog}`, `attribute_changed{focused_app}`, `phase_transitioned` are natural OSWorld observables once the desktop producer emits them; reactive sensorimotor priors generalise |
| `env_wrappers:tetris` / `candy_crush` → `osworld` | 15-30% | Tile-clearing priors are very specific |
| `gym_v` games → `browsergym` | 15-30% | Same shape transfers; AXTree action space is more specific than gym joystick |
| `gym_v` games → `tir_bench` / `visual_toolbench` (image-VR) | **15-35%** | The `visual_reasoning_wrapper` tool registry IS the env — hops dispatch to live tool calls (`grounded_detect`, `describe_region`, `count_value`, `compute_ratio`, `compare_values`, `verify_claim`). Predicates fire on derivation-log entries + grounded-entity changes. Image-VR path is fully wired today via `bind_visual_reasoning_executor(adapter, image=img)` (461-LOC `VisualReasoningExecutor`). |
| `gym_v` games → `video_holmes` / `siv_bench` (video-VR) | **15-30%** | Same mechanism as image-VR via `tools_video_visual.build_video_visual_registry(frames=...)` (strict superset of the image registry — every image tool plus temporal ones: `get_frame`, `find_moment`, `track_object`, `compare_frames`). Only missing piece is a ~120-200 LOC `VideoExecutor` mirroring `VisualReasoningExecutor` against the video registry. |

**Implication** (revised 2026-05-02): the four single-shot QA corpora are step-able transfer destinations after all — the `visual_reasoning_wrapper` tool registry is the env, and game-skill predicate types map onto live tool-call observations (`entity_grounded` ↔ `grounded_detect` hit, `entity_value_increased` ↔ `count_value` increase, `phase_transitioned` ↔ first `verify_claim`). Image-VR is the **cheapest** of the five cross-domain probes (no replay executor, no schema producer, no live VM); video-VR is the second-cheapest once a small `VideoExecutor` port lands. Phase 6's transfer matrix can still use the three-experiment partition below, but with **revised admit-rate expectations**:

- **Experiment A — sensorimotor transfer (5×5)**: env_wrappers + gym_v + browsergym + osworld + video as both sources and targets. Mechanism applies; expect 15-50% admit rates.
- **Experiment B — declarative-reasoning transfer (4×4 within VR/video)**: siv_bench + tir_bench + video_holmes + visual_toolbench as both sources and targets. Same predicate-firing mechanism as Experiment A but evaluated against the tool-registry derivation log; needs `register_success_fn("visual_reasoning", make_qa_success_fn)` to plug MCQ exact-match + LLM-judge scoring.
- **Experiment C — cross-cluster (game ↔ VR/video)**: previously framed as a negative-result baseline; **the tool-registry mechanism flips this expectation upward to 15-35% (image) / 15-30% (video)** because predicate evaluators run against `_DerivationLog` and grounded-entity tables rather than pre/post env-state diffs. Game-source → VR-destination cells now fold into Experiment A's expected range; only QA-source → game-destination cells remain a genuinely-mismatched cross-cluster cell.

### 9.3.5 What unblocked the measurement (shipped 2026-05-02)

| Owed deliverable | Shipped path | LOC | Caveat |
|---|---|---:|---|
| osworld real adapter | `harness/osworld_executor.py` + `harness/osworld_success.py` | ~380 | upgraded 2026-05-02 PM: dispatcher binds `harness/_osworld_per_sample_executor.py:TaskAwareOsworldExecutor` over `harness/_executor_helpers/osworld_client.py` (HTTP to live `happysixd/osworld-docker` fleet) when cold-start tree + container fleet present; falls back to the stub otherwise |
| osworld schema producer | `harness/osworld_schema_producer.py` | 226 | -- |
| osworld few-shot demos | `harness/few_shot_demos_osworld.py` | 217 | -- |
| browser executor + producer + demos + adapter `set_executor` + `register_success_fn` | `harness/browsergym_executor.py` + `harness/browser_schema_producer.py` + `harness/few_shot_demos_browsergym.py` + `harness/browser_success.py` + `harness/adapters/browser_adapter.py` patch | ~700 across 5 | upgraded 2026-05-02 PM: dispatcher binds `harness/_browser_per_sample_executor.py:TaskAwareBrowserExecutor` (JSON-RPC subprocess hosting real Playwright `gym.Env` in `browsergym` conda env via `harness/_executor_helpers/browser_helper.py`) when cold-start tree present; falls back to the stub otherwise |
| VR demos + qa_success + cycle `--target visual_reasoning` | `harness/few_shot_demos_vr.py` + `harness/qa_success.py` + `labeling_supplement/_phase4_target_dispatch.py::_build_visual_reasoning_target` | ~310 | upgraded 2026-05-02: dispatcher binds `harness/_vr_per_sample_executor.py:TaskAwareVisualReasoningExecutor` for real per-sample image loading + VLM tool dispatch when cold-start frames present |
| video executor + demos + bind + qa_success + cycle `--target video` | `harness/video_executor.py` + `harness/few_shot_demos_video.py` + `harness/adapters/video_adapter.py::bind_video_executor` + `harness/video_qa_success.py` + dispatcher branch | ~471 | upgraded 2026-05-02: dispatcher binds `harness/_video_per_sample_executor.py:TaskAwareVideoReasoningExecutor` for real frame decode + VLM tool dispatch when cold-start `video_meta` present |

The unified Stage 6 driver
`python -m labeling_supplement._phase4_transfer_matrix` runs the full
NxN matrix today and emits
`cross_domain_results/_final/<run_id>/_report.md` carrying the G1-G6
acceptance gates ([`labeling_supplement/_phase4_transfer_report.py`](../labeling_supplement/_phase4_transfer_report.py)).
Stub-pathology caveat: G6 fires on game-target cells precisely because
the gymv stub identity-passes predicates — the gate is doing exactly
what it should. Replacing the Stage 1-4 stubs with reality-grounded
executors is the next step; admit rates measured before then are
upper-bounded by what stubs can echo.

For the canonical, severity-ranked code-level gap inventory (Tier 1: 4
stub executors; Tier 2: 2 missing `vlm_wrapper/<domain>_adapter.py` files
for video / visual_reasoning; Tier 3: per-domain runtime
predicate-translators; plus Tiers 4-6 covering Phase-1.5b TODOs, Stage 5
LLM-clustered fallback, and pre-Phase-5/6 backlog), see
[`../implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md)
§12.

---

## 10. Anti-goals (mirrors `PLAN-HARNESS.md` §20.8)

- **Do not** build a parallel transfer framework here. Both runners are dispatchers.
  Adapter logic, success scorers, and proposal mints belong in `harness/`,
  `labeling/`, and `crafter/`.
- **Do not** redefine cell semantics, gate thresholds, or ablation metrics
  inside this folder. Those live in
  [`PLAN-HARNESS.md` §5.1, §10, §20](../plans/05-harness/PLAN-HARNESS.md) and
  [`PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md).
  If a runner needs different semantics, fix the upstream plan first.
- **Do not** fork [`labeling/_protocol_lift.py`](../labeling/_protocol_lift.py)
  inside `extract/`. The verb taxonomy, slot binder, and effect miner are
  imported verbatim. If a corpus genuinely needs a different verb table, the
  fix lands in `labeling/`.
- **Do not** skip Phase 0 because A2–A4 sound more interesting. Per
  [`PLAN-HARNESS.md` §20.7](../plans/05-harness/PLAN-HARNESS.md), if `A4 − A0 ≈ 0`
  on the smoke slice the rest of the suite is not worth running — Phase 0 is
  the cheapest way to find that out.
- **Do not** collapse Q1, Q2, Q3, Q4 into one number. The point of the suite
  is that they are separable.
- **Do not** treat `extract/`'s `verified_domains` as ground truth.
  OSWorld success in particular is a heuristic (`last_action=DONE`); a real
  evaluator pass is the only authoritative signal. `report.success_source`
  records which heuristic decided each verdict.
- **Do not** write into `verified_domains` / `bank_snapshot_id` / `AuditRecord`
  from the ablation runner. Those are the Orchestrator's exclusive surface
  ([`crafter-harness-orchestrator-roles.md` §3](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md)).
  Both runners read them. The extraction layer writes its own
  `verified_domains` field on per-corpus records, which is conceptually
  separate (these are fresh records, not promotions of existing ones).

---

## 11. TL;DR

- **Folder = two cooperating measurement layers.** Extraction layer
  (`extract/`, shipped) lifts 6 corpora into the canonical `{report,
  skill}` shape. Ablation runner (top-level, planned) measures harness
  contribution via 5 cells × {intra-gymv, cross-domain} probes.
- **No new harness logic, no canonical-lift fork, no new bank writes
  by the ablation runner.**
- **`extract/` ships today**: `python -m skill_transfer_test.extract.runner --corpora all --max-samples 100`. Smoke v5 results in [`extract/README.md` §5](./extract/README.md).
- **Top-level `runner.py` = dispatcher** over `dump_harness_io_gpt54.py` + `decide_promotion_gpt54.py`. Phase 0 is dominated (1–4 hr, $0 API spend, smoke slice). Run it before investing in Phases 1–4.
- **Phases 1, 2, 3 land outside this folder** — protocol lift, task
  axis, gymv executor are upstream changes consumed here. **Phase 1.5
  is fully inside this folder** (`extract/`, shipped).
- **Phase 5 = first offline promotion cycle.** Same execution graph; satisfies
  [`harness/README.md` §17](../harness/README.md) keystone (`bank.runnable()`
  becomes non-empty).
- **Limits at §9** must be stated in every report — both the
  ablation-runner limits (§9.1) and the extraction-layer limits (§9.2).
  Don't quietly outgrow them.

If you're opening this folder to start Phase 0 (ablation runner), the next step is:

1. Confirm D1–D7 in
   [`implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md` §8](../implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md).
2. Skim
   [`labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py)
   to understand its existing CLI surface.
3. Implement `cell_configs/a0_no_harness.yaml` + `cell_configs/a1_harness_lite.yaml`
   + a 50-line top-level `runner.py` dispatcher + `tests/test_smoke_a0_a4_one_source.py`
   in smoke mode (only A0/A1).
4. Run on Airstriker, eyeball `runs/<ts>/reports/a.md`, decide whether to invest in Phase 1.

If you're opening this folder to consume `extract/` records (Phase 6
cross-domain transfer matrix), the next step is:

1. Read [`extract/README.md`](./extract/README.md) §5-§7 (smoke v5
   metrics + v0 limitations).
2. Walk `skill_transfer_test/skill_bank_local/<run_id>/<corpus>/{per_sample,per_episode}/skill_bank.jsonl`.
   Schema is in [`extract/README.md` §8](./extract/README.md).
3. Quote `report.success_source` and the §9.2 limitations in any
   report that uses these records.
