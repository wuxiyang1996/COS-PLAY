# `skill_transfer_test/extract/` — cross-corpus skill bank lift, 6 corpora

> **Status:** full v5 shipped 2026-05-01 (LLM-free path only) — covers
> **all 2,334 GPT 5.4 cold-start tasks** across 6 corpora; emitted
> **1,083 skill records** (885 verified). Three audit rounds, 18 issues
> triaged, 16 fixed, 1 confirmed-not-a-bug, 1 documented-as-v0-limit.
> Canonical `SkillBankAgent` path stubbed; gated on API budget.
>
> Stage-0 static audits (`audits/vocab_jaccard.py`, `audits/predicate_firing_static.py`,
> `audits/slot_binding_feasibility.py`, `audits/_runner.py`) shipped 2026-05-02
> (closes TODO-6); per-corpus `archetype/skill_bank.jsonl` bank kind
> (`archetype_aggregator.py`) shipped 2026-05-02 (closes TODO-1). All six
> stages of the Phase-5/6 measurement plan
> ([`implementation_notes/legacy/phase5-cross-domain-measurement.md`](../../implementation_notes/legacy/phase5-cross-domain-measurement.md))
> on `main`.
>
> **Quick reproduce:**
>
> ```bash
> python -m skill_transfer_test.extract.runner \
>     --corpora all --run-id full_v5
> # writes: skill_transfer_test/skill_bank_local/full_v5/<corpus>/<sub>/skill_bank.jsonl
> ```
> **Cross-refs (design rationale lives there, not here):**
> [`implementation_notes/cross-domain-transfer-suite-rollout.md`](../../implementation_notes/cross-domain-transfer-suite-rollout.md)
> §5.5 (cross-corpus skill bank lift)
> and §11.5 (empirical transferability assessment, mirrored as
> `skill_transfer_test/README.md` §9.3) for whether these skills transfer
> across domains,
> [`skill_transfer_test/TODO.md`](../TODO.md) (single source of truth for
> open Phase-1.5b follow-ups — archetype bank, unified index, vocab-Jaccard
> audit, three test files),
> [`labeling/readme.md`](../../labeling/readme.md) (env_wrappers + gym_v
> extractors this package generalises),
> [`labeling/_protocol_lift.py`](../../labeling/_protocol_lift.py) (the
> typed-hop lift this package imports verbatim — no fork).

---

## 1. What this package does

A **local thin** extractor that lifts GPT-5.4 cold-start rollouts from
the four extra corpora into the canonical `{report, skill}` shape that
[`labeling/extract_skillbank_gpt54.py`](../../labeling/extract_skillbank_gpt54.py)
already produces for env_wrappers / gym_v. After this package runs, all
six corpora — env_wrappers, gym_v, browsergym, osworld,
visual_reasoning (4 benchmarks), vr_video — share one disk-format and
can be fed to the same Phase-6 cross-domain transfer matrix.

It does **none** of the heavy lifting: typed-hop classification,
schema-index construction, effect mining, slot binding, and verb
taxonomy all import directly from
[`labeling/_protocol_lift.py`](../../labeling/_protocol_lift.py). What
this package adds is per-corpus glue: locating rollouts on disk,
splitting `experiences[]` (sequence corpora) or
`schema + answer_reasoning` (single-shot corpora) into prose-protocol
shape, and post-processing the lifted hops to handle the cross-corpus
quirks the canonical lift wasn't designed for.

---

## 2. File layout

```
skill_transfer_test/extract/
├── README.md                 ← this file
├── __init__.py               ← module docstring + lift-architecture recap
├── _corpus_specs.py          ← CorpusSpec registry (6 entries: name, lift_kind,
│                                modality, default_input_root, sample_glob,
│                                archetype_cluster_field, extra)
├── runner.py                 ← CLI: --corpora all|<list> --max-samples N
│                                --output-root <path> --run-id <name>
├── single_shot_lift.py       ← VTB / TIR-Bench / Video-Holmes / SIV-Bench
│                                lift driver (LLM-free, rule-based)
├── sequence_lift.py          ← browsergym / osworld lift driver. Two paths:
│                                lift_corpus_per_episode  (LLM-free; ships today)
│                                lift_corpus_with_agent   (canonical, NotImplementedError)
├── archetype_aggregator.py   ← 519 LOC. Clusters per-sample skills by
│                                provenance.cluster_key; emits
│                                archetype/skill_bank.jsonl (closes TODO-1).
│                                Shipped 2026-05-02.
└── audits/                   ← Stage 0 static-feasibility oracle (Phase-5/6
    │                           plan). Shipped 2026-05-02. Closes TODO-6.
    ├── __init__.py
    ├── _loaders.py           ← 292 LOC. Shared bank discovery walks both
    │                           labeling/skill_bank_out and skill_bank_local.
    ├── _target_vocabularies.py ← 112 LOC. Aspirational predicate vocabularies
    │                           per target_domain (visual_reasoning, video,
    │                           osworld, browser).
    ├── vocab_jaccard.py      ← 369 LOC. Reproduces §11.5.1 Jaccard tables
    │                           (game vs cross-domain protocol_ops / slot_types
    │                           / predicates) at vocab_jaccard.{md,json}.
    ├── predicate_firing_static.py ← 230 LOC. Per-skill predicate-firing
    │                           feasibility against target schema.
    ├── slot_binding_feasibility.py ← 199 LOC. Per-skill slot-type
    │                           feasibility (which `${slot}` patterns can a
    │                           given target schema bind?).
    └── _runner.py            ← 218 LOC. Orchestrates the three audits;
                                emits cross_domain_results/_phase0/<run_id>/
                                upper_bounds.csv consumed as the G6 ceiling
                                by labeling_supplement/_phase4_transfer_report.py.
```

### Output layout

```
skill_transfer_test/skill_bank_local/<run_id>/
├── rollup.json                                          ← per-corpus summary
├── browsergym/
│   ├── extraction_summary.json
│   └── per_episode/skill_bank.jsonl                     ← sequence-lift output
├── osworld/
│   ├── extraction_summary.json
│   └── per_episode/skill_bank.jsonl
├── siv_bench/
│   ├── extraction_summary.json
│   ├── per_sample/skill_bank.jsonl                      ← single-shot output
│   └── archetype/skill_bank.jsonl                       ← archetype-aggregator output
├── tir_bench/
│   ├── extraction_summary.json
│   ├── per_sample/skill_bank.jsonl
│   └── archetype/skill_bank.jsonl
├── video_holmes/
│   ├── extraction_summary.json
│   ├── per_sample/skill_bank.jsonl
│   └── archetype/skill_bank.jsonl
└── visual_toolbench/
    ├── extraction_summary.json
    ├── per_sample/skill_bank.jsonl
    └── archetype/skill_bank.jsonl
```

Three valid output flavours: `per_sample/` (single-shot lift),
`per_episode/` (sequence lift), and `archetype/` (per-cluster aggregation
emitted by `archetype_aggregator.py`; provenance carries
`bank_kind="archetype"` + `n_members` + `member_skill_ids`).

---

## 3. Two lift architectures

### 3.1 Sequence-segment lift — `sequence_lift.py`

For corpora whose rollouts are multi-step interactive episodes with an
``experiences[]`` spine: **browsergym, osworld** (and, by inheritance,
env_wrappers / gym_v — but those already have dedicated extractors).

Two modes:

| Mode | Function | Ships today | Output |
|---|---|---|---|
| LLM-free per-episode | `lift_corpus_per_episode` | yes | one `{report, skill}` record per episode (~8 hops) |
| Canonical agent-driven | `lift_corpus_with_agent` | **no** — raises `NotImplementedError` | sub-skills via `SkillBankAgent` segmentation + clustering |

The LLM-free path concatenates each step's `experiences[i].subgoal`
(falling back to `intentions`, then to a templated action line),
deduplicates byte-identical consecutive prose, caps at 8 steps, and
routes the result through `lift_protocol_to_typed_hops`. Each episode
becomes one skill — coarser than the canonical path's sub-episode
granularity, but sufficient for Phase 1.5 smoke and for transfer
experiments that match by name + protocol shape.

### 3.2 Single-shot lift — `single_shot_lift.py`

For corpora where each input file is one rollout — schema +
answer_reasoning + answer + gold_answer + correct + judge — and there
is no `experiences[]` to segment: **VTB, TIR-Bench, Video-Holmes,
SIV-Bench**.

Lift contract per sample:

```
schema           → GameSchemaIndex (auto-mined entity labels +
                   explicit e\d+ IDs)
answer_reasoning → prose hops (sentence-tokenised, narrative-prose
                   rewritten for the lift's first-token classifier;
                   originals restored to hop.notes after lift)
answer           → COMMIT-hop notes
gold_answer      → report.expected_answer
correct          → verified_status / report.overall_pass_rate
```

Records are tagged with the corpus's native `archetype_cluster_field`
(Video-Holmes `question_type`, SIV-Bench `dimension`, VTB `eval_focus`,
TIR-Bench `raw_sample.task`). TIR-Bench's `archetype_cluster_field`
landed in v4 per §6.2 Bug 11 (the v3 audit found `raw_sample.{task_type,
subset, category, ...}` were all None and only `raw_sample.task`
carried the categorical info); the full_v5 emitted records ship with
**10-11 distinct task families** (`refcoco`, `rotation_game`, `math`,
`maze`, `contrast`, `word_search`, `color`, `symbolic`,
`visual_search`, `instrument`, `ocr`).

---

## 4. CLI

```bash
# all six corpora, 100 samples / episodes each, default output root:
python -m skill_transfer_test.extract.runner \
    --corpora all --max-samples 100

# one corpus, custom run id:
python -m skill_transfer_test.extract.runner \
    --corpora siv_bench --max-samples 200 --run-id baseline_v1

# include incorrect single-shot samples (default: skip them):
python -m skill_transfer_test.extract.runner \
    --corpora visual_toolbench --include-incorrect
```

`runner.py` is a dispatcher — it picks the right driver per
`CorpusSpec.lift_kind` (`single_shot` → `single_shot_lift.lift_corpus`;
`sequence` → `sequence_lift.lift_corpus_per_episode`) and writes a
`rollup.json` summarising all corpora.

---

## 5. Empirical findings — smoke v5 + full v5 (2026-05-01)

Two parallel runs both completed 2026-05-01:

1. **`smoke_v5`** — `--corpora all --max-samples 100` (regression check)
2. **`full_v5`** — `--corpora all` (full coverage over all GPT 5.4 cold-start tasks)

### 5.0 Full GPT 5.4 cold-start coverage

> *Yield = % of cold-start tasks producing a record. Sequence corpora lift every episode (yield → 100%); single-shot corpora drop `correct=False` samples by default, so single-shot yield ≈ actor accuracy on that benchmark.*

```
Corpus           | Cold-start | Lifted | Yield  | Effects | Real-bound | Fallback | Mean hops
--------------------------------------------------------------------------------------------
browsergym       |        301 |    301 |  100%  |  100%   |    46.4%   |   0.7%   |   12.3
osworld          |         30 |     30 |  100%  |  100%   |    64.4%   |   0.6%   |   15.6
siv_bench        |        382 |    220 |   58%  |  100%   |    42.8%   |   3.8%   |    5.9
tir_bench        |        308 |    105 |   34%  |  100%   |    50.4%   |   6.8%   |    5.7
video_holmes     |       1000 |    396 |   40%  |  100%   |    45.9%   |   3.3%   |    5.9
visual_toolbench |        313 |     31 |   10%  |  100%   |    56.9%   |   5.6%   |    5.7
TOTAL            |       2334 |   1083 |   46%  |  100%   |    47.0%   |   ~3%    |   ~7
```

Output: `skill_transfer_test/skill_bank_local/full_v5/<corpus>/<bank_kind>/skill_bank.jsonl`.

**Verified skills** (`verified_domains` populated, i.e. transfer-ready):
browsergym 122/301 (41%), osworld 11/30 (37%), siv_bench 220/220
(100%), tir_bench 105/105 (100%), video_holmes 396/396 (100%),
visual_toolbench 31/31 (100%) → **885 verified skills total**.
Sequence-corpora yield is gated by actor success rate; single-shot
corpora's lifted records are all from `correct=True` cold-start
samples and so are 100% verified by construction.

**Cluster-key coverage** (transfer-experiment readiness):

```
siv_bench        : 10 archetypes (Relation Inference, Human Attribute
                   Identification, Attitude Inference, Environment
                   Perception, Facial Expression Recognition,
                   Counterfactual Prediction, Intent Inference,
                   Action Recognition, Emotion Inference, Factual
                   Prediction)
tir_bench        : 11 task families (refcoco 7, rotation_game 14,
                   math 14, maze 16, contrast 7, word_search 6,
                   color 10, symbolic 8, visual_search 13,
                   instrument 8, ocr 2)
video_holmes     : 7 archetypes (MHR 56, IMC 60, TCI 45, CTI 46,
                   TA 63, SR 74, PAR 52)
visual_toolbench : 2 archetypes (hybrid_tool_reasoning 19,
                   region_switch_qa 12)
```

### 5.1 Headline numbers (smoke_v5; for regression-test continuity)

Run: `--corpora all --max-samples 100`. Output:
`skill_transfer_test/skill_bank_local/smoke_v5/`.

| Corpus | Lift kind | Lifted | Mean hops | Fallback rate | Real-bound payloads | Effects-add coverage |
|---|---|---:|---:|---:|---:|---:|
| browsergym | sequence | 100/100 | 16.0 | 0.0% | 53.0% | 100% |
| osworld | sequence | 30/30 | 15.6 | 0.6% | 66.0% | 100% |
| siv_bench | single_shot | 60/100 | 6.0 | 3.1% | 43.8% | 100% |
| tir_bench | single_shot | 41/100 | 5.8 | 5.9% | ~47% | 100% |
| video_holmes | single_shot | 39/100 | 5.9 | 1.7% | 45.3% | 100% |
| visual_toolbench | single_shot | 13/100 | 5.8 | 5.3% | ~48% | 100% |

For comparison: env_wrappers gold standard fallback ≈ 3%, gym_v ≈
45.8% (the cautionary baseline in
[`cross-domain-transfer-suite-rollout.md`](../../implementation_notes/cross-domain-transfer-suite-rollout.md) §5.5.6).

Sequence-corpora protocols doubled in size from v3 (8 → 16 hops)
because each kept step now produces TWO hops: a GATHER/REASON hop for
the agent's `subgoal` text and a COMMIT hop for the `action`
actually emitted. Without the second hop, OSWorld lost 99% of action
information (subgoal text is abstract; only 3/234 v3 osworld hops
referenced the actor verb). v4 osworld hops carry concrete
``pyautogui.click(464, 47)`` / ``pyautogui.hotkey('ctrl', 'shift', 't')``
payloads.

### 5.2 OSWorld success extraction (tri-state)

OSWorld's episode-level `outcome=True` is a misleading default; the
trustworthy signal is `experiences[-1].action`:

| Source | Verdict | Count (n=30) | `verified_domains` |
|---|---|---:|---|
| `osworld:last_action=DONE` | True (passed) | 11 | `["desktop"]` |
| `osworld:last_action=FAIL` | False (failed) | 4 | `[]` |
| `osworld:last_action=incomplete` | None (hit max steps) | 15 | `[]` |

Recorded in `report.success_source` for every record so downstream
consumers can audit the heuristic.

### 5.3 Effects contracts (now populated)

v4 ships a per-corpus effect miner so contracts are never empty.

**Sequence corpora** (`mine_sequence_episode_effects`):

| Predicate | Args | Bucket | Source |
|---|---|---|---|
| `task_status` | `value=success\|failure\|incomplete, corpus=...` | add (success/incomplete) / del (failure) | episode outcome |
| `last_action` | `verb=DONE\|FAIL\|click\|...` | add (success) / del (failure) | last `experiences[-1].action` head |
| `actor_used_action` | `verb=...` | add | distinct action verbs across episode |
| `visited_entity` | `label=..., n_subgoal_refs=...` | add | entity labels referenced ≥2× in subgoals |

**Single-shot corpora** (`mine_single_shot_effects`):

| Predicate | Args | Bucket | Source |
|---|---|---|---|
| `answer_emitted` | `value=...` | add | always |
| `answer_matches_gold` | `gold=...` | add (when `correct=True`) | gold comparison |
| `answer_diverged_from_gold` | `answer=..., gold=...` | del (when `correct=False`) | gold comparison |
| `entity_grounded` | `e=..., ontology=...` | add | each `e\d+` cited in reasoning AND in schema |

### 5.4 Verb distribution (sample, smoke_v4)

```
browsergym:  INSPECT 787, EXECUTE 800 (action hops), COMPARE 12, ...
osworld:     INSPECT, EXECUTE, COMPARE, MOVE, EVALUATE, CONTINUE
siv_bench:   INSPECT 155, VERIFY 61, EXECUTE 60, EVALUATE 34, COMPARE 33
video_holmes: INSPECT 108, VERIFY 41, EXECUTE 39, COMPARE 18, EVALUATE 17
```

Sequence corpora's EXECUTE count is now ~half of all hops because the
v4 action-hop addition explicitly synthesises `Execute the action X.`
prose for each kept step, which the classifier resolves to EXECUTE in
first-token mode.

### 5.5 Cluster-key coverage (single-shot corpora)

```
siv_bench (10 keys, 60 records): Relation Inference, Human Attribute
   Identification, Counterfactual Prediction, Action Recognition,
   Intent Inference, Factual Prediction, Environment Perception,
   Attitude Inference, Emotion Inference, Facial Expression Recognition
video_holmes (7 keys): MHR, IMC, TCI, CTI, TA, SR, PAR
visual_toolbench (2 keys): hybrid_tool_reasoning, region_switch_qa
tir_bench (10 task families, raw_sample.task): refcoco, rotation_game,
   math, maze, contrast, word_search, color, symbolic, visual_search,
   instrument
```

(v3 had `tir_bench cluster_key=None` for all records; v4 fixed by
pointing `archetype_cluster_field` at `raw_sample.task` per the
2026-05-01 audit.)

---

## 6. Bugs found by post-smoke audits (2026-05-01)

Two audit rounds: **round 1** (v1 → v3) caught 8 bugs and fixed 6.
**Round 2** (v3 → v4) caught 7 more and fixed all 7 — including the
two previously-deferred items.

### 6.1 Round 1 (v1 → v3)

| # | Bug | Severity | Status |
|---|---|---|---|
| 1 | `notes` polluted with synthetic verb prefixes (`"penalize that the page is …"`) — the classifier-friendly rewrite leaked into the visible record | Med | **fixed v3** — `restore_original_notes` post-pass |
| 2 | Over-aggressive `PENALIZE` rule fired 128× in 100 browsergym episodes (`does\s+not` / `cannot` / `no\s+evidence` → false positives on descriptive prose) | Med | **fixed v3** — pattern narrowed to explicit avoidance heads (`avoid`, `reject`, `skip`, `discard`, `exclude`, `ignore`) |
| 3 | Browser/OSWorld schema_index was empty — canonical `_parse_schema_block_entities` requires `ontology=` attribute; browser uses `type=element` only, osworld uses `role=…` | High | **fixed v3** — `parse_schema_entities_cross_corpus` handles all 3 declaration formats; browser 0 → 52-59 labels, osworld 0 → 68-76 labels per episode |
| 4 | `"any"`-typed slots (EVALUATE.subject/.criterion, COMPARE.lhs/.rhs) never bind by canonical-lift design — 95% of payloads were `${slot}` placeholders | Med | **fixed v3 (e\d+)**, **extended v4 (label fallback)** — see #12 |
| 5 | `effects_add` / `effects_del` contracts always empty — `success_criteria` strings don't trigger any `_PREDICATE_TRIGGERS` (gaming-centric) | Med | **fixed v4** — `mine_single_shot_effects` + `mine_sequence_episode_effects` (see §5.3) |
| 6 | Browser/osworld protocols 23-61 hops long (env_wrappers gold is 4-10) — no LLM segmenter, consecutive duplicate subgoals | High | **fixed v3** — `_extract_prose_per_step` dedups + caps at `max_steps=8`; v4 added action hops (length 15-16, still under v1 baseline) |
| 7 | OSWorld `outcome=True` for **every** episode including ones ending with `FAIL` action — `verified_domains` was lying | High | **fixed v3** — `_episode_outcome_success(corpus_name="osworld")` reads `experiences[-1].action` (tri-state) |
| 8 | Skill-name dups (8× same name in siv_bench) | Low | **fixed v4** — `_derive_skill_name` appends a stable 5-char hash of `task_id` |

### 6.2 Round 2 (v3 → v4)

| # | Bug | Severity | Status |
|---|---|---|---|
| 9 | Trailing whitespace in browser entity-label bindings — 84.6% of browser real bindings were `'navigation '`; `_parse_entity_decl_attrs` only stripped one quote-pair layer | Med | **fixed v4** — paired-quote stripping + idempotent whitespace normalisation; 84.6% → 0% dirty bindings |
| 10 | Action information lost in OSWorld — `subgoal` prose is abstract, only 3/234 v3 osworld hops referenced the actor verb. Browser preserved ~91% in subgoal text but still benefited from explicit hops | High | **fixed v4** — each kept step now becomes intent_hop + action_hop carrying concrete `pyautogui.click(...)` / `click("e20")` payloads |
| 11 | TIR-Bench `cluster_key=None` for all records — `raw_sample.{task_type, subset, category, ...}` are all None; only `raw_sample.task` carries the categorical info | Med | **fixed v4** — `archetype_cluster_field = "raw_sample.task"`; 10 distinct task families now populated |
| 12 | Browser/OSWorld `"any"`-slot binding limited to `e\d+` — reasoning prose mostly cites entity labels not canonical IDs, so 100% of `"any"` slots remained unbound for sequence corpora | Med | **fixed v4** — `_bind_entity_refs_in_payloads` falls back to schema-label matching when no `e\d+` is found; osworld 31.2% → 66.0%, browser 50.3% → 53.0% |
| 13 | Post-binder ignored slot types — entity labels were filling `direction` enum slots (e.g. `MOVE.direction='Close'`) | Low | **fixed v4** — post-binder skips `enum`/`effect_predicate` slot types; 0 mis-bound enum slots in v4 |
| 14 | PENALIZE residuals (3-9 occurrences across siv_bench / video_holmes) | n/a | **not a bug** — audit confirmed all are true positives ("There is no evidence …" sentences) |
| 15 | TIR-Bench had 39% skill-name collisions (16/41 records); siv_bench 12% (7/60) | Low | **fixed v4** alongside #8 (hash-suffix in skill name) |

### 6.3 Round 3 (v4 → v5)

| # | Bug | Severity | Status |
|---|---|---|---|
| 16 | `visual_toolbench lifted=13/100` (87% drop). Cause: `correct=False` for ~87% of cold-start samples; lift filters them by default | n/a | **v0 limit** — not a code bug, the actor's 13% accuracy is the floor; surface via `--include-incorrect` to extract unverified skills |
| 17 | OSWorld action-verb head extraction collapsed every `pyautogui.click(...)` / `pyautogui.hotkey(...)` / `pyautogui.typewrite(...)` to head `pyautogui` (split on the FIRST dot, not the last) — the actually-useful verbs were lost. Same bug in `last_action`. v4 distribution was `{pyautogui:30, DONE:11, WAIT:8, FAIL:4}` — a constant, not a fingerprint | **High** | **fixed v5** — `_action_verb_head` keeps the last dotted segment before the call paren; v5 distribution is `{click:30, press:28, hotkey:19, DONE:11, WAIT:8, doubleClick:5, typewrite:4, FAIL:4, rightClick:1, double_click:1}` |
| 18 | `skill_id` collisions on cold-start rerun records (1 each in tir_bench / visual_toolbench): re-runs of the same task produced duplicate `skill_id` values | Low | **fixed v5** — `lift_corpus` / `lift_corpus_per_episode` track `seen_skill_ids` and append `#run<N>` suffix on collision; original task_id preserved in `provenance.base_task_id` |

### 6.3a Round 4 (2026-05-02) — Phase-5/6 Stage 0 audit suite

Not a code-bug round: on 2026-05-02 the
[Phase-5/6 measurement plan](../../implementation_notes/legacy/phase5-cross-domain-measurement.md)'s
Stage 0 oracle shipped, reproducing the §11.5.1 / `skill_transfer_test/README.md` §9.3.1
Jaccard numbers programmatically and adding two new static-feasibility
checks. All four scripts live under `extract/audits/`:

| # | Script | What it does |
|---|---|---|
| A1 | `audits/vocab_jaccard.py` (369 LOC) | Walks every bank under `labeling/skill_bank_out/` + `skill_transfer_test/skill_bank_local/full_v5/`; computes per-cluster (game vs cross-domain) `protocol_ops` / `slot_types` / `hop_predicates` / `contract_predicates` / `predicates_combined` Jaccard; emits `vocab_jaccard.{md,json}`. Reproduces 0.82 / 1.00 / 0.00 / 0.00 / 0.00 against §11.5.1. |
| A2 | `audits/predicate_firing_static.py` (~150 LOC) | Per-skill predicate-firing feasibility: which target schemas can satisfy each skill's contract predicates? Emits `predicate_firing_per_skill.jsonl` + `predicate_firing_static.json`. |
| A3 | `audits/slot_binding_feasibility.py` (~150 LOC) | Per-skill slot-type feasibility: can each `${slot}` placeholder bind under each target's entity vocabulary? Emits `slot_binding_per_skill.jsonl` + `slot_binding_feasibility.json`. |
| A4 | `audits/_runner.py` (~150 LOC) | Orchestrates A1-A3; emits `cross_domain_results/_phase0/<run_id>/upper_bounds.csv` consumed by `labeling_supplement/_phase4_transfer_report.py` as the G6 acceptance-gate ceiling (`measured <= upper_bound + 0.10`). |

### 6.4 Round-3 final probe (v5) — checks that came back clean

| Probe | Result |
|---|---|
| Records with <4 hops | 0 across all 6 corpora |
| `verified_domains` ↔ `judge_correct` contradictions | 0 across all 6 corpora |
| Payload value types | 100% `str` (no `None` / `int` / `list` pollution) |
| `entity_grounded.ontology` = MISSING/unknown | 0 across single-shot corpora; all resolve to `tracked_entity` / `goal_indicator` / `selectable_entity` / etc. |
| `visited_entity` empty/whitespace labels | 0 across browser + osworld; labels are sensible words (`window`, `menu`, `help`, `close`, `chrome`, `files`, `media`, `video`, `file`, ...) |
| Per-record duplicate effect predicates | 0 across all 6 corpora |
| Hop notes empty / corrupted | 0 across all 6 corpora |

### 6.5 Effect of fixes — cumulative across 5 smoke runs

```
                          v1       v2      v3       v4       v5
fallback_rate browsergym  10.3%   41.3%   0.0%     0.0%     0.0%
              osworld      9.7%   43.8%   1.3%     0.6%     0.6%
              siv_bench    7.3%   16.0%   3.1%     3.1%     3.1%
              tir_bench    3.8%   17.7%   5.9%     5.9%     5.9%
              video_holmes 7.4%   17.8%   1.7%     1.7%     1.7%
              VTB          9.3%   24.0%   5.3%     5.3%     5.3%
real_bound %  browsergym   0%     0%      50.3%    53.0%    53.0%
              osworld      0%     0%      31.2%    66.0%    66.0%
              VR/video    1-9%    1-9%   40.7-48%  43.8-48.1% same
effects-add %               0%      0%      0%      100%     100%
mean_hops     browsergym   32     ~24     8         16       16
              osworld      48.7   ~23     7.8       15.6     15.6
notes poll.   all corpora   N      N      0%        0%       0%
osworld       100%(bad)    bad    bad    11/30(ok) 11/30(ok) 11/30(ok)
   verified_domains
tir_bench cluster_key       —      —     None      10 fam.  10 fam.
skill-name dups (siv)        8      8      7         0        0
   "          "  (tir)       ?      ?     16         0        0
osworld actor verbs        --     --     --       1 (bad)   10 (good)
   distinct heads
skill_id dups (tir/VTB)    --     --     --        2         0
```

---

## 7. v0 limitations (state in every report)

Every record this package emits today carries the limitations below.
None of them are bugs — they are deliberate v0 trade-offs. When the
canonical `SkillBankAgent` path lands (§5.5.4 Phase 1.5 deferred
deliverable in
[`cross-domain-transfer-suite-rollout.md`](../../implementation_notes/cross-domain-transfer-suite-rollout.md)),
limitations 1, 3, 4 lift mechanically.

1. **Effects contracts are now populated, but with episode-level
   predicates only.** v4 ships per-corpus effect miners
   (`mine_single_shot_effects`, `mine_sequence_episode_effects`) so
   `effects_add` / `effects_del` are no longer empty — every record
   carries `task_status`, `last_action`, `actor_used_action`,
   `visited_entity` (sequence) or `answer_emitted`,
   `answer_matches_gold`, `entity_grounded` (single-shot). The
   canonical `mine_effects` from `labeling/_protocol_lift.py` still
   doesn't fire on QA / browser / desktop reasoning prose because its
   `_PREDICATE_TRIGGERS` table is gaming-centric; **per-hop**
   predicate mining (vs the per-record / contract-level miners
   shipped today) is a Phase-2 deliverable. Effects-aware transfer
   matching is therefore enabled for episode-level scoring but not
   yet for hop-level rigor.

2. **`"any"`-typed slots are post-bound heuristically.** The
   canonical lift deliberately leaves EVALUATE.subject /
   COMPARE.lhs / etc. as `${slot}` placeholders. The post-binder in
   `_bind_entity_refs_in_payloads` substitutes (a) `e\d+` references
   mined from the hop's notes, then (b) entity labels from the
   schema_index that appear in the notes. v4 also respects
   `enum`/`effect_predicate` slot types so directional / predicate
   slots aren't filled with entity labels. Good enough for Phase 6
   transfer matching, but not as principled as an ontology-aware
   binder.

3. **Per-episode lift granularity for sequence corpora.** Without an
   LLM-driven segmenter, every browsergym / osworld episode becomes
   ONE skill (now 15-16 hops alternating intent + action). The
   canonical `SkillBankAgent` path produces N sub-skills per episode
   via segment + cluster + materialise — not wired here yet.

4. **`verified_domains` only populated for fully-passed episodes.**
   OSWorld `incomplete` (15/30 episodes hit max steps) → `verified_domains=[]`.
   This is correct, but it means the verified yield from sequence
   corpora is materially smaller than the lifted yield.

5. **BrowserGym actor success rate is low on AssistantBench.** First
   100 alphabetical episodes: 0/100 verified. Across the full corpus:
   122/301 (~40%). Not a lift bug — the cold-start GPT-5.4 actor just
   fails most early-list tasks.

6. **OSWorld success heuristic is best-effort.** `last_action=DONE`
   doesn't *guarantee* the task succeeded — only that the agent
   declared completion. A real OSWorld evaluator pass is the only
   ground truth. `report.success_source` records which heuristic
   decided each verdict so downstream consumers can audit.

7. **Effects predicates are episode-level, not hop-level.** The
   v4 effect miners attach predicates to the skill record's
   `contract`, not to individual hops. A learned, hop-level
   predicate miner is a Phase-2 deliverable.

---

## 8. Output schema (locked, smoke v3)

Each line of `skill_bank.jsonl` is one `{report, skill}` record. The
fields below are stable; downstream consumers may rely on them.

```json
{
  "report": {
    "skill_id": "<task_id or episode_id>",
    "n_instances": 1,
    "n_steps": 8,                       // sequence corpora only
    "overall_pass_rate": 1.0 | 0.0 | null,  // null = osworld:incomplete
    "lift_stats": {
      "n_hops": 8,
      "n_first": 8, "n_rescued": 0, "n_fallback_exec": 0,
      "verbs": {"INSPECT": 6, "COMPARE": 1, "EXECUTE": 1},
      "fallback_rate": 0.0
    },
    "expected_answer": "D",             // single-shot only
    "model_answer": "D",                // single-shot only
    "judge_correct": true | false | null,
    "success_source": "osworld:last_action=DONE",  // sequence only
    "actor_verb_distribution": {"pyautogui": 4, "DONE": 1},  // sequence only
    "reward_total": 0.0,                // sequence only
    "n_explicit_entity_refs": 3         // single-shot only
  },
  "skill": {
    "skill_id": "<unique>",
    "name": "answer/<question_head>" | "episode/<goal_head>",
    "strategic_description": "<question or task instruction>",
    "applicable_domains": ["video"],
    "feasible_domains": ["video"],
    "verified_domains": ["video"] | [],
    "verified_tasks": ["<task_id>"] | [],
    "evidence_role": "COMMIT",
    "protocol": [                       // typed hops; canonical shape
      {"op": "INSPECT", "payload": {"target": "e1"},
       "slot_types": {"target": "container_entity"},
       "preconditions": [], "effects_add": [], "effects_del": [],
       "evidence_role": "GATHER",
       "notes": "<original prose, not the rewritten version>",
       "lift_mode": "first" | "rescued" | "fallback_exec"}
      // ...
    ],
    "protocol_raw": {"steps": [...], "success_criteria": [...], "abort_criteria": []},
    "contract": {
      "effects_add": [
        {"type": "answer_emitted", "args": {"value": "D"},
         "from_phrase": "agent emitted answer: D"},
        {"type": "answer_matches_gold", "args": {"gold": "D"},
         "from_phrase": "answer matches gold: D"},
        {"type": "entity_grounded", "args": {"e": "e1", "ontology": "tracked_entity"},
         "from_phrase": "reasoning chain cites schema entity e1"}
      ],
      "effects_del": []
    },  // §7 limitation 1 — episode-level only; per-hop predicates pending.
        // The discriminator field is "type"; each entry carries a
        // human-readable "from_phrase" recording which sentence/clause
        // triggered the predicate (audit aid).
    "provenance": {
      "corpus": "siv_bench",
      "benchmark": "siv_bench",
      "modality": "video",
      "bank_kind": "per_sample" | "episode" | "archetype",
      // archetype records add: "n_members": int, "member_skill_ids": [...],
      // "representative_skill_id": "...", "aggregation": "direct",
      // "aggregator_version": "v1", "aggregated_at": "...".
      "cluster_key": "Relation Inference" | null,
      // ... (corpus-specific extras)
    },
    "tags": ["siv_bench", "video", "single_shot", "Relation Inference"],
    "n_instances": 1,
    "source_type": "single_shot_qa" | "sequence_per_episode",
    "status": "draft", "retired": false, "version": 1,
    "created_at": "...", "updated_at": "..."
  }
}
```

---

## 9. What this package is NOT

- **Not a fork of `labeling/_protocol_lift.py`.** Heavy lifting is
  imported. If a corpus genuinely needs a different verb taxonomy or
  effect-predicate set, the fix lands in `labeling/`, not here.
- **Not a runtime / executor.** Lifted records are static JSONL —
  the harness's transfer machinery consumes them downstream.
- **Not a replacement for the canonical `SkillBankAgent` path.**
  `lift_corpus_with_agent` is a stub deliberately — the LLM-free
  per-episode mode is for measurement plumbing, not for production
  skill quality.
- **Not authoritative for `verified_domains`** beyond the
  `success_source` heuristic. OSWorld in particular needs a real
  evaluator pass to be ground-truth.

---

## 10. TL;DR

- **Six corpora, two architectures, ~3,820 LOC total, no canonical-lift
  fork.** Single-shot for VR/video QA; sequence-segment per-episode
  for browser/osworld. Breakdown: ~1,870 LOC `extract/` lift drivers
  (`single_shot_lift.py` 792 + `sequence_lift.py` 816 + `runner.py` 108 +
  `_corpus_specs.py` 124 + `__init__.py` 27 + `tests/`), 519 LOC
  `archetype_aggregator.py` (clusterer; closes TODO-1; shipped
  2026-05-02), ~1,440 LOC `audits/` subpackage (Stage 0
  static-feasibility oracle for the Phase-5/6 measurement plan;
  shipped 2026-05-02).
- **Headline result: full GPT 5.4 cold-start coverage** for the four
  corpora this package owns —
  1,083 records (885 verified) across **`browsergym + osworld + 4
  VR/video corpora`** (VTB, TIR-Bench, Video-Holmes, SIV-Bench).
  Note: the env_wrappers + gym_v records are emitted by
  [`labeling/extract_skillbank_gpt54.py`](../../labeling/extract_skillbank_gpt54.py),
  not by this package; this README's scope is the **non-game** half
  of the unified bank.
- **smoke_v5 + full_v5 (2026-05-01)** cover all 2,334 GPT 5.4
  cold-start tasks across 6 corpora; 1,083 records / 885 verified;
  fallback rates 0.0%-5.9% (env_wrappers gold ≈ 3%, gym_v ≈ 45.8%);
  real-bound payloads 53-66% on sequence corpora; records carry the
  canonical `{report, skill}` shape.
- **Eighteen issues found by three post-smoke audit rounds; sixteen
  fixed (Bugs 1-13, 15, 17, 18), one confirmed-not-a-bug (#14
  PENALIZE residuals = true positives), one documented as a v0 limit
  (#16 visual_toolbench low yield = actor accuracy floor).** Per-hop
  effect predicates and per-episode sequence-corpus granularity
  remain v0 trade-offs that lift when the canonical SkillBankAgent
  path is wired.
- **Run with**
  `python -m skill_transfer_test.extract.runner --corpora all --run-id full_v5`.
  Output lands in `skill_transfer_test/skill_bank_local/<run_id>/`.
- **Don't read this folder as Phase-6 ground truth.** It's a
  measurement layer for transfer experiments. §7 limitation 7
  (episode-level effect predicates only) and the OSWorld success
  heuristic (Bug 7 / §7 limitation 6) must be quoted in any report
  that consumes these records.

---

## 11. Stage 0 oracle (Phase-5/6 measurement plan)

The four audit scripts in `audits/` ship the static feasibility oracle
that Stage 1-6 of the
[Phase-5/6 measurement plan](../../implementation_notes/legacy/phase5-cross-domain-measurement.md)
validates against. Run them via:

```bash
python -m skill_transfer_test.extract.audits._runner
```

Outputs land at `cross_domain_results/_phase0/<run_id>/upper_bounds.csv`
(gitignored). Every later stage's measured admit rate must satisfy
`measured <= upper_bound + slack(0.10)` (the G6 acceptance gate
evaluated by
[`labeling_supplement/_phase4_transfer_report.py`](../../labeling_supplement/_phase4_transfer_report.py)).
