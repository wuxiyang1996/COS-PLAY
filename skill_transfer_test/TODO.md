# `skill_transfer_test/TODO.md` — Phase 1.5b deferred items

> **Scope:** open follow-ups from Phase 1.5 (LLM-free cross-corpus skill
> bank lift, shipped 2026-05-01 — `skill_transfer_test/extract/`,
> 1,083 records / 885 verified across 6 corpora).
>
> **Single source of truth for status only.** Each item below is described
> in detail in the design memo
> [`implementation_notes/cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md)
> §5.5.4 / §5.5.4a / §5.5.7a / §11.5.1. Treat the memo as authoritative for
> *what* each item is and *why*; treat this file as authoritative for
> *whether it has shipped*. When you ship one, flip the checkbox here and
> link the commit/PR — do **not** delete the row from §5.5.4a in the memo
> (the memo records the original spec; this file records execution state).
>
> **Status legend:** `[ ]` open · `[~]` in progress · `[x]` shipped · `[-]` cancelled (superseded).
>
> **Last updated:** 2026-05-02 (file created consolidating the seven items
> previously scattered across §5.5.4a, §5.5.7a, and §11.5.1 of the rollout
> memo; later same-day, the rollout memo §11.5.4 estimates were revised
> upward for game→VR/video transfer after a `visual_reasoning_wrapper`
> audit; later still, **TODO-6 closed** as part of Stage 0 of the
> [Phase-5/6 measurement plan](../implementation_notes/legacy/phase5-cross-domain-measurement.md)
> — three audit scripts shipped under `skill_transfer_test/extract/audits/`;
> later still on the same day, **TODO-1 closed** as part of Stage 5 —
> `extract/archetype_aggregator.py` shipped + `labeling_supplement/_phase5_matrix.py`
> driver shipped, producing the within-VR/video 4x4 transfer matrix; later
> still on the same day, **all 6 stages of the Phase-5/6 plan shipped** —
> Stage 6 added `labeling_supplement/_phase4_transfer_matrix.py` (full NxN
> driver) + `labeling_supplement/_phase4_transfer_report.py` (Experiment-A/B/C
> + Stage 0 upper-bound comparison + 6 acceptance gates); 2026-05-02 (later still),
> **TODO-2 / TODO-3 / TODO-4 / TODO-5 all closed in one batch** —
> `extract/_unify.py` (cross-domain analogue of `labeling.unify_skill_index`
> tagged with corpus + bank_kind for all 6 corpora) shipped, plus three
> test files under `extract/tests/` covering corpus-spec field validation
> (20 tests), single-shot lift envelope shape across 4 visual/video
> benchmarks (12 tests, no fixture files needed — uses real samples
> opportunistically with skip-on-missing), and runner end-to-end smoke
> with `_unify` round-trip (2 tests). The same batch also fixed the
> Phase-5/6 §12.1 Tier 1 critical gap by wiring per-sample PIL.Image
> loading into the visual_reasoning dispatcher
> (`harness/_vr_per_sample_executor.py` +
> `labeling_supplement/_phase4_target_dispatch.py` plumbing) so Stage 1
> now binds a real `VisualReasoningExecutor` per task instead of leaving
> the adapter on its inherited deterministic stub.

---

## Phase 1.5b — open

### `[x]` TODO-1 — `extract/archetype_aggregator.py` *(shipped 2026-05-02)*

Emit the **archetype bank kind** for VR/video corpora (clusters of
per-sample skills). Two strategies, selectable per `CorpusSpec`:

- **direct** — group by `provenance.cluster_key` (mirrored from
  `CorpusSpec.archetype_cluster_field` at lift time: `eval_focus`
  for VTB, `task` for TIR-Bench, `question_type` for Video-Holmes,
  `dimension` for SIV-Bench). Shipped.
- **LLM-clustered** — call `gpt-5.5` with the sample question + extracted
  protocol, ask for a topic tag, then cluster by tag (VTB, TIR-Bench).
  **Not shipped**; required only when `direct` produces fewer than 3
  archetypes (currently only VTB, which has 2 distinct
  `eval_focus` values). Tracked separately as a Phase-2 enhancement.

| Field | Value |
|---|---|
| **Output** | `skill_transfer_test/skill_bank_local/<run_id>/<corpus>/archetype/skill_bank.jsonl` (one `{report, skill}` envelope per cluster_key, loaders-compatible with `labeling_supplement._harness_io_helpers.load_bank_records`) |
| **Acceptance** | `tir_bench` (11), `video_holmes` (7), `siv_bench` (10) — **PASS** (≥3 archetypes). `visual_toolbench` ships 2 — **FAIL** under `direct`; gated on the LLM-clustered fallback (Phase-2). The matrix-mode driver in [`labeling_supplement/_phase5_matrix.py`](../labeling_supplement/_phase5_matrix.py) handles 2-archetype VTB transparently (cells with VTB-as-source contribute 2 verdicts each). |
| **Memo refs** | §5.5.2 (granularity), §5.5.4 (file table), §5.5.4a (deferred row), §5.5.7a row 4, [`implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md) §8 |
| **Shipped LOC** | 376 (`archetype_aggregator.py`) |
| **Shipped as part of** | Stage 5 of Phase-5/6 measurement plan (also ships `labeling_supplement/_phase5_matrix.py` — within-VR/video 4x4 driver — and a 4-LOC extension to `labeling_supplement/_harness_io_helpers.py::record_from_bank_entry` so the loader reads both legacy `eff_add`/`eff_del` and cross-domain `effects_add`/`effects_del` contract-key spellings). |

---

### `[x]` TODO-2 — `extract/_unify.py` *(shipped 2026-05-02)*

Cross-corpus unified skill index. The cross-domain bank layout
(`<output_root>/<corpus>/<bank_kind>/skill_bank.jsonl` with
`contract.effects_add` typed predicate dicts) is fundamentally different
from the legacy game-bank layout
(`<root>/<source>/skill_bank.jsonl` with `contract.eff_add` flat string
lists), so we couldn't reuse `labeling.unify_skill_index.unify_roots`
directly -- shipped a sibling module with the same output contract but
a cross-domain-aware reader. The "thin wrapper" framing in the original
spec turned out to need a custom row->entry projection.

| Field | Value |
|---|---|
| **Output** | `skill_transfer_test/skill_bank_local/<run_id>/_unified/{skill_index.jsonl,skill_catalog_all.json,skill_rag_index.json}` |
| **Acceptance** | 6 distinct `corpus` tags in the unified index; non-empty `skill_rag_index.json` for every corpus -- **PASS** (verified on `full_v5`: 1113 skills across 6 corpora, 10 banks). |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 5 |
| **Shipped LOC** | ~270 (`_unify.py`, including module-level docstring and `_coerce_version` for the `aggregator_version="v1"` -> `int(1)` translation across legacy + cross-domain rows) |
| **Run as** | `python -m skill_transfer_test.extract._unify --output-root skill_transfer_test/skill_bank_local/<run_id>` (idempotent; never mutates per-corpus banks) |

---

### `[x]` TODO-3 — `extract/tests/test_corpus_specs.py` *(shipped 2026-05-02)*

Field-validation tests for the six `CorpusSpec`s registered in
`_corpus_specs.py`. Tests are pure / hermetic where they can be -- the
`archetype_cluster_field` resolution test is opt-in and skipped per
corpus when its `default_input_root` is missing on disk, so CI runners
without the cold-start data still exercise the pure-shape checks.

`action_parser_ref` was dropped from the original spec because no such
field exists on `CorpusSpec` (the lift drivers route on `lift_kind`
directly). The test instead validates `lift_kind`, `modality`, `domain`,
`default_input_root`, `sample_glob`, `archetype_cluster_field` (when
set), plus `get_spec()` / `all_names()` / `all_specs()` round-trips and
the `KeyError` raised on unknown corpora.

| Field | Value |
|---|---|
| **Acceptance** | `pytest skill_transfer_test/extract/tests/test_corpus_specs.py` green; runs in <5 s -- **PASS** (20 tests, 0.05 s). |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 6 (partial) |
| **Shipped LOC** | ~155 (`test_corpus_specs.py`) |

---

### `[x]` TODO-4 — `extract/tests/test_single_shot_lift.py` *(shipped 2026-05-02)*

Lift-coverage tests across the four single-shot QA corpora (VTB,
TIR-Bench, Video-Holmes, SIV-Bench). For each corpus, drives
`lift_one_sample` against ONE real `correct=True` sample on disk and
asserts the `{report, skill}` envelope satisfies every contract the
downstream `runner.py` / `archetype_aggregator.py` / `_unify.py`
consumers rely on (skill_id consistency, protocol shape with `op` /
`payload` / `evidence_role` per typed hop, single-shot v4 predicates in
`contract.effects_add`, provenance fields, report book-keeping).

A second test pins the negative path (`correct=False` returns None
unless `include_incorrect=True`); a third verifies entity-reference
binding (cited `e\d+` IDs in `answer_reasoning` land in at least one
hop's payload/notes). Fixtures are NOT checked in -- each parametrized
case finds the first `correct=True` sample under `default_input_root`
and skips when the cold-start tree is missing.

| Field | Value |
|---|---|
| **Acceptance** | `pytest skill_transfer_test/extract/tests/test_single_shot_lift.py` green -- **PASS** (12 tests, 0.10 s). |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 6 (partial) |
| **Shipped LOC** | ~245 (`test_single_shot_lift.py`) |

---

### `[x]` TODO-5 — `extract/tests/test_runner_smoke.py` *(shipped 2026-05-02)*

End-to-end smoke for `runner.py` + `_unify.py`. Drives `runner.main`
against the visual / video corpora that have on-disk data with
`--max-samples 2`, checks every requested corpus produced a per-corpus
`per_sample/` (single-shot) or `per_episode/` (sequence)
`skill_bank.jsonl`, then runs `_unify.unify_root` on the run-id root and
verifies `_unified/{skill_index.jsonl,skill_catalog_all.json,skill_rag_index.json}`
shape: every flat row carries `corpus` + `bank_kind`, the catalog
groups by corpus, and the RAG index has one entry per skill with unique
IDs. Skips on absent cold-start data; uses `tmp_path` so no on-disk
bytes leak into `skill_bank_local/`.

| Field | Value |
|---|---|
| **Acceptance** | `pytest skill_transfer_test/extract/tests/test_runner_smoke.py` green -- **PASS** (2 tests, 0.08 s). Closes rollout memo §5.5.7a "All 3 unit-test files pass" row. |
| **Memo refs** | §5.5.4 (file table), §5.5.4a, §5.5.7a row 6 |
| **Shipped LOC** | ~180 (`test_runner_smoke.py`) |
| **Depends on** | TODO-2 (`_unify.unify_root`) -- shipped in the same batch |

---

### `[x]` TODO-6 — `extract/audits/vocab_jaccard.py` *(shipped 2026-05-02)*

Reproducible audit script for the §11.5.1 (rollout memo) / §9.3.1
(`skill_transfer_test/README.md`) vocabulary-alignment Jaccard numbers.
Walks both the game banks
(`labeling/skill_bank_out/run_*/{env_wrappers,gym_v}/*`) and the
cross-domain banks (`skill_transfer_test/skill_bank_local/full_v5/*`);
emits a per-layer Jaccard table (protocol ops / slot-type ontology /
hop-level predicates / contract-level predicates / combined predicates).

| Field | Value |
|---|---|
| **Output** | `cross_domain_results/_phase0/<run_id>/vocab_jaccard.{json,md}` (gitignored). Run as `python -m skill_transfer_test.extract.audits.vocab_jaccard`. |
| **Acceptance** | Numbers reproduce §11.5.1 within ±0.05 Jaccard — **PASS** (measured: protocol_ops=0.82, slot_types=1.00, predicates_combined=0.00; identical to §11.5.1 reference within rounding). |
| **Memo refs** | §11.5.1 (caveat), `skill_transfer_test/README.md` §9.3.1, `implementation_notes/legacy/phase5-cross-domain-measurement.md` §3 (Stage 0) |
| **Shipped LOC** | 369 (vocab_jaccard.py) + 292 (`_loaders.py` shared) + 112 (`_target_vocabularies.py` shared) |
| **Shipped as part of** | Stage 0 of Phase-5/6 measurement plan (also closes the static predicate-firing + slot-binding feasibility audits — `audits/predicate_firing_static.py`, `audits/slot_binding_feasibility.py`, `audits/_runner.py`). The §11.5.1 / §9.3.1 caveat ("the Jaccard numbers above are an analytical estimate ...") can now be replaced with a `(generated by skill_transfer_test/extract/audits/vocab_jaccard.py on 2026-05-02)` attribution. |

---

## Cancelled / superseded

### `[-]` TODO-7 — `extract/run_extract.sh` (cancelled)

Originally specced as a shell wrapper:
`bash run_extract.sh --corpus all` runs the 6 corpora sequentially.
**Superseded by** the `python -m skill_transfer_test.extract.runner` CLI
which already supports `--corpora all|<list> --max-samples N
--output-root <path> --run-id <name>`. No follow-up needed; the §5.5.4a
"deferred" annotation is technically a *cancellation*, not a deferral.

| Memo refs | §5.5.4 (file table), §5.5.4a |
|---|---|

---

## Out of scope for Phase 1.5b

Not tracked in this file:

- **§7 v0 limitations of `extract/README.md`** — these are deliberate
  trade-offs (per-hop predicate mining is Phase-2; ontology-aware slot
  binder is design-deferred; canonical `SkillBankAgent` LLM path is API-
  budget-gated). They are *philosophical* limits, not action items.
  When the canonical path lands, those limits lift; until then they
  belong in the README, not here.
- **Phases 0, 1, 2, 3, 4, 5, 6 from the rollout memo** — these are
  *new* workstreams (executors, harness producers, transfer matrix),
  not Phase 1.5 follow-ups. They live in the design memo §4 / §5 /
  §6 / §7 / §8 / §9 / §10 with their own acceptance gates.
- **Phase-5/6 cross-domain measurement plan (game→{osworld, browser,
  image-VR, video-VR}, plus within-VR/video transfer)** — being
  designed in chat as of 2026-05-02; will land as a separate memo
  (`implementation_notes/legacy/phase5-cross-domain-measurement.md`). The
  measurement-blocker LOC subset is ~310 (image-VR) / ~310-380
  (video-VR) / ~500 (browser) / ~470 (osworld) per the revised rollout
  memo §11.5.5; image-VR has no adapter work because
  `bind_visual_reasoning_executor` already wires the real
  `VisualReasoningExecutor` (461 LOC). TODO-1 (`archetype_aggregator`)
  is a hard prerequisite for the cross-corpus VR/video transfer cells
  but NOT for the within-corpus or game→VR cells.
- **Workspace-wide "Not yet delivered"** — see
  [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) and
  [`harness/README.md`](../harness/README.md) §"Suggested work-order".

---

## Cross-refs

- **Design rationale:** [`implementation_notes/cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md) §5.5 (Phase 1.5), §11.5 (transferability)
- **Shipped feature doc:** [`skill_transfer_test/extract/README.md`](extract/README.md)
- **Top-level workstream:** [`skill_transfer_test/README.md`](README.md)
- **Previous audit rounds:** [`skill_transfer_test/extract/README.md`](extract/README.md) §6 (18 issues / 16 fixed / 1 not-a-bug / 1 v0-limit-doc)
