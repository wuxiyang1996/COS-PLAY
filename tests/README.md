# `tests/` — Invariant, smoke, and backbone-pin tests

The full suite is the load-bearing safety net for the new MVP (`common/`, `data_structure/extensions/`, `skill_bank/`, `harness/`, `orchestrator/`, `crafter/`). It must stay green for any change inside those packages.

```bash
cd Multi-hop-Reasoning-VLM-Agent
python -m pytest tests/ -v
# Expected: 29 passed
```

No GPU required, no network access required (GPT-4o calls are mocked).

---

## File map

| File | Count | What it tests |
|---|---|---|
| `test_invariants.py` | 14 | The **mechanically-enforced invariants** from the root README §"Mechanically-enforced invariants". One test per invariant clause, plus regression tests for the source/transfer asymmetry (#7) and the `verified_domains`-is-gate-owned ordering (#8) |
| `test_smoke.py` | 2 | End-to-end wiring: a tiny rollout exercising every Phase A/B/C module. Builds a fake env + actor, runs one outer episode, ingests a synthetic failure into the crafter, verifies that all artefact subdirectories (`episodes/`, `proposals/`, `failures/`, `evaluations/`, `releases/`) are populated. **Not** a correctness test — it's a wiring test |
| `test_backbone_model.py` | 13 | The GPT-4o pin from `common/models.py`. Asserts that `BACKBONE_MODEL` / `BACKBONE_TEACHER_MODEL` / `BACKBONE_JUDGE_MODEL` all default to `gpt-4o`, that `is_deferred(name)` flags every Qwen variant in `DEFERRED_MODELS`, that the env-var override path (`VLM_AGENT_BACKBONE_MODEL` …) works, and that no module under `common/`, `harness/`, `orchestrator/`, `crafter/`, or `skill_bank/` hardcodes a model name |
| `test_few_shot_transfer.py` | — | `harness.FewShotAdapter` integration: K-demo adaptation, `eligible_domains` emission, and the `record_transfer_verification` → `lifecycle.promote` ordering that owns invariant 8 |
| `conftest.py` | — | Shared fixtures: `bank_root` (tmp four-store layout), `repo` + `lifecycle`, fake `Env` / `Actor`, in-memory `ArtifactStore`, mocked GPT-4o teacher |

---

## Invariants checklist (must stay green)

| # | Invariant | Test |
|---|---|---|
| 1 | G0 evidence-driven (`SkillEpisode.finalize` raises on empty evidence; lifecycle rejects ACTIVE on empty `expected_evidence_roles`) | `test_invariants.py::test_g0_evidence_driven` |
| 2 | No-memory (`SkillEpisodeStep.__post_init__` rejects `QUERY_MEM` / `WRITE_MEM`) | `test_invariants.py::test_no_memory_step_kinds_rejected` |
| 3 | T1.3d lane-(a) ACTIVE gate (lifecycle rejects ACTIVE when `metrics["retrievals"] < min_retrievals_per_skill`; default 0 = off, orchestrator wires `OrchestratorConfig.gate_thresholds.min_retrievals_per_skill`) | `test_invariants.py::test_active_promotion_requires_min_retrievals_when_threshold_set` |
| 4 | Bank-write isolation (`SkillStore.put` raises `StoreLockedError` outside lifecycle) | `test_invariants.py::test_bank_write_isolation` |
| 5 | Gate-bound promotion (`PromotionOrchestrator.promote` rejects FAIL + content-hash drift; refuses ACTIVE on `LIMITED_PASS`) | `test_invariants.py::test_gate_bound_promotion` |
| 6 | Crafter scope (crafter writes only DRAFT; never imports `skill_bank.stores`; never touches `active_store`) | `test_invariants.py::test_crafter_scope` |
| 7 | Source/transfer asymmetry (every ACTIVE skill has ≥1 entry from `SOURCE_DOMAINS` **and** ≥1 from `TRANSFER_TARGET_DOMAINS`) | `test_invariants.py::test_source_transfer_asymmetry` |
| 8 | `verified_domains` is gate-owned (only `record_transfer_verification` may write; `PromotionOrchestrator` calls it before status transition) | `test_invariants.py::test_verified_domains_gate_owned` + `test_few_shot_transfer.py` |

---

## Adding a test

If you add a new mechanically-enforced rule, follow the existing pattern:

1. Add the runtime check in the owning module (`common/`, `skill_bank/lifecycle.py`, etc.) so the violation raises an exception at write time, not at read time.
2. Add a row to the root [`readme.md`](../readme.md) §"Mechanically-enforced invariants" table with the invariant text and the exact enforcement point.
3. Add a `test_invariants.py::test_<short_name>` that exercises both the legitimate path (does not raise) and the violation path (raises the expected exception type and message fragment).

If your change touches the backbone-model registry, also extend `test_backbone_model.py`. If it touches Phase-A/B/C wiring, exercise it through `test_smoke.py` so an end-to-end run still succeeds.

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Mechanically-enforced invariants" and §"Quick start" → `python -m pytest tests/ -v`.
- [`../skill_bank/README.md`](../skill_bank/README.md) — the lifecycle invariants this suite enforces.
- [`../harness/README.md`](../harness/README.md) and [`../orchestrator/README.md`](../orchestrator/README.md) — the modules `test_smoke.py` exercises end-to-end.
