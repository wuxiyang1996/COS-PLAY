# NeurIPS 2026 SkillBridge — writing stubs

This folder collects the prose stubs the reviewers asked for, plus the
corresponding generator commands so each table / figure has a
re-runnable provenance footnote.

| stub | paper section | generator |
| ---- | ------------- | --------- |
| `05.2_skill_dynamics.md`         | §5.2 Skill-bank dynamics                | `scripts/skillbridge_analysis/plot_skill_dynamics.py` + `_long_tail` + `_lifetime` |
| `05.3_lifecycle_gating_ablation.md` | §5.3 Lifecycle / intention ablations | `scripts/run_coevolution.py` flags + `scripts/run_skillbridge_eval.sh` |
| `05.5_cross_domain_transfer.md`  | §5.5 Cross-domain transfer matrix       | `scripts/skillbridge_eval/run_transfer_matrix.py` + `run_few_shot_sweep.py` |
| `05.6_runtime_overhead.md`       | §5.6 Runtime / token overhead           | `scripts/skillbridge_analysis/plot_runtime_overhead.py` |
| `06_limitations.md`              | §6 Limitations                          | _(prose)_ |
| `algorithms.md`                  | Alg. 1 (episode step) + Alg. 2 (bank update) | _(prose)_ |
| `_consistency_sweep.md`          | F2 cleanup notes                        | _(grep checklist)_ |

All stubs use the **canonical SkillBridge architectural vocabulary**:

* **Decision Agent** — the actor with three LoRAs
  (`action_taking`, `skill_selection`, `intention`).
* **Skill Bank Agent** — the offline-learning pipeline composed of
  *Extractor*, *Crafter*, and *Promotion gate*.  Owns the persistent
  Skill Bank.
* **Harness** — the **runtime** validation layer
  (`harness/SkillHarness`).  Two phases: eligibility filtering (pre-LLM)
  and `validate_invocation` (post-LLM).  **Not** a separate agent.

Whenever you see "harness agent" or "validator agent" in older drafts,
rewrite as "harness (runtime layer)".
