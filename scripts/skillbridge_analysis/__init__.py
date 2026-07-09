"""SkillBridge post-hoc analysis + plotting utilities.

Each module in this package consumes one of the JSONL streams emitted
by :mod:`trainer.coevolution._run_loggers` (or the run's existing
``audit.jsonl`` / ``promotion_decisions_out/*``) and produces a figure
for the NeurIPS submission:

* :mod:`plot_skill_dynamics`   — promotion / rejection / deprecation
  curves + crafter-mutation pie chart.
* :mod:`plot_skill_long_tail`  — per-skill retrieval frequency CDF.
* :mod:`plot_skill_lifetime`   — skill lifetime distribution.
* :mod:`plot_failure_modes`    — harness veto-code pie chart.
* :mod:`plot_runtime_overhead` — per-component ms / token bar chart.
* :mod:`plot_skill_flow_map`   — Sankey from raw segments → skills.
* :mod:`compute_significance`  — paired-bootstrap p-values per benchmark.
* :mod:`case_study_skill_trace` — provenance trail for a given skill_id.

All scripts are stand-alone CLIs; running them with ``-h`` prints the
expected JSONL paths.
"""
