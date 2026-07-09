"""Cross-domain post-training eval drivers for SkillBridge (block C).

Each module loads a trained SkillBridge checkpoint (LoRA adapters +
skill bank), runs N held-out tasks against the appropriate environment
wrapper, and emits a uniform ``eval_result.json`` so the aggregator
can produce a single benchmark table.

Modules:

  * :mod:`eval_actor`        — actor wrapper (LoRA + bank + optional
    harness validation).  Stable interface used by every domain
    runner.
  * :mod:`eval_gymv`         — held-out gymv games via env_wrappers
    (canonical, fully implemented).
  * :mod:`eval_browsergym`   — BrowserGym tasks (delegates to
    cold_start/generate_cold_start_actor_browsergym.py with a
    SkillBridge actor swap).
  * :mod:`eval_osworld`      — OSWorld desktop tasks.
  * :mod:`eval_visual_reasoning` — visual reasoning benchmarks.
  * :mod:`eval_video`        — video QA / planning.
  * :mod:`eval_aggregator`   — reads every per-domain result JSON
    and emits a single Markdown + CSV table.

Entry point: ``scripts/run_skillbridge_eval.sh`` drives all 5 in
sequence (or in parallel under ``--parallel``).
"""
