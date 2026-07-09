# VisualWebArena (VWA) — archived 2026-05-03

This directory archives the VWA-specific scripts, plans, and task subset
that were active in `main` between 2026-04-29 and 2026-05-03. **VWA is
no longer a supported benchmark in the COS-PLAY pipeline.** Decision
captured in chat — short version: *the agent code is fine, the VWA
environment stack is unmaintained academic infrastructure with too many
non-research bugs to be worth absorbing*.

## TL;DR — why we dropped it

VWA = `BrowserGym → visualwebarena (CMU 2024) → Playwright →
Chromium → 4 Docker web services + 1 homepage → OpenAI judge`.
Six layers, each with its own assumptions, none tested end-to-end.
Across one week of work we hit **10 distinct bugs** that had nothing
to do with the agent algorithm:

| # | Bug | Layer |
|---|---|---|
| 1 | `visualwebarena` package eager-imports `OpenAI()` at module load | upstream |
| 2 | Evaluator hardcodes `gpt-4-1106-preview` (deprecated 2024-04) | upstream |
| 3 | `jykoh/classifieds:latest` ships with broken `WEB_PATH` (no trailing `/`) | upstream container |
| 4 | BrowserGym obs key `axtree_object` vs `axtree_txt` confusion | wrapper |
| 5 | `set_of_marks` flag silently dropped `fill()` candidates | wrapper |
| 6 | `task.timeout = 10000` hardcoded → cold-start `goto` timeouts | upstream |
| 7 | `task.validate()` runs LLM judge on **every** step | upstream design |
| 8 | Goal-image host on different port than generic webarena homepage | upstream docs |
| 9 | 4 separate Docker services per task, all stateful | architecture |
| 10 | Docker daemon manifest revalidation hangs >20 min | infrastructure |

Patching all 10 to get a cold baseline cost more time than the
benchmark was worth for our skill-discovery story. AssistantBench
gives us 6/10 of those bugs *for free* (no Docker fleet, no per-step
LLM judge, real Internet) and is the new primary web env. See
`implementation_notes/legacy/...` for cross-refs.

## What's in here

| File | Was at | Purpose |
|---|---|---|
| `visualwebarena_env.sh`           | `cold_start/visualwebarena_env.sh`            | Sourceable VWA URL endpoints (`CLASSIFIEDS`, `REDDIT`, `SHOPPING`, `WIKIPEDIA`, `HOMEPAGE`, `VWA_JUDGE_MODEL`) |
| `install_visualwebarena_sites.sh` | `install/install_visualwebarena_sites.sh`     | Bring up classifieds (OSClass+MySQL) Docker stack + apply `WEB_PATH` patch + invoke judge-model patch |
| `patch_vwa_judge_model.sh`        | `install/patch_vwa_judge_model.sh`            | Idempotent in-file patch over upstream `visualwebarena/evaluation_harness/helper_functions.py` to swap deprecated `gpt-4-1106-preview` for `$VWA_JUDGE_MODEL` (default `gpt-4o`) |
| `vwa-improvement-plan.md`         | `implementation_notes/vwa-improvement-plan.md` | 12-section diagnostic, fix log, and follow-up plan covering the §1-12 timeline of all 10 bugs above + smoke results |
| `osworld-vwa-200-baseline-plan.md`| `implementation_notes/osworld-vwa-200-baseline-plan.md` | 4-model VWA baseline plan (200-task pinned subset, stratified by site × image × difficulty). Sister doc to `osworld-4model-baseline-plan.md` (still active) |
| `browsergym_visualwebarena_200.txt`| `cold_start/task_samples/browsergym_visualwebarena_200.txt` | The pinned 200/910 subset (seed=0, covers 116/152 templates). Built by `cold_start/task_samples/build_browsergym_diverse_200.py` |

## What survived in main (intentionally)

The agent-side improvements that VWA debugging *motivated* are general
BrowserGym infrastructure and stayed in `main` because they help every
other suite (AssistantBench / WebArena / MiniWoB):

* **`cold_start/generate_cold_start_actor_browsergym.py`** —
  `_count_som_telemetry`, anti-thrash override (#6d), anti-repeat
  override (#6e), terminal-action plumbing (#12), search-first
  heuristic in the system prompt. All keep their VWA-flavoured
  inline comments as historical breadcrumbs (e.g. "...catches the
  May-3 visualwebarena.96 click('211') 7× loop...") because removing
  them would erase the rationale for *why* the mechanism exists.
* **`browsergym_wrapper/tools.py`** — relaxed `fill()` candidate
  filter (no longer requires `set_of_marks=True` for textboxes).
* **`tests/test_browsergym_anti_thrash.py`,
  `tests/test_browsergym_anti_repeat.py`,
  `tests/test_browsergym_action_candidates.py`,
  `tests/test_browsergym_terminal_actions.py`** — 116 regression
  tests covering all of the above.
* **`cold_start/run_coldstart_actor_browsergym.sh`** — multi-suite
  launcher; the `_NEED_VISUALWEBARENA` auto-source branch was
  removed when this archive was created.

## Resurrection instructions

If you ever need to revive VWA (paper revision, reviewer demand,
new student joins and wants to redo it from scratch), the workflow is:

```bash
# From repo root.
git mv legacy/visualwebarena/visualwebarena_env.sh           cold_start/
git mv legacy/visualwebarena/install_visualwebarena_sites.sh install/
git mv legacy/visualwebarena/patch_vwa_judge_model.sh        install/
git mv legacy/visualwebarena/vwa-improvement-plan.md         implementation_notes/
git mv legacy/visualwebarena/osworld-vwa-200-baseline-plan.md implementation_notes/
git mv legacy/visualwebarena/browsergym_visualwebarena_200.txt cold_start/task_samples/

# Restore the auto-source branch in run_coldstart_actor_browsergym.sh
# (see git log -p around the archive commit for the exact diff).
```

The judge-model patch script is idempotent and re-runnable. The
classifieds installer is also idempotent (the `WEB_PATH` patch hook
will detect "already normalised" and skip).

Empirically, expect ~2-3 days to rebuild the full stack from a fresh
machine: 1 day Docker fleet + DB seed, 0.5 day judge patch + cookie
warm-up, 0.5 day driver re-validation, 1 day actual baseline
production run.
