# OSWorld 4-model baseline plan (50-task subset)

> **Status (2026-05-03 AM):** 🟡 **PLAN — not yet executed.** This memo
> defines the minimum-viable cross-model OSWorld baseline that supplies
> teacher-comparison numbers for the Phase-5/6 cross-domain transfer
> story. Code prerequisites (multi-provider routing + steering bug fix)
> already landed; what remains is a small `--eval_mode low` shortcut, a
> deterministic subset sampler, and the actual cross-machine launch.

> **Cross-refs:**
> - [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) lines 40, 339-342 — Phase-5/6 Stage 6 NxN matrix; OSWorld is one of 4 cross-domain *targets*, not the headline KPI.
> - [`implementation_notes/cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md) §11.5.4 — admit-rate bands the transfer matrix is meant to land in.
> - [`cold_start/run_osworld_multimodel.sh`](../cold_start/run_osworld_multimodel.sh) — provider → model-id mapping (already shipped).
> - [`cold_start/run_coldstart_actor_osworld.sh`](../cold_start/run_coldstart_actor_osworld.sh) — base launcher (already shipped).
> - [`cold_start/smoke_multimodel.py`](../cold_start/smoke_multimodel.py) — 60-second LLM-only bring-up smoke (already shipped).
> - [`cold_start/osworld_steering.py`](../cold_start/osworld_steering.py) — opt-in memory / reflection / self-verify; `reasoning_effort='minimal'` → `'low'` bug fix landed 2026-05-03.

---

## 1. Why this baseline exists (one paragraph)

The project's main contribution is **skill discovery, adaptation, and
transfer** measured by the Phase-5/6 cross-domain admit-rate matrix.
OSWorld is one of four *target* domains in that matrix, not the
headline number. To frame the transfer result we need a defensible
**teacher OSWorld pass-rate** for each frontier API model, so the
publication can read:

> "Our gymv-derived skills transfer to OSWorld with admit rate X%.
> For context, frontier teachers achieve native OSWorld pass-rates of
> GPT-5.4: A%, Claude Sonnet 4.6: B%, Gemini 2.5 Pro: C%, Qwen3-VL
> 235B: D%."

A 50-task subset (5 per domain × 10 domains) gives ±7% binomial CI on
the global pass-rate and ±20% per-domain — enough for the comparison
table. The full 250 / 361-task run is reserved for a possible
reviewer-rebuttal pass and is **not on the critical path**.

## 2. Scope and non-goals

In scope:

* Run the same 50-task subset on 4 API models with identical agent
  configuration (7 improvements all on, `--eval_mode low`).
* Produce a single comparison CSV / scoreboard JSON the publication
  table can render directly.
* Save full rollouts so the trajectories can later be lifted into the
  skill bank if useful (delta over the 30 OSWorld skills already in
  `skill_transfer_test/skill_bank_local/full_v5/osworld/per_episode/skill_bank.jsonl`).

Out of scope:

* Beating the OSWorld leaderboard. We are explicitly not chasing the
  43.9% Claude-Sonnet number — `--eval_mode low` will land 2-3 pp
  below `medium`/`high` for gpt-5.4 and that is fine.
* Re-running the cross-domain transfer matrix (Phase-5/6 Stage 6).
  That is a separate, downstream activity that consumes these
  baselines.
* Touching the trainer co-evolution loop or any of the C/A/D layers
  in `coevolution-cross-domain-integration.md` §1.

## 3. Subset selection — 50 tasks, deterministic

Source catalog: `/workspace/OSWorld/evaluation_examples/test_nogdrive.json`
(361 tasks across 10 domains).

| Domain | Total | Sampled |
|---|---|---|
| `chrome` | 46 | 5 |
| `gimp` | 26 | 5 |
| `libreoffice_calc` | 47 | 5 |
| `libreoffice_impress` | 47 | 5 |
| `libreoffice_writer` | 23 | 5 |
| `multi_apps` | 93 | 5 |
| `os` | 24 | 5 |
| `thunderbird` | 15 | 5 |
| `vlc` | 17 | 5 |
| `vs_code` | 23 | 5 |
| **Total** | **361** | **50** |

Selection rule: per domain, sort tasks by `task_id` lexicographically
(so the choice is reproducible), then take the first 5.  No
random-shuffle — we want *the same 50 tasks* across all four models so
the comparison is paired (each model is scored on the identical set).

**Deliverable:** a small JSON catalog file
`evaluation_examples/test_nogdrive_subset50_v1.json` containing the
sampled 50 task ids, written by a one-shot script
`scripts/build_osworld_subset50.py`. The script emits a manifest line
to `runs/osworld_baseline_50/manifest.json` recording the source
catalog hash and the sampling rule, so reviewers can reproduce.

Pinning the file (not just the seed) means future updates to
`test_nogdrive.json` upstream cannot silently change which 50 tasks
the baseline covers.

## 4. Models, IDs, and per-provider parameters

| Provider tag | OpenRouter / OpenAI id | Route | `--eval_mode` |
|---|---|---|---|
| `gpt5` | `gpt-5.4` | OpenAI direct (via `--api_key` from `api_keys.py`) | `low` (NEW — see §5) |
| `claude-sonnet` | `anthropic/claude-sonnet-4.6` | OpenRouter | `low` |
| `gemini-pro` | `google/gemini-2.5-pro` | OpenRouter | `low` |
| `qwen3-vl` | `qwen/qwen3-vl-235b-a22b-instruct` | OpenRouter | `low` |

For the three non-OpenAI providers, `reasoning_effort` is silently
dropped by the driver — `--eval_mode low` for them only sets
`temperature_action=0.0`, `temperature_schema=0.0`, `done_nudge_step=999`,
`max_steps=75`. The headline knob (`reasoning_effort=low`) is
GPT-only.

Common flags across all four runs:

```
--task_catalog evaluation_examples/test_nogdrive_subset50_v1.json
--episodes 1
--eval_mode low
--no_reuse_env                    # protocol-clean per-task VM
--pause_after_action 1.5          # web-page settle (was 3.0 in earlier runs)
--max_entities 35
--enable_memory
--enable_reflection
--enable_self_verify
--skill_bank_path skill_transfer_test/skill_bank_local/full_v5/osworld/per_episode/skill_bank.jsonl
--skill_retrieval_top_k 3
--resume
--no_save_frames                  # disk-pressure guard; per-step JSON sidecar still written
-v
```

Per-provider differences (handled by `run_osworld_multimodel.sh`):
provider tag → model id → output dir suffix. Nothing else differs.

## 5. Code prerequisite — add `--eval_mode low`

Currently `--eval_mode {medium, high, max}` is supported (see
[`generate_cold_start_actor_osworld.py:3187`](../cold_start/generate_cold_start_actor_osworld.py)).
We need a `low` tier that maps to:

| Knob | `low` | `medium` | `high` | `max` |
|---|---|---|---|---|
| `reasoning_effort` | `low` | `medium` | `high` | `high` |
| `temperature_action` | 0.0 | 0.0 | 0.0 | 0.0 |
| `temperature_schema` | 0.0 | 0.0 | 0.0 | 0.0 |
| `done_nudge_step` | 999 | 999 | 999 | 999 |
| `max_steps` | 75 | 75 | 75 | 100 |

LOC change: ~5 lines in the existing `eval_tier` block of `main()`.
`run_osworld_multimodel.sh:provider_default_eval_mode()` then flips
its defaults (`high` → `low` everywhere; Qwen3-VL stays `low` as it
already does).

Also: `cold_start/smoke_multimodel.py` already passes
`reasoning_effort='high'` for its action-call probe; for parity with
the chosen tier in production, it should default to `low` and accept
a `--reasoning-effort` override. ~5 LOC.

## 6. Wall-clock and cost budget

Per-task assumptions (35 step average, 7 improvements on,
`--eval_mode low`, `--no_reuse_env`):

| Provider | $/task | step latency | task wall-clock | 50-task wall-clock (1 machine) | 50-task wall-clock (4 machines, this is 1/provider) | 50-task LLM cost |
|---|---|---|---|---|---|---|
| GPT-5.4 (low) | $2.46 | ~25s | ~20 min | ~17 h | ~17 h on its own machine | $123 |
| Claude Sonnet 4.6 | $1.59 | ~13s | ~11 min | ~9 h | ~9 h on its own machine | $80 |
| Gemini 2.5 Pro | $0.79 | ~17s | ~13 min | ~11 h | ~11 h on its own machine | $40 |
| Qwen3-VL 235B-A22B Instruct | $0.10 | ~10s | ~9 min | ~7.5 h | ~7.5 h on its own machine | $5 |
| **All 4 (1 machine each, in parallel)** | — | — | — | **wall-clock = max ≈ 17 h** | — | **~$248** |
| **All 4 (single machine, sequential)** | — | — | — | ~44 h ≈ 1.8 d | — | ~$248 |

Notes:

* Numbers ignore docker-VM boot/reboot overhead beyond the ~3 min/task
  already folded into "task wall-clock". A single VM cold boot at the
  *start* of each provider's run adds ~3-4 min — negligible.
* gpt-5.4 cost line includes the ~70K hidden thinking tokens at
  `reasoning_effort=low` (~1K tokens × ~70 reasoning calls per task,
  billed at $15/M output).
* OpenRouter pricing pulled live from `/api/v1/models` on 2026-05-03
  AM. Recheck before launch if the run is delayed >1 week.

## 7. Pipeline and output layout

```
WORKSPACE_ROOT/
├── evaluation_examples/
│   └── test_nogdrive_subset50_v1.json          ← built by step 0
├── runs/osworld_baseline_50/
│   ├── manifest.json                           ← subset hash + seed + cmds
│   ├── gpt5/                                   ← --output_dir for gpt5 run
│   │   ├── chrome/<task_id>/episode_000.json
│   │   ├── ...
│   │   ├── batch_rollout_summary.json
│   │   └── run_metadata.json
│   ├── claude-sonnet/...
│   ├── gemini-pro/...
│   ├── qwen3-vl/...
│   └── compare/
│       ├── pass_at_1_overall.csv               ← 4 rows × {n, pass, rate, CI}
│       ├── pass_at_1_per_domain.csv            ← 4 × 10 grid
│       ├── per_task.csv                        ← 50 × 4 ⊂ 0/1 matrix
│       └── readme.md                           ← human-readable rendering
```

Wall-clock isolation: each provider run writes to its own
`<provider>/` subdir. No risk of port collision between provider runs
on a single machine — they're sequential (provider A finishes before
B starts). For multi-machine parallelism, each machine clones the
catalog file, runs **one** provider, and rsyncs back to a coordinator
host that owns `compare/`.

## 8. Aggregation script (~150 LOC, new file)

`scripts/aggregate_osworld_baseline.py`:

* Reads each `<provider>/batch_rollout_summary.json` and the
  per-task `episode_000.json`s.
* Honors `eval_score is None` as 0.0 (re-uses
  `cold_start/load_rollouts.py --treat-null-as-zero`).
* Emits the four CSVs above + a markdown summary.
* Validates: every (provider, task_id) cell is filled. Flags partial
  runs with a clear "K/50 missing" note rather than silently
  reporting biased numbers.

## 9. Pre-flight checklist (per machine)

Run **before** the actual eval on each machine:

```bash
# 0. The codebase is already cloned at the same commit on every machine.
cd /workspace/Multi-hop-Reasoning-VLM-Agent

# 1. api_keys.py present at the workspace root with at minimum
#    openrouter_api_key (and openai_api_key for the gpt5 machine).
ls /workspace/api_keys.py

# 2. OSWorld VM image present (~23 GB Ubuntu.qcow2).
ls -la docker_vm_data/Ubuntu.qcow2

# 3. The osworld conda env is healthy and DesktopEnv imports.
conda activate osworld && python -c "import desktop_env"

# 4. The Docker image is pulled (warns are fine; OSWorld pulls on first use).
docker image inspect happysixd/osworld-docker >/dev/null

# 5. 60-second LLM-only smoke for the provider this machine will run.
python cold_start/smoke_multimodel.py --provider <claude-sonnet|gemini-pro|qwen3-vl|gpt5>

# 6. Build / copy the subset catalog (only the coordinator host runs the build).
python scripts/build_osworld_subset50.py
```

Hard fail conditions: any of 0-4 missing → fix before launching.

## 10. Execution sequence

1. **(coordinator only)** Land the three small code prereqs:
   * `--eval_mode low` (~5 LOC in `generate_cold_start_actor_osworld.py`).
   * `provider_default_eval_mode()` switch from `high` → `low` for
     non-Qwen providers in `run_osworld_multimodel.sh` (~6 LOC).
   * `scripts/build_osworld_subset50.py` (~50 LOC).
   * `scripts/aggregate_osworld_baseline.py` (~150 LOC).
   * Smoke that all four `provider_default_eval_mode` paths still
     `--print_resolved` cleanly.
2. **(coordinator)** Run `python scripts/build_osworld_subset50.py`.
   Commit the resulting `evaluation_examples/test_nogdrive_subset50_v1.json`
   so every machine pulls the identical file.
3. **(per machine)** Pre-flight checklist §9 → smoke for the
   provider it will run.
4. **(per machine)** Launch:
   ```bash
   bash cold_start/run_osworld_multimodel.sh \
        --provider <provider_tag> \
        --task_catalog evaluation_examples/test_nogdrive_subset50_v1.json \
        --episodes 1 \
        --eval_mode low \
        --no_reuse_env \
        --pause_after_action 1.5 \
        --max_entities 35 \
        --enable_memory --enable_reflection --enable_self_verify \
        --skill_bank_path skill_transfer_test/skill_bank_local/full_v5/osworld/per_episode/skill_bank.jsonl \
        --skill_retrieval_top_k 3 \
        --resume \
        --output_dir runs/osworld_baseline_50/<provider_tag> \
        --no_save_frames -v \
        2>&1 | tee /tmp/osworld_baseline_50_<provider_tag>.log
   ```
5. **(coordinator)** Once a provider's run finishes, rsync its
   `<provider>/` subdir back to the coordinator host's
   `runs/osworld_baseline_50/<provider>/`.
6. **(coordinator)** When all four are in:
   `python scripts/aggregate_osworld_baseline.py
        --root runs/osworld_baseline_50/`. This writes `compare/`.
7. **(coordinator)** Eyeball `compare/readme.md`. Done.

## 11. Acceptance criteria

The baseline is "done" when:

| Check | How to verify |
|---|---|
| All 4 providers ran the identical 50-task subset | `compare/per_task.csv` has 50 × 4 = 200 cells, none missing. |
| No silent failures hidden by `eval_score=None` | `aggregate_osworld_baseline.py` prints `null_count_per_provider` and it is the same number that the launcher reported truncations for. Otherwise audit the 5+ outliers. |
| Per-provider pass-rate is statistically distinct | 95% binomial CIs on the global pass-rate either separate the providers visually or the report explicitly says "within statistical noise". |
| Skill-retrieval was actually used | Each provider's `run_metadata.json` shows `skill_bank_loaded=30, retrieval_top_k=3, retrieval_calls > 0`. |
| All 7 improvements were active | Same file shows `enable_memory=True, enable_reflection=True, enable_self_verify=True` and the `osworld_steering` warnings count is < 2 per task on average (the bug we fixed must not regress). |

If any check fails, debug before publishing. Do not publish a partial
table.

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `gpt-5.4` direct API has another tools-vs-`reasoning_effort` quirk we did not catch in May 2026 | Medium | The May-2026 patch in `_chat_completion` already strips `reasoning_effort` for tool-bearing gpt-5.x calls on direct OpenAI. The new `low` tier still flows through that gate. Verify with a 1-task launch before the full 50. |
| Gemini 2.5 Pro returns `finish_reason=error` on ~5% of vision calls (safety blip) | High | Driver fall-back (candidate-list selection) handles this. The aggregation script flags providers with > 10% empty-content steps so the issue surfaces. |
| OpenRouter rate-limit (free-tier quotas, regional outages) | Low–Medium | All four runs use the same OPENROUTER_API_KEY. Stagger machine launches by ~30 s if running concurrently. The driver retries with 60 s back-off (already exercised in earlier runs). |
| Docker port collision when two providers share a machine | N/A (sequential per machine) | Plan never schedules > 1 OSWorld VM on the same machine simultaneously. |
| One task hangs past 75-step cap and burns 30+ minutes for a `eval_score=None` row | High (truncation rate ~46% historically) | We accept truncation. The aggregation honours `--treat-null-as-zero`. Truncations affect all four providers equally and so do not bias the comparison. |
| `eval_score=None` is silently mis-reported as success in the per-task CSV | Low | §11 acceptance check explicitly verifies this. |
| The `osworld_steering.py` `reasoning_effort='minimal'` regression returns | Low | Locked by the bug-fix patch (defaults are `'low'` now). Add a unit test `test_osworld_steering_default_reasoning_effort` covering all 3 dataclasses. |

## 13. Open questions

These are non-blocking for landing the prereqs but worth resolving
before launch:

1. **Confidence intervals.** Wilson interval or Clopper-Pearson? The
   project elsewhere uses Wilson for binomial CIs (see
   `evaluation/scoreboard.py`). Re-use it here for consistency.
2. **Should we also score `--eval_mode medium` for gpt-5.4 on the
   same subset?** Cost: 50 tasks × $5/task × 1 = $250 extra, ~33 h
   wall-clock. Yields a "low vs medium" data-point for the
   publication footnote. **Recommendation:** defer; can be added
   later if reviewers ask for it.
3. **Do we run the same baselines on the full 250 / 361 in addition
   to the 50?** Only if budget allows and only on Qwen3-VL ($25 + 1.5
   d single machine) — the small marginal cost makes a "fuller
   single-model run" cheap insurance. Other providers stay at 50.
4. **Should the launcher itself do the rsync to the coordinator?** No —
   coupling the launcher to a coordinator host is a footgun if the
   launcher gets reused outside this baseline. Manual rsync stays.

## 14. LOC budget for the prereqs

| Item | File | LOC | Risk |
|---|---|---|---|
| `--eval_mode low` tier | `cold_start/generate_cold_start_actor_osworld.py` | ~5 | Trivial |
| `provider_default_eval_mode` flip | `cold_start/run_osworld_multimodel.sh` | ~6 | Trivial |
| Subset builder | `scripts/build_osworld_subset50.py` | ~50 | Low |
| Aggregator | `scripts/aggregate_osworld_baseline.py` | ~150 | Low–Medium (CSV / CI math) |
| Smoke `--reasoning-effort` flag | `cold_start/smoke_multimodel.py` | ~5 | Trivial |
| Steering default-test | `tests/test_osworld_steering_defaults.py` | ~30 | Trivial |
| **Total** | | **~250 LOC** | |

All six items are pure-additions; no existing code path's defaults
change.

## 15. Sequencing rationale

We land prereqs **before** any launch because:

1. The `--eval_mode low` tier is the single biggest cost lever
   ($1138 → $123 for gpt-5.4 alone) and we want it baked into
   `argparse.eval_tier` so misuse is impossible (rather than
   relying on operators to remember three CLI flags).
2. The deterministic subset file pins the experiment so every
   machine cannot accidentally diverge.
3. The aggregator written first means the very first finished
   provider can be eyeballed in isolation and we catch any
   per-provider plumbing issue before all four have spent
   wall-clock.
4. The unit test on steering defaults locks the May-2026 bug fix in
   regression-test territory before the next refactor passes
   through that file.

The prereqs are small enough (~250 LOC) and risk-free enough that
landing them in one short session before kicking off the eval is
strictly faster than launching, hitting an issue, restarting.
