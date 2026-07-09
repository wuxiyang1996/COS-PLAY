# VisualWebArena 4-model baseline plan (200-task pinned subset)

> **Status (2026-05-03 AM):** 🟡 **PLAN — partially executed.** The 3
> code prereqs (driver `--reasoning_effort` default flip, per-task
> watchdog, sourceable env file) **already landed on `main` as
> `90628b8`**. The 200-task pinned subset already exists at
> `cold_start/task_samples/browsergym_visualwebarena_200.txt`
> (built earlier by `build_browsergym_diverse_200.py` seed=0).
> What remains is a 4-provider router for BrowserGym (mirrors
> OSWorld's `run_osworld_multimodel.sh`), the upstream Classifieds +
> VWA-homepage install for 100% task coverage, and the cross-machine
> launch.

> **Cross-refs:**
> - [`implementation_notes/osworld-4model-baseline-plan.md`](osworld-4model-baseline-plan.md) — sister plan, same skill-transfer story, different target domain. **Most decisions below mirror it.**
> - [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) lines 40, 339-342 — Phase-5/6 Stage 6 NxN matrix; VWA is one of 4 cross-domain *targets*.
> - [`cold_start/run_coldstart_actor_browsergym.sh`](../cold_start/run_coldstart_actor_browsergym.sh) — base BrowserGym launcher (already shipped).
> - [`cold_start/generate_cold_start_actor_browsergym.py`](../cold_start/generate_cold_start_actor_browsergym.py) — driver. May 3 commit `90628b8` flipped `--reasoning_effort` default to `low` and added the per-task `[WATCHDOG]` log.
> - [`cold_start/visualwebarena_env.sh`](../cold_start/visualwebarena_env.sh) — sourceable VWA env file. Documents the 52-classifieds-task + 72-image-task partial-coverage limit when the upstream install hasn't been run.
> - [`cold_start/task_samples/browsergym_visualwebarena_200.txt`](../cold_start/task_samples/browsergym_visualwebarena_200.txt) — pinned 200/910 subset, seed=0, stratified by `(site × has_image × overall_difficulty)`. Covers 116/152 templates.
> - [`cold_start/task_samples/build_browsergym_diverse_200.py`](../cold_start/task_samples/build_browsergym_diverse_200.py) — generator script (already shipped).
> - [`install/install_visualwebarena_sites.sh`](../install/install_visualwebarena_sites.sh) — Classifieds (OSClass+MySQL) installer; run-on-this-machine 2026-05-03 (containers up + DB seeded + patch hook integrated).
> - [`install/patch_vwa_judge_model.sh`](../install/patch_vwa_judge_model.sh) — idempotent text-patch over the upstream ``visualwebarena/evaluation_harness/helper_functions.py`` to replace the deprecated ``gpt-4-1106-preview`` judge with ``$VWA_JUDGE_MODEL`` (default ``gpt-4o``). Without it, **18 / 200 pinned tasks fail at ``env.step()``** because VWA's ``task.validate()`` runs the evaluator every step, and 10 ``string_match+fuzzy_match`` + 8 ``page_image_query`` tasks need an LLM judge that current keys can no longer reach.
> - 3-task smoke output: `Cold-start-out-browsergym/vwa_smoke3/` — 268/444/279 ran with the new defaults at ~21 s/step weighted, validating no `full7-smoke`-style inflation.
> - Post-install 3-task re-smoke: tasks 433 (reddit+image, ``program_html`` evaluator), 92 (classifieds, ``url_match``), 135 (classifieds+image, ``string_match+fuzzy_match``). All three reset and step clean **after** the judge-model patch; before it, 135 hit ``NotFoundError: gpt-4-1106-preview does not exist or you do not have access to it``.

---

## 1. Why this baseline exists (one paragraph)

The project's main contribution is **skill discovery, adaptation, and
transfer** measured by the Phase-5/6 cross-domain admit-rate matrix.
VisualWebArena is one of four *target* domains in that matrix
(alongside OSWorld, AssistantBench, MiniWoB++). For the publication
we need a defensible **teacher VWA pass-rate** for each frontier API
model so the comparison reads:

> "Our gymv-derived skills transfer to VWA with admit rate X%. For
> context, frontier teachers achieve native VWA pass-rates of GPT-5.5:
> A%, Claude Sonnet 4.6: B%, Gemini 2.5 Pro: C%, Qwen3-VL 235B: D%."

The pre-existing 200-task subset gives **±3.5%** binomial CI on the
global pass-rate (a stronger margin than OSWorld's 50-task ±7%
because VWA's web actions are cheaper per step → bigger n is
affordable). The full 910-task suite is reserved for a possible
reviewer-rebuttal pass; **not on the critical path**.

## 2. Scope and non-goals

In scope:

* Run the same 200-task pinned subset on 4 API models with identical
  agent configuration (`--reasoning_effort low`, `--max_steps 12`).
* Produce a 4-CSV + readme.md comparison report identical in shape to
  OSWorld's `runs/osworld_baseline_50/compare/` so the publication
  table renders directly.
* Save full rollouts so the trajectories can later feed the skill
  bank (delta over the 0 VWA skills currently in the bank).

Out of scope:

* Beating the VWA leaderboard. We're not chasing the GPT-4V 16.4%
  number. `--reasoning_effort low` plus `--max_steps 12` will land
  somewhat below `medium` / `high` with `--max_steps 30` (the
  literature setting), and that is fine — the comparison among the
  4 teachers is what we care about, not the headline.
* Re-running the cross-domain transfer matrix (Phase-5/6 Stage 6).
* Adding any steering / memory / reflection / self-verify modules to
  the BrowserGym driver. The OSWorld experience (May 3 `full7-smoke`
  → 95.6 s/step) is the reason we're keeping VWA bare-bones.

## 3. Subset selection — 200 tasks, deterministic, **already pinned**

Source catalog: `visualwebarena/test_raw.json` (910 tasks shipped
with the upstream `visualwebarena` Python package).

Pinned file: `cold_start/task_samples/browsergym_visualwebarena_200.txt`.
Seed=0. Stratified by `(primary_site × has_image × overall_difficulty)`
with proportional quotas + a +1 floor per non-empty stratum. Covers
**116 of 152 distinct intent templates**.

| Site combination | n | classifieds dep? | image dep? |
|---|---|---|---|
| `('shopping',)` | 101 | no | mixed |
| `('classifieds',)` | 52 | **yes** | mixed |
| `('reddit',)` | 37 | no | mixed |
| `('reddit', 'wikipedia')` | 7 | no | mixed |
| `('reddit', 'shopping')` | 3 | no | mixed |
| **Total** | **200** | **52 need classifieds** | **72 need VWA-homepage input_images** |

| Difficulty | n |
|---|---|
| `easy` | 46 |
| `medium` | 82 |
| `hard` | 72 |

**No new file to build.** The existing pin already covers the same
"identical 200 tasks across all 4 models" property the OSWorld plan
calls out via §3 — every model is scored on the same paired set.

If the upstream `test_raw.json` ever changes, **regenerate** with
`python cold_start/task_samples/build_browsergym_diverse_200.py` and
**redo the comparison from scratch** (do NOT keep mixing pre- and
post-update results).

## 4. Models, IDs, and per-provider parameters

Mirror OSWorld plan §4 exactly. The same 4 providers, the same
`api_keys.py`-routed setup:

| Provider tag | OpenRouter / OpenAI id | Route | `reasoning_effort` |
|---|---|---|---|
| `gpt5` | `gpt-5.5` | OpenAI direct (via `--api_key` from `api_keys.py`) | `low` ← driver default since `90628b8` |
| `claude-sonnet` | `anthropic/claude-sonnet-4.6` | OpenRouter | `low` (silently dropped — non-reasoning model) |
| `gemini-pro` | `google/gemini-2.5-pro` | OpenRouter | `low` (silently dropped) |
| `qwen3-vl` | `qwen/qwen3-vl-235b-a22b-instruct` | OpenRouter | `low` (silently dropped) |

Note: BrowserGym driver currently defaults to **`gpt-5.5`** (newer than
the `gpt-5.4` OSWorld plan picked). Pin the same model that OSWorld
uses for cross-benchmark comparability. **Decision needed (§13 #1).**

Common flags across all 4 runs:

```
--tasks $(cat cold_start/task_samples/browsergym_visualwebarena_200.txt | tr '\n' ' ')
--episodes 1
--max_steps 12                    # web tasks need more headroom than the launcher's 8 default
                                  # (smoke task 444 hit max at 8 mid-Magento-nav). Literature
                                  # uses 15-30; 12 is the budget compromise that fits the
                                  # ~9 hr/model wall-clock target.
--reasoning_effort low            # default since 90628b8 (BrowserGym driver)
--temperature_action 0.0
--temperature_schema 0.0
--no_save_frames                  # disk guard (per-step JSON sidecar still written)
--resume
-v
```

Per-provider differences (will be handled by a new
`run_browsergym_multimodel.sh` mirroring OSWorld's, see §5):
provider tag → `--model` → `--api_endpoint` → `--api_key` → output dir.
Nothing else differs.

## 5. Code prerequisite — `run_browsergym_multimodel.sh`

OSWorld has `cold_start/run_osworld_multimodel.sh` mapping
`--provider gpt5 / claude-sonnet / gemini-pro / qwen3-vl` to the right
model id + endpoint + key. **There is no equivalent for BrowserGym
yet.** Write one with the same shape:

* Same `--list_providers` table.
* Same `provider_default_eval_mode` style (but BrowserGym has no
  eval_mode tier — instead set `--reasoning_effort low` and forward
  the rest).
* Same `provider_to_model` + `provider_to_endpoint` maps.
* Auto-source `cold_start/visualwebarena_env.sh` when the chosen
  provider/task plan involves VWA.

LOC budget: **~120 LOC**, mostly mechanical copy of the OSWorld
script with the launcher-target flipped to
`run_coldstart_actor_browsergym.sh`.

Three smaller pieces also need to land before the 200-task launch:

| Item | File | LOC | Risk |
|---|---|---|---|
| `run_browsergym_multimodel.sh` | new in `cold_start/` | ~120 | Low (clone-and-edit OSWorld variant) |
| Aggregator | `scripts/aggregate_vwa_baseline.py` | ~120 | Low (pattern-shared with `aggregate_osworld_baseline.py`) |
| Watchdog regression test | `tests/test_browsergym_watchdog_log.py` | ~30 | Trivial |
| **Total** | | **~270 LOC** | |

The aggregator can lift > 80% of `aggregate_osworld_baseline.py` —
both read per-task `rollout_summary.json` written by the same
`load_rollouts.aggregate_run_pass_at_1` helper.

## 6. Wall-clock and cost budget

Real per-step latency from the May 3 3-task smoke (gpt-5.5 `low`,
`max_steps=8`, image-free non-classifieds tasks, headless Chromium
on the live WA stack):

| Task | Site | steps | elapsed | s/step | Watchdog |
|---|---|---|---|---|---|
| 268 | reddit (medium) | 3 | 73.4 s | 24.5 | ok (reward=1.00) |
| 444 | shopping (medium) | 8 | 237.7 s | 29.7 | ok (reward=0) |
| 279 | reddit+wiki (hard) | 8 | 92.1 s | 11.5 | FAST (reward=0, error page) |
| **Mean (weighted)** | | **19** | **403.2 s** | **~21.2** | — |

VWA per-step time is dominated by **page complexity**, not
`reasoning_effort`. Magento (shopping) ≈ 30 s/step, Postmill (reddit)
≈ 22 s/step, error pages ≈ 12 s/step.

Weighted to the 200-set distribution:
`(101·30 + 52·22 + 37·22 + 7·22 + 3·25) / 200 ≈ 26 s/step`.

Per-task wall-clock at `--max_steps 12` (no early-exit):
`reset 15 s + 12 step × 26 s = 327 s ≈ 5.5 min`.
With ~25 % early-exit on solved tasks (smoke saw 1 / 3): mean ≈ **4 min/task**.

| Provider | $/task | step latency | task wall-clock | 200-task wall-clock (1 machine, sequential) | 200-task LLM cost |
|---|---|---|---|---|---|
| GPT-5.5 (low) | $0.35 | ~24 s | ~4.5 min | ~15 h | $70 |
| Claude Sonnet 4.6 | $0.65 | ~16 s | ~3.5 min | ~12 h | $130 |
| Gemini 2.5 Pro | $0.30 | ~17 s | ~3.7 min | ~12 h | $60 |
| Qwen3-VL 235B-A22B Instruct | $0.07 | ~20 s | ~4.5 min | ~14 h | $15 |
| **All 4 (single machine, 4 background processes)** | — | — | — | **wall-clock = max ≈ 15 h** | **~$275** |
| **All 4 (single machine, sequential)** | — | — | — | ~53 h ≈ 2.2 d | ~$275 |

Notes:

* Numbers reflect the May 3 smoke s/step and the 200-set's heavy
  shopping skew. ±25 % uncertainty on the wall-clock until 50 tasks
  are in.
* Budget assumes the upstream `install_visualwebarena_sites.sh` +
  VWA-homepage container are running so all 200 tasks can complete.
  Without them, 52 + (some) × `with_image` tasks fail at reset and
  cost / wall-clock both drop ~25 %, but the comparison loses the
  classifieds slice.
* The 4-process parallel option works on this machine specifically
  because (a) GPUs 4-7 are idle, (b) VWA needs no GPU, (c) ~56 CPU
  cores are unused after the live training fleet, (d) 1.3 TB RAM
  free. See [§9 — Pre-flight] for the actual checks.

## 7. Pipeline and output layout

```
WORKSPACE_ROOT/
├── cold_start/task_samples/
│   └── browsergym_visualwebarena_200.txt        ← already exists
├── runs/vwa_baseline_200/
│   ├── manifest.json                            ← step 0 of §10
│   ├── gpt5/                                    ← --output_dir for gpt5 run
│   │   ├── visualwebarena.351/episode_000.json
│   │   ├── visualwebarena.351/rollout_summary.json
│   │   ├── ...
│   │   ├── batch_rollout_summary.json
│   │   └── run_metadata.json
│   ├── claude-sonnet/...
│   ├── gemini-pro/...
│   ├── qwen3-vl/...
│   └── compare/
│       ├── pass_at_1_overall.csv                ← 4 rows × {n, pass, rate, Wilson 95}
│       ├── pass_at_1_per_domain.csv             ← 4 × 5 site grid
│       ├── per_task.csv                         ← 200 × 4 ⊂ 0/1 matrix
│       └── readme.md                            ← human-readable rendering
```

Two structural differences from OSWorld:

1. **No `<provider>/<domain>/<task_id>/...`** — BrowserGym writes
   one directory per `target_safe_id` (= `visualwebarena.NNN`),
   *flat* under `<provider>/`. The aggregator must group by site
   *post-hoc* by reading the upstream `test_raw.json` site field.
2. **No `evaluation_examples/test_nogdrive_subset200_v1.json`
   equivalent** — the BrowserGym launcher consumes a flat `--tasks
   $(...)` list, so the pinned subset stays as a `.txt` line list,
   not a JSON catalog.

## 8. Aggregation script (~120 LOC, new file)

`scripts/aggregate_vwa_baseline.py`:

* Reads each `<provider>/<safe_id>/rollout_summary.json` and groups
  by site via the upstream `test_raw.json`.
* Honors `eval_score is None` as 0.0 (re-uses
  `cold_start/load_rollouts.py:aggregate_run_pass_at_1`).
* Emits the 4 CSVs above + a markdown summary.
* Validates: every (provider, task_id) cell is filled. Flags partial
  runs with a clear "K/200 missing" note.
* Cross-provider agreement breakdown (same as OSWorld aggregator).

Difference from `aggregate_osworld_baseline.py`: instead of
`<domain>/<task>` walks the layout `<safe_id>/`, and adds a
"site" axis via the test_raw.json lookup. The Wilson 95 % helper +
CSV / markdown writers can be lifted verbatim.

## 9. Pre-flight checklist (per machine)

Run **before** the actual eval on each machine:

```bash
cd /workspace/Multi-hop-Reasoning-VLM-Agent

# 1. api_keys.py present at workspace root with at minimum
#    openrouter_api_key (and openai_api_key for the gpt5 machine).
ls /workspace/api_keys.py

# 2. browsergym conda env is healthy and required imports succeed,
#    AND the upstream judge-model patch is in place.
/workspace/miniconda3/envs/browsergym/bin/python -c \
    "import browsergym.core, browsergym.visualwebarena, playwright; \
     print('OK')"
SP=$(/workspace/miniconda3/envs/browsergym/bin/python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')
grep -q '_VWA_JUDGE_MODEL' \
    "${SP}/visualwebarena/evaluation_harness/helper_functions.py" \
    && echo "  judge-model patch: OK" \
    || { echo "  judge-model patch: MISSING — run install/patch_vwa_judge_model.sh"; exit 1; }

# 3. Live WA stack is up. Specifically these 6 services serving HTTP 2xx:
for url in \
    http://localhost:7770       \  # shopping (Magento)
    http://localhost:9999       \  # reddit (Postmill)
    http://localhost:8888       \  # wikipedia (kiwix)
    http://localhost:4399       \  # WA homepage (shared with VWA today)
    http://localhost:9980       \  # Classifieds (OSClass) — ONLY if §4-5 install ran
    ; do
    code=$(curl -m 5 -s -o /dev/null -w '%{http_code}' "$url")
    echo "  $url -> $code"
done

# 4. visualwebarena_env.sh present (auto-sourced by the launcher).
ls cold_start/visualwebarena_env.sh

# 5. Live machine resources can absorb 4-process parallel.
free -h | grep ^Mem | awk '{ printf "  RAM free: %s of %s (need ≥4 GiB headroom for 4 procs)\n", $4, $2 }'
nproc | awk '{ printf "  CPU cores: %d (need ≥8 free)\n", $1 }'
df -h /workspace | awk 'NR==2 { printf "  Disk free: %s (need ≥40 GiB for 4 model output dirs)\n", $4 }'

# 6. 60-second LLM-only smoke for the provider this machine will run.
/workspace/miniconda3/envs/browsergym/bin/python \
    cold_start/smoke_multimodel.py --provider $PROVIDER

# 7. Real-rendering 1-task smoke for the same provider.
bash cold_start/run_coldstart_actor_browsergym.sh \
    --tasks browsergym/visualwebarena.268 \
    --episodes 1 --max_steps 8 \
    --output_dir Cold-start-out-browsergym/preflight_$PROVIDER \
    -v 2>&1 | grep '\[WATCHDOG\]'
```

Hard fail conditions: any of 1-4 missing → fix before launching.
Step 5 is informational (the actual check we lean on while the
training fleet is running). Step 7 must show `[WATCHDOG ok]` or
`[WATCHDOG FAST]` — anything else means re-investigate before the
4-model run.

## 10. Execution sequence

1. **(coordinator only)** Land the three small code prereqs (§5):
   * `cold_start/run_browsergym_multimodel.sh` (~120 LOC).
   * `scripts/aggregate_vwa_baseline.py` (~120 LOC).
   * `tests/test_browsergym_watchdog_log.py` (~30 LOC).
2. **(coordinator only)** Optional: run the upstream
   `bash install/install_visualwebarena_sites.sh` to bring up
   Classifieds (~10 min) + the VWA-specific homepage container
   (~20 min DIY). Required only if the run includes classifieds /
   image-bearing tasks. **Decision tree:**
   * Skip → 91 / 200 fully runnable + (image-free non-classifieds);
     remaining 109 fail at reset. Cell-level loss in
     `compare/per_task.csv`.
   * Run → 200 / 200 runnable. ~30-50 min one-shot install cost.
3. **(per machine)** Pre-flight checklist §9.
4. **(per machine)** Launch:
   ```bash
   bash cold_start/run_browsergym_multimodel.sh \
        --provider <provider_tag> \
        --tasks_file cold_start/task_samples/browsergym_visualwebarena_200.txt \
        --episodes 1 \
        --max_steps 12 \
        --reasoning_effort low \
        --resume \
        --output_dir runs/vwa_baseline_200/<provider_tag> \
        --no_save_frames -v \
        2>&1 | tee /tmp/vwa_baseline_200_<provider_tag>.log
   ```
   For 4-process parallel on this same machine:
   ```bash
   for prov in gpt5 claude-sonnet gemini-pro qwen3-vl; do
       nohup bash cold_start/run_browsergym_multimodel.sh \
           --provider "$prov" \
           --tasks_file cold_start/task_samples/browsergym_visualwebarena_200.txt \
           --max_steps 12 --reasoning_effort low --resume \
           --output_dir runs/vwa_baseline_200/"$prov" \
           --no_save_frames -v \
           > "runs/vwa_baseline_200/$prov.log" 2>&1 &
   done
   ```
5. **(coordinator)** Once all four providers' runs finish:
   `python scripts/aggregate_vwa_baseline.py
        --root runs/vwa_baseline_200/`. Writes `compare/`.
6. **(coordinator)** Eyeball `compare/readme.md`. Done.

## 11. Acceptance criteria

The baseline is "done" when:

| Check | How to verify |
|---|---|
| All 4 providers ran the identical 200-task subset | `compare/per_task.csv` has 200 × 4 = 800 cells, none missing — OR a clear partial-coverage note for the 52 + 72 cells skipped by the operator's choice in step 2 of §10. |
| No silent failures hidden by `eval_score=None` | `aggregate_vwa_baseline.py` prints `null_count_per_provider` and it matches the launcher's reported truncations within ±5 %. Otherwise audit the outliers. |
| Per-provider pass-rate is statistically distinct | 95 % Wilson CIs on the global pass-rate either separate the providers visually OR the report explicitly says "within statistical noise". |
| Watchdog showed no SLOW (>30 s/step) episodes | `grep '\[WATCHDOG SLOW\]' runs/vwa_baseline_200/*.log` returns 0 lines. If it returns >5 lines, investigate before publishing — the May-3 OSWorld `full7-smoke` lesson. |
| reasoning_effort actually applied for gpt-5.5 | `runs/vwa_baseline_200/gpt5/<safe_id>/rollout_summary.json` has `reasoning_effort=low` in the metadata, and gpt-5.5 step latencies stay in the 18-25 s range (anything >40 s implies the param was ignored). |
| Judge-model patch is in place | `grep _VWA_JUDGE_MODEL $(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/visualwebarena/evaluation_harness/helper_functions.py` returns 2 hits. Otherwise 18/200 tasks (10 fuzzy_match + 8 page_image_query) will silently fail at `env.step()` and pollute the comparison. |

If any check fails, debug before publishing. Do not publish a partial
table.

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `gpt-5.5` direct API has the same tools-vs-`reasoning_effort` quirk gpt-5.4 had | Medium | The May-3 patch in `_chat_completion` (OSWorld driver) strips `reasoning_effort` for tool-bearing gpt-5.x calls. The BrowserGym driver inherits the same `_chat_completion` shape — verify the patch is also there before launch. **Open** [§13 #2]. |
| Upstream judge-model patch gets reverted by `pip install` / env rebuild | Medium | `install/patch_vwa_judge_model.sh` is idempotent text-patch over `site-packages/visualwebarena/evaluation_harness/helper_functions.py`. **Add it to the conda-env bring-up runbook**, and run it as part of the pre-flight on every machine. The pre-flight check in §9 step 2 below verifies the sentinel `_VWA_JUDGE_MODEL` exists in the helper file. |
| Gemini 2.5 Pro safety blip on vision calls (5 % empty `finish_reason=error`) | High | Driver fall-back already handles. Aggregator flags providers with > 10 % empty-content steps. |
| OpenRouter quota / regional outage | Low–Medium | Stagger launches by ~30 s. Driver retries with 60 s back-off. |
| Same WA stack mutate-contamination across 4 parallel workers | Medium | Postmill `vote/comment/post` and Magento `cart/checkout` are mutate-heavy; if all 4 workers happen to land on the same task at the same moment they can step on each other's session state. **Mitigation**: launcher already uses task-id-keyed working dirs, but the underlying DB state is shared. Practical fix: scheduling order = round-robin by site so 4 workers naturally hit different services. The 200-set is well-distributed; in practice only ≤2 workers will be on shopping at once. |
| One task hangs past 12-step cap and burns 10+ min for `eval_score=None` row | High (truncation rate ~50 % at this step budget) | Accept truncation. Aggregation honors `--treat-null-as-zero`. Truncations affect all 4 providers equally → don't bias the comparison. |
| Disk full mid-run (`/workspace` is at 80 %; ~290 GB free) | Low–Medium | `--no_save_frames` keeps per-task ~5 MB. 200 × 4 = 800 tasks × 5 MB = 4 GB total. Pad: monitor with `df -h /workspace` between providers. |
| RAM/CPU starvation against the live training fleet | Low | Snapshot 2026-05-03 03:05 AM: 56 idle cores, 1.3 TB free RAM. 4 Chromium workers ≈ 4 GB RAM + ~8 cores. Budget = 1 % of free pool. |

## 13. Open questions

1. **Pin gpt-5.4 or gpt-5.5 for cross-benchmark comparability?**
   OSWorld plan picked `gpt-5.4`. BrowserGym driver defaults to
   `gpt-5.5`. If we want a single GPT row that works for both
   benchmarks, we should pin the *same* model. **Recommendation**:
   default to `gpt-5.4` for both (more conservative since
   `gpt-5.4` is the one actually shipped in `api_keys.py`-routed
   tests). This needs a `--model gpt-5.4` flag passthrough on the
   new BrowserGym multimodel launcher.
2. **Verify the May-3 `_chat_completion` reasoning-effort-strip
   patch is in the BrowserGym driver too**, or port it. The
   OSWorld driver `generate_cold_start_actor_osworld.py` had it
   landed; the BrowserGym driver `generate_cold_start_actor_browsergym.py`
   has its own `_chat_completion` (different file) that may or
   may not have inherited the same patch. Audit and align before
   launching.
3. **Should `max_steps=12` or `max_steps=15`?** 12 fits the
   ~9-15 hr/model budget; 15 is closer to literature but pushes to
   ~16 hr/model. **Recommendation**: 12 for the first pass (we
   value the 4-model comparison more than the absolute pass-rate);
   bump to 15 only if reviewers ask for it.
4. **Do we score `--reasoning_effort medium` for gpt-5.5 on the
   same subset as a footnote?** Cost: ~$120 extra, ~7 hr extra
   wall-clock. Yields a "low vs medium" data-point. **Recommendation:**
   defer; can be added later.
5. **Do we run on the full 910-task suite for any single model?**
   Only Qwen3-VL (~$70 + 14 hr single machine) is cheap enough to
   make a "fuller single-model" run insurance against subset-bias
   accusations.

## 14. LOC budget for the prereqs

| Item | File | LOC | Risk |
|---|---|---|---|
| Multi-provider router | `cold_start/run_browsergym_multimodel.sh` | ~120 | Low (clone OSWorld) |
| Aggregator | `scripts/aggregate_vwa_baseline.py` | ~120 | Low (lift 80 % of OSWorld aggregator) |
| Watchdog regression test | `tests/test_browsergym_watchdog_log.py` | ~30 | Trivial |
| Optional: `--tasks_file` flag (read newline-list) | `cold_start/run_coldstart_actor_browsergym.sh` | ~10 | Trivial |
| Optional: gpt-5.4 model pin in launcher | `cold_start/run_browsergym_multimodel.sh` | ~3 | Trivial |
| **Total** | | **~280 LOC** | |

All items are pure-additions; no existing code path's defaults
change beyond what `90628b8` already shipped.

## 15. Sequencing rationale

We land prereqs **before** any launch because:

1. The watchdog log (already shipped) is the single best
   inflation guard we have, and the regression test locks it in
   place before the next refactor passes through the driver.
2. The deterministic 200-pin file already exists and is committed,
   so no machine can accidentally diverge.
3. The aggregator written first means the very first finished
   provider can be eyeballed in isolation; we catch any
   per-provider plumbing issue before all 4 have spent wall-clock.
4. The multimodel launcher is the most likely place a typo (wrong
   model id, wrong endpoint) will silently waste an entire 12-hour
   run; landing it as a separate small commit with `--list_providers`
   self-check makes that mistake near-impossible.

The prereqs are small enough (~280 LOC) and risk-free enough that
landing them in one short session before kicking off the eval is
strictly faster than launching, hitting an issue, restarting.

## 16. What's already done as of 2026-05-03 04:30 AM

| Item | Where | Status |
|---|---|---|
| Driver `--reasoning_effort` default `unset → low` | `cold_start/generate_cold_start_actor_browsergym.py` | ✓ commit `90628b8` |
| Per-task `[WATCHDOG]` log | same | ✓ commit `90628b8` |
| Sourceable `cold_start/visualwebarena_env.sh` | new | ✓ commit `e30c473` (post-install update) |
| 200-task pinned subset | `cold_start/task_samples/browsergym_visualwebarena_200.txt` | ✓ pre-existing |
| Real-rendering 3-task smoke (gpt-5.5 low) | `Cold-start-out-browsergym/vwa_smoke3` | ✓ ~21 s/step weighted, all watchdog ok/FAST |
| Classifieds (OSClass + MySQL) on :9980 | live container | ✓ docker compose up, DB seeded |
| VWA-specific homepage on :4400 | live container `vwa_homepage` | ✓ serves `static/input_images/{classifieds,reddit,shopping}/` |
| Judge-model patch (gpt-4-1106-preview → $VWA_JUDGE_MODEL) | `install/patch_vwa_judge_model.sh` | ✓ idempotent text-patch + auto-applied by install script |
| Post-install 3-task re-smoke (433 / 92 / 135) | inline | ✓ all three reset+step clean after judge-model patch |
| Multi-provider router for BrowserGym | new | ⏳ not yet (§5) |
| Aggregator | new | ⏳ not yet (§8) |
