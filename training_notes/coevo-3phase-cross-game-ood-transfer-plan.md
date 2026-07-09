# Co-evolution 3-phase training plan — source GRPO → in-domain few-shot → OOD transfer

> **Status (2026-05-03 PM, refreshed):** 🟢 **PLAN — game roster
> data-refreshed; curriculum mechanism + shared-bank pipeline locked.**
> Phase-1 curriculum is **6 games trained sequentially at 15 GRPO steps
> each (90 steps total)** with bank + LoRA carry-over between games.
> The roster was **swapped 2026-05-03 PM** after the new
> `Cold-start-out-gymv/latest` SFT teacher data showed that two of the
> originally locked Phase-1 picks (`StreetsOfRage2`, `Strider`) had
> very different teacher reward to what the older eval suggested
> (Strider: GPT-5.4 = 0, Qwen3-VL = 0; SoR2 healthy at 200–400 across
> all 4 frontier teachers but pairs more cleanly with `AlteredBeast` as
> a Phase-2 in-genre transfer target than as a Phase-1 source). The
> updated roster pulls the data-richest 4 gymv games into Phase 1
> (`ThunderForceIII`, `AlteredBeast`, `Columns`, `DynamiteHeaddy`) and
> uses the remaining 4 gymv games as Phase-2 transfer targets, each
> paired in-genre with a Phase-1 source so the cross-game skill
> translator (`skill_agents/skill_bank/translate_for_target.py`) has the
> closest possible source vocabulary to re-ground onto. The shared-bank
> + per-boundary translation pipeline (`config.bank_mode='shared'` +
> `TRANSLATE_ON_BOUNDARY=1`) landed alongside this refresh and is the
> default carrier for Phase-2 (§7) so the "shared bank rescues
> partial-signal games" hypothesis becomes directly testable on
> `Strider`. 2 decisions still pending pre-launch
> (§11.1 hyperparam audit; §11.4 keep/drop the 5-step inference-only
> Phase-2 baseline). The plan extends the COS-PLAY paper's per-game
> training (Appendix C / Table 3) into a cross-game skill-transfer +
> cross-domain few-shot adaptation experiment; the **cross-game
> transfer** protocol and **OOD few-shot adaptation** are explicitly
> identified as open problems in the paper's Limitation / Future Work
> section and are this plan's novel contribution.

> **Cross-refs:**
> - [COS-PLAY paper](https://arxiv.org/pdf/2604.20987) — Appendix C,
>   Table 3 (per-game hyperparams); §5.3 + Table 2 (within-game skill
>   reusability); §6 Conclusion (cross-domain transfer = future work).
> - [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md)
>   — backbone stack, the 5 LoRA adapters, current Stage-1 readiness.
> - [`implementation_notes/coevolution-cross-domain-integration.md`](../implementation_notes/coevolution-cross-domain-integration.md)
>   — the C/A/D-layer cross-domain measurement infrastructure that
>   Phase-3 of this plan reuses.
> - [`scripts/run_2048.sh`](../scripts/run_2048.sh) /
>   [`run_tetris.sh`](../scripts/run_tetris.sh) /
>   [`run_super_mario.sh`](../scripts/run_super_mario.sh) — existing
>   per-game wrappers; this plan adds two more (`run_candy_crush.sh`
>   and a generic-gymv wrapper) and a 3-phase orchestrator script.
> - [`trainer/coevolution/config.py`](../trainer/coevolution/config.py)
>   — `CoEvolutionConfig.curriculum_schedule`, `--seed-bank-dir`,
>   `total_steps`, GRPO hyperparam fields.
> - [`canvases/coevolution-architecture-overview.canvas.tsx`](../../../.cursor/projects/workspace/canvases/coevolution-architecture-overview.canvas.tsx)
>   — visual outline of the per-step training cycle.

---

## 1. Why this plan exists (one paragraph)

The project's headline contribution is **skill discovery, adaptation,
and transfer**. The COS-PLAY paper (Appendix C / Table 3) shows each
game converges in 7–25 GRPO steps with the bank crystallising into 6–64
reusable skills, but **all six paper experiments train a single game in
isolation**. The paper's own Conclusion explicitly defers
cross-game and cross-domain transfer: "We also aim to improve
cross-domain transfer, so that learned skills generalize across
multi-modal games, agentic environments, and broader visual reasoning
tasks." This plan turns that future work into a concrete 3-phase
curriculum: (1) reproduce the paper's per-game training on six gymv
games, (2) measure how fast the merged skill bank adapts to held-out
games, (3) measure how fast it adapts to entirely new domains
(`video`, `visual_reasoning`, optionally `browser` / `osworld`).

## 2. Scope and non-goals

In scope:

* **Phase 1** — sequential GRPO curriculum across six source games
  (4 gymv + `candy_crush` + `tetris`) at **15 steps each (90 total)**;
  bank + LoRA carry over between games and a snapshot is dropped after
  each game so any Phase-1 game can be re-evaluated post-hoc against
  its own end-of-game checkpoint.
* **Phase 2** — few-shot adaptation on six held-out games (the four
  unused gymv benchmark games + `2048` + `super_mario_bros`) seeded
  with the post-Phase-1 bank + LoRA, reported at two adaptation
  budgets (5-step inference-only baseline + 15-step GRPO main result).
* **Phase 3** — same two-budget few-shot protocol on the cross-domain
  targets `video` and `visual_reasoning` (cheap VLM-only envs);
  `browser` / `osworld` deferred behind a go/no-go gate after
  Phase-1 + Phase-2 results land.
* A **per-game snapshot capture** at the end of each Phase-1 game so
  the rolling-curriculum LoRA never strictly dominates per-game
  best-checkpoint reporting (§4.4 below).
* An **independent-runs-merge ablation** (kept as Option A in §11.2)
  to back-stop the curriculum design if late-game LoRA drift erases
  early-game competence; only triggered if §12 R6 fires.

Out of scope:

* Multi-player social games (Avalon, Diplomacy). Paper's per-game
  training already covers them; their bank dynamics differ enough
  (self-play, role-conditioned skills) that they are a separate
  experimental track.
* Beating the OSWorld / VWA leaderboards. Phase 3's role is
  "few-shot adapts faster than no-bank baseline", not "SOTA".
* Replacing the per-step Layer-A transfer gate or Layer-D dashboard
  hook. They run as before; this plan only adds an outer-level
  curriculum on top.
* Re-running cold-start SFT. Phase 1 starts from the existing
  cold-started `runs/sft_coldstart_20260502_025737` adapters.

## 3. Paper grounding — what's settled vs novel

| Question | Answer | Evidence |
|---|---|---|
| Per-game step budget | **7–25 steps**, mean ~15 | Appendix C, Table 3 |
| Per-game hyperparams | LR 1e-5–5e-5, KL 0.04–0.08, clip 0.10–0.20, max-epochs 2–4, advantage clip on harder games | Appendix C, Table 3 |
| Bank size after training | 6 (Tetris) – 64 (Diplomacy) skills, average ~30/skill instances | §5.3, Table 2 |
| Cross-game GRPO transfer | **Not measured** in the paper | §6 Conclusion (future work) |
| Cross-domain few-shot adaptation | **Not measured** in the paper | §6 Conclusion (future work) |
| "Few-shot" definition in paper | "≤25 iterations to reach strong performance from scratch" — about the **whole** training, not adaptation | Abstract + §5 |

This plan uses (1)–(3) verbatim, treats (4)–(5) as the novel
contribution, and is careful **not** to conflate (6)'s "few-shot"
with our Phase-2 / Phase-3 definition (see §7 below).

## 4. Phase 1 — sequential source GRPO curriculum

### 4.1 Game roster (refreshed 2026-05-03 PM, data-driven from new SFT cold-start)

Six games trained one after another at **15 GRPO steps each** (90
steps total). Two come from COS-PLAY Table 3 (`tetris`, `candy_crush`)
so the merge-rate of bank / reward against the paper still has an
anchor; four come from the
[`gymv_wrapper` 8-game benchmark scope](../baselines/README.md#gym-v-benchmark-scope)
(commit `4f97dd6`) — re-picked from the new `Cold-start-out-gymv/latest`
4-backbone teacher table so the SFT cold-start LoRAs train on demos
where every Phase-1 game has **non-zero teacher reward across all 4
frontier teachers (GPT-5.4 / Claude-4.6-Sonnet / Gemini-3.1-Pro /
Qwen3-VL-235B)**:

| # | Slug | Genre | Teacher band (min–max across 4 frontier rows) | Why Phase 1 (mining source) |
|---|---|---|---|---|
| 1 | `gymv_thunder_force_iii` | shmup | 269–750 (GPT 306, Claude 269, Gemini 725, Qwen3-VL 750) | Strong, varied teacher signal across all four teachers; weapon-switching mechanic mines diverse skills. Manageable scale (~hundreds). |
| 2 | `gymv_altered_beast` | beat-em-up | 119–425 (GPT 119, Claude 294, Gemini 425, Qwen3-VL 263) | All teachers score; combat + transformations = rich action vocabulary. Genre-orthogonal to TF3 (different action patterns ⇒ different mined skills). |
| 3 | `gymv_columns` | puzzle | 63–160 (GPT 154, Claude 63, Gemini 99, Qwen3-VL 132) | Only puzzle gymv we have; small healthy scale; spatial-reasoning genre ⇒ partial transfer to `tetris` / `candy_crush`. |
| 4 | `gymv_dynamite_headdy` | action-platformer | 75–94 (GPT 94, Claude 94, Gemini 81, Qwen3-VL 75) | Most diverse mechanics in the gymv set (Headdy's modular abilities). In-genre source for `Strider` Phase-2 transfer test. |
| 5 | `candy_crush` | match-3 | n/a (paper Figure 4) | LR 5e-5 · KL 0.05 · clip 0.20 · max_ep 4 · no adv-clip · ep/step 8 |
| 6 | `tetris` | spatial puzzle | n/a (paper Figure 4) | LR 2e-5 · KL 0.08 · clip 0.10 · max_ep 2 · adv-clip 3.0 · ep/step 8 |

Curriculum order is **as listed above** — start with the highest-signal
shmup (`ThunderForceIII`, all 4 teachers ≥ 269), swap genre
(`AlteredBeast` beat-em-up) to keep the curator from over-fitting one
action vocabulary, swap again to puzzle (`Columns`) so the
skill-segmenter learns sparse-reward grids, then `DynamiteHeaddy` for
its richer action-platformer mechanics, and finish on the two paper
Table-3 games (`candy_crush`, `tetris`) as terminal anchors comparable
to Figure 4. The two intentionally-omitted gymv games are
`SpaceHarrierII` (~30× larger reward scale than every other gymv game
— moved to Phase 2 as a scale-invariance transfer test rather than
letting it dominate Phase-1 cross-game aggregates) and
`StreetsOfRage2` / `Airstriker` / `Strider` (paired in-genre as
Phase-2 transfer targets — see §7.1). Re-ordering is a §11 deferred
follow-up if R6 (LoRA drift) fires.

> **Why this differs from the prior locked roster.** The earlier picks
> (`SpaceHarrierII`, `StreetsOfRage2`, `Columns`, `Strider`) came from
> a 4-backbone success-rate baseline that didn't reflect the
> `frame_skip=8` cold-start data. In the new table:
>
> * `SpaceHarrierII` teacher reward sits at **14 469–29 431** — ~30×
>   the next-largest gymv game (`ThunderForceIII` at 269–750). Even
>   with per-game normalization (§4.5) the GRPO advantage variance
>   during a SH2 phase would drown the signal-to-noise on per-step
>   monitoring; moving it to Phase 2 turns the scale outlier into a
>   *test* ("does TF3-mined shmup skill survive the 30× reward jump?")
>   rather than a noise source.
> * `Strider` is partial-signal (GPT-5.4 = 0, Qwen3-VL = 0; Claude /
>   Gemini score 31 / 113). Mining skills from a poisoned 50 %-zero
>   teacher distribution is exactly the SFT pathology we identified in
>   the prior roster's SoR2 / Strider regression — moving Strider to
>   Phase 2 lets the shared-bank + translation pipeline attempt the
>   rescue rather than handicapping Phase 1 with poisoned demos.
> * `StreetsOfRage2` is now healthy in this newer data (200–408 across
>   all 4 teachers) but pairs more cleanly with `AlteredBeast` as the
>   in-genre Phase-2 transfer target ("does AB-mined beat-em-up skill
>   transfer to SoR2?") than as a parallel Phase-1 source.

### 4.5 Reward normalization across phases

Per-game raw rewards span ~3 orders of magnitude across the 8 gymv
games (Strider 0–112 → SpaceHarrierII 14 k–29 k). Within a phase this
is a non-issue — GRPO normalizes advantages inside the rollout group,
so absolute reward magnitude doesn't reach the optimizer. **Across
phases**, however, every aggregate metric (W&B mean, Layer-D
dashboard transfer matrix, best-checkpoint selection, curriculum
phase-success thresholds) is reward-magnitude-biased toward whichever
phase has the largest absolute reward.

The plan adds a **teacher-anchored, additive normalization layer**:

```
r_norm[game] = clip(r_raw[game] / r_teacher_anchor[game], 0.0, 2.0)
```

* **Anchors** — read from `Cold-start-out-gymv/latest/<game>/rollout_summary.json`
  at orchestrator startup (auto-derived from the actual demos the SFT
  trained on). Falls back to a hardcoded table baked from the new
  4-backbone teacher data when the cold-start file is missing:

| Game | Anchor source | Anchor (max across 4 frontier teachers) |
|---|---|---|
| `gymv_thunder_force_iii` | new SFT data | 750.0 (Qwen3-VL) |
| `gymv_altered_beast` | new SFT data | 425.0 (Gemini) |
| `gymv_columns` | new SFT data | 160.8 (GPT-5.4 upper-CI) |
| `gymv_dynamite_headdy` | new SFT data | 100.0 (GPT-5.4 / Claude upper-CI) |
| `gymv_space_harrier_ii` | new SFT data | 29 431.0 (Claude) |
| `gymv_streets_of_rage_2` | new SFT data | 408.8 (Gemini) |
| `gymv_airstriker` | new SFT data | 97.5 (Gemini) |
| `gymv_strider` | new SFT data | 112.5 (Gemini) |
| `candy_crush` / `tetris` / `super_mario` / `twenty_forty_eight` | paper Table 3 + run_<game>.sh baseline | TBD — populated from baselines on first run |

* **Where it's applied** — additive to the existing
  `harness.RewardLogger` JSONL: each `RewardLogEntry` and
  `GRPOStepLogEntry` gets a `reward_normalized` field next to the raw
  `score` / `reward`. W&B logs both:
  * `reward/raw/{game}` — per-game raw, unchanged behaviour
  * `reward/normalized/{game}` — per-game normalized, used by all
    cross-phase aggregates and curriculum thresholds
* **Where it's NOT applied** — GRPO advantage computation
  (`grpo_training.py`) is untouched; the optimizer keeps reading raw
  rewards because group normalization already handles within-batch
  variance.
* **Interpretation** — `r_norm = 1.0` ⟺ matches teacher; `0.5` ⟺ half
  teacher; the 2.0 ceiling stops a lucky-spike episode from owning
  the dashboard. `None` (when anchor missing or zero) ≠ `0` so
  dashboards can distinguish "no anchor" from "scored zero".

Total Phase-1 step budget: **6 × 15 = 90 steps**. Wall-clock
~36 min/step ⇒ **~54 h sequentially**; the curriculum is sequential
**by design** under the chosen Option C (§11.2) — bank + LoRA carry
over between games, so Phase-1 cannot be parallelised across games.

> The paper's per-game step budgets (`tetris`=7, `candy_crush`=10) are
> **shorter** than 15. We over-budget here on purpose: paper-budgeted
> single-game runs converge in a single LoRA, whereas curriculum-mode
> 15-step plateaus give the curator + segmenter more rollouts to
> consolidate skills before the next game arrives. If the post-curriculum
> reward on either paper game lags Figure 4 by >20 %, switch the
> ablation: rerun those two with paper budgets in the independent-runs
> merge mode (§11.2 Option A back-stop).

### 4.2 Per-game snapshot

After each game's 15 steps complete, snapshot to
`runs/phase1_curriculum_<timestamp>/snapshots/<NN>_<game>/` with
`adapters/` (full 5-LoRA state), `skillbank/skill_bank.jsonl` (the
single rolling bank — not a per-game bank, since curriculum-mode
shares one bank across games), `_phase_meta.json`, and
`rewards/step_NNNN.jsonl`. wandb run tagged
`phase1-curriculum-step<N>-<game>`.

The post-curriculum state — `snapshots/06_tetris/` — is the seed for
Phase 2 / Phase 3 (no separate merge utility needed under Option C).

### 4.3 Sanity bar

Two anchors:

1. **Paper anchor (weak under curriculum mode):** end-of-`candy_crush`
   reward and end-of-`tetris` reward within **±30 %** (loosened from
   ±20 % in the original draft because curriculum mode adds
   bank-borrowing across games, so absolute rewards don't have to
   match Figure 4 strictly).
2. **Per-game baseline anchor (strong):** end-of-game reward for each
   gymv game ≥ the **4-backbone median baseline** from
   `baselines/README.md` § "Gym-V benchmark scope". For `Columns`
   that's ~89 % per-episode success; for the other three it's ≥ 78 %.
   The trained-LoRA actor must clear the **un-trained-actor** bar by
   construction or there is no reason to continue.

If both anchors fail, abort and inspect; if only the paper anchor
fails (i.e. paper games regress because curriculum carry-over erased
their skills), trigger §11.2 Option A independent-runs ablation.

## 5. Bank handling under sequential curriculum

Under Option C (§11.2), all six Phase-1 games share **one rolling
skill bank**. The bank is grown game-by-game by the live curator +
crafter (atomic save per `SkillBankMVP.save()`); no separate offline
merge step is required. The post-curriculum bank
`runs/phase1_curriculum_<timestamp>/snapshots/06_tetris/skillbank/skill_bank.jsonl`
is the single seed for Phase 2 / Phase 3.

Expected post-curriculum size: ~30–60 unique skills (paper baselines
imply 6 + 10 + 10 + 20 ≈ 46 across the four known games; the four
gymv games typically add 5–15 each but with heavy overlap, so the
rolling-bank curator dedupe should keep total growth tame).

> **Independent-runs merge ablation (kept on the shelf).** If R6
> (LoRA drift) fires and we need clean per-game baselines, the
> independent-runs path produces six separate banks that need a
> merge utility — see §11.2 Option A. The **`scripts/merge_banks.py`**
> contract for that fallback path is:
>
> ```
> merge_banks.py \
>   --inputs runs/phase1_indep_<game>_*/skillbank/<game>/skill_bank.jsonl … (6 total) \
>   --output runs/source_merged_<timestamp>/skill_bank.jsonl \
>   --dedupe-policy curator-score-keep-best \
>   --dedupe-threshold 0.85   # cosine sim on contract effects
> ```
>
> Behaviour: group by canonical contract signature
> (`schema_hash(eff_add ∪ eff_del ∪ eff_event)`); within a group keep
> highest curator score; cross-group dedupe at the cosine threshold;
> write atomically (mirroring `SkillBankMVP.save()`); emit
> `_merge_meta.json` with pre/post counts and per-input contribution.
> Only land this script if the ablation actually triggers.

## 6. LoRA handling under sequential curriculum

Option C carries one rolling LoRA state across all six games — the
`adapters/` directory at the end of game N is the warm start for
game N+1, and the post-curriculum state is the warm start for
Phase 2 / Phase 3.

| Option | Mechanism | Status |
|---|---|---|
| **L2 — sequential warm-start (chosen)** | Load adapters from game N at end-of-N → continue GRPO on game N+1 → repeat | **Active path under §11.2 Option C.** Per-game snapshot in §4.2 lets us pick "post-curriculum" or "best-per-game" at eval time. |
| **L1 — pick best** | Use the single-game adapters from the **best-reward** Phase-1 game as Phase-2 cold start | Kept as Phase-2 ablation. Cheap to add since we already snapshot per game. |
| **L3 — weight-merge** | Average / SLERP-merge across six per-game adapters | **Only relevant if Option A back-stop fires** (independent-runs path). Implement `scripts/merge_adapters.py` only if R6 triggers it. |

The chosen path's risk — **late-game games dominate the LoRA gradient
state and erase early-game competence** — is captured as R6 (§12) with
two mitigations: per-game snapshot retention (§4.2) so we can always
fall back to a per-game best-checkpoint, and a Phase-1 sanity bar
(§4.3) that explicitly checks each game's end-of-game reward against
its 4-backbone baseline.

## 7. Phase 2 — in-domain few-shot adaptation

### 7.1 Held-out roster (refreshed 2026-05-03 PM, paired in-genre with Phase-1 sources)

Six games not seen in Phase 1 — four gymv benchmark leftovers and
two paper Table-3 games. Each Phase-2 game is **paired with a
Phase-1 source by genre**, so the cross-game skill translator
(`skill_agents/skill_bank/translate_for_target.py`) has the closest
possible source vocabulary to re-ground onto at the phase boundary.
The translator + shared bank pipeline runs by default for Phase 2
(`config.bank_mode='shared' + TRANSLATE_ON_BOUNDARY=1`), so the
"shared bank rescues partial-signal games" hypothesis becomes
directly testable on `Strider`:

| # | Slug | Genre | Teacher band (4-backbone min–max) | In-genre pair (Phase-1 source) | What this pair tests |
|---|---|---|---|---|---|
| 1 | `gymv_streets_of_rage_2` | beat-em-up | 202–408 | `gymv_altered_beast` (Phase-1 #2) | **Pure within-genre lift.** Same Genesis 6-button beat-em-up vocabulary; should be the cleanest cross-game transfer signal. Healthy teacher both sides. |
| 2 | `gymv_space_harrier_ii` | shmup | 14 469–29 431 | `gymv_thunder_force_iii` (Phase-1 #1) | **Reward-scale invariance test.** Same shmup family but ~30× larger reward magnitude. Tests whether normalized-reward training transfers without rescaling skill scores. |
| 3 | `gymv_airstriker` | shmup | 52–97 | `gymv_thunder_force_iii` (Phase-1 #1) | **Easier in-genre transfer.** Simpler vertical shmup; smaller reward scale than TF3. Sanity check — if this fails, none of the cross-game claims hold. |
| 4 | `gymv_strider` | action-platformer | 0–112 | `gymv_dynamite_headdy` (Phase-1 #4) | **Hardest case — partial-signal rescue.** GPT-5.4 / Qwen3-VL teacher = 0; Claude / Gemini = 31 / 112. Direct test of "shared bank + translator rescues a poisoned-SFT game". Shared mode vs per_game mode here is the cleanest A/B for the whole pipeline. |
| 5 | `twenty_forty_eight` | grid puzzle | n/a (paper Table 3) | `tetris` (Phase-1 #6) + `gymv_columns` (Phase-1 #3) | **Grid-puzzle pairing.** Two Phase-1 puzzle sources should give the translator richer material than a single source. |
| 6 | `super_mario` | action / scrolling | n/a (paper Table 3) | (no direct Phase-1 pair; closest is `gymv_dynamite_headdy`) | **Hardest cross-genre.** No in-genre Phase-1 source — tests how far the translator can stretch when no analogue exists. Negative result here is publishable as a transfer-distance bound. |

Pre-launch screen for each held-out game: "≥ 3 episodes complete in
2 min on the **un-seeded** Qwen3.5-9B actor". For the four gymv
games this is already satisfied by the new
`Cold-start-out-gymv/latest/<game>/rollout_summary.json` sweep
(`frame_skip=8`, 16 episodes per teacher). For `twenty_forty_eight` and
`super_mario` the GamingAgent / Orak wrappers ship 50 / 100 max
steps and have been baseline-stable on the existing scripts.

### 7.2 Two-budget reporting protocol

The paper's "few-shot" terminology refers to the *entire* training
being short (≤25 steps from scratch). Our Phase-2 / Phase-3
"few-shot" means **adaptation given a pre-trained bank + LoRAs**.
To prevent reviewer confusion, we report **both** budgets per
held-out game:

| Budget | Purpose | Bank | LoRA | Reports |
|---|---|---|---|---|
| **5 steps · inference-only** | Zero-shot bank reuse — does the merged Phase-1 bank fire on the new game *at all*? | merged | frozen Phase-1 (no GRPO) | reward, % steps with skill bound, harness rejection rate |
| **15 steps · GRPO** | Online adaptation — does adding GRPO compound the bank advantage over no-bank baseline? | merged, hot-reloaded | unfrozen, full 5-LoRA GRPO | reward curve, new-skill discovery rate, bank growth |

Add a **no-seed control** at 15 steps GRPO (same hyperparams, empty
bank) per held-out game so the lift attributable to the bank is
unambiguous.

Phase-2 wall-clock: 6 games × (5 + 15 + 15) steps × 36 min ≈
**126 h** (5.25 days) sequentially / **~21 h** with 6× parallel
(matches Phase-1 GPU budget).

### 7.3 Headline number

`reward(seed-bank @ 5 steps inference) / reward(no-seed @ 15 steps GRPO)`
— "**how many GRPO steps does the bank save us per held-out game?**"

If this ratio is ≥ 1.0 on ≥ 4 of 6 held-out games, the cross-game
transfer claim holds. Below that we either need more Phase-1 source
games or richer skill abstractions before publishing.

## 8. Phase 3 — out-of-domain few-shot adaptation

### 8.1 Target roster (cost-stratified)

| Target | Per-step cost | Stage-1 inclusion | Why |
|---|---|---|---|
| `video` | ~30 min · VLM-only | **YES** (mandatory) | cheap, mature wrapper, clean signal |
| `visual_reasoning` | ~30 min · VLM-only | **YES** (mandatory) | cheap, mature wrapper, clean signal |
| `browser` | ~50–70 min · Playwright + WebArena/VWA mirrors | **gated** | mirrors landed 2026-05-03 (`docs(vwa)` commit `e30c473`); stability not yet stress-tested at training-loop scale |
| `osworld` | ~80–120 min · Docker + VM | **gated** | per-step cost 5×; current Pass@1 ~4.8% (see [`canvases/osworld-results-diagnosis.canvas.tsx`](../../../.cursor/projects/workspace/canvases/osworld-results-diagnosis.canvas.tsx)) — even a 2× lift might fall inside CI |

Go/no-go for `browser` / `osworld` decided after Phase-2 numbers land.

### 8.2 Adaptation protocol (same as Phase 2)

Two budgets per target: 5-step inference-only + 15-step GRPO with the
merged-bank seed. **No** no-seed control here — Phase 3's claim is
"the gymv-derived bank transfers across domains", not "GRPO works on
this domain in isolation".

Phase-3 wall-clock (mandatory targets only): 2 targets × 20 steps ×
36 min ≈ **24 h**. Browser adds ~14 h, OSWorld ~32 h if gates open.

### 8.3 Headline number

% of skills in the merged Phase-1 bank that **bind successfully**
(harness `validate_choice` returns true) ≥ 1× on each target, plus
the reward delta vs random-skill-selection baseline. Mirrors Layer-A
transfer-gate semantics so the metric is comparable to the in-loop
gate from `_transfer_hook.py`.

## 9. Concrete launch sequence

Two new orchestration scripts (curriculum + holdout) plus reuse /
extension of existing `run_<game>.sh`:

```
scripts/
  run_phase1_curriculum.sh    ← NEW · single sequential 6-game run with
                                 bank+LoRA carry-over, snapshots after
                                 each game (mirrors run_all.sh phase
                                 structure but uses the locked roster
                                 from §4.1)
  run_phase2_holdout.sh       ← NEW · 6 held-out games × 2 budgets
                                 (5-step infer-only + 15-step GRPO)
                                 + no-seed control per held-out game
  run_phase3_ood.sh           ← NEW · video + visual_reasoning × 2 budgets
  run_candy_crush.sh          ← NEW · template'd from run_2048.sh
                                 (per-game wrapper still useful for
                                 baseline / single-game smoke tests)
  run_2048.sh / tetris.sh /
   super_mario.sh             ← VERIFY paper hyperparams wired (§11.1)

  merge_banks.py              ← DEFERRED · only land if Option A
  merge_adapters.py             back-stop fires (R6 in §12)
```

`run_phase1_curriculum.sh` is the canonical orchestrator (lands
2026-05-03; mirrors `run_all.sh`'s snapshot-per-phase pattern). Its
`PHASES` array reflects the refreshed roster from §4.1:

```bash
PHASES=(
    "1:gymv_thunder_force_iii:Thunder Force III"
    "2:gymv_altered_beast:Altered Beast"
    "3:gymv_columns:Columns"
    "4:gymv_dynamite_headdy:Dynamite Headdy"
    "5:candy_crush:Candy Crush"
    "6:tetris:Tetris"
)
ITERS_PER_PHASE=15
```

`scripts/run_phase2_holdout.sh` (NEW — lands alongside the refreshed
roster) iterates over the §7.1 transfer roster:

```bash
PHASES=(
    "1:gymv_streets_of_rage_2:Streets of Rage 2"
    "2:gymv_space_harrier_ii:Space Harrier II"
    "3:gymv_airstriker:Airstriker"
    "4:gymv_strider:Strider"
    "5:twenty_forty_eight:2048"
    "6:super_mario:Super Mario Bros"
)
ITERS_PER_PHASE=15        # 15-step GRPO main result; 5-step infer-only is a separate driver
BANK_MODE=shared           # default for Phase 2 — one shared bank
TRANSLATE_ON_BOUNDARY=1    # auto-translate prior bank onto each target's action vocab
```

All 8 gymv slugs are wired in
[`env_wrappers/gymv_temporal_nl_wrapper.py:GYMV_TEMPORAL_GAMES`](../env_wrappers/gymv_temporal_nl_wrapper.py)
(`gymv_space_harrier_ii`, `gymv_streets_of_rage_2`, `gymv_columns`,
`gymv_strider`, `gymv_altered_beast`, `gymv_airstriker`,
`gymv_dynamite_headdy`, `gymv_thunder_force_iii`) and registered in
[`trainer/coevolution/config.py:SKILL_BANK_GAMES`](../trainer/coevolution/config.py)
plus `GAME_MAX_STEPS` (200 each) so `--games <slug>` resolves
end-to-end without further plumbing.

Recommended dual-stack runtime layout (8×H200):

```
GPUs 0–3 → 9B trainer + actor vLLM :8000
GPUs 4–7 → 35B-A3B teacher + judge vLLM :8001
                                  (one instance shared across all phases)
[optional] GPU 8 → scripts/dashboard_sidecar.py (Layer-D, opt-in)
```

Phase-1 is **strictly sequential** under Option C — the single rolling
LoRA + bank cannot be parallelised across games. Phase 2 / Phase 3,
however, can run six held-out games concurrently if a multi-machine
deployment is available, since each held-out game starts from the same
post-curriculum snapshot and writes to its own output directory.

## 10. Wall-clock & cost

| Phase | Step budget | Sequential wall-clock | Parallel wall-clock |
|---|---|---|---|
| 1 — sequential curriculum | 6 × 15 = 90 | ~54 h | ~54 h (sequential by design — Option C cannot parallelise) |
| Per-game snapshot capture | n/a — inline | inline | inline |
| 2 — held-out adaptation | 6 × (5+15+15) = 210 | ~126 h | ~21 h (6× parallel — each holdout reads same Phase-1 snapshot) |
| 3 — OOD (video + VR) | 2 × 20 = 40 | ~24 h | ~12 h (2× parallel) |
| **Total (mandatory)** | **340 steps** | **~204 h ≈ 8.5 days** | **~87 h ≈ 3.6 days** |

OOD `browser` / `osworld` (gated): +14 h / +32 h respectively.

The parallel column for Phase 1 collapses to sequential because the
chosen Option C carries one rolling LoRA + bank across all six games.
If R6 (LoRA drift) fires and we revert to Option A, Phase-1 parallel
wall-clock recovers to ~12 h on 6× single-GPU vLLM (or a
multi-machine deployment).

LLM-as-judge spend: $0 (35B-A3B local). SFT teacher (gpt-5.5) is not
re-invoked in this plan — the cold-start data is reused from the
existing SFT adapters.

## 11. Open decisions

### 11.1 Per-game wrappers — paper hyperparam audit + new gymv wiring  🔴 OPEN

Two sub-tasks before Phase-1 launch:

**(a) Hyperparam audit.** The wrappers `run_2048.sh` / `run_tetris.sh` /
`run_super_mario.sh` already exist but have not been audited against
Table 3 since the 35B-judge / atomic-save commits landed (2026-05-03).
Diff each wrapper against the Table 3 row; record deltas (and rationale,
if any) in this note's §13 changelog. `super_mario` only matters at
Phase 2 now (see §7.1) but its wrapper is still the source of truth
for the hyperparam row.

**(b) gymv adapter wiring.** `trainer/coevolution/episode_runner.py`
currently only registers four games in `GAMINGAGENT_GAMES /
ORAK_SUBPROCESS_GAMES` (`twenty_forty_eight / candy_crush / tetris /
super_mario`). Phase-1 needs the four `Temporal/*` games wired in
similarly so `--games <slug>` resolves end-to-end. Smallest plumb:
- a `GYMV_TEMPORAL_GAMES` set in `episode_runner.py`;
- a per-slug → `Temporal/<EnvId>` map (see `TEMPORAL_GAME_SPECS` in
  `gymv_wrapper/temporal_visual_grounding.py`);
- a make-env branch that wraps `gym_v.make(env_id)` with the new
  `StochasticFrameSkip(n=8, stickprob=0)` wrapper landed in
  commit `4f97dd6` and the `TemporalVisualGroundingWrapper`.

### 11.2 Cross-game transfer mechanism (A vs B vs C)  🟢 RESOLVED 2026-05-03 PM

| Option | Mechanism | Status |
|---|---|---|
| A | Independent runs · merge banks · merge adapters | **Back-stop only** — kept as the recovery path if R6 (LoRA drift) fires; requires the `merge_banks.py` / `merge_adapters.py` scripts deferred in §9. |
| B | Single run with `config.games=[6]` (default behaviour) | *Rejected* — concurrent training every step erases per-game step-budget control. |
| **C** | **Sequential curriculum with bank + LoRA carry-over (`config.curriculum_schedule={0:[g1], 15:[g2], 30:[g3], …}`) and per-game snapshots** | **CHOSEN.** Matches user's intent ("one by one training, skill transfer when switching games"). Risk: late-game LoRA drift erases early-game competence — captured as R6 (§12). Mitigations: per-game snapshots in §4.2, per-game-baseline anchor in §4.3, Option A as documented back-stop. |

### 11.3 Phase-1 source roster + Phase-2 held-out roster  🟢 RESOLVED 2026-05-03 PM (refreshed)

* **Phase 1 (6 games, 15 steps each, sequential):** four gymv benchmark
  picks (`gymv_thunder_force_iii`, `gymv_altered_beast`, `gymv_columns`,
  `gymv_dynamite_headdy`) + two paper Table-3 games (`candy_crush`,
  `tetris`). All 6 have non-zero teacher reward across **all 4
  frontier teachers** in the new `Cold-start-out-gymv/latest` data
  ⇒ no poisoned-SFT slot.
* **Phase 2 (6 held-out games, 5+15-step adaptation):** four gymv
  benchmark leftovers (`gymv_streets_of_rage_2`, `gymv_space_harrier_ii`,
  `gymv_airstriker`, `gymv_strider`) + two paper Table-3 games
  (`twenty_forty_eight`, `super_mario`). Each Phase-2 gymv game is
  paired in-genre with a Phase-1 source so the cross-game translator
  has the closest possible source vocabulary.

Rationale: gymv picks are data-driven from the new 4-backbone teacher
table (Cold-start-out-gymv/latest, `frame_skip=8`). The data showed
that the previously locked Phase-1 set had two problems —
`SpaceHarrierII` is a ~30× reward-scale outlier that biases every
cross-phase aggregate, and `Strider` is partial-signal (GPT-5.4 = 0,
Qwen3-VL = 0). Both moved to Phase 2 where the scale-jump and
poisoned-rescue become *features* (testable hypotheses) rather than
liabilities. The remaining Phase-1 picks have median teacher reward
in 75–425 (TF3, AB, Columns, DH) — comparable order of magnitude, no
single phase will dominate aggregates. Full picks + per-teacher
numbers in §4.1 (Phase 1) and §7.1 (Phase 2).

### 11.4 Phase-2 budget — keep both budgets, or drop 5-step infer-only?  🔴 OPEN

If wall-clock pressure forces compression, drop the 5-step
inference-only baseline first (it's the cheapest to add later as a
post-hoc eval since it doesn't need GRPO). Keep the no-seed 15-step
control unconditionally — without it the lift claim is unfalsifiable.

## 12. Risks

* **R1 — Phase 1 reward drift vs paper.** If our reproduction of
  `tetris` / `candy_crush` lands outside **±30 %** of Figure 4
  endpoints (loosened from ±20 % under curriculum mode — §4.3),
  the comparison story is weakened. Mitigation: §11.1 hyperparam
  audit; 2-step smoke per game before committing the full 15-step
  run; trip to Option A (§11.2) if both paper games regress.
* **R2 — Late-curriculum bank collapse / dilution.** Under Option C
  the rolling bank can either (a) over-prune — curator
  conservatively rejects new gymv skills because they overlap effects
  with earlier paper-game skills — or (b) over-grow — late-game
  skills accumulate without dedupe and bury earlier ones in
  selection. Mitigation: monitor `bank_size` per Phase-1 step in
  wandb; expect ~30–60 entries post-curriculum (§5); if bank size <
  20 by step 30 or > 100 by step 75, abort and inspect curator scores.
* **R3 — Phase-2 no-seed control beats seeded 15-step.** Plausible if
  the held-out games are too distant from Phase-1 source distribution.
  Mitigated structurally by the §7.1 in-domain pairing (every held-out
  game pairs with a Phase-1 source game by genre). Report all six
  held-out results even if negative.
* **R4 — Cross-domain Phase-3 zero binding.** If no Phase-1 skill
  binds on `video` / `visual_reasoning`, the contract effects must be
  too gymv-grid-specific. Mitigation: paper §6 already calls this out
  as future work; partial result is publishable; skill-bridging
  abstraction (paper's PolySkill cross-ref) is the recovery path.
* **R5 — 35B judge family bias inflates Phase-2 / Phase-3 admit rates.**
  Mitigation: schedule a 5 % gpt-5.5 spot-check over the final reports
  (per `common/models.py` "Judge family-bias spot-check"); flip
  `VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5` for the spot-check pass.
* **R6 — Sequential-curriculum LoRA drift (the cost of Option C).** The
  late games (positions 5–6: `candy_crush`, `tetris`) dominate the
  final LoRA gradient state; early-games (positions 1–2:
  `gymv_thunder_force_iii`, `gymv_altered_beast`) may regress 20–40 %
  in reward by end-of-curriculum.
  Mitigations:
  - **Per-game snapshots** (§4.2) — every game's end-of-15-step
    LoRA + bank is captured; Phase-2 ablation can swap in the
    per-game best instead of the post-curriculum state.
  - **Per-game baseline anchor** (§4.3) — each game's end-of-its-own-15-steps
    reward must clear its 4-backbone baseline; if game N regresses
    below its baseline by end-of-curriculum, we report that as a
    finding, not a bug.
  - **Option A back-stop** (§11.2) — if R6 fires hard (>50 % regression
    on ≥3 of 6 games), revert to independent-runs + offline
    bank/adapter merge; cost is +12–46 h Phase-1 wall-clock and the
    `merge_banks.py` / `merge_adapters.py` scripts deferred in §9.

## 13. Changelog

* **2026-05-03 PM (refreshed)** — game roster swapped per the new
  `Cold-start-out-gymv/latest` 4-backbone teacher table.
  * **Phase 1** now `gymv_thunder_force_iii`, `gymv_altered_beast`,
    `gymv_columns`, `gymv_dynamite_headdy`, `candy_crush`, `tetris`
    (was: `SpaceHarrierII`, `StreetsOfRage2`, `Columns`, `Strider`,
    `candy_crush`, `tetris`). All 6 Phase-1 games now have non-zero
    teacher reward across **all 4 frontier teachers** in the new SFT
    data ⇒ zero poisoned-SFT slots.
  * **Phase 2** now `gymv_streets_of_rage_2`, `gymv_space_harrier_ii`,
    `gymv_airstriker`, `gymv_strider`, `twenty_forty_eight`,
    `super_mario` (was: `AlteredBeast`, `Airstriker`, `DynamiteHeaddy`,
    `ThunderForceIII`, `2048`, `super_mario_bros`). Each Phase-2 gymv
    game now has a Phase-1 in-genre source so the cross-game skill
    translator has the closest possible vocabulary to re-ground onto.
  * **§4.5 Reward normalization** added — teacher-anchored, additive
    layer in `RewardLogger`. Anchors auto-derive from
    `Cold-start-out-gymv/latest/<game>/rollout_summary.json` at
    startup with a hardcoded fallback table baked from the new
    4-backbone teacher data. Applied to W&B aggregates, curriculum
    thresholds, and the Layer-D dashboard transfer matrix; **not**
    applied to GRPO advantage (already group-normalized).
  * **Shared bank + per-boundary translation pipeline** is now the
    default Phase-2 carrier (`config.bank_mode='shared'` +
    `TRANSLATE_ON_BOUNDARY=1`). Phase 4 of Phase 2 (`gymv_strider`)
    becomes the cleanest A/B test of the whole pipeline:
    per_game-mode reward (expected ~0 from poisoned SFT) vs
    shared-mode reward (expected lift if cross-game translation
    works). See `skill_agents/skill_bank/translate_for_target.py`.
  * §9 PHASES arrays for both `run_phase1_curriculum.sh` and the new
    `run_phase2_holdout.sh` updated; §11.3 lock entry refreshed.
* **2026-05-03 PM (initial)** — game roster + curriculum mechanism
  locked.
  Phase-1 source: 4 gymv benchmark picks (`SpaceHarrierII`,
  `StreetsOfRage2`, `Columns`, `Strider`) + 2 paper games
  (`candy_crush`, `tetris`); Phase-2 holdout: 4 gymv leftovers +
  `2048` + `super_mario_bros`. **15 steps each / 90 total**.
  §11.2 Option C (sequential curriculum with bank+LoRA carry-over)
  selected over the original Option A; §5 / §6 / §10 rewritten to
  match. R6 added to §12 covering the LoRA-drift cost of Option C
  with three mitigations. §11.1 expanded to cover the new gymv
  adapter wiring task in `trainer/coevolution/episode_runner.py`.
  Sanity bar in §4.3 loosened to ±30 % vs paper Figure 4 to account
  for cross-game bank-borrowing under curriculum mode.
* **2026-05-03 AM** — initial draft. Per-game step budget grounded in
  COS-PLAY paper Appendix C / Table 3. Plan structure, bank-merge
  utility, two-budget reporting protocol, OOD cost stratification,
  and 5 risks captured. **3 open decisions** still blocked launch
  (§11.1 hyperparam audit, §11.3 two new gymv games, §11.4 budget
  trim if needed).
