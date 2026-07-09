# Pre-training readiness audit — what's missing before fast-loop GRPO launch

> **Status:** audit captured + **revised 2026-05-01 (revision 4 —
> S0 + S1 wrappers landed).** Synthesis of four parallel plan-vs-repo
> audits (Action Agent, Skill Bank, Skill Crafter, Harness, plus the
> cross-cutting visual-grounding and trainer scaffolds), reconciled
> against the on-disk SFT inventory under [`runs/`](../runs/)
> (revision 2), the recorded lane decision in
> [`skill-lane-decision.md`](legacy/skill-lane-decision.md) (revision 3 —
> lane (a): skills are retrieval payloads), and the **2026-05-01
> S0/S1 closure** (revision 4 — every S0 code/doc item shipped:
> `enable_protocol_patching` flag, threshold YAMLs, SFT manifest +
> load-smoke + schema-gen probe scripts, offline-promotion-cycle
> wrapper, lane-(a) banners on the four PLAN docs + harness README +
> roles doc; outstanding S0/S1 work is now GPU-bound execution of
> the shipped scripts, not new code or governance). Action items are
> tracked in the Tier 1–4 sections below; sequencing in §5;
> **the consolidated "Not Done" list is in §0**.
> **Last reviewed:** 2026-05-01 (revision 4).
> **Cross-refs:**
> [`runs/sft_schema_gen/schema_gen_20260430_091831/`](../runs/sft_schema_gen)
> (Phase-1 SFT, eval token-acc 0.9838 — see §0.2),
> [`runs/sft_coldstart/`](../runs/sft_coldstart)
> (5 LoRA SFT seeds: `action_taking`, `skill_selection`, `segment`,
> `contract`, `curator` — see §0.2),
> [`implementation_notes/legacy/skill-lane-decision.md`](legacy/skill-lane-decision.md)
> (**decided 2026-05-01:** lane (a) — skills are retrieval payloads,
> not runnable programs; closes T1.3),
> [`implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](legacy/crafter-harness-orchestrator-roles.md)
> (§7 — historical context for the lane decision; superseded by
> `skill-lane-decision.md`),
> [`implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md`](legacy/harness-usability-and-intra-gymv-transfer.md)
> (§6.2 keystone — `bank.runnable()` is empty until the offline loop fires once),
> [`implementation_notes/legacy/protocol-lift-design.md`](legacy/protocol-lift-design.md)
> (lane (b) implementation hook — `labeling/_decorate_skill_records.py`),
> [`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](legacy/single-vs-two-mdp-tradeoff.md)
> (**decided + shipped:** no `hop_select` LoRA, no inner MDP — see T3.6;
> remaining work is plan-doc cleanup only),
> [`plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (§5.0 fast-loop = gymv only),
> [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) (§13 Phase-1 SFT bar),
> [`plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) (§17 single reward sink, §20.2 ablations),
> [`plans/00-system/PLAN-SYSTEM-NORTHSTAR.md`](../plans/00-system/PLAN-SYSTEM-NORTHSTAR.md) (§5 stop/go, §7.3 release scoreboard),
> [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) (§4 eval-suite loader, §9 `skill_gate.yaml`),
> [`plans/08-cross-cutting/PLAN-EXPERIENCE-EXTENSION.md`](../plans/08-cross-cutting/PLAN-EXPERIENCE-EXTENSION.md) (§3 five typed extension records),
> [`plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md),
> [`plans/legacy/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md`](../plans/legacy/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md) (§3.2 HopTrace, §4.7 Crafter input contract),
> [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) (kept in sync with this audit and S0–S2 closures).

This memo records the synthesis of four parallel plan-vs-repo audits run
on 2026-05-01 to answer one question: **"what are we still missing
before we start to run the training?"**

It exists because the same gaps kept being re-discovered in independent
audits (visual-grounding readiness, trainer co-evolution scaffold, Crafter
lane decision, Harness wire-up). Consolidating them here means the next
contributor opening `trainer/coevolution/` does not re-litigate which
items are blockers, which are deferred-by-design, and which are content
gaps masquerading as structural ones.

---

## 0. Not done — consolidated checklist (post-SFT discovery)

The first audit pass (revision 1) had `schema_gen` SFT as the single
biggest blocker. Inspection of [`runs/`](../runs/) on 2026-05-01
showed all five SFT adapters (`schema_gen` + 4 cold-start LoRAs) are
**already trained and on disk**. This revision drops T1.1 from the
"hard blocker" tier and re-prioritises the remaining work.

### 0.1 What's still not done — ranked by sprint

| ID | Sprint | Item | Why it blocks training | One-line fix |
|---|---|---|---|---|
| ~~T1.2~~ | ~~S1~~ | ~~`bank.runnable()` is empty~~ → **wrapper landed 2026-05-01; executed 2026-05-02.** Driver: [`scripts/run_offline_promotion_cycle.sh`](../scripts/run_offline_promotion_cycle.sh). Post-condition green: **375 / 489** cold-start rows writeback-eligible (`ACTIVE` / `PROVISIONAL` / `SHADOW`) — see [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §S1. | — |
| ~~T1.3~~ | ~~S0~~ | ~~Lane decision (retrieval payload vs. runnable program) is unrecorded~~ → **closed 2026-05-01: lane (a) — context-only skills.** See [`skill-lane-decision.md`](legacy/skill-lane-decision.md). Skills are RAG-style retrieval payloads / procedural guidance for the actor LLM; the harness is an eligibility filter + validator, not an executor. **All lane follow-ups T1.3a–T1.3f shipped S0; T1.3b–T1.3d shipped S2 (2026-05-02)** — see §0.4. | — |
| **T1.4** | S3 | Real env executors for transfer-target adapters | `osworld` / `video` are pure stubs; `browser` / `visual_reasoning` have helpers but the trainer never calls them. Blocks G3a `min_target_domains_verified ≥ 1` ⇒ no skill ever leaves `CANDIDATE` even after promotion fires. | Pick one target (`browser` is plan default), bind `set_executor`, ship a few-shot demo file. |
| **T2.1** | S3 | Few-shot demo libraries for the four transfer targets | Compounds with T1.4 — without target demos `FewShotAdapter.adapt(...)` returns `target_domain_demo_unavailable`. | Mirror `harness/few_shot_demos_gymv.py` for the chosen target. |
| ~~T2.2~~ | ~~S2~~ | ~~`orchestrator/eval_suite.py` (G5 non-regression loader)~~ → **shipped 2026-05-02.** New module hosts canonical `EvalSuite` (re-exported by `harness.gate_runner` to avoid the import cycle) plus `EvalSuiteSpec` / `Scoreboard` / `EvalSuiteLoader`. `GateService.evaluate(eval_suite=)` consumes a frozen `EvalSuite` directly; mixing `eval_suite=` with `(baseline_score, post_score)` raises. Starter suite: [`evaluation/suites/gymv-smoke-v1/suite.yaml`](../evaluation/suites/gymv-smoke-v1/suite.yaml). | — |
| ~~T2.3~~ | ~~S2~~ | ~~Eval E0 driver + canonical scoreboard~~ → **shipped 2026-05-02.** `evaluation/answer_evaluator.py` (F1–F7 priority classifier), `evaluation/scoreboard.py` (10x10 canonical table + companion tables, markdown + JSON sidecar), `evaluation/driver.py` (writes per-instance JSONL + suite-level scoreboard JSON consumed by T2.2). `RunRelease` now carries `eval_suite_id` + `scoreboard_path` and folds both into `content_hash`. | — |
| ~~T2.4~~ | ~~S2~~ | ~~`harness/reward_logger.py` not wired into GRPO~~ → **shipped 2026-05-02.** `RewardLogger.log_grpo_record(...)` is a kind-discriminated JSONL sink (`grpo_step` vs. `skill_episode`); `episode_runner.run_episode_async` emits at both `action_taking` and `skill_selection` `GRPORecord` append sites, propagated through `rollout_collector.collect_rollouts` to `orchestrator.run_training_loop_async`. `CoEvolutionConfig.reward_log_path` auto-resolves under the rewards dir when blank. | — |
| ~~T2.5~~ | ~~S0~~ | ~~`configs/skill_gate.yaml` + `configs/failure_routing.yaml` missing~~ → **shipped 2026-05-01.** [`configs/skill_gate.yaml`](../configs/skill_gate.yaml) + [`configs/failure_routing.yaml`](../configs/failure_routing.yaml) (versioned policy v1.0.0; lane-(a) failure taxonomy with `lane_b_primary_mode` overrides). | — |
| ~~T2.6~~ | ~~S0~~ | ~~`IMPLEMENTATION-STATUS.md` stale (last 2026-04-21)~~ → **refreshed 2026-05-01** to point at all six checkpoint paths + Day-7→10 wire-up + `legacy_writeback` + `enable_protocol_patching` + threshold YAMLs + SFT manifest + offline-promotion driver. | — |
| T3.1 | S4 | Five typed extension records (`RunRecord`, `GroundingRecord`, `SkillInvocationRecord`, `FailureRoutingRecord`, `AnswerSupportRecord`) | Blocks multi-domain claims; not the fast loop. | Either ship the five dataclasses or update the plans to bless the existing analogs. |
| T3.2 | S4 | Failure router (R1–R5) entirely missing | L2/L4 failures dropped or consumed ad-hoc; contradicts §10 anti-goal #4. | New module `orchestrator/failure_router.py` + queue tree + dashboards. |
| T3.3 | S4 | `HopTrace` artefact + inner-hop logging | Phase B mining (typed `SkillIR`, transferable reasoning skills) has no input. Reconcile with the single-MDP decision (could be offline-only). | Add `HopTrace` as offline log on Action Agent; *do not* re-add online inner-hop LLM. |
| T3.4 | S4 | Canonical cross-domain ontology types (`Agent`, `UIElement`, `EvidenceSpan`, …) | Same blocker as T1.3 lane (b); typed validators can't run. | Module + dataclasses derived from PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS §3.4. |
| T3.5 | S4 | Online runtime adapters (`HarnessSkillProvider`, `RunnerActorAdapter`, `EnvLike`, `legacy_bridge`) | Day-10 `SkillHarnessHook` covers the trainer; these only block live-online deployment. | Defer until post-S2. |
| ~~T3.6~~ | ~~S0 (doc-only)~~ | ~~`hop_select` LoRA absence~~ → **bundled with T1.3e/T1.3f and shipped 2026-05-01.** The four PLAN-doc lane-(a) banners (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION) explicitly mark `hop_select` / `inner_mdp` references obsolete and point at [`single-vs-two-mdp-tradeoff.md`](legacy/single-vs-two-mdp-tradeoff.md). PLAN-ACTION-AGENT §5 / §5.5 retain the historical `hop_select` rationale; the lane-(a) banner in the four downstream plans is sufficient governance. | — |
| T4.* | post-launch | 7 open questions (hop-vocab, Q4 ablation, Phase-2 SFT trigger, counterfactuals, K-thresholds, Stage-5 window, Crafter input contract) | Tuning, not blocking. | See §5. |

### 0.2 What's already done — reconciliation against `runs/`

The Phase-1 SFT pass and all four cold-start LoRA seeds **landed
2026-04-30**:

| Adapter | Path | Base | Train rows | Epochs | Final eval loss | Final eval token-acc | Wall-clock |
|---|---|---|---|---|---|---|---|
| `schema_gen` | [`runs/sft_schema_gen/schema_gen_20260430_091831/`](../runs/sft_schema_gen) (final + ckpts 200/400/477) | Qwen3.5-35B-A3B | 477 steps × bs 16 effective | 1 | **0.0861** | **0.9838** | 6 h 30 m |
| `action_taking` | [`runs/sft_coldstart/decision/action_taking/`](../runs/sft_coldstart/decision/action_taking) | Qwen3.5-9B | 30 482 | 2 | 0.0889 | — | 12 h 27 m |
| `skill_selection` | [`runs/sft_coldstart/decision/skill_selection/`](../runs/sft_coldstart/decision/skill_selection) | Qwen3.5-9B | 29 532 | 2 | **0.0685** | — | 12 h 11 m |
| `segment` | [`runs/sft_coldstart/skillbank/segment/`](../runs/sft_coldstart/skillbank/segment) | Qwen3.5-9B | 4 343 | 4 | 0.3234 | — | 4 h 25 m |
| `contract` | [`runs/sft_coldstart/skillbank/contract/`](../runs/sft_coldstart/skillbank/contract) | Qwen3.5-9B | 1 316 | 5 | 0.0748 | — | 1 h 25 m |
| `curator` | [`runs/sft_coldstart/skillbank/curator/`](../runs/sft_coldstart/skillbank/curator) | Qwen3.5-9B | 193 | 15 | 1.218 ⚠️ | — | 24 m |

LoRA shape: `r=16`, `α=32`, dropout 0.05, 9-projection target set
(`q/k/v/o_proj`, `in_proj_qkv`, `out_proj`, `gate/up/down_proj`).
`schema_gen` adapter is 16 MB; the 5 cold-start adapters are 148 MB
each.

#### Quality flags discovered during inventory

* **`schema_gen` eval token-acc 0.9838 (eval loss 0.086).** PLAN-VISUAL-GROUNDING-MILESTONES §13 specifies *exact-match* schema accuracy thresholds (≥ 70 % `gymv`, ≥ 50 % `browser`); the trainer reports `mean_token_accuracy` (token-level CE-derived). The numbers strongly suggest the bar is met but **must be re-verified with an exact-match probe** before pinning `schema_gen_ckpt_id` into `RunRelease`. New T1.1 sub-item → §0.3.
* **`segment` eval≈train (0.32 vs. 0.30).** Either Stage-2 boundary preference is high-entropy or the LR schedule is under-trained. Flag for Phase-F warm-up convergence test; not a blocker.
* **`curator` eval 1.22 vs. train 0.49 over 15 epochs on 193 rows.** Severe overfit on a tiny corpus. Mitigations: (a) treat curator as a tie-breaker only in early GRPO; (b) regenerate corpus with more games; (c) stronger early stopping. **Now the top quality concern.** New T2.7 → §3.
* ~~**`schema_gen` is on the 35B-A3B control-plane base, not the 9B actor.**~~ **T2.8 closed** — split-base topology documented in [`legacy/vllm-topology.md`](legacy/vllm-topology.md) (implicit option (ii): `schema_gen` offline / separate worker).
* ~~**`sft_summary.json` is per-GPU**, not run-wide.~~ **T2.9 closed** — [`runs/sft_coldstart/sft_summary_all.json`](../runs/sft_coldstart/sft_summary_all.json) + [`scripts/build_sft_manifest.py`](../scripts/build_sft_manifest.py).
* ~~**`*.partial_20260430_091054` log shards**~~ **T2.10 closed 2026-05-02** — smoke load on all six adapters; report `runs/sft_coldstart/_smoke/smoke_all.json` (§0.3).

#### What this changes in the readiness ledger

* T1.1 (perception SFT) is **closed — pending exact-match verification**.
* All Phase-F GRPO LoRA targets (`skill_select` / `action_execute` / `CONTRACT` / `CURATOR` / `SEGMENT`) have **SFT warm-starts on disk**. Phase F is unblocked on weight inputs.
* ~~The single biggest remaining Tier-0 blocker was **T1.2** (offline promotion once).~~ **Closed 2026-05-02** — see §0.1 row `T1.2` and [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §S1. The **training-critical verification path** is now **T1.1′ after T2.11 SFT re-run** (exact-match probe on freshly trained adapters) plus actually **launching** fast-loop GRPO (S2 outstanding).

### 0.3 New / promoted action items from the SFT discovery

* **T1.1′ (executed 2026-05-02 — both §13 thresholds missed; root cause is target-modules drift on the LoRA, see T2.11 below).** Probe at `evaluation/probe_schema_gen_exact_match.py` ran end-to-end on `n=5` held-out gymv triples with `--max-new-tokens 768` against `runs/sft_schema_gen/schema_gen_20260430_091831/`. Result: `exact_match_rate = 0.000 (0/5)`, `path_a_accept_rate = 0.000 (0/5)`, `overall_field_acc = 0.000`, elapsed = 1241 s. Report at `runs/sft_coldstart/_probe/probe_schema_gen_n5.json`. **The thresholds are missed *because the LoRA delta isn't being applied*, not because the SFT under-learned** — see T2.11.
* **T2.7 (curator overfit mitigation)** — see §3.7.
* **T2.8 (vLLM topology check)** — see §3.8.
* **T2.9 (run-wide SFT manifest)** — see §3.9.
* **T2.10 (audit `*.partial_*` shards)** — **closed 2026-05-02.** `evaluation/smoke_load_sft_adapters.py` ran on all six adapters (5× Qwen3.5-9B + 1× Qwen3.5-35B-A3B) with `device_map="auto"`, dtype=bfloat16, sharded across GPUs 0-3 under the active vLLM tower (~40 GB free per GPU). Every adapter loaded, ran one forward pass, and produced finite logits on the expected shape `(1, 2, 248320)`. **No torn `*.partial_*` shards detected.** Report at `runs/sft_coldstart/_smoke/smoke_all.json`.
* **T2.12 (NEW — SFT throughput uplift, 2026-05-02).** Re-running the six SFT jobs (T2.11 remedy) needs the slow-loop scaffold to actually be fast. Audited the kernel/optimizer stack and landed three drop-in upgrades + flagged two CUDA-toolkit-bound items:

    | Win | Status | Where |
    |---|---|---|
    | **liger-kernel** Qwen3.5 / Qwen3.5-MoE / Qwen3-VL fused patches (CE / RMSNorm / RoPE) | ✅ installed (`liger-kernel`); auto-applied via `trainer/SFT/speed_utils.py::apply_liger_kernel`. Both train scripts call it before model load. | new `trainer/SFT/speed_utils.py` |
    | **Paged AdamW 8-bit** optimizer (cuts optimizer memory ~4×) | ✅ installed (`bitsandbytes` 0.49.2); `pick_optim()` defaults to `"paged_adamw_8bit"` when bnb is available, else `"adamw_torch_fused"`. Plumbed into both train scripts via `--optim` (or auto-pick). | `trainer/SFT/{train,schema_gen/train}.py` |
    | **TF32 + group-by-length + dataloader workers + pin-memory** (no-deps wins) | ✅ default-on. Both train scripts call `enable_tf32()` and pass `group_by_length=True`, `dataloader_pin_memory=True`, `dataloader_num_workers=4` (override via `--dataloader_workers`). | both train scripts |
    | **flash-attn 2** (full-attention layer fast path) | ✅ **closed 2026-05-03.** Host CUDA-13.0 toolkit was installed (`/usr/local/cuda → cuda-13.0`, `nvcc` 13.0.88 on PATH). `flash_attn==2.8.3` already in `game-ai-agent`; `flash_attn_func` smoke-tested on H200 bf16. | env: `game-ai-agent` |
    | **mamba_ssm + causal_conv1d** (linear-attention CUDA kernels — biggest single Qwen3.5 win, ~5–10× on the GatedDeltaNet path) | ✅ **closed 2026-05-03.** `causal_conv1d 1.6.1` was already present; `pip install --no-build-isolation mamba-ssm` now succeeds against torch 2.11.0+cu130 (3 min source build, 350 MB wheel, `mamba_ssm 2.3.1`). End-to-end smoke at `/tmp/smoke_mamba.py`: `Mamba` block forward + `selective_scan_fn` + `selective_state_update` + `rmsnorm_fn` + `causal_conv1d_fn` + `flash_attn_func` all green. | env: `game-ai-agent` |

    **Batch-size / gradient-checkpoint defaults (Levers A + C).** Audited the per-adapter overrides in `trainer/SFT/config.py` and bumped them so every cold-start adapter now uses **`batch_size=16, grad_accum=1`** (effective batch held at 16 for loss-curve continuity with the previous schedule):

    | Adapter | Before | After | Same effective batch? |
    |---|---|---|---|
    | `skill_selection` / `action_taking` | bs=16 ga=1 | bs=16 ga=1 | yes (already optimal) |
    | `segment` | bs=4 ga=4 | **bs=16 ga=1** | yes — ~25 % wall-clock gain |
    | `contract` | bs=8 ga=2 | **bs=16 ga=1** | yes |
    | `curator` | bs=4 ga=4 | **bs=16 ga=1** | yes — biggest wall-clock win (15 epochs × small corpus) |

    Cold-start `gradient_checkpointing` default flipped from `True` → **`False`** — at bs=16, seq=2048, paged_adamw_8bit, the 9B base + ungated activations land around ~35 GB on the H200's 143 GB. Buys another ~30–40 % throughput. The 35B-A3B `schema_gen` path keeps `gradient_checkpointing=True` (separate `SchemaGenConfig` field; that base genuinely needs it).

    **Lever B exposed via CLI.** Added `--scale_effective_batch <factor>` and `--scale_lr <factor>` flags to `trainer/SFT/train.py` so a caller can opt past effective-16 with the linear-LR-scale rule baked in (e.g. `--scale_effective_batch 2.0 --scale_lr 2.0` → effective-32 + 4e-4). Defaults stay at 1.0 so loss curves don't drift silently.

    **Multi-GPU per adapter (Lever D — 8× H200 utilisation).** The earlier parallel launcher pinned one adapter to one GPU; with 8 H200s and 5 cold-start adapters, three GPUs sat idle. Added `--gpus_per_adapter N` to `trainer/SFT/train.py`: when `>1`, each adapter is launched under `accelerate launch --num_processes N` so HF Trainer data-parallels via DDP. Effective batch becomes `per_device_bs × N × grad_accum` so the call should be paired with `--scale_lr N` (linear-scale rule). Schedule examples on 8 H200s:

    | Recipe | Adapters / chunk | Wall-clock vs. 1-GPU-each baseline |
    |---|---|---|
    | `--gpus_per_adapter 1` (default) | 5 chunks of 1 GPU; 3 GPUs idle | 1.0× (baseline) |
    | `--gpus_per_adapter 2 --scale_lr 2.0` | 4 chunks of 2 GPUs; 5 adapters round-robined → chunk[0] gets two adapters | ~1.7× faster (each adapter ~2× via DDP, but chunk[0] does two sequentially) |
    | `--gpus_per_adapter 4 --scale_lr 4.0` | 2 chunks of 4 GPUs; chunks get 3 + 2 adapters sequentially | ~2-3× faster — biggest single jump |
    | `--gpus_per_adapter 8 --scale_lr 8.0` | 1 chunk of 8 GPUs; adapters trained sequentially under DDP | ~2× faster vs. 1-GPU-each baseline; useful when wanting tightest LR re-tune |

    For the **`schema_gen` 35B-A3B path** the existing
    `bash trainer/SFT/schema_gen/run_schema_gen.sh --num-gpus N` launcher already
    wraps DeepSpeed ZeRO-3 (config at
    `trainer/SFT/schema_gen/configs/ds_zero3.yaml`). With 4× H200 the 70 GB bf16
    base shards into ~18 GB chunks per GPU, leaving ~125 GB headroom — `bs=4 ga=4`
    is comfortably feasible (was `bs=1 ga=16`). All T2.11/T2.12 changes (correct
    LoRA recipe, liger-kernel, paged-AdamW) flow through transparently because
    they're model-side, not launcher-side.

    **Expected uplift on the SFT re-run (no CUDA-toolkit unblock):**
    * Lever A (collapse grad_accum into bs): ~10–25 %
    * Lever C (no gradient_checkpointing): ~30–40 %
    * Liger-kernel + paged-8bit + tf32 + group-by-length: ~30–40 % (on top, partially overlapping)

    Combined: **~1.7–2.2× wall-clock on the 9B cold-start path** before any CUDA-13 toolkit work. **With toolkit unblock (flash-attn + mamba_ssm) — now landed 2026-05-03:** another 2–3× on the linear-attention-heavy 35B-A3B `schema_gen` run, plus ~10–20 % on the 9B GatedDeltaNet path (was running a Python recurrence fallback).

* **T2.11 (LoRA target-modules drift between SFT-time and load-time) — *recipe fix landed 2026-05-02; SFT re-run still required*.** During T2.10 the smoke transformers warned about "missing adapter keys" on every adapter — `linear_attn.{out_proj, in_proj_qkv}` and `mlp.{gate_proj, up_proj, down_proj}` for many layers. Quantitatively the LoRA params actually loaded into the cold-start 9B adapters total **36.96 M** and the schema_gen 35B-A3B total **8.36 M**. Expected order-of-magnitude (r=16 × hidden×2 × n_target_modules × n_layers) is ~42 M for the 9B adapters (≈87 % loaded — borderline) and ~36 M for schema_gen (only ≈23 % loaded — three quarters of the LoRA delta is silently dropped).

    **Root cause (confirmed by inspecting `transformers/models/qwen3_5/modeling_qwen3_5.py`).** Qwen3.5's `Qwen3_5DecoderLayer` is a *hybrid* block: each layer is one of two types selected by `config.layer_types[i]`. `"full_attention"` layers expose the classic `q_proj/k_proj/v_proj/o_proj`; `"linear_attention"` layers expose `linear_attn.{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj}` (a Mamba-style GatedDeltaNet). Both layer types share `mlp.{gate_proj, up_proj, down_proj}`. The classic seven-projection `target_modules` carried forward from Qwen2/Qwen3 recipes matches *zero* GatedDeltaNet legs, so on Qwen3.5-35B-A3B (mostly linear-attention layers) only the few full-attention layers + every MLP got a LoRA. PEFT's substring matching is silent on a miss, so the trainer never noticed. Empirically: schema_gen recipe used the classic-7 → 23 % coverage; cold-start recipes added only `in_proj_qkv` + `out_proj` → 87 % coverage (the missing 13 % is `in_proj_z` + `in_proj_b` + `in_proj_a`).

    **Remedy landed (2026-05-02).** New module [`trainer/SFT/lora_targets.py`](../trainer/SFT/lora_targets.py) is the single source of truth for Qwen-family `target_modules`:
    * `QWEN3_5_LORA_TARGETS` — full hybrid-stack list, **12 entries**: `q,k,v,o_proj` + `in_proj_{qkv,z,b,a}` + `out_proj` + `gate,up,down_proj`. Resolver returns this for any `text_arch` containing `qwen3_5` (covers `qwen3_5`, `qwen3_5_moe`, `qwen3_5_text`).
    * `QWEN_CLASSIC_LORA_TARGETS` — classic-7, returned for every other Qwen family (Qwen2/2.5/3 dense and MoE, Qwen3-VL, Qwen3-VL-MoE).
    * `assert_lora_coverage(peft_model, model_arch, …)` — post-`get_peft_model` sanity check that counts wrapped layers per projection name; aborts in strict mode when any required leg has zero matches. **Eliminates the silent-miss class of bugs.**

    All five LoRA-creating pipelines now resolve through this module:
    * `trainer/SFT/config.py::SFTConfig.resolve_target_modules` → delegates.
    * `trainer/SFT/schema_gen/config.py::SchemaGenConfig.resolve_target_modules` (new method) → delegates; the hardcoded classic-7 default that caused the bug has been removed.
    * `trainer/coevolution/config.py::prepare_adapters` → delegates (the live GRPO loop now writes/reads the same LoRA shape as SFT, fixing a second silent drift at the SFT→GRPO boundary). The earlier "skip the tiny `in_proj_a/b/z` gating projections" comment was wrong about `in_proj_z` (whose output dim is `value_dim`, not `num_v_heads` — it's the same size as `out_proj`).
    * `skill_agents/grpo/fsdp_trainer.py::_train_one_adapter` and `_fsdp_train_worker_multi` (2026-05-03 follow-up) → both random-init LoRA fallback branches were still using the legacy 6-leg list `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj]` (also missing `down_proj`). Now resolve through `lora_targets`.
    * `skill_agents/lora/model.py::LoraModel.prepare_for_training` (2026-05-03 follow-up) → same legacy-7 list (had `down_proj` but missing the GatedDeltaNet legs); now resolves through `lora_targets`.
    * `configs/skillbank_lora.yaml` — updated to the 12-entry list.
    * Tests: [`tests/test_lora_targets.py`](../tests/test_lora_targets.py) — 7 new tests cover Qwen3.5 leg presence, classic-Qwen list shape, explicit-override priority, fallback for unknown archs, and both passing + failing branches of `assert_lora_coverage`. All pass (36/36 in the lora_targets/few_shot/smoke/invariants suite as of 2026-05-03 post-fix).

    **Outstanding work — re-run the six SFT jobs against the corrected recipe.** The on-disk weights from the previous run *cannot* be salvaged (the dropped legs were never trained → no on-disk tensor to rewrite into the new namespace; option (b) "key-rewriter" from §0.3 is therefore ruled out). Options (a) re-train and (c) transformers downgrade remain; (a) is the chosen path because it both fixes the recipe and aligns SFT-time with GRPO-time forever. **T1.1′ remains blocked until SFT re-runs complete.**

    **2026-05-03 update — SFT re-run completed (`runs/sft_coldstart_20260502_025737/` + `runs/sft_schema_gen/schema_gen_20260502_025751/`).** All six adapters now carry the 12-leg `target_modules` in `adapter_config.json` and are referenced by the run-wide manifest at `runs/sft_coldstart/sft_summary_all.json`. *However,* the post-rerun smoke surfaced a **second, independent SFT→GRPO/vLLM boundary bug** — see T2.13 below.

* **T2.13 (NEW — SFT/GRPO loader-class mismatch caused silent LoRA no-op at the SFT→vLLM boundary; closed 2026-05-03).** While re-running the post-T2.11 smoke under `evaluation/smoke_load_sft_adapters.py`, the load was 5/5 OK on a forward pass *but* PEFT emitted `Found missing adapter keys` warnings for **every** layer in **every** cold-start adapter — symptom: `lora_B` slots stayed at zero-init, so `BAx = 0` and the cold-start delta was a silent no-op when stitched into the live GRPO/vLLM tower. The 35B-A3B `schema_gen` adapter was unaffected.

    **Root cause.** Qwen3.5 ships an umbrella `Qwen3_5ForConditionalGeneration` config with `text_config` + `vision_config`. Two different transformers auto-classes return two different module trees:
    * `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` (LM-only sub-model). LoRA attaches at `model.layers.<i>.…`, saved keys: `base_model.model.model.layers.<i>.…lora_A.weight`.
    * `AutoModelForImageTextToText` → `Qwen3_5ForConditionalGeneration` (full multimodal model with frozen vision tower). LoRA attaches at `model.language_model.layers.<i>.…`, saved/expected keys: `base_model.model.model.language_model.layers.<i>.…lora_A.weight`.

    `trainer/SFT/train.py` loaded via `AutoModelForCausalLM` (LM-only keys), but **every production loader** — `trainer/coevolution/prepare_adapters`, `evaluation/smoke_load_sft_adapters`, the `vllm serve --model Qwen/Qwen3.5-9B` server, and `trainer/SFT/schema_gen/train.py` — uses `AutoModelForImageTextToText` (VLM keys with `language_model.` prefix). PEFT's substring-key matching is silent on a structural-prefix mismatch; the LoRA modules were instantiated empty in the live tower, the warning was treated as noise, and the cold-start signal was being dropped at the SFT→GRPO/vLLM boundary.

    **Remedy landed 2026-05-03 (two-part fix).**
    1. **One-shot key remap** of all five legacy LM-only-keyed cold-start adapters via [`evaluation/fix_lora_keys_for_vlm_loader.py`](../evaluation/fix_lora_keys_for_vlm_loader.py). Rewrites `base_model.model.model.layers.<i>.` → `base_model.model.model.language_model.layers.<i>.` in `adapter_model.safetensors` (496 keys per adapter), leaves `adapter_config.json` untouched (`target_modules` are leaf-name patterns, not full paths), and writes a `.pre_vlm_remap` backup of the original. Applied in-place to:
       * `runs/sft_coldstart_20260502_025737/decision/{action_taking,skill_selection}/<name>/`
       * `runs/sft_coldstart_20260502_025737/skillbank/{segment,contract,curator}/<name>/`

       The 35B-A3B `schema_gen` adapter was already VLM-keyed (its training script uses `AutoModelForImageTextToText`) and was correctly skipped by the remapper.

    2. **Permanent fix across every base-model loader used by SFT, GRPO, and runtime LoRA inference** (2026-05-03):
       * [`trainer/SFT/train.py`](../trainer/SFT/train.py) — flipped to `AutoModelForImageTextToText` on multimodal configs (was `AutoModelForCausalLM`).
       * [`skill_agents/grpo/fsdp_trainer.py`](../skill_agents/grpo/fsdp_trainer.py) — both `_train_one_adapter` (single-adapter FSDP path) and `_fsdp_train_worker_multi` (multi-adapter persistent FSDP path used by the production co-evolution loop via `run_fsdp_grpo_multi`) now mirror the same loader-class selection. Without this, **every GRPO step would have written its trained LoRA delta with LM-only-keyed safetensors**, which would have re-introduced T2.13 at every checkpoint.
       * [`skill_agents/lora/model.py`](../skill_agents/lora/model.py) `LoraModel._load_base_model` — same fix.
       * Logged at INFO in every site so future re-trains surface the loader choice next to the model name.

       Selection is identical everywhere: `hasattr(cfg, "text_config") or hasattr(cfg, "vision_config")` ⇒ `AutoModelForImageTextToText`, else `AutoModelForCausalLM`. Mirrors `evaluation/probe_schema_gen_exact_match.py` and `evaluation/smoke_load_sft_adapters.py`.

    **Verification (2026-05-03).** Re-ran `evaluation/smoke_load_sft_adapters.py --skip schema_gen --device auto` against the post-remap manifest. 5/5 OK with `n_loaded_params=43,278,336` per adapter, **0 missing-key warnings**, finite logits on `(1, 2, 248320)`. Out-of-band PEFT load against `Qwen3_5ForConditionalGeneration` confirmed `22,831,104` non-zero `lora_B` params (≈ 50 % of total LoRA budget — the SFT-trained delta is now actually live). Report at `runs/sft_coldstart/_smoke/smoke_all_post_remap.json`. Per-adapter remap log at `runs/sft_coldstart_20260502_025737/_remap_report.json`.

    **Why this wasn't caught earlier.** The T2.10 smoke ran a forward pass and asserted finite logits — both pass on a base-only model with zero LoRA delta. The T2.11 audit caught a *related* recipe-side drift (`target_modules` coverage) and (correctly) ruled out a key-rewriter as a remedy *for that specific bug*, since the dropped legs had no on-disk tensors. T2.13 is the inverse situation: the tensors exist, only the structural-prefix names disagree, so a key rewriter is the right tool here.

* **T2.13′ (NEW — 1-shot ICL wiring closed in production callers, 2026-05-03).** Sibling task to T2.13 surfaced during the same sweep: while [`evaluation/smoke_schema_gen_5tasks.py`](../evaluation/smoke_schema_gen_5tasks.py) explicitly threads `get_few_shot_examples → build_adaptive_system_prompt(few_shot_examples=…)` (and the probe shows *prefix-match* climbing from ~0.2 % zero-shot → ~38 % 1-shot on the post-T2.11 schema_gen adapter), three production callers were still issuing zero-shot prompts:
    * [`vlm_wrapper/tool_loop.py`](../vlm_wrapper/tool_loop.py) — used by every gym-v / browsergym / image-qa / video-qa adapter (the main production VLM tool loop).
    * [`osworld_wrapper/adapter.py`](../osworld_wrapper/adapter.py) — `generate_label` for OSWorld desktop frames.
    * [`visual_grounding_tests/generate_osworld_text_schema.py`](../visual_grounding_tests/generate_osworld_text_schema.py) — text-only schema head used to validate the cascaded ground baseline.

    All three now look up the curated example block via `get_few_shot_examples(domain, n=N, task_id=…, fallback_domain=…)` and pass `few_shot_examples=…` into `build_adaptive_system_prompt`. Default `N=1` (env-overridable via `VLM_FEW_SHOT_N`; set `0` for zero-shot). For env_wrappers / gymv `task_id` we let it cascade through `{domain}.{task_slug}.txt → {domain}.txt → gymv.txt`, matching the cold-start labeler's resolution order. Fast-loop GRPO is unaffected (those four games (`twenty_forty_eight, tetris, candy_crush, super_mario`) are wrapped via `env_wrappers/gamingagent_nl_wrapper.py`'s text-only path and never touch `build_adaptive_system_prompt`); the change matters for cross-domain Stage-2 inference, the Phase-5/6 measurement matrix, and any base-VLM eval that runs *without* the schema_gen LoRA stitched in.

* **T2.14 (NEW — vLLM 0.20 `deep_gemm_warmup` hard-fail on bf16 weights, closed 2026-05-03).** Surfaced during the post-T2.13 vLLM-live smoke and again during the 1-step `scripts/run_coevolution.py` dry-run: vLLM 0.20+ unconditionally enumerates FP8-eligible linear layers during `kernel_warmup` → `deep_gemm_warmup` → `_fp8_linear_may_use_deep_gemm` → `get_mk_alignment_for_contiguous_layout`, which raises `RuntimeError: DeepGEMM backend is not available or outdated` if the (DeepSeek) `deep_gemm` package is missing — *even on bf16/fp16 weights* like Qwen3.5-9B that have no FP8 path. The crash hits at engine-init, before `Application startup complete`, so vLLM never serves a single request and the orchestrator times out at 600 s with `0/N healthy`.

    **Remedy (permanent).** [`trainer/coevolution/vllm_server.py`](../trainer/coevolution/vllm_server.py) (`_launch_wave`) now sets `env.setdefault("VLLM_USE_DEEP_GEMM", "0")` for every spawned `vllm serve` instance. `setdefault` so callers that have built/installed `deep_gemm` and want FP8 kernels can re-enable by exporting `VLLM_USE_DEEP_GEMM=1` from outside the orchestrator. Verified end-to-end: a 5-LoRA single-instance smoke (port 8001, GPU 0) now reaches `Application startup complete` in 205 s and answers /v1/completions for base + all 5 LoRA adapters; the 4× TP=1 wave (orchestrator default) reaches `4/4 healthy` in ≤ 9 min on cold cache and serves 150 vLLM calls in the dry-run.

* **T2.15 (NEW — `harness_filter_diag` UnboundLocalError collapsed every Phase-A rollout when skill bank was empty / sticky-guidance, closed 2026-05-03).** Surfaced during the same 1-step coevo dry-run *after* T2.13 + T2.14 were closed: every one of the 8 spawned episodes errored after 2 attempts with `cannot access local variable 'harness_filter_diag' where it is not associated with a value`, leaving `Phase A+B: 16.9s, 8 episodes collected, 0 consumed` and skipping the GRPO step entirely (`No GRPO records for 'skill_selection'`, `No GRPO records for 'action_taking'`).

    **Root cause.** [`trainer/coevolution/episode_runner.py`](../trainer/coevolution/episode_runner.py) initialised `harness_filter_diag: Optional[Dict[str, Any]] = None` at *inner* scope — inside the `if bank_available and (need_reselect or last_guidance is None):` block (~L990). The read site at the experience-dict assembly (~L1384) is at *outer* per-step scope. When the bank was empty (cold-start step 0) or sticky-guidance kept us out of the inner block, the variable was never bound → UnboundLocalError on every rollout in the wave.

    **Remedy.** Hoisted the init out of the inner block to the same scope as the existing `harness_validate_diag: Optional[Dict[str, Any]] = None` outer init (~L1053). The inner re-assignment from `harness_hook.filter_candidates(...)` still lives where it was. Verified end-to-end: re-run of the same dry-run reached `Step 0 complete: 154.6s | 1 eps | mean_reward=561.00 | per_game=[candy_crush=561.0] | 2 skills (+2) across 1 games | 150 vLLM calls`, GRPO `action_taking` ran 50 samples in 111.2 s on 4 GPUs, and the post-step `Adapter hot-reload: 20/20 successful across 4 instances (5 adapters)` confirmed the updated LoRA stitched back into the live vLLM towers.

### 0.4 New action items from the lane decision (T1.3 closed → lane (a))

The lane decision recorded in [`skill-lane-decision.md`](legacy/skill-lane-decision.md)
opens six follow-ups that replace the original open-ended T1.3:

| ID | Sprint | Item | One-line fix |
|---|---|---|---|
| ~~T1.3a~~ | ~~S0~~ | ~~Default-disable the Repairer in the live trainer Crafter~~ → **shipped 2026-05-01.** `SkillCrafterService(enable_protocol_patching=False)` default + `_crafter_hook.run_crafter_step(enable_protocol_patching=…)` plumb-through + `CoEvolutionConfig.crafter_enable_protocol_patching` + `scripts/run_coevolution.py --enable-protocol-patching` opt-in. New tests at [`tests/test_crafter_lane_a_flag.py`](../tests/test_crafter_lane_a_flag.py) verify the default fall-through to the Hypothesizer; offline driver [`labeling_supplement/reflect_per_episode_gpt54.py`](../labeling_supplement/reflect_per_episode_gpt54.py) opts back in. | — |
| ~~T1.3b~~ | ~~S2~~ | ~~Add `RewriteProposal` / `MergeProposal` alias~~ → **shipped 2026-05-02.** [`data_structure/extensions/bank_mutation_proposal.py`](../data_structure/extensions/bank_mutation_proposal.py) (`RewriteProposal`, `MergeProposal = ComposeProposal`); [`tests/test_rewrite_and_merge_proposal.py`](../tests/test_rewrite_and_merge_proposal.py); `GateService._run_static` accepts `RewriteProposal`. | — |
| ~~T1.3c~~ | ~~S2~~ | ~~Lane-(a) `BANK_GAP` / `RETRIEVAL_MISLEAD` / `STALE_DESCRIPTION` live path~~ → **shipped 2026-05-02.** [`common/enums.py`](../common/enums.py) `LANE_A_RECOVERY_STRATEGIES`; [`crafter/service.py`](../crafter/service.py) `_run_failure_dispatch` routes past Repairer; [`tests/test_lane_a_failure_taxonomy.py`](../tests/test_lane_a_failure_taxonomy.py). | — |
| ~~T1.3d~~ | ~~S2~~ | ~~`min_retrievals_per_skill` at ACTIVE~~ → **shipped 2026-05-02.** [`configs/skill_gate.yaml`](../configs/skill_gate.yaml) + [`GateThresholds`](../orchestrator/config.py); [`SkillLifecycleManager`](../skill_bank/lifecycle.py) enforces at ACTIVE promotion. | — |
| ~~T1.3e~~ | ~~S0 (doc-only)~~ | ~~Update [`harness/README.md`](../harness/README.md) §22 + [`crafter-harness-orchestrator-roles.md`](legacy/crafter-harness-orchestrator-roles.md) §7.3 / §7.5~~ → **shipped 2026-05-01.** `harness/README.md` §22 lane block + archived roles memo §7.3 banner + §7.5 supersedure note + new "Post-decision rule" sub-section. | — |
| ~~T1.3f~~ | ~~S0 (doc-only)~~ | ~~Update plan documents (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION)~~ → **shipped 2026-05-01.** Lane-(a) banner blocks at the top of all four PLAN docs, each pointing at [`skill-lane-decision.md`](legacy/skill-lane-decision.md) + [`single-vs-two-mdp-tradeoff.md`](legacy/single-vs-two-mdp-tradeoff.md) and explicitly marking `hop_select` / `inner_mdp` references obsolete. | — |

**Lane closure:** **T1.3a–T1.3f** are all shipped (S0 governance + S2 Crafter/gate mechanics). Nothing remains open in this table.

---

## 1. TL;DR

The trainer scaffolding runs end-to-end on `gymv`, **six** SFT warm-start
artefacts are on disk (`schema_gen` + five cold-start LoRAs; see §0.2),
and **lane (a)** is decided and enforced in code — skills are retrieval
payloads; live Crafter defaults with `enable_protocol_patching=False`
(see [`skill-lane-decision.md`](legacy/skill-lane-decision.md)). **T1.3a–f
are shipped** (S0 + S2 — §0.4). **T1.2 has executed** — offline promotion
flipped hundreds of rows so `bank.runnable() != []` ([`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §S1).

**S2 code items closed 2026-05-02:** eval suite + scoreboard driver +
GRPO `RewardLogger`, lane-(a) proposals + failure taxonomy +
`min_retrievals_per_skill`, curator warmup weighting, etc. (§6 table).

The **remaining gate before trusting perception metrics** is **T2.11 →
re-run all six SFT jobs**, then **re-run T1.1′** (`evaluation/probe_schema_gen_exact_match.py`)
— the 2026-04-30 checkpoints were trained / loaded under incomplete LoRA
coverage (see §0.3). **Outstanding product work:** actually **launch**
fast-loop GRPO on `gymv` with `crafter_promotion_enabled` / audit knobs,
then S3/S4 transfer + router + extension records per §0.1.

Cadence rule unchanged:
[`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)
— fast loop stays `gymv`-first; cross-domain executors remain slow-loop.

What we **can** already exercise:

- Cascaded grounding, gate stages 0–4; G0 on `SkillEpisode`
- Lifecycle / split stores, promotion + rollback, offline promotion mirror
- `gymv` few-shot transfer + typed `BankMutationProposal` / `RewriteProposal`
- Day-10 trainer ↔ harness hook (`SkillHarnessHook`)
- Phase-5/6 **measurement** stack (deterministic-stub tier — not mechanism validation)

---

## 2. Tier 1 — Hard blockers

Training is unsafe or a silent no-op without these.

### T1.1 `schema_gen` Phase-1 SFT checkpoint — **closed (pending exact-match probe)**

> ✅ **Status flipped 2026-05-01 (revision 2).** Phase-1 SFT trained
> 2026-04-30; checkpoint at
> [`runs/sft_schema_gen/schema_gen_20260430_091831/`](../runs/sft_schema_gen)
> with intermediate ckpts at `checkpoint-{200,400,477}`. Final eval
> token-accuracy **0.9838** / eval loss **0.0861** on hold-out
> (`mean_token_accuracy` reported by trainer). Adapter is 16.8 MB,
> r=16 / α=32 over the 7-projection target set on the
> Qwen3.5-35B-A3B base. Training run config in
> `train_config.json` confirms domains `gymv` + `env_wrappers` +
> `browser` + `image_qa` + `video_qa`.

**Outstanding sub-item (T1.1′, blocker for `RunRelease` pinning):**
the trainer reports `mean_token_accuracy`, but
[`PLAN-VISUAL-GROUNDING-MILESTONES.md` §13](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)
specifies **exact-match schema-validity rate** thresholds (Path A
≥ 70 % `gymv`, ≥ 50 % `browser`; field accuracy ≥ 85 % / ≥ 75 %).
Token-accuracy 0.9838 strongly suggests the bar is met but is not the
same metric. Build a 30-min probe at
`evaluation/probe_schema_gen_exact_match.py` that loads the LoRA over
the 35B-A3B base, runs it on a held-out slice of
`labeling/output/grounding/{gymv,env_wrappers,browser}/` triples, and
emits the exact-match rate per domain.

**Fix:** run the probe; if it passes, pin `schema_gen_ckpt_id =
"schema_gen_20260430_091831"` into `RunRelease`. If it fails, retrain
with the same scaffold against a stricter loss or more epochs (the
existing `train.py` is reusable as-is).

### T1.2 §17 keystone — `bank.runnable()` is empty until the offline loop fires once

```51:57:Multi-hop-Reasoning-VLM-Agent/skill_bank/repository.py
def runnable(self, *, include_shadow: bool = True) -> List[SkillRecord]:
    out: List[SkillRecord] = []
    for r in self.active.all():
        if r.status == SkillStatus.SHADOW and not include_shadow:
            continue
        out.append(r)
    return out
```

`runnable()` reads only the active store. Cold-start ingest writes
**everything** to `CANDIDATE`. Until `PromotionOrchestrator.promote`
graduates a batch `CANDIDATE → SHADOW`, the trainer's harness hook sees
`[]` and every `filter_candidates` call admits the legacy bank by
missing-fallback — i.e. the harness is silently bypassed. This is the
keystone called out in
[`harness-usability-and-intra-gymv-transfer.md` §6.2](legacy/harness-usability-and-intra-gymv-transfer.md).

**Fix:** run
[`labeling_supplement/decide_promotion_gpt54.py`](../labeling_supplement/decide_promotion_gpt54.py)
at least once on the cold-start corpus before launching co-evolution; or
have the trainer's first warm-up step gate on a non-empty `runnable()`.

### T1.3 Lane decision — **closed: lane (a), skills are retrieval payloads**

> ✅ **Status flipped 2026-05-01 (revision 2).** Decision recorded in
> [`skill-lane-decision.md`](legacy/skill-lane-decision.md). A skill is a
> *semantic retrieval payload* and *procedural guidance* for the
> actor LLM — name, description, preconditions / effects / role
> labels, optionally NL `protocol`. Skills are **not** runnable
> programs invoked by the harness at training time. The actor's
> live decision loop (`select_skill → update_intention →
> take_action`) and the Day-10 `SkillHarnessHook` (which calls only
> `filter_candidates` + `validate_choice`, never `run_skill`) already
> implement the lane-(a) wire; the decision formalises what was
> implicit.

**What this changes for training:**

* The Repairer's protocol-edit path is parked behind a feature flag
  (T1.3a, S0). Live runs default to lane-(a) Crafter behaviour.
* The lane-(b) machinery that already shipped (`labeling/_protocol_lift.py`
  v2.1, `harness/gymv_executor.py`, `harness/gymv_success.py`,
  `harness/replay_validator.py` action-walk mode, `FewShotAdapter`
  intra-source-domain transfer) **stays in tree** as offline gate /
  diagnostic infrastructure — used by `decide_promotion_gpt54.py`
  and `_phase4_transfer_cycle.py`, not by the live actor.
* Multi-domain `feasible_domains ≥ 2` invariant for `ACTIVE` is
  replaced by `min_retrievals_per_skill` (T1.3d, S2) — unblocks
  `ACTIVE` promotion on single-domain `gymv` banks.
* `FailureClass` taxonomy gains three retrieval-centric classes
  (`BANK_GAP`, `RETRIEVAL_MISLEAD`, `STALE_DESCRIPTION`) for live
  Crafter use (T1.3c, S2).
* Plan documents need a documentation pass — the four PLAN docs
  (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS,
  PLAN-COMPONENTS-IMPLEMENTATION) were written for lane (b) and need
  retrofitting (T1.3f, S0 doc-only).

**Rollback condition:** revisit only if both (i) the actor's
`skill_selection` LoRA + retrieval scoring saturates **and**
(ii) NORTHSTAR §7.3 Joint Success Rate is still below the headline
target after exhausting the
[`single-vs-two-mdp-tradeoff.md` §"Escalation order"](legacy/single-vs-two-mdp-tradeoff.md)
(better `strategic_description`, pattern-tag abstraction layer,
two-call inference inside one MDP). Even then the next escalation is
MCTS-with-a-forward-model on games and tool-augmented harness ops on
VR / video — **not lane (b) directly**.

### T1.4 Real env executors for transfer-target adapters

Only `gymv` has a real executor 1. `osworld`, `video` are pure
`StubTransferTargetAdapter`. `browser` and `visual_reasoning` have helper
bindings (`vlm_wrapper/browser_adapter.py`,
`visual_reasoning_wrapper/skill_executor.bind_executor`) but the trainer
never calls them.

For the fast loop this is fine —
[`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)
pins fast-loop = `gymv` only. For G3a transfer, every target except
`gymv` returns `target_domain_demo_unavailable` ⇒
`min_target_domains_verified ≥ 1` is unreachable ⇒ **no skill ever
leaves `CANDIDATE` even after promotion fires.**

**Fix order:** `gymv` only for Phase-A fast loop → revisit one
transfer-target executor + demo library before claiming any cross-domain
win.

---

## 3. Tier 2 — Critical-path infrastructure

Training fires but signal is degraded.

### T2.1 Few-shot demo libraries for the four transfer targets

[`harness/few_shot_demos_gymv.py`](../harness/few_shot_demos_gymv.py)
exists; analogues for `browser` / `osworld` / `video` / `visual_reasoning`
do not. Compounds with T1.4.

### T2.2 `orchestrator/eval_suite.py` — G5 non-regression eval driver

`orchestrator/gate_service.py::_run_non_regression` accepts
caller-supplied `(baseline, post)` or `EvalSuite`. The frozen-suite
loader specified in
[`PLAN-UNIFIED-SKILL-GATE.md` §4](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)
is missing. Without it the §5 stop/go rules in
[`PLAN-SYSTEM-NORTHSTAR.md`](../plans/00-system/PLAN-SYSTEM-NORTHSTAR.md)
cannot fire on automated promotions.

### T2.3 Eval E0 driver + canonical scoreboard

[`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) lists
`evaluation/driver.py` + `evaluation/answer_evaluator.py` as not yet
delivered.
[`PLAN-SYSTEM-NORTHSTAR.md` §7.3](../plans/00-system/PLAN-SYSTEM-NORTHSTAR.md):
every release **MUST** emit `releases/{release_id}/scoreboard.md` with
the 10-column canonical table + 4 companion tables, all sharing one
`eval_suite_id` + `bank_snapshot_id`. Without this the **Joint Success
Rate** column has no producer and the GRPO loop has no headline.

### ~~T2.4~~ `harness/reward_logger.py` not wired into the GRPO buffer — **closed 2026-05-02**

Per [`PLAN-HARNESS.md` §17](../plans/05-harness/PLAN-HARNESS.md) the harness `RewardLogger` is the canonical sink. **Shipped:** `RewardLogger.log_grpo_record(...)` (kind-discriminated JSONL: `grpo_step` vs. `skill_episode`); `episode_runner.run_episode_async` emits at both GRPO append sites; path through `rollout_collector` → `orchestrator.run_training_loop_async`; `CoEvolutionConfig.reward_log_path`. *(Historical note: shaping for policy gradients may still flow via `decision_agents/reward_func.py`; the audit sink is no longer forked from JSONL logging.)*

### ~~T2.5~~ Policy-as-config YAMLs — **closed 2026-05-01**

| Artefact | Required by | Status |
|---|---|---|
| `configs/skill_gate.yaml` | [`PLAN-UNIFIED-SKILL-GATE.md` §9](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | **Shipped** — [`configs/skill_gate.yaml`](../configs/skill_gate.yaml) (`policy_version` + drift annotations) |
| `configs/failure_routing.yaml` | [`PLAN-FAILURE-ROUTING.md`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md) | **Shipped** — [`configs/failure_routing.yaml`](../configs/failure_routing.yaml) (lane-(a) taxonomy + overrides) |

### ~~T2.6~~ `IMPLEMENTATION-STATUS.md` is stale — **closed** (refreshed 2026-05-01, maintained 2026-05-02)

Previously lagged the Day-7 → 10 hook trio, `legacy_writeback`, `runs/sft_*`, and threshold YAMLs. **Now** co-maintained with this audit — see [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) header + §S0–S2.

### T2.7 `curator` SFT overfit mitigation (added 2026-05-01)

The `curator` adapter at
[`runs/sft_coldstart/skillbank/curator/`](../runs/sft_coldstart/skillbank/curator)
trained 15 epochs on **193 rows** with eval loss **1.218** vs. train
loss **0.491**. The corpus is too small for the LoRA capacity. This
LoRA participates in the medium-timescale Bank GRPO loop alongside
`segment` and `contract`; if its outputs are unreliable on held-out
bank states the medium loop will fight the actor. **Mitigations
(non-blocking, in priority order):**

1. Treat `curator` as a tie-breaker only in early GRPO (low weight in
   the per-step Bank reward composition) — gated via a
   `curator_weight` knob in `trainer/coevolution/config.py`.
2. Regenerate the curator corpus by running
   [`labeling_supplement/decide_skill_crafting_gpt54.py`](../labeling_supplement/decide_skill_crafting_gpt54.py)
   over more games / more episodes, then re-train.
3. Add early-stopping at `eval_loss` plateau detection (currently the
   trainer rides the full 15-epoch budget).

### ~~T2.8~~ vLLM topology check for split-base inference — **closed**

**Documented** in [`legacy/vllm-topology.md`](legacy/vllm-topology.md): `schema_gen` on Qwen3.5-35B-A3B vs GRPO on Qwen3.5-9B — separate workers; trainer defaults align with offline / split-tower operation (`model_name` in [`trainer/coevolution/config.py`](../trainer/coevolution/config.py)).

### ~~T2.9~~ Run-wide SFT manifest — **closed**

[`runs/sft_coldstart/sft_summary_all.json`](../runs/sft_coldstart/sft_summary_all.json) emitted by [`scripts/build_sft_manifest.py`](../scripts/build_sft_manifest.py). *(Legacy `sft_summary.json` may remain per-GPU; drivers should prefer the run-wide manifest.)*

### ~~T2.10~~ Audit `*.partial_20260430_091054` shards — **closed 2026-05-02**

Smoke load on all six adapters — **no torn partial shards**; report `runs/sft_coldstart/_smoke/smoke_all.json` (§0.3).

---

## 4. Tier 3 — Architectural gaps

Won't break the fast loop, but block multi-domain claims.

### T3.1 The five typed extension records ([`PLAN-EXPERIENCE-EXTENSION.md` §3](../plans/08-cross-cutting/PLAN-EXPERIENCE-EXTENSION.md))

| Spec'd | In repo? | Closest analog |
|---|---|---|
| `RunRecord` | ❌ | `RunRelease` covers ~60 %, lacks budget summary + per-run promotion / rollback decision |
| `GroundingRecord` | ❌ as a Python dataclass | `<evidence_refs>` schema section + `parse_evidence_refs` exist; typed dataclass missing |
| `SkillInvocationRecord` | ❌ | `SkillEpisode` covers most fields but plan calls for separate per-attempt record |
| `FailureRoutingRecord` | ❌ | `FailureTrace` lacks `severity`, `recoverability`, `route_to`, `parent_failure_id` — the routing-critical fields |
| `AnswerSupportRecord` | ❌ | reconstructed from `Episode.answer_support_chain` ad-hoc |

**Pick one canon:** either ship the five records or update the plans to
bless the existing alternates.

### T3.2 Failure router — entirely missing

[`PLAN-FAILURE-ROUTING.md`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md)
R0–R5 deliverables — none ship:

- `orchestrator/failure_router.py` (R1)
- `orchestrator/failure_router_runner.py` (R2)
- Detector shims on Harness / Grounding / Judge / Budget (R3)
- `failures/<class>/<id>` queue tree
- Two-step write contract (`failures/incoming/` → `failures/<class>/`)
- Target-rate dashboards (R5)

Today every L2 / L4 failure is either consumed ad-hoc by the Crafter or
dropped, contradicting §10 anti-goal #4.

### T3.3 `HopTrace` artefact + inner-hop logging on the Action Agent

`grep -n HopTrace decision_agents/` returns zero. Per
[`PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md` §3.2](../plans/legacy/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md),
Phase B mining (typed `SkillIR`, inner-hop reasoning skill discovery) has
no input substrate until this lands. `SkillEpisodeStep` carries
inner-step info but isn't logged as the typed `HopTrace` artefact
required.

**Note:** [`single-vs-two-mdp-tradeoff.md`](legacy/single-vs-two-mdp-tradeoff.md)
deliberately retired the inner-MDP scaffold for latency reasons. The
two notes need to be reconciled — either Phase B of the
transferable-reasoning edits is deferred, or `HopTrace` is reintroduced
as an **offline artefact** (typed log) without restoring the online
inner LLM call.

### T3.4 Canonical cross-domain ontology types

`Agent`, `UIElement`, `EvidenceSpan`, `FrameSpan`, `RegionPatch`,
`IdentityHypothesis`, … — zero hits. Without them,
`SkillRecord.protocol` stays `List[Dict[str, Any]]`, and HARNESS
§10b/c/d typed validators (`slot_binding_validator`,
`ontology_remap_validator`, `adapter_compatibility_checker`) cannot do
typed-level checks. **Same blocker as T1.3 lane (b).**

### T3.5 Online runtime adapters (audit §16.2 – §16.4)

| Spec | In repo? |
|---|---|
| `HarnessSkillProvider` | ❌ — actor still consumes `SkillBankProvider`. Trainer routes harness via the orthogonal `SkillHarnessHook` instead. |
| `RunnerActorAdapter` | ❌ — `EpisodeRunner.run` expects `ActorLike.choose_action(state, eligible)`; current `ActorAgent.run` returns text actions. |
| `EnvLike` shim in `vlm_wrapper/` | ❌ — protocol exists at `orchestrator/runner.py:49`; no production-env bridge. |
| `skill_bank/legacy_bridge.py` | ❌ — only one-way [`legacy_writeback.py`](../skill_bank/legacy_writeback.py) ships. |

These don't block the fast loop (the Day-10 hook bypasses them) but
block any future "live online runtime" execution path.

### T3.6 `hop_select` LoRA absence — **decided: single-MDP, plans need updating**

> ✅ **Status corrected 2026-05-01 (revision 2).** This is **not** an
> open question — the lane has been picked, the code has shipped, and
> the SFT corpus on disk reflects the decision. The only remaining
> work is documentation drift on the plan side.

**What the live actor actually does (one MDP):**

* [`decision_agents/actor_agent.py`](../decision_agents/actor_agent.py) `_run_inner_mdp` was **deleted** (~430 LOC); `_pick_action` chooses directly from `harness.valid_actions(state)` ([`actor_agent.py:24`](../decision_agents/actor_agent.py)).
* [`decision_agents/inner_mdp.py`](../decision_agents/inner_mdp.py) is **deleted entirely** — `HopAction` / `HopPolicy` / `HopStep` / `HopTrace` / `HeuristicHopPolicy` / `parse_hop_action` raise `AttributeError` with a `DeprecationWarning` via the package shim.
* The legacy `hop_policy=` / `max_hops_per_step=` kwargs on `ActorAgent.__init__` (and `run_actor_episode`) only exist as deprecation-warning shims for back-compat ([`actor_agent.py:286-298`](../decision_agents/actor_agent.py)); they are ignored at runtime.
* What were inner-MDP operators are now first-class harness actions: `GROUND` → `VRHarness.step(LOOK)`, `RETRIEVE` → `VRHarness.step(RETRIEVE)`, `CONCLUDE` → `VRHarness.step(NOTE)`, `EXECUTE` → folded into the env-action loop. See [`decision_agents/README.md` §"Migration of inner-MDP operators"](../decision_agents/README.md).
* Five `Harness` implementations (`GymHarness`, `BrowserHarness`, `OSWorldHarness`, `VRHarness`, `VideoHarness`) own the per-task action vocabulary; `info["valid_actions"]` is sourced from `harness.valid_actions(state)`.
* GRPO LoRAs are exactly **two**: `skill_selection` + `action_taking`. The `hop_select` LoRA was explicitly dropped because hop decisions are now action decisions, both rolled into `action_taking`. **The on-disk SFT seed corpus at [`runs/sft_coldstart/decision/`](../runs/sft_coldstart/decision) confirms this — only `action_taking/` and `skill_selection/` are trained, no `hop_select/`.**

**Why the single-MDP decision stands**
([`single-vs-two-mdp-tradeoff.md` §"Escalation order"](legacy/single-vs-two-mdp-tradeoff.md)):

1. Latency win — the actor has a non-trivial vLLM TTFT and an inner
   MDP doubles the call count without obvious return.
2. Games specifically — the inner-hop alphabet (`GROUND` / `CHECK` /
   `RETRIEVE`) doesn't fit retro action games; the cold-start
   collectors and SFT corpus would not transfer.
3. Cross-task transfer — one `action_taking` LoRA trained on games +
   web + VR + video (all five harnesses) is the cheapest and most
   transfer-friendly design. Forking into a `hop_select` LoRA breaks
   `harness.action_kind` cost bucketing and the
   "GPT-5.4 → Qwen3.5-9B SFT cold-start works for both LoRAs from the
   same trace" property.

**What's still left to do — plan-side documentation hygiene only**
(the code is settled). All references in `plans/` that imply a
three-LoRA actor (`schema_gen` / `skill_select` / `hop_select` /
`action_execute`) need to be rewritten to reflect the shipped
two-LoRA actor (`skill_selection` / `action_taking`) plus the
SFT-only `schema_gen` adapter on the 35B-A3B base. The list:

| Plan file | Section to update |
|---|---|
| [`plans/02-action-agent/PLAN-ACTION-AGENT.md`](../plans/02-action-agent/PLAN-ACTION-AGENT.md) | §5.3 hop alphabet should reference `Harness.valid_actions` instead of an inner-MDP `hop_select`; §6 phase plan drops `hop_select` from the GRPO target list |
| [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](../plans/03-skill-bank/PLAN-SKILL-BANK.md) | §1.5 hop vocabulary should reconcile with the harness action sets |
| [`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) | Phase F GRPO target list — remove `hop_select` |
| [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) | Mark the single-MDP pivot as shipped (Patch #13 in `decision_agents/README.md`) |

**Effort:** ~30 min find-and-replace across 4 plan files. **No code
changes required.** This is the same documentation-drift pattern as
T3.6's S0 entry in §0.1; the cleanup belongs in the same governance
slot as ongoing plan-doc hygiene (~~T2.6~~ status refresh is done; IMPLEMENTATION-STATUS still benefits when phases shift).

---

## 5. Tier 4 — Open questions

Settle before scaling, not before starting.

1. **Hop-vocabulary inconsistency** between
   [`PLAN-SKILL-BANK.md` §1.5](../plans/03-skill-bank/PLAN-SKILL-BANK.md)
   (`{GROUND, CHECK, RETRIEVE, COMMIT, ACT/EXECUTE, VERIFY}` + `CONCLUDE`)
   and
   [`PLAN-ACTION-AGENT.md` §5.3](../plans/02-action-agent/PLAN-ACTION-AGENT.md)
   (`{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`). Matters for G0
   role-vs-hop check.
2. **Q4 ablation unmeasured** — [`PLAN-HARNESS.md` §20.2](../plans/05-harness/PLAN-HARNESS.md):
   until A0 (no Harness) and A4 (full system) are run on the intra-`gymv`
   probe, the architectural claim that the **Actor (not the 72B) is the
   policy** is unverified. The `skill_transfer_test/` framework is
   sketched in
   [`harness-usability-and-intra-gymv-transfer.md` §5](legacy/harness-usability-and-intra-gymv-transfer.md).
3. **Phase-2 teacher-SFT trigger is qualitative-only** ("if narrow
   failures persist"); no quantitative threshold.
4. **Counterfactual prediction reliability** — Crafter §6.9
   auto-synthesis at 3+ recurrences, no verifier-quality test.
5. **Numerical thresholds unpinned:** shadow-cycle K,
   `few_shot.k_shot_default` / `max`,
   `transfer_min_target_domains_verified`, slow-loop N episodes,
   NORTHSTAR §5.6 noise bands (blank for first 3 releases).
6. **Stage 5 two-step (`PROVISIONAL → ACTIVE`)** — `PromotionOrchestrator.promote`
   does not visibly enforce the `provisional_active_window_episodes: 200`
   window or `shadow_origin_penalty: 0.30`.
7. **Crafter input contract** (`active_bank_skills`,
   `candidate_protocol_clusters`, `failed_hop_patterns`,
   `failure_cases`, `transfer_mismatch_reports`) per
   [`PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md` §4.7](../plans/legacy/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md)
   — not present as a typed `CrafterInput` dataclass.

---

## 6. Recommended sequencing (concrete sprint plan)

> **Revised 2026-05-01 (revision 2):** S0 pre-flight is lighter — the
> `schema_gen` Phase-1 SFT is already trained, so the only remaining
> S0-data item is the exact-match probe (T1.1′).

| Sprint | Track | Items | Why now |
|---|---|---|---|
| **S0 — pre-flight** ✅ shipped 2026-05-01 | data | ~~**T1.1′**~~ — probe [`evaluation/probe_schema_gen_exact_match.py`](../evaluation/probe_schema_gen_exact_match.py) **ran 2026-05-02 on legacy checkpoints — failed** (LoRA not applied — T2.11). **Outstanding:** **re-run six SFT jobs** under [`trainer/SFT/lora_targets.py`](../trainer/SFT/lora_targets.py), then **re-run T1.1′** | exact-match bar + RunRelease pinning |
| | code | ~~**T1.3a**~~ — `enable_protocol_patching: bool = False` flag on `SkillCrafterService`, threaded through `_crafter_hook` / `CoEvolutionConfig` / `run_coevolution.py`; tests in [`tests/test_crafter_lane_a_flag.py`](../tests/test_crafter_lane_a_flag.py) | lane-(a) default — Repairer parked behind feature flag |
| | governance | ~~T1.3~~ — **closed**, see [`skill-lane-decision.md`](legacy/skill-lane-decision.md). **All T1.3a–f shipped** (S0 + S2 — §0.4). | — |
| | governance | ~~T2.6~~ — `IMPLEMENTATION-STATUS.md` refreshed to reference `runs/sft_*`, Day-7 → 10 hooks, lane-(a) decision, threshold YAMLs, SFT manifest, offline-promotion driver | — |
| | governance | ~~**T1.3e + T1.3f + T3.6**~~ — `harness/README.md` §22 lane block + `crafter-harness-orchestrator-roles.md` §7.3 / §7.5 banner+supersedure + four PLAN docs banner (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION); T3.6 obsoletes `hop_select` references in those banners | doc drift closed |
| | governance | ~~**T2.9**~~ — [`scripts/build_sft_manifest.py`](../scripts/build_sft_manifest.py) emits [`runs/sft_coldstart/sft_summary_all.json`](../runs/sft_coldstart/sft_summary_all.json) | — |
| | governance | ~~**T2.10**~~ — [`evaluation/smoke_load_sft_adapters.py`](../evaluation/smoke_load_sft_adapters.py) ran 2026-05-02 on all six adapters; **no torn `*.partial_*` shards** — report `runs/sft_coldstart/_smoke/smoke_all.json` | — |
| **S1 — fire offline once** ✅ shipped + executed 2026-05-02 | data | ~~**T1.2**~~ — [`scripts/run_offline_promotion_cycle.sh`](../scripts/run_offline_promotion_cycle.sh) ran end-to-end; writeback green (**375 / 489** rows). §17 post-condition satisfied — see [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §S1 | — |
| | governance | ~~T2.5~~ — [`configs/skill_gate.yaml`](../configs/skill_gate.yaml) + [`configs/failure_routing.yaml`](../configs/failure_routing.yaml) shipped (policy v1.0.0; lane-(a) failure taxonomy + drift annotations) | — |
| **S2 — fast loop launch** ✅ code items closed 2026-05-02 | code | ~~**T1.3b**~~ — `RewriteProposal` + `MergeProposal` alias landed in [`data_structure/extensions/bank_mutation_proposal.py`](../data_structure/extensions/bank_mutation_proposal.py); gate `_run_static` accepts both | — |
| | code | ~~**T1.3c**~~ — `BANK_GAP` / `RETRIEVAL_MISLEAD` / `STALE_DESCRIPTION` shipped in [`common/enums.py`](../common/enums.py) (`LANE_A_RECOVERY_STRATEGIES`); `FailureDiagnoser` + `_run_failure_dispatch` route them past the Repairer to the Hypothesizer | — |
| | code | ~~**T1.3d**~~ — `min_retrievals_per_skill` knob shipped on [`GateThresholds`](../orchestrator/config.py) + [`configs/skill_gate.yaml`](../configs/skill_gate.yaml); `SkillLifecycleManager` enforces it at the `ACTIVE` transition; legacy multi-domain Stage-0 echo removed | — |
| | training | ~~**T2.7**~~ — `curator_weight` + `curator_warmup_steps` shipped on [`CoEvolutionConfig`](../trainer/coevolution/config.py); `set_curator_warmup(...)` called once per outer step from [`orchestrator.run_training_loop_async`](../trainer/coevolution/orchestrator.py); `_dynamic_curator_reward` multiplies the base reward by the linear ramp | — |
| | training | ~~**T2.4**~~ — `RewardLogger.log_grpo_record(...)` is the single sink (kind-discriminated JSONL); wired through `episode_runner` → `rollout_collector` → `orchestrator` with a `reward_log_path` config knob (auto-resolves under the rewards dir) | — |
| | training | Launch fast-loop GRPO on `gymv` only ([`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)) — actor + bank LoRAs at 5–10 : 1 cadence, all 5 LoRAs warm-started from `runs/sft_coldstart/` | Phase 1 of [`PLAN-ACTION-AGENT.md` §6](../plans/02-action-agent/PLAN-ACTION-AGENT.md) |
| | governance | ~~**T2.2**~~ — [`orchestrator/eval_suite.py`](../orchestrator/eval_suite.py) loader + `EvalSuite` canonical home; `GateService.evaluate(eval_suite=)` consumes it directly. Starter suite at [`evaluation/suites/gymv-smoke-v1/`](../evaluation/suites/gymv-smoke-v1/) | — |
| | governance | ~~**T2.3**~~ — [`evaluation/answer_evaluator.py`](../evaluation/answer_evaluator.py) (F1–F7), [`evaluation/scoreboard.py`](../evaluation/scoreboard.py) (canonical 10x10 + companion tables, markdown + JSON sidecar), [`evaluation/driver.py`](../evaluation/driver.py) (writes per-instance JSONL + suite scoreboard JSON consumed by T2.2). `RunRelease` carries `eval_suite_id` + `scoreboard_path` | — |
| | governance | ~~**T2.8**~~ — split-base vLLM topology documented in [`implementation_notes/legacy/vllm-topology.md`](legacy/vllm-topology.md) | — |
| **S3 — first transfer probe** | data | T1.4 + T2.1 — pick one transfer target (probably `browser` per the plan's first-arena), build demo library, plug `set_executor` | unblocks G3a → first `ACTIVE` skill possible |
| | governance | A0 vs A4 ablation on intra-`gymv` probe (Q4 + [`harness-usability-and-intra-gymv-transfer.md`](legacy/harness-usability-and-intra-gymv-transfer.md)) | answers Q4: is the Actor doing real work? |
| **S4 — multi-domain hardening** | architecture | T3.1 (extension records) + T3.2 (failure router) + T3.3 (`HopTrace`) + T3.4 (ontology types) | required for cross-domain claims, not for `gymv` |

---

## 7. What to **not** block on before training

- **`hop_select` LoRA (T3.6)** — design has been deliberately rejected
  and **the single-MDP code has shipped**
  ([`single-vs-two-mdp-tradeoff.md`](legacy/single-vs-two-mdp-tradeoff.md);
  [`decision_agents/actor_agent.py`](../decision_agents/actor_agent.py)
  has no `_run_inner_mdp`; `inner_mdp.py` is deleted; SFT seeds in
  [`runs/sft_coldstart/decision/`](../runs/sft_coldstart/decision)
  are exactly 2 LoRAs). The remaining work is plan-doc cleanup, not
  re-litigation. **Do not re-add `hop_select` to the trainer.**
- **Online runtime adapters (T3.5)** — the Day-10 `SkillHarnessHook`
  covers the trainer surface; the runtime adapters are independent work
  for live deployment.
- **Real executors for `osworld` / `video` / `visual_reasoning`** —
  fast loop is `gymv`-only; these are 4-week projects each per
  [`harness-usability-and-intra-gymv-transfer.md` §2](legacy/harness-usability-and-intra-gymv-transfer.md).
- **Phase B of transferable-reasoning edits** — defer until `HopTrace`
  design is reconciled with the single-MDP decision.

---

## 8. Verdict

**S0–S1 governance + drivers are shipped; S2 structural code items are
shipped (2026-05-02).** What remains before calling the fast loop
*"trustworthy end-to-end"* is mostly **execution + weights**:

- ~~T1.1 (schema SFT checkpoint on disk)~~ → **still true**, but **T1.1′
  must be re-run after T2.11 SFT re-run** — old checkpoints load with
  incomplete LoRA legs (§0.3).
- ~~T1.2 (`bank.runnable()` non-empty)~~ → **closed 2026-05-02**
  ([`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §S1).
- ~~T1.3 + T1.3a–f~~ → **closed** — lane (a), Repairer default-off, docs,
  and **T1.3b/c/d** Crafter + gate mechanics ([`skill-lane-decision.md`](legacy/skill-lane-decision.md), §0.4).
- ~~T2.4 / T2.5 / T2.7~~ → **shipped** (reward JSONL sink, threshold YAMLs,
  curator warmup ramp — §6).
- **Next operational step:** launch **fast-loop GRPO on `gymv`** with the
  trainer harness + promotion hooks; treat **T2.11 SFT re-run** as blocking
  any strong claim from `schema_gen`.

Everything else (T1.4/T2.1 transfer targets, T3.* router / extension
records, live `HarnessSkillProvider`) can land asynchronously without
blocking the first `gymv-only` GRPO epoch — see §7.

---

## 9. Subagent audits — converged findings

The four parallel audits all landed on overlapping conclusions; they are
recorded here so the synthesis above is reproducible without re-running
them.

### 9.1 Plan-discipline audit (Action Agent + Skill Bank + Skill Crafter + Harness)

The plans converge on a strict frozen-first / phase-A-then-phase-B
discipline:

- **(A) Phase order:** GRPO targets the **9B Actor first**. *Plan
  side* lists `schema_gen` / `skill_select` / `hop_select` /
  `action_execute`; *live side* (after the single-MDP pivot — see
  T3.6) is `schema_gen` (SFT-only on the 35B-A3B control plane) +
  `skill_selection` + `action_taking`. The Bank's medium-timescale
  `CONTRACT` / `CURATOR` / `SEGMENT` LoRAs second (5–10 actor
  cycles : 1 bank cycle); only narrow SFT on the frozen 32B / 72B
  Crafter as a last resort.
- **(B) Cold-start prerequisites:** Tier-0 / Tier-1 seed trajectories,
  the five-stage bank pipeline, the unified gate plumbing
  (`SkillLifecycleManager` + four-store split + `skill_gate.yaml`),
  schema-as-input wiring, and Crafter Phase 0 multi-pass procedures
  with the typed-proposal schema.
- **(C) Hard prerequisites:** timescale separation; the G0–G5
  acceptance-gate stack on every episode (G0 evidence-driven invariant
  runs continuously); the frozen game-source-domain non-regression
  slice; the general-protocol invariant (no per-domain sub-banks);
  Layer A guardrail before Layer B GRPO; and Harness Phase 0+1
  (`SkillEpisode` + `SkillHarness` + `AdapterRegistry`) before any new
  trainable model is added.
- **(D) Openly flagged unresolved items:** hop-vocabulary inconsistency
  between Bank (`CONCLUDE`) and Action Agent (`COMMIT` + `VERIFY`); the
  documented 7B / 8B drift risk on free-form `hop_select`; "new skill
  vs. new adapter" ownership; counterfactual prediction reliability;
  the unverified "Actor is the real policy" claim until Harness
  ablations A0–A4 run; and "verifier quality" being the silent
  load-bearing assumption.

### 9.2 Phase-and-artefact audit

- **Phase ordering A → F is strict;** GRPO surface lives in Phase F.
- **Ten architectural invariants** in
  [`PLAN-COMPONENTS-IMPLEMENTATION.md` §7](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md);
  four-way Actor / Harness / Bank / Orchestrator separation.
- **Source-domain (`gymv`) vs. transfer-target asymmetry** is binding;
  NORTHSTAR §5 stop / go rules are binding.
- **Per-phase acceptance gates:** Phases A.5 / B.7 / C.7 in
  PLAN-COMPONENTS; Phase D shadow-first / six-gate; Phase E scoreboard
  reproducibility; eval phases E0 / E1 / E2; NORTHSTAR phase-emphasis
  exits.
- **Required artefacts:** full module file map under
  `src/{harness, crafter, orchestrator, skill_bank, common}`; eleven
  record types incl. `SkillEpisode` / `GateVerdict` / `AuditRecord`;
  `artifacts/*` storage layout; canonical scoreboard with four
  companion tables incl. few-shot transfer; `AuditRecord` six-field
  contract; three judge surfaces; `BudgetController`.
- **Open questions:** unpinned thresholds for K-cycle shadow stability
  and few-shot K-shot; deferred Hypothesizer Best-of-N and
  `PatchProposal` / `RetireProposal`; stub adapters for `browser` /
  `osworld` / `video` / `visual_reasoning`; episode-local-only state
  surface; judge-revision ownership when agreement < 0.85.

**Bottom line:** GRPO co-evolution can only start safely once Phases A–E
are green, the §0a.6 control-path interface is the sole call pattern,
evidence-role G0 is enforced on every `SkillEpisode`,
`PromotionOrchestrator` transactions are atomic and audited, the
canonical scoreboard CI emission contract is operational, and the
cadence asymmetry (fast / medium = `gymv`, slow = few-shot transfer) is
wired.

### 9.3 Plan-vs-repo cross-check (8 plan documents)

- **Visual-grounding plumbing** (cascade, semantic validator,
  `evidence_refs`, labeling scripts) is **shipped**. ~~the Phase-1
  `schema_gen` SFT checkpoint is only scaffolded, not trained~~ →
  **revision 2:** Phase-1 SFT trained 2026-04-30 (`runs/sft_schema_gen/schema_gen_20260430_091831/`,
  eval token-acc 0.9838); the remaining S0 verification step is the
  exact-match probe (T1.1′), not the training run.
- **G0–G5 gate stack is structurally complete**
  (`SkillLifecycleManager`, `GateService` / `GateRunner`,
  `FewShotAdapter`, `PromotionOrchestrator`, source / target asymmetry
  in `common/enums.py`); missing pieces are `configs/skill_gate.yaml`,
  `orchestrator/eval_suite.py`, and few-shot demo libraries for the
  four transfer targets.
- **Failure routing is the largest hole:** none of the five
  PLAN-EXPERIENCE-EXTENSION records (`RunRecord`, `GroundingRecord`,
  `SkillInvocationRecord`, `FailureRoutingRecord`, `AnswerSupportRecord`)
  exist as dataclasses, and `orchestrator/failure_router.py` +
  `configs/failure_routing.yaml` + the `failures/<class>/<id>` queue
  tree are absent.
- **Edits status:** Harness control-plane edits and the lightweight
  visual-grounding edits have largely landed; the
  transferable-reasoning-skills edits are the most incomplete — typed
  `SkillIR`, canonical cross-domain ontology types
  (`Agent` / `UIElement` / `EvidenceSpan` / etc.), `HopTrace` logging,
  and the inner-hop discovery pipeline are all still on paper, which
  directly blocks cross-domain transfer.

### 9.4 Trainer co-evolution scaffold audit

The trainer co-evolution scaffold is real and substantive: orchestrator,
episode runner, GRPO with 5 LoRAs (`skill_selection`, `action_taking`,
`segment`, `contract`, `curator`), vLLM lifecycle / hot-reload, per-game
skill bank pipeline, Crafter + Promotion hooks, harness hook. **All
five GRPO LoRA targets have SFT warm-starts on disk** as of 2026-04-30
(`runs/sft_coldstart/decision/{action_taking,skill_selection}/`,
`runs/sft_coldstart/skillbank/{segment,contract,curator}/`). Blocking
gaps before §6 launch:

- `hop_select` LoRA does not exist — **single-MDP design has shipped**
  (T3.6). `inner_mdp.py` is deleted; only `skill_selection` and
  `action_taking` are in the GRPO / SFT corpus. Remaining work is
  plan-doc cleanup, not re-implementation.
- `schema_gen` is SFT-only (intentional per
  [`PLAN-EDITS-VISUAL-GROUNDING-LIGHTWEIGHT.md`](../plans/legacy/10-edits/PLAN-EDITS-VISUAL-GROUNDING-LIGHTWEIGHT.md))
  and not in the GRPO `ADAPTER_MAP`. **Revision 2:** the SFT
  checkpoint is now on disk
  (`runs/sft_schema_gen/schema_gen_20260430_091831/`, base
  Qwen3.5-35B-A3B);   needs **T1.1′ after T2.11 SFT re-run** (exact-match probe); ~~T2.8~~ topology is documented.
- `HarnessSkillProvider`, `RunnerActorAdapter`, and the `vlm_wrapper`
  `EnvLike` shim are unimplemented.
- `skill_bank/legacy_bridge.py` is missing (one-way
  [`legacy_writeback.py`](../skill_bank/legacy_writeback.py) ships).
- ~~`harness/reward_logger.py` is **not** wired into GRPO~~ → **shipped**
  (`RewardLogger.log_grpo_record` JSONL sink — §0.1 ~~T2.4~~).
- Only `gymv` has a real adapter executor (`browser` / `osworld` /
  `video` / `visual_reasoning` ship as stubs with `set_executor` hooks).
- `GateRunner` exposes 5 stages, not the plan's G0–G5.
- ~~`IMPLEMENTATION-STATUS.md` is stale~~ → **refreshed** (2026-05-02 —
  §0.1 ~~T2.6~~).
- **Revision 2 (added 2026-05-01):** the `curator` SFT eval loss
  (1.218) flags an overfit risk that needs gating in early GRPO (T2.7).

---

## 10. Headline

Training is **not** ready to start safely in the strict sense the plans
require until **`schema_gen` is re-verified after the T2.11 LoRA recipe
fix** (six-way SFT re-run → **T1.1′** exact-match probe). Revision 3
closed the lane question — **lane (a): skills are retrieval payloads /
procedural guidance for the actor LLM, not runnable programs**
([`skill-lane-decision.md`](legacy/skill-lane-decision.md)). **Closed since the prior headline draft:** ~~T1.3a~~ (Repairer default-off), ~~T1.2~~ (offline promotion executed), ~~T2.4~~ (reward JSONL sink), ~~T2.5~~ (threshold YAMLs), ~~T2.6~~ (IMPLEMENTATION-STATUS refresh), ~~T2.8–T2.10~~ (topology doc, run-wide manifest, smoke shards).

**Remaining S0–S2 critical path (condensed):**

1. **Re-run six SFT jobs** on the corrected [`trainer/SFT/lora_targets.py`](../trainer/SFT/lora_targets.py) recipe (**T2.11**), then **T1.1′** (exact-match probe §13).
2. **Launch fast-loop GRPO on `gymv`** ([`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)) with warm-started adapters.
3. **Gate `curator` LoRA** at low weight in early GRPO (**T2.7**) — mitigations §3.7.

The Day-7 → 10 work closed the structural harness/orchestrator wire;
what remains is **training-quality verification** (LoRA coverage +
probe), actually **starting** GRPO, and **curator** risk management —
not re-litigating lane (a) or offline promotion.
