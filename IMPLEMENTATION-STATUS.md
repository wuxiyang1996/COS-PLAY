# Implementation status — P0, Phase A, B, C MVP

Last updated: 2026-05-03 (S0+S1 — lane-(a) decision shipped, SFT
checkpoints reconciled, pre-flight tooling landed; **T2.11 LoRA
target-modules recipe fix + T2.12 SFT throughput uplift** landed
2026-05-02; **T2.12 CUDA-13 toolkit unblock (flash-attn 2.8.3 +
causal_conv1d 1.6.1 + mamba_ssm 2.3.1)**, **T2.13 SFT/GRPO loader-class
key-remap fix (cold-start LoRA keys now carry `language_model.` prefix
to match vLLM's multimodal loader; permanent loader-class fix landed in
`trainer/SFT/train.py`, `skill_agents/grpo/fsdp_trainer.py` (both
`_train_one_adapter` + `_fsdp_train_worker_multi`), and
`skill_agents/lora/model.py`; T2.11 random-init fallback drift in the
same files also closed)**, **T2.13′ 1-shot ICL wiring in production
schema-gen callers (`vlm_wrapper/tool_loop.py`,
`osworld_wrapper/adapter.py`,
`visual_grounding_tests/generate_osworld_text_schema.py`)**, **T2.14
vLLM 0.20 `deep_gemm_warmup` hard-fail unblock (orchestrator now sets
`VLLM_USE_DEEP_GEMM=0` for spawned `vllm serve` instances on bf16/fp16
weights — `trainer/coevolution/vllm_server.py`)**, and **T2.15
`harness_filter_diag` UnboundLocalError when bank empty / sticky-
guidance (init hoisted to per-step scope in
`trainer/coevolution/episode_runner.py`)** all landed today; the same
day a 1-step `scripts/run_coevolution.py` dry-run ran end-to-end on
candy_crush — `Step 0 complete: 154.6s | 1 eps | mean_reward=561.00 |
2 skills (+2) | 150 vLLM calls`, GRPO action_taking 50 samples in
111.2 s on 4 GPUs, 20/20 LoRA hot-reloads — **co-evolution loop
greenlit for Stage-1 launch**; **Phase-5/6 measurement Stages 0-6
shipped (deterministic-stub tier; see Phase-5/6 §12 gap inventory at
`implementation_notes/legacy/phase5-cross-domain-measurement.md`)**).

This document tracks what has been implemented from
[`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md)
and where each module lives. It coexists with — and does not replace —
the legacy code under `decision_agents/`, `skill_agents/`,
`vlm_wrapper/`, and `data_structure/`.

> **Cross-references for the current readiness audit**:
> - [`implementation_notes/pre-training-readiness-audit.md`](implementation_notes/pre-training-readiness-audit.md) — open / closed items by sprint (S0–S4).
> - [`implementation_notes/legacy/skill-lane-decision.md`](implementation_notes/legacy/skill-lane-decision.md) — T1.3 closed: skill = retrieval payload (lane (a)).
> - [`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](implementation_notes/legacy/single-vs-two-mdp-tradeoff.md) — T3.6 decided + shipped: single-MDP (no `hop_select` LoRA, no `inner_mdp.py`).
> - [`runs/sft_coldstart/sft_summary_all.json`](runs/sft_coldstart/sft_summary_all.json) — run-wide manifest of all six trained SFT adapters (T2.9).

## Delivered

| Phase | Module | Files |
| --- | --- | --- |
| Common | Shared types and enums | `common/{enums,ids,state_schema,typing,models}.py` |
| P0 | Extension records | `data_structure/extensions/{skill_episode,skill_record,gate_verdict,skill_evaluation,bank_mutation_proposal,failure_trace,run_release}.py` |
| Skill bank | Split-storage + lifecycle | `skill_bank/{stores,repository,lifecycle}.py` |
| Phase A | Harness MVP | `harness/{skill_adapter,adapter_registry,eligibility,reward_logger,replay_validator,skill_harness}.py`, `harness/adapters/{_common,gymv_adapter,browser_adapter}.py` |
| Phase B | Orchestrator MVP | `orchestrator/{config,artifact_store,budget,snapshot_manager,gate_service,promotion_orchestrator,runner}.py` |
| Phase C | Crafter MVP | `crafter/{failure_memory,failure_diagnoser,composer,generalizer,hypothesizer,service}.py` |
| Crafter Phase D | `Repairer` + `PatchProposal` plumbing (`SkillCrafterService.propose_repair`, repair-first dispatch in `cycle()`) | `crafter/repairer.py`, `crafter/service.py` |
| Crafter Phase F | Frozen Qwen3-VL-32B / 235B-A22B teacher registry + `SkillCrafterService.{with_qwen3_vl_teacher, from_env, set_teacher_model}` (default `Qwen/Qwen3.5-35B-A3B`; Qwen3-VL is an opt-in upgrade path) | `common/models.py`, `crafter/service.py` |
| Harness Day 7-9 | `GateRunner` (offline gate surface), action-level `ReplayValidator`, `validate_invocation`, `RejectedSkillSink`, expanded `SkillEpisode*` / `SkillEvaluationRecord` fields, `PromotionOrchestrator` reproducibility anchors, `--persist` writeback for the Stage-3a transfer cycle | `harness/{gate_runner,replay_validator,skill_harness,eligibility,rejected_skill_sink}.py`, `data_structure/extensions/{skill_episode,skill_evaluation}.py`, `orchestrator/promotion_orchestrator.py`, `labeling_supplement/_phase4_transfer_cycle.py` |
| Trainer Day 10 | `SkillHarnessHook` wires the harness's two LLM-free surfaces into the co-evolution loop: pre-LLM eligibility filter + post-LLM `validate_invocation` veto in Phase A, and `RejectedSkillSink → record_false_binding_pattern` drain in Phase B′. Opt-in via `--harness-enabled` / `crafter_promotion_enabled`. | `trainer/coevolution/{_harness_hook,_crafter_hook,episode_runner,rollout_collector,orchestrator,config}.py`, `scripts/run_coevolution.py` |
| Lane-(a) flag (T1.3a, S0) | `SkillCrafterService(enable_protocol_patching=False)` default — Repairer / `PatchProposal` mint path is parked under the lane-(a) decision; dispatcher's existing `_STATUS_NO_OP` → Hypothesizer fall-through carries the failure signal. Threaded through `_crafter_hook → CoEvolutionConfig → run_coevolution.py --enable-protocol-patching`. | `crafter/service.py`, `trainer/coevolution/{_crafter_hook,config,orchestrator}.py`, `scripts/run_coevolution.py` |
| Threshold YAMLs (T2.5, S0) | `configs/skill_gate.yaml` (single source of truth for G0–G5 thresholds + drift annotations) and `configs/failure_routing.yaml` (FailureClass → Crafter mode dispatch table, including the new lane-(a) `BANK_GAP / RETRIEVAL_MISLEAD / STALE_DESCRIPTION` taxonomy). Both carry `policy_version` for audit-log drift attribution. | `configs/{skill_gate,failure_routing}.yaml` |
| SFT manifest + tools (T2.9 / T2.10 / T1.1′, S0) | Run-wide manifest for all six trained adapters; load-smoke and exact-match probes for pre-flight verification before launching co-evolution. | `scripts/build_sft_manifest.py`, `runs/sft_coldstart/sft_summary_all.json`, `evaluation/{smoke_load_sft_adapters,probe_schema_gen_exact_match}.py` |
| Offline promotion cycle (T1.2, S1) | One-shot wrapper for the §17 keystone — drives `decide_promotion_gpt54.py` + `legacy_writeback.writeback_promotion` once to flip cold-start banks from CANDIDATE to ACTIVE/PROVISIONAL/SHADOW so `bank.runnable() != []`. Asserts the post-condition before exiting. | `scripts/run_offline_promotion_cycle.sh`, `labeling_supplement/decide_promotion_gpt54.py`, `skill_bank/legacy_writeback.py` |
| Phase-5/6 measurement (Stages 0-6) — **image-VR + video + Tier 2 + Tier 3 closed 2026-05-02** | Cross-domain transfer measurement infrastructure: Stage 0 audit oracle (vocab Jaccard / predicate firing / slot binding), Stage 1-4 cross-domain executors + success_fns + schema producers + few-shot demo loaders, Stage 5 archetype aggregator + within-VR/video 4x4 matrix driver, Stage 6 NxN matrix driver + unified report generator with G1-G6 acceptance gates. After 2026-05-02 follow-up waves, image-VR (Stage 1) and video (Stage 2) cells measure real admit-rates via `harness/_{vr,video}_per_sample_executor.py`'s `TaskAware*Executor` wrappers, Tier 2 `vlm_wrapper/{visual_reasoning,video}_adapter.py` shims ship, and Tier 3's per-domain runtime predicate-translator (`harness/predicate_translator.py` + 28 unit tests + 4-target dispatcher wiring) bridges game-vocab effects (e.g. `cumulative_reward_increased`) onto target-vocab predicates (`[answer_emitted, answer_matches_gold]`). Remaining critical-path: just osworld/browser real-env executors, both gated on CI sandbox provisioning. See [`implementation_notes/legacy/phase5-cross-domain-measurement.md`](implementation_notes/legacy/phase5-cross-domain-measurement.md) §12 for the updated inventory. | `skill_transfer_test/extract/audits/`, `harness/{qa,video_qa,osworld,browser}_success.py`, `harness/{video,osworld,browsergym}_executor.py`, `harness/{osworld,browser}_schema_producer.py`, `harness/few_shot_demos_{vr,video,osworld,browsergym}.py`, `labeling_supplement/{_phase4_target_dispatch,_phase5_matrix,_phase4_transfer_matrix,_phase4_transfer_report}.py`, `skill_transfer_test/extract/archetype_aggregator.py` |
| Tests | Invariant + smoke + backbone-model + crafter Phase D/F + lane-(a) flag + harness Day 7-10 + trainer Day 10 hook | `tests/{conftest,test_invariants,test_smoke,test_backbone_model,test_crafter_*,test_few_shot_transfer,test_gate_runner,test_replay_validator_action_walk,test_lifecycle_*,test_validate_invocation,test_skill_episode_field_expansion,test_promotion_orchestrator_anchors,test_phase4_persist,test_stub_executor_typed_hops,test_rejected_skill_sink,test_trainer_harness_hook,test_crafter_lane_a_flag}.py` (433 passing as of S0 close; one pre-existing unrelated failure in `test_schema_predicates::test_extra_whitespace_tolerated`) |

### Backbone models — three-tier stack

Single source of truth: `common/models.py`.

  - `BACKBONE_MODEL = "Qwen/Qwen3.5-9B"` — actor (`decision_agents`) +
    skill-bank (`skill_agents`) trained policy. LoRA-adapter target for
    `trainer/SFT/` and `trainer/coevolution/`.
  - `BACKBONE_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"` — frozen
    control-plane backbone shared by the crafter, harness, and
    orchestrator. Served separately via
    `inference/serve_qwen35_35b_a3b.sh`.
  - `BACKBONE_JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"` (revised 2026-05-03)
    — eval-driver judge / skill-evaluation judge / promotion gates
    (E0 / E1 / E2). **Same weights as `BACKBONE_TEACHER_MODEL`,
    different role**: a single `serve_qwen35_35b_a3b.sh` instance
    services both. Per-model dispatch is wired via
    `API_func._candidate_vllm_urls` + `VLLM_BASE_URL_MAP`. Override
    to `gpt-5.5` via `VLM_AGENT_BACKBONE_JUDGE_MODEL` for paper /
    formal eval where within-Qwen-family bias must be controlled.
  - `BACKBONE_SFT_TEACHER_MODEL = "gpt-5.5"` — cold-start data
    generation (`cold_start/`, `labeling/`) consumed by `trainer/SFT/`.
    Stays on the frontier model because cold-start labels are baked
    once into SFT adapters and never re-run during training.

The Qwen3-VL Phase-F teachers (`Qwen/Qwen3-VL-32B`,
`Qwen/Qwen3-VL-235B-A22B`) and the older 8B / 32B / 72B Qwen tracks are
**deferred**. They remain reachable through dedicated entrypoints:

  - `scripts/qwen3_decision_agent.py`
  - `scripts/qwen3_skillbank_agent.py`
  - `inference/run_qwen3_8b_eval.py`
  - `inference/run_academic_benchmarks.py`
  - `skill_agents/lora/` (training-time only)
  - `SkillCrafterService.with_qwen3_vl_teacher(...)`

No library default points at them. To re-enable a deferred track,
either pass `model="..."` explicitly or set one of
`VLM_AGENT_BACKBONE_MODEL` / `VLM_AGENT_BACKBONE_TEACHER_MODEL` /
`VLM_AGENT_BACKBONE_JUDGE_MODEL` / `VLM_AGENT_BACKBONE_SFT_TEACHER_MODEL`
at process start.

Live defaults flipped in the 2026-04-28 model-stack migration:

  - `decision_agents/agent.py` `VLMDecisionAgent.DEFAULT_MODEL`: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `decision_agents/actor_agent.py` `DEFAULT_MODEL`: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `decision_agents/agent_helper.py` `DEFAULT_LLM_MODEL`: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `API_func.ask_model(model=None, ...)` default: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `orchestrator/config.py` `TeacherConfig.model_name`: `gpt-4o` → `Qwen/Qwen3.5-35B-A3B`
  - `orchestrator/config.py` `JudgeConfig.model_name`: `gpt-4o` → `gpt-5.5`
  - `crafter/service.py` `SkillCrafterService._teacher`: `gpt-4o` → `Qwen/Qwen3.5-35B-A3B`
  - `cold_start/` and `labeling/` `MODEL_GPT54` / `--label_model`: `gpt-5.4` / `gpt-5-mini` → `gpt-5.5`
  - `skill_agents/lora/config.py` `MultiLoraConfig.base_model_name_or_path`: `Qwen/Qwen3-8B` → `Qwen/Qwen3.5-9B`

Live defaults flipped in the 2026-05-03 judge consolidation:

  - `common/models.py` `BACKBONE_JUDGE_MODEL`: `gpt-5.5` → `Qwen/Qwen3.5-35B-A3B`
    (judge is now consolidated onto the local control-plane teacher,
    saving judge API spend; `gpt-5.5` remains the documented override
    for paper / formal eval via `VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5`).
  - `skill_agents/skill_evaluation/config.py` `LLMJudgeConfig.model`:
    `None` → `BACKBONE_JUDGE_MODEL` (fixes a latent self-judging bug where
    the judge silently fell back to the 9B actor).
  - `orchestrator/config.py` `JudgeConfig.model_name` (default still
    `BACKBONE_JUDGE_MODEL`): now resolves to `Qwen/Qwen3.5-35B-A3B`.
  - `API_func.ask_vllm`: added `VLLM_BASE_URL_MAP` per-model URL
    dispatch so a single call can route 35B requests to :8001 while
    9B requests stay on :8000. Contract pinned by
    `tests/test_api_func_routing.py`.
  - `scripts/use_35b_judge.sh` (new): one-shot helper that exports the
    URL map and the (now redundant but auditable) judge env var.

### Invariants enforced (mechanical, with tests)

1. **G0 — evidence-driven** (`SkillEpisode.finalize` rejects evidence-free
   non-action successes; `SkillLifecycleManager` rejects ACTIVE
   promotion when `expected_evidence_roles` is empty).
2. **No-memory** (`SkillEpisodeStep.__post_init__` rejects any action
   type starting with `QUERY_MEM` or `WRITE_MEM`).
3. **General-protocol** (`SkillLifecycleManager` rejects ACTIVE
   promotion when `feasible_domains < 2`; `GateService.Stage 0` flags
   it during evaluation).
4. **Bank-write isolation** (`SkillStore.put`/`remove` raise
   `StoreLockedError` unless called via the lifecycle manager).
5. **Gate-bound promotion** (`PromotionOrchestrator.promote` rejects
   `FAIL` verdicts and content-hash drift; refuses ACTIVE promotion on
   `LIMITED_PASS`).
6. **Crafter scope** (`SkillCrafterService` materialises new skills only
   as DRAFT records; never touches `active_store`).

## Not yet delivered (next sessions)

> Single source of truth for sprint sequencing:
> [`implementation_notes/pre-training-readiness-audit.md`](implementation_notes/pre-training-readiness-audit.md)
> §6 — five-sprint plan (S0–S4). The list below mirrors the
> still-open items there; flip an entry to "Delivered" *and* tick
> the audit row when you ship.

### S0 — pre-flight (✅ shipped 2026-05-01; outstanding = GPU-bound execution only)

- ☑ **T1.3a** — `SkillCrafterService(enable_protocol_patching=False)` default,
  threaded through `_crafter_hook → CoEvolutionConfig → run_coevolution.py
  --enable-protocol-patching`. Tests in `tests/test_crafter_lane_a_flag.py`.
- ☑ **T2.5** — `configs/{skill_gate,failure_routing}.yaml` shipped (policy
  v1.0.0; lane-(a) failure taxonomy with `lane_b_primary_mode` overrides;
  drift annotations).
- ☑ **T2.9** — `runs/sft_coldstart/sft_summary_all.json` regenerable via
  `python scripts/build_sft_manifest.py`.
- ☑ **T2.6** — This file refreshed to point at the Day-7→10 hooks, the
  `runs/sft_*` corpus, the lane-(a) flag, the threshold YAMLs, the SFT
  manifest, and the offline-promotion driver.
- ☑ **T1.3e + T1.3f + T3.6** — Lane-(a) banner blocks shipped on
  `harness/README.md` §22, `implementation_notes/legacy/crafter-harness-orchestrator-roles.md`
  §7.3 / §7.5 (with new "Post-decision rule" sub-section), and the four
  PLAN docs (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS,
  PLAN-COMPONENTS-IMPLEMENTATION). Each banner explicitly marks
  `hop_select` / `inner_mdp` references obsolete and points at
  `single-vs-two-mdp-tradeoff.md`.

Pre-flight execution results (2026-05-02):

- ☑ **T2.10** — `evaluation/smoke_load_sft_adapters.py` ran on all six adapters
  under the live vLLM tower (~40 GB free per GPU; cold-start 9B adapters on a
  single GPU, schema_gen 35B-A3B sharded across GPUs 0-3). Every adapter
  loaded, ran one forward pass, and produced finite logits on the expected
  shape. **No torn `*.partial_*` shards detected.** Report:
  `runs/sft_coldstart/_smoke/smoke_all.json`.
- ☒ **T1.1′** — `evaluation/probe_schema_gen_exact_match.py` ran end-to-end on
  `n=5` held-out gymv triples (1241 s wall-clock under CPU offload pressure).
  **Both §13 thresholds missed**: `exact_match=0/5`, `path_a=0/5`,
  `overall_field_acc=0.000`. Report: `runs/sft_coldstart/_probe/probe_schema_gen_n5.json`.
  **Diagnosed root cause:** LoRA target-modules drift (T2.11 — see
  `pre-training-readiness-audit.md` §0.3). The smoke loader warned about
  "missing adapter keys" on every adapter; quantitatively, schema_gen loaded
  only **8.36 M** LoRA parameters where the SFT-time delta should have been
  ~36 M (≈ 23 % loaded), and the cold-start 9B adapters loaded 36.96 M of an
  expected ~42 M (≈ 87 %). **T2.11 (NEW) blocks T1.1′ re-pass.**

T2.11 + T2.12 follow-ups (2026-05-02):

- ☑ **T2.11 — recipe fix landed.** New single-source-of-truth helper
  [`trainer/SFT/lora_targets.py`](trainer/SFT/lora_targets.py) defines the
  full Qwen3.5 hybrid-stack `target_modules` (12 entries: `q,k,v,o_proj` +
  `in_proj_{qkv,z,b,a}` + `out_proj` + `gate,up,down_proj`) and a
  `assert_lora_coverage(...)` post-`get_peft_model` sanity check that
  fail-fasts when any architecturally-required projection has zero
  wrapped layers. All three LoRA recipe sites now delegate to this
  helper: `trainer/SFT/config.py`, `trainer/SFT/schema_gen/config.py`
  (the hardcoded classic-7 default that caused the bug is removed),
  `trainer/coevolution/config.py::prepare_adapters`, and
  `configs/skillbank_lora.yaml`. The previously-claimed "skip the tiny
  `in_proj_a/b/z` gating projections" rationale was incorrect about
  `in_proj_z` (whose output dim is `value_dim`, not `num_v_heads`); it is
  the same size as `out_proj`, so dropping it cost ~13 % of cold-start
  coverage. New tests at [`tests/test_lora_targets.py`](tests/test_lora_targets.py)
  (7 tests, all passing) lock in the leg-presence invariants. **Open
  follow-up:** re-run the six SFT jobs against the corrected recipe;
  on-disk weights from the first run cannot be salvaged because the
  dropped legs were never trained → no on-disk tensor exists to
  rewrite into the new namespace (option (b) "key-rewriter" from §0.3
  is therefore ruled out).
- ☑ **T2.12 — SFT throughput uplift landed.** New helper
  [`trainer/SFT/speed_utils.py`](trainer/SFT/speed_utils.py) wires three
  drop-in upgrades into both train scripts:
  *(a)* `apply_liger_kernel(arch)` — picks the right
  `apply_liger_kernel_to_qwen3_5{,_moe}` patch and applies it before
  the model loads (fused CE / RMSNorm / RoPE; ~30–40 % uplift).
  *(b)* `pick_optim()` — defaults to `paged_adamw_8bit` when
  `bitsandbytes` is available (cuts optimizer memory ~4×), else
  `adamw_torch_fused`.
  *(c)* `enable_tf32()` — H200/Hopper-safe TF32 for residual fp32 ops.
  Both train scripts also now pass `group_by_length=True`,
  `dataloader_pin_memory=True`, `dataloader_num_workers=4`, and expose
  `--no_gradient_checkpointing` / `--use_liger_kernel` /
  `--no_liger_kernel` / `--optim` / `--strict_lora_coverage` /
  `--dataloader_workers` flags. **Still blocked on infra**: `flash-attn`
  and `mamba_ssm` / `causal_conv1d` (the GatedDeltaNet CUDA kernels) need
  a CUDA-13 host toolkit installed (current host is 12.8; torch was
  built against 13.0). The Python recurrence fallback for
  `linear_attention` layers is the single biggest remaining throughput
  hole — once the toolkit is installed, expect another 2–3× on the 35B-A3B
  schema_gen run.

### S1 — fire offline once (the §17 keystone) — ✅ shipped + executed 2026-05-02

- ☑ **T1.2** — `bash scripts/run_offline_promotion_cycle.sh` ran end-to-end
  on the latest cold-start corpus (17 pair banks under
  `labeling/skill_bank_out/run_20260430_030637`). 224 proposals across 17
  pairs, 186 PROMOTE decisions (limited_pass), 38 ROLLBACK. After the
  legacy-writeback step the §17 post-condition fires green:
  **375 / 489 entries flagged `_writeback_status ∈ {active, provisional,
  shadow}` across all 17 pairs** (writeback exit_code = 0,
  `bank.runnable() != []` → trainer launch unblocked). Wrapper bug fix
  (WritebackReport attribute access + correct post-condition field
  `skill._writeback_status`) shipped in this same pass. Artefacts under
  `labeling_supplement/promotion_decisions_out/run_offline_cycle_20260501_235935/`.

### S2 / S3 / S4 (later sprints)

- **S2 — Live + audit guards** ✅ **code items closed 2026-05-02.**
  - **T1.3b** — `RewriteProposal` + `MergeProposal` alias landed in
    `data_structure/extensions/bank_mutation_proposal.py`; gate
    `_run_static` accepts both (`source_type` bypass + `base_skill_id`
    lineage check).
  - **T1.3c** — `BANK_GAP` / `RETRIEVAL_MISLEAD` / `STALE_DESCRIPTION`
    in `common/enums.py` (`LANE_A_RECOVERY_STRATEGIES`); the live
    `_run_failure_dispatch` short-circuits the Repairer for those and
    routes to the Hypothesizer (legacy `INVARIANT_VIOLATION → HOP_INSERTION`
    backward-compat preserved).
  - **T1.3d** — `min_retrievals_per_skill` knob on `GateThresholds` +
    `configs/skill_gate.yaml`; `SkillLifecycleManager` enforces it at
    the `ACTIVE` transition.
  - **T2.2** — `orchestrator/eval_suite.py` (canonical home of
    `EvalSuite`, plus `EvalSuiteSpec` / `Scoreboard` / `EvalSuiteLoader`);
    `GateService.evaluate(eval_suite=)` consumes the loader output
    directly. Starter suite at `evaluation/suites/gymv-smoke-v1/`.
  - **T2.3** — `evaluation/{answer_evaluator, scoreboard, driver}.py`.
    `AnswerEvaluator` emits F1–F7 in spec priority; `ScoreboardAssembler`
    builds the canonical 10×10 table + 4 companion tables;
    `EvalDriver` writes per-instance JSONL + suite-level scoreboard JSON
    consumed by T2.2; `RunRelease` carries `eval_suite_id` +
    `scoreboard_path` (folded into `content_hash`).
  - **T2.4** — `RewardLogger.log_grpo_record(...)` is the single
    JSONL sink (kind discriminator: `grpo_step` vs. `skill_episode`).
    Wired through `episode_runner.run_episode_async` (both
    `action_taking` and `skill_selection` `GRPORecord` append sites)
    → `rollout_collector.collect_rollouts` →
    `orchestrator.run_training_loop_async`. `CoEvolutionConfig.reward_log_path`
    auto-resolves under the rewards dir when blank.
  - **T2.7** — `curator_weight` + `curator_warmup_steps` on
    `CoEvolutionConfig`; `set_curator_warmup(...)` called once per
    outer step from `run_training_loop_async`; `_dynamic_curator_reward`
    multiplies the base reward by the linear ramp.
  - **T2.8** — split-base vLLM topology documented in
    `implementation_notes/legacy/vllm-topology.md`.
  - **Outstanding for S2:** launch fast-loop GRPO on `gymv` only
    (Phase 1 of `PLAN-ACTION-AGENT.md` §6); enable
    `crafter_promotion_enabled=True`; add `policy_version` audit
    drift rows.
- **S3 — Curriculum + scale-out.** Multi-game GRPO; integrate the audit's
  curriculum-graduation thresholds.
- **S4 — Full eval + co-evolution.** Eval E0 / E1 / E2 drivers; cross-task
  transfer using the `evaluation_dataset/` pool + holdout manifests.

### Cross-cutting items not yet sprinted

- **P1 — Visual-grounding stabilisation** under `vlm_wrapper/grounding.py`.
- ~~**P2 — Eval E0 driver** (`evaluation/driver.py`,
  `evaluation/answer_evaluator.py`, slice/report scaffolds).~~ → **shipped
  2026-05-02 (T2.3).** See above.
- **Phase D / E / F** of `PLAN-COMPONENTS-IMPLEMENTATION.md`
  (training cadence, multi-domain rollout, full eval).
- **Phase-5/6 real-env binding (all 4 Tier 1 + Tier 2 + Tier 3 closed 2026-05-02)** -- the
  deterministic-stub tier of Phase-5/6 (see Delivered table) provides
  infrastructure validation only; mechanism validation now closed end-to-end:
  - **Tier 1**: 4 harness executors needed real-env binders -- all 4
    closed as of 2026-05-02 PM:
    * `harness/visual_reasoning` per-sample image loading -- **CLOSED**
      via `harness/_vr_per_sample_executor.py`'s
      `TaskAwareVisualReasoningExecutor` + dispatcher rewire in
      `labeling_supplement/_phase4_target_dispatch.py`.
    * `harness/video_executor.py` -- **CLOSED** via
      `harness/_video_per_sample_executor.py`'s
      `TaskAwareVideoReasoningExecutor` + `discover_task_to_video_meta`
      + dispatcher rewire (1000+ task->video_meta mappings discovered
      against `Cold-start-out-visual-reasoning-video/video_holmes/`).
      Verb-routing keeps both InnerAction and legacy video-domain
      verb sets exercising end-to-end.
    * `harness/osworld_executor.py` (real `pyautogui`) -- **CLOSED**
      via `harness/_osworld_per_sample_executor.py`'s
      `TaskAwareOsworldExecutor` + `discover_task_to_osworld_meta` +
      `harness/_executor_helpers/osworld_client.py`'s `OsworldClient`
      and `OsworldContainerPool` (HTTP client over the
      `happysixd/osworld-docker` Flask server). Smoke-verified
      end-to-end: `pyautogui.click(x=100, y=100, button='left',
      clicks=1)` actually executed in container `recursing_wilson`.
      516 task->meta entries discovered across 14 OSWorld domains
      (chrome, vlc, gimp, libreoffice_*, ...). 13-container fleet
      preloaded.
    * `harness/browsergym_executor.py` (real BrowserGym/Playwright)
      -- **CLOSED** via `harness/_browser_per_sample_executor.py`'s
      `TaskAwareBrowserExecutor` + `discover_task_to_browser_meta` +
      `harness/_executor_helpers/browser_helper.py` (JSON-RPC
      subprocess hosting `gym.make("browsergym/<task>")` in the
      `browsergym` conda env). Smoke-verified end-to-end against
      `Cold-start-out-browsergym/miniwob.email-inbox-star-reply/`:
      real `click("47")` returned `terminated=True` (task completed)
      in 13.4s including helper boot. 125 unique miniwob tasks
      discovered.

    **Retraction 2026-05-02:** A prior revision of this section
    framed items 3-4 as "infra-blocked, deferred -- needs an OSWorld
    VM in CI / Playwright in CI". That was wrong: the workspace
    already shipped dedicated `osworld` and `browsergym` conda envs
    with all dependencies, the upstream OSWorld + BrowserGym sources
    (editable installs), `Xvfb` + `xvfb-run` on PATH, 13
    pre-warmed `happysixd/osworld-docker` containers running for
    >35h, and the WebArena Docker stack. The actual gating
    constraint was code-side wiring, not infra. With per-sample
    executors + helper plumbing now landed, both items shipped
    without a CI sandbox change.

    See
    [`implementation_notes/legacy/phase5-cross-domain-measurement.md`](implementation_notes/legacy/phase5-cross-domain-measurement.md)
    §12.1.
  - **Tier 2**: ~~author `vlm_wrapper/video_adapter.py` and
    `vlm_wrapper/visual_reasoning_adapter.py` from scratch~~ -- **CLOSED
    2026-05-02.** Both shims ship as ~25-LOC re-exports over
    `visual_reasoning_wrapper.{skill_executor, video_skill_executor}`;
    original ~600-800-LOC-per-adapter estimate was ~10x off because the
    heavy machinery (registries, OmniParser-v2, Florence-2, video
    decode, cross-frame analysis) already shipped under
    `visual_reasoning_wrapper/`. See §12.2.
  - **Tier 3**: ~~design and ship per-domain runtime
    predicate-translators~~ -- **CLOSED 2026-05-02.** Shipped as
    `harness/predicate_translator.py` (~250 LOC) + 28 unit tests in
    `tests/test_predicate_translator.py` + dispatcher wiring across
    `_phase4_target_dispatch._build_{visual_reasoning,video,osworld,browser}_target`.
    `PREDICATE_TRANSLATIONS` table covers (gymv, *) for all 4
    cross-domain targets with mappings validated against
    `TARGET_PREDICATE_VOCAB` so translation actually unblocks cells
    rather than just shifting the static-vocab miss. See §12.3 and the
    sibling memo `implementation_notes/cross-domain-transfer-suite-rollout.md`
    §11.5.0.
  - Closing measurement: re-run Stage 6 NxN driver
    (`labeling_supplement/_phase4_transfer_matrix.py`) on the now-fully-wired
    pipeline; expect G6 to pass and §11.5.4's 15-35% / 15-30% bands to
    become measured admit rates rather than projections across all 25 cells.
  - **Co-evolution-loop integration (UPDATED 2026-05-02 PM —
    Layers C / A / D LANDED)**:
    The four-layer plan in
    [`implementation_notes/coevolution-cross-domain-integration.md`](implementation_notes/coevolution-cross-domain-integration.md)
    is partially shipped. Today (2026-05-02 PM) Layers C, A, D were
    implemented and tested:
    - **Layer C (commit `bc07599`)** — predicate translator splice
      into `SkillHarnessHook.filter_candidates`. Cross-domain skill
      contracts get their `effects_{add,del}` rebound through
      `harness.predicate_translator.translate_skill_contract` before
      the eligibility filter — and the LLM — sees them. New
      diagnostic counters `n_predicate_translations_applied/_failed`
      ride on `HarnessStepStats.to_json()` for the wandb sink.
      `tests/test_trainer_harness_hook.py` (5 new + 21 prior, 26
      total).
    - **Layer A (commit `10b23b3`)** —
      `trainer/coevolution/_transfer_hook.py` (798 LOC) +
      `crafter_transfer_*` config knobs +
      `configs/failure_routing.yaml` `cross_domain_taxonomy:` block
      + orchestrator wire. Re-evaluates each just-promoted skill's
      cross-domain admit rate via subprocess to
      `_phase4_transfer_matrix.py`; rolls back promotions failing
      `crafter_transfer_admit_band[0]` on every target by atomic
      JSONL row drop. `tests/test_transfer_hook.py` (28 tests).
    - **Layer D (commit `bf83fec`)** —
      `trainer/coevolution/_dashboard_hook.py` (649 LOC) +
      `crafter_dashboard_*` config knobs + orchestrator
      end-of-step wire. Periodic Stage-6 N×N matrix sweep on a bank
      snapshot; emits G1-G5 acceptance gates + per-cluster admit
      rates as wandb / TB scalars under the `cross_domain/...`
      namespace. `tests/test_dashboard_hook.py` (39 tests).

    All three CAD layers are off by default; flip
    `crafter_transfer_gate_enabled` / `crafter_dashboard_enabled` to
    opt in. Combined CAD test suite: 93 tests, 0 skipped, ~1.5s
    wall-clock under pytest.

    **Layer B** (cross-domain admit-rate as a GRPO reward channel)
    remains DESIGN-only — out of CAD scope per the user's
    instruction; deferred to a follow-up session.
- **Actor rewire**: replace `decision_agents.skill_interface
  .SkillBankProvider` with a `HarnessSkillProvider` that wraps
  `SkillHarness.select_eligible_skills`. (Note: the Day-10
  `SkillHarnessHook` already covers this for the trainer's
  co-evolution loop, but the `decision_agents` library API has not
  yet been ported.)
- **Legacy bridge**: one-way migration of `skill_agents/skill_bank`
  Stage-3 records into the new `SkillRecord` (planned in
  `skill_bank/legacy_bridge.py`).
- **Live frozen-teacher inference for Qwen3-VL** — the `SkillCrafterService`
  surface for swapping to `Qwen/Qwen3-VL-{32B,235B-A22B}` is wired
  (`with_qwen3_vl_teacher`, `from_env`, `set_teacher_model`), but the
  actual model invocation path through `API_func.ask_model` for these
  IDs still needs provider-side routing (vLLM / HF endpoint).

## How to run the tests

```bash
cd Multi-hop-Reasoning-VLM-Agent
python -m pytest tests/ -v
```

Expected as of S0 close: `433 passed, 1 deselected` (the deselected test is
the pre-existing `test_schema_predicates::test_extra_whitespace_tolerated`
which is unrelated to the readiness audit and tracked separately).

## Module dependency graph

```
common/                  # leaf
  ↑
data_structure/extensions/   # leaf (uses common)
  ↑
skill_bank/             # uses common + extensions
  ↑                ↑
harness/             |  # uses common + extensions
  ↑                ↑
orchestrator/        |  # uses common + extensions + skill_bank + harness
  ↑
crafter/                # uses common + extensions + skill_bank + orchestrator (only ArtifactStore)
```

The crafter never imports `harness` (it operates on traces, not live
adapters) and never imports `skill_bank.stores` (only the lifecycle
manager). The harness never imports `orchestrator` or `crafter`.
