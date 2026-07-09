# 01-visual-grounding — Stage 1: Visual Grounding

Pixels → structured `<state>` schema. The first stage of the canonical pipeline. Everything downstream consumes the schema produced here.

## Status (repo snapshot — 2026-05-02)

**Shipped:** Phase-1 `schema_gen` SFT weights on disk (`runs/sft_schema_gen/`); probe + smoke drivers (`evaluation/{probe_schema_gen_exact_match,smoke_load_sft_adapters}.py`); split-base vLLM governance — [`implementation_notes/legacy/vllm-topology.md`](../../implementation_notes/legacy/vllm-topology.md).  
**Open:** Re-run **six SFT jobs** after [`trainer/SFT/lora_targets.py`](../../trainer/SFT/lora_targets.py) (T2.11), then re-pass **T1.1′** — see [`implementation_notes/pre-training-readiness-audit.md`](../../implementation_notes/pre-training-readiness-audit.md) §0.3.

| Document | Purpose |
|----------|---------|
| [`PLAN-VISUAL-GROUNDING.md`](PLAN-VISUAL-GROUNDING.md) | Design of the VLM parser, canonical `<state>` schema, three grounding heads (heuristic / vision / OmniParser / tool loop), domain adapters (Gym-V, BrowserGym, OSWorld, video), schema completeness guarantee (§12), Qwen3-VL-8B training plan. |
| [`PLAN-VISUAL-GROUNDING-MILESTONES.md`](PLAN-VISUAL-GROUNDING-MILESTONES.md) | Concrete week-by-week build-out: 5-stage inference pipeline, routing policy (Path A / B / C), training phases 0–4, 7 ablations, success criteria. |
| [`PLAN-VISUAL-SKILLS.md`](PLAN-VISUAL-SKILLS.md) | *Optional* — transferable visual grounding **strategies** as skills (multi-step perception programs that compose tools). Sits between perception tools and reasoning skills; uses `belief / binding` effect contracts rather than world-effect contracts. |

Cross-cutting consumers: [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) (inner-MDP `GROUND`), [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) (contract learning), [Uncertainty Calibration](../08-cross-cutting/PLAN-UNCERTAINTY-CALIBRATION.md) (per-field uncertainty source).

Back to [plans/README.md](../README.md).
