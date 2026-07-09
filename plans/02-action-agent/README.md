# 02-action-agent — Stage 2: Action Agent

**Runtime posture (2026-05):** **single-MDP** actor — two GRPO LoRAs (`skill_selection`, `action_taking`) only; no `hop_select` / `inner_mdp.py` ([`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](../../implementation_notes/legacy/single-vs-two-mdp-tradeoff.md)).  
This folder's `PLAN-ACTION-AGENT.md` still documents historical **two-level / inner-hop** design for background; lane-(a) makes retrieved skills **context**, not executed protocols ([`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md)).

Consumes the structured `<state>` from [Stage 1](../01-visual-grounding/) and selects environment actions guided by skills retrieved from [Stage 3](../03-skill-bank/).

| Document | Purpose |
|----------|---------|
| [`PLAN-ACTION-AGENT.md`](PLAN-ACTION-AGENT.md) | Historical **two-level MDP** narrative + inner-action alphabet (still useful for offline diagnostics / HopTrace thinking). **Live stack:** single-MDP + lane-(a) skills — see banners at top of PLAN + legacy memos linked above. |

## Status (repo snapshot — 2026-05-02)

**Shipped:** `ActorAgent` single-step loop; Day-10 `SkillHarnessHook`; cold-start SFT seeds under `runs/sft_coldstart/decision/`.  
**Open:** Run **fast-loop GRPO on `gymv`** (IMPLEMENTATION-STATUS §S2 outstanding); reconcile PLAN prose with shipped MDP when editing — audit §7–§8.

Cross-cutting consumers: [Harness](../05-harness/PLAN-HARNESS.md) (the Actor's online policy that consumes Harness-filtered `eligible_skills`), [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (`InnerHopRecord` / `ActionRecord` artifacts), [First Eval Target](../00-system/PLAN-EVAL-FIRST-TARGET.md) (`reasoning_trace_summary`, Avg Hops).

Back to [plans/README.md](../README.md).
