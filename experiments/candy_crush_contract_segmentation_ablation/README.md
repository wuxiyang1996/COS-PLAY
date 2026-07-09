# Candy Crush Contract and Segmentation Ablations

This folder defines the Candy Crush rebuttal ablations for isolating
which parts of the SkillBridge skill-bank pipeline are necessary.

The experiments are intentionally single-game first (`candy_crush`) so
the comparison answers mechanism questions without cross-game transfer
confounds.

## Core Questions

1. Is the predicate-level effect contract necessary, or is the natural
   language skill protocol enough?
2. Is the contract mechanism doing more than counting start/end
   predicate deltas?
3. Is rollout-to-skill segmentation a core contribution, or can simple
   heuristic cuts recover the same benefit in the weak SkillRL setting?

## Required Ablations

### A1: w/o effect contract

Remove the predicate-level effect contract.

Keep:
- Natural-language skill protocol.
- Skill discovery / skill text exposed to the actor.
- Normal rollout collection on Candy Crush.

Remove:
- Predicate-level `effects_add` / `effects_del` contracts.
- Contract matching in harness eligibility.
- Contract completion reward.
- Contract-GRPO reward channel.

Question answered:

> Is the contract necessary, or is a natural-language skill protocol
> sufficient?

Expected interpretation:
- If performance and skill reuse remain close to the full system, the
  contract may be mostly decorative for Candy Crush.
- If skill admission becomes noisy, reuse drops, or actor reward falls,
  the contract is carrying nontrivial grounding / verification signal.

### A2: raw delta contract

Use the single start/end predicate delta of each segment as the contract.

Keep:
- Predicate representation.
- Natural-language skill protocol.
- Contract-shaped fields in the skill record.

Remove:
- Multi-instance consensus.
- Frequency thresholds across segment instances.
- Contract verification / refinement.
- Contract completion reward derived from verified effects.

Question answered:

> Is the contract only predicate counting, or does consensus +
> verification add value?

Expected interpretation:
- If raw deltas match the full system, then multi-instance contract
  learning may be over-engineered for Candy Crush.
- If raw deltas increase false admits or unstable skills, the full
  contract learner is doing more than counting predicates.

## Recommended Ablation

### A3: heuristic-only segmentation

Use only heuristic boundaries. Do not use learned segmentation,
preference ranking, or DP / beam segment decoding.

Keep:
- Same downstream contract learner as the full system unless paired with
  A1 or A2.
- Same Candy Crush rollout budget and actor model.

Remove:
- Learned segment scorer.
- Segment decoding over candidate boundaries.
- LLM / preference-teacher segment ranking.

Question answered:

> Is rollout-to-skill segmentation a core contribution, and how is this
> different from the weak SkillRL setting?

Expected interpretation:
- If heuristic-only segmentation is close to full, segmentation is not
  the main driver on Candy Crush.
- If heuristic-only segmentation produces lower-quality skills or worse
  actor reward, the learned rollout-to-skill layer is a necessary bridge
  from weak rollouts to reusable skills.

## Recommended Table

| Variant | Effect contract | Contract reward | Contract matching | Segmentation | Candy Crush reward | Skill reuse | False admit rate | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | consensus + verified | on | on | learned decode | TBD | TBD | TBD | main system |
| w/o effect contract | off | off | off | learned decode | TBD | TBD | TBD | answers contract necessity |
| raw delta contract | single start/end delta | off | raw only | learned decode | TBD | TBD | TBD | answers predicate-counting concern |
| heuristic-only segmentation | consensus + verified | on | on | heuristic only | TBD | TBD | TBD | answers segmentation contribution |

## Metrics to Report

- `mean_reward`: final Candy Crush rollout reward.
- `area_under_reward_curve`: sample-efficiency summary.
- `n_skills_active`: final active skill count.
- `skill_reuse_rate`: fraction of actor steps that select a bank skill.
- `harness_admit_rate`: admitted candidates / retrieved candidates.
- `false_admit_rate`: admitted skills whose expected effect does not
  verify on the post-state.
- `contract_pass_rate`: verification pass rate for contract-bearing
  skills.
- `segments_per_episode`: segmentation granularity diagnostic.
- `mean_segment_length`: segmentation granularity diagnostic.

## Files

- `experiment_matrix.yaml`: canonical experiment definitions.
- `run_candy_crush_ablation.sh`: launcher wrapper and CLI contract.
- `RESULTS_TEMPLATE.md`: table template for recording runs.

