# Algorithms 1 & 2

> **Reviewer ask**: "the system has too many moving parts to follow
> from prose; please add formal algorithm boxes."

Below are the two pseudocode boxes the paper currently lacks.  Both
match the implementation 1:1 (file references on the right).

## Algorithm 1 — SkillBridge inner loop (one episode)

```
Algorithm 1: SkillBridge Episode Step                          [ref]
Input : env state s_t, intention z_{t-1}, bank B,
        adapters {A_action, A_select, A_int}, harness H
Output: action a_t, intention z_t, audit row r_t

 1: z_t ← UPDATE_INTENTION(s_t, z_{t-1}, A_int)               # _intention_update
 2: K   ← bank-cap from CoEvolutionConfig.actor_bank_cap_k
 3: C   ← B.select(s_t, z_t, top_K=K)                         # query.SkillQueryEngine.select
 4: C'  ← H.filter_candidates(C)                              # _harness_hook.filter_candidates
                                                              #   logs harness_log/rejections.jsonl
 5: u_t ← skill_selection(s_t, z_t, C', A_select)             # qwen3_decision_agent.pick_skill
 6: ok, diag ← H.validate_invocation(u_t)                     # logs harness_log/validate.jsonl
 7: if not ok then
 8:     u_t ← null   ;   record veto in r_t
 9: a_t ← action_taking(s_t, z_t, u_t, A_action)              # qwen3_decision_agent.pick_action
10: r_t ← {step, episode_id, z_t, u_t, ok, …}
11: return a_t, z_t, r_t
```

Wired by `trainer.coevolution.episode_runner.run_episode_async`.

## Algorithm 2 — SkillBridge bank update (one trainer step)

```
Algorithm 2: SkillBridge Bank Update                                  [ref]
Input : trace τ_n, bank B_n, harness H_n
Output: bank B_{n+1}, harness H_{n+1}

 1: F   ← extractor(τ_n)                                              # _extractor / segmenter
 2: P_d ← crafter.deterministic(F)                                    # _crafter_hook (rule-based)
 3: P_l ← crafter.llm(F, B_n)        if config.crafter_enabled        # _llm_crafter
 4: P   ← P_d ∪ P_l
 5: for s in P do                                                     # _promotion_hook
 6:     verdict ← gate(s) using {gated, permissive}                   # decide_promotion_gpt54.main
 7:     if verdict.PASS then B'.add(s) at status DRAFT/PROVISIONAL
 8: B_{n+1} ← lifecycle.advance(B')                                   # skill_bank.lifecycle
                                                                      #   logs lifecycle_log/transitions.jsonl
 9: H_{n+1} ← H_n.bind(B_{n+1})                                       # SkillHarness.rebind
10: return B_{n+1}, H_{n+1}
```

Wired by `trainer.coevolution.orchestrator._run_trainer_step` end-of-step block.

## Notation crib sheet

| symbol | meaning | source |
| ------ | ------- | ------ |
| `s_t`  | env state at step t | `env_wrappers.subprocess_env` |
| `z_t`  | intention at step t (a free-form short string with optional tag prefix) | `intention_log/switches.jsonl` |
| `u_t`  | chosen skill at step t (or null) | `harness_log/validate.jsonl` |
| `B_n`  | skill bank at trainer step n | `skill_bank.SkillBankMVP` |
| `H_n`  | runtime harness layer (eligibility filter + validator) at step n | `harness/SkillHarness` |
| `τ_n`  | per-step trace = `{(s_t, z_t, u_t, a_t, r_t)}_t` | `reward_log.jsonl` |
| `F`    | "facts" — per-trace structural features fed to the crafter | `_extractor` |
