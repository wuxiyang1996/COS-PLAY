# Decision Agent — GRPO + LoRA flavour (Qwen3-VL-8B-Instruct)

Online-policy Actor backed by **`Qwen/Qwen3-VL-8B-Instruct`** served
through vLLM with multi-LoRA hot-swap. This is the **student** that the
GPT-4o collector in [`decision_agents/SFT/`](../SFT/) feeds via
cold-start SFT, and that the GRPO loop in
[`trainer/coevolution/`](../../trainer/coevolution/) trains online.

---

## What it ships

| File | Role |
|------|------|
| [`actor_qwen_vl.py`](actor_qwen_vl.py) | `QwenVLActor` — subclass of `ActorAgent` that overrides `_call_llm` to issue async multimodal chat completions through `AsyncVLLMClient`, and adds `step_async` for runners on an event loop. |
| [`rollout_logger.py`](rollout_logger.py) | `GRPORolloutLogger` — emits `trainer.common.metrics.RolloutStep` rows the GRPO trainer ingests directly. |

Everything else — schema parsing, skill interface, skill tracker,
inner-MDP, reward shaping — is reused unchanged from
[`decision_agents/`](..). The actor body **is** the legacy
`ActorAgent`; only the LLM seam and the rollout-logger plumbing are
new.

---

## Wiring at a glance

```
                       Qwen/Qwen3-VL-8B-Instruct
                        (one vLLM instance)
                                ▲
                                │ chat.completions
                                │ + image_url parts
                                │ + adapter=action_taking
                                │
              AsyncVLLMClient ──┘
              (trainer/coevolution/vllm_client.py)
                                ▲
                                │
       ┌────────────────────────┴────────────────────────┐
       │  QwenVLActor.step()  (decision_agents/grpo/)    │
       │    └─ ActorAgent.step()  (parent class)         │
       │         ├─ parse <state> schema                 │
       │         ├─ infer_intention                      │
       │         ├─ SkillTracker.should_reselect         │
       │         ├─ harness.valid_actions(state)         │
       │         └─ _pick_action → _call_llm(...) ───────┘
       │
       │  observe_result(reward, done) ───┐
       │                                  ▼
       │                         GRPORolloutLogger
       │                            log_step(...)
       │                                  │
       │                      RolloutStep[r_env, r_follow,
       │                                  r_cost, r_total,
       │                                  action_type, ...]
       └──────────────────────────────────┬──────────────┐
                                          ▼              │
                                 RolloutRecord ──→ trainer.coevolution.grpo_training
```

---

## Quick start

### Sync runner

```python
from decision_agents.grpo import (
    QwenVLActor, GRPORolloutLogger, DEFAULT_QWEN_VL_MODEL,
)
from decision_agents.core import VisualInput
from trainer.coevolution.vllm_client import AsyncVLLMClient

vllm = AsyncVLLMClient(
    base_url="http://localhost:8000/v1",
    model=DEFAULT_QWEN_VL_MODEL,
    default_temperature=0.3,
    default_max_tokens=256,
)
logger = GRPORolloutLogger(env_name="tetris", game_name="tetris")

actor = QwenVLActor(
    vllm_client=vllm,
    rollout_logger=logger,
    adapter="action_taking",
)

obs, info = env.reset()
logger.start_episode(seed=42)
done = False
while not done:
    decision = actor.step(
        observation=str(obs),
        schema_text=info.get("schema_text"),
        valid_actions=info.get("valid_actions"),
        images=[VisualInput(image_path=info["screenshot"])],
    )
    obs, reward, term, trunc, info = env.step(decision.action)
    done = bool(term or trunc)
    actor.observe_result(decision, reward=reward, done=done)
record = logger.finalize_episode(score=info.get("score", 0.0), won=info.get("won", False))
```

### Async runner

```python
import asyncio

async def run():
    actor = QwenVLActor(vllm_client=vllm, rollout_logger=logger)
    logger.start_episode(seed=42)
    obs, info = env.reset(); done = False
    while not done:
        decision = await actor.step_async(
            observation=str(obs),
            schema_text=info.get("schema_text"),
            valid_actions=info.get("valid_actions"),
            images=[VisualInput(image_path=info["screenshot"])],
        )
        obs, reward, term, trunc, info = env.step(decision.action)
        done = bool(term or trunc)
        actor.observe_result(decision, reward=reward, done=done)
    return logger.finalize_episode(
        score=info.get("score", 0.0), won=info.get("won", False),
    )

record = asyncio.run(run())
```

---

## LoRA adapter routing

`QwenVLActor` always invokes the `action_taking` LoRA on the LLM call
(matches `trainer.SFT.config.DECISION_ADAPTERS` and
`trainer.coevolution.vllm_client.ADAPTER_MAP`). The **base model** is
used implicitly when:

* the SFT-trained adapter is missing on the vLLM server (the client
  auto-detects a 4xx and retries against the base — see
  `_is_adapter_missing` in `vllm_client.py`);
* the caller passes `adapter=None` for an ablation run.

Other LoRA adapters (`skill_selection`, `segment`, `contract`,
`curator`) are **not** owned by this module — `skill_selection` is hit
by the skill provider (`SkillBankProvider` calling vLLM separately),
and the SkillBank-side adapters live in `skill_agents/`. Keeps the
GRPO loop straightforward: one adapter per agent role.

---

## What the GRPO trainer reads

`GRPORolloutLogger.finalize_episode` returns a
`trainer.common.metrics.RolloutRecord` whose every field maps onto
something the `DecisionGRPOTrainer` already reads:

| Field | Source | Used for |
|-------|--------|----------|
| `r_env` | env reward | base policy gradient |
| `r_follow` | `RewardComputer` (skill-contract effect match) | skill-following bonus |
| `r_cost` | `RewardComputer` (query / call / switch costs) | per-step regularisation |
| `r_total` | sum | advantage |
| `action_type` (`primitive` / `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`) | `ActorDecision` flags | per-action-type rate metrics |
| `active_skill_id` | `SkillTracker.active_skill_id` | skill-switch rate |
| `query_key` | actor intention/summary | mean query-key length |
| `intentions` | `infer_intention` output | strategy-C ablations |

So the **only** thing a GRPO trainer needs to do to consume these
records is what it already does for the legacy text-only Actor — no
schema changes.

---

## Why a separate flavour?

The plan in `plans/02-action-agent/PLAN-ACTION-AGENT.md` distinguishes
between (a) the **online, trainable** Actor (this sub-package) and (b)
an **offline, frozen** teacher used for label generation
([`decision_agents/SFT/`](../SFT/)). Splitting them in two folders:

- keeps the vLLM / async / LoRA machinery off the GPT-4o code path
  (and vice versa), so a developer touching the SFT collector never
  has to reason about event loops;
- makes it impossible to accidentally ship the GPT-4o teacher into a
  GRPO rollout (the trainer ingests only `RolloutRecord`s emitted by
  the GRPO logger);
- cleanly mirrors the two-stage pipeline laid out in
  `vlm_wrapper/README.md`: GPT-4o → SFT → Qwen3-VL-8B → GRPO.
