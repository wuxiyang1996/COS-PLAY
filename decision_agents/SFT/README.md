# Decision Agent — SFT data-collection flavour

GPT-4o-driven Actor that gathers per-step labels for the cold-start SFT
pipeline in [`trainer/SFT/`](../../trainer/SFT/). This is the **teacher**
side of the distillation loop laid out in
[`vlm_wrapper/README.md`](../../vlm_wrapper/README.md) — the records
this collector produces are exactly what the Qwen3-VL-8B student in
[`decision_agents/grpo/`](../grpo/) is fine-tuned and then GRPO-trained
on.

---

## What it ships

| File | Role |
|------|------|
| [`actor_gpt4o.py`](actor_gpt4o.py) | `GPT4oCollectorActor` — subclass of `ActorAgent` that overrides `_call_llm` to send GPT-4o multimodal chat completions and writes a per-step record after each `step()`. |
| [`sft_recorder.py`](sft_recorder.py) | `SFTRecorder`, `SFTRecord` — append-only JSONL writer producing the **exact** layout `trainer/SFT/data_loader.py` already understands: `<out>/<game>/{skill_selection,action_taking}.jsonl`. |
| [`run_collect.py`](run_collect.py) | CLI entrypoint that drives the collector against a Gym-like env factory. |

Everything else the actor needs — schema parsing, skill interface,
skill tracker, inner-MDP, reward shaping — is reused from
[`decision_agents/`](..) directly. There is **no fork** of the per-step
pipeline; only the LLM seam and a recorder are added.

---

## Output format

Per-step rows match `trainer/SFT/data_loader._normalise_example` and
`_align_action_taking_to_coevolution` field by field:

```json
{
  "prompt":      "<full action-prompt the actor saw, incl. valid actions>",
  "completion":  "SUBGOAL: [TAG] ...\nREASONING: ...\nACTION: 3",
  "intention":   "[NAVIGATE] reach the staircase",
  "active_skill": "skill_navigate_to_target_v2",
  "image": {
      "path":      "rollouts/.../step_0007.png",
      "mime_type": "image/png",
      "width": 1280,
      "height": 720
  },
  "parse_path":   "llm:exact",
  "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]", "no-op"]
}
```

The first three fields drive the existing cold-start trainer; the
`image` block is silently passed through (the loader's
`row.get("intention", "")` pattern ignores unknown keys), so the
rollout artefact stays forward-compatible with the multimodal Stage B
training pipeline described in `vlm_wrapper/README.md`.

Files land at:

```
<out_dir>/
  tetris/
    skill_selection.jsonl
    action_taking.jsonl
  super_mario/
    ...
  _manifest.json          # row counts per (game, adapter)
```

The default `<out_dir>` is
`labeling/output/gpt54_skill_labeled/grpo_coldstart`, which is exactly
what `trainer.SFT.config.SFTConfig.decision_data_dir` points at — no
trainer-side configuration changes needed.

---

## Quick start

### Programmatic

```python
from decision_agents.SFT import GPT4oCollectorActor, SFTRecorder
from decision_agents.core import VisualInput

recorder = SFTRecorder(output_dir="labeling/output/gpt54_skill_labeled/grpo_coldstart")

actor = GPT4oCollectorActor(
    recorder=recorder,
    game="tetris",
    model="gpt-4o",
)

obs, info = env.reset()
while not done:
    decision = actor.step(
        observation=str(obs),
        schema_text=info.get("schema_text"),
        valid_actions=info.get("valid_actions"),
        task="Clear lines as fast as possible.",
        images=[VisualInput(image_path=info["screenshot"])],   # optional
    )
    obs, reward, term, trunc, info = env.step(decision.action)
    actor.observe_result(decision, reward=reward, done=(term or trunc))
recorder.write_manifest()
```

### CLI

```bash
python -m decision_agents.SFT.run_collect \
    --env-factory my_envs.tetris:make_env \
    --game tetris \
    --episodes 50 \
    --max-steps 200 \
    --out labeling/output/gpt54_skill_labeled/grpo_coldstart \
    --image-info-key screenshot \
    --schema-info-key schema_text
```

---

## How GPT-4o is reached

`_call_llm` routes through one of two paths:

1. **Vision** — when at least one `VisualInput` is staged for the step,
   it builds OpenAI chat-completion content parts (`[text, image_url, ...]`)
   and calls `openai.OpenAI(...).chat.completions.create(...)`. The
   client is built lazily, preferring OpenRouter when
   `open_router_api_key` is set in `API_func` (matching the project's
   default routing).
2. **Text fallback** — when no images are staged, it uses
   `API_func.ask_gpt(prompt, model="gpt-4o", ...)` so legacy callers
   keep working.

Both paths preserve the `_extract_action_from_reply` decoding stack
that `ActorAgent._pick_action` already runs (exact → numbered →
entity-ref → edit-distance → token-overlap → loose), so the
`completion` field stored in the SFT row is always the raw GPT-4o
reply, not a normalised one. That preserves the supervision signal the
trainer expects.

---

## Why a separate flavour?

The plan in `plans/02-action-agent/PLAN-ACTION-AGENT.md` distinguishes
between (a) the **online, trainable** Actor (Qwen/Qwen3.5-9B + GRPO LoRA, see
`decision_agents/grpo/`) and (b) an **offline, frozen** teacher used
only for label generation. GPT-4o fills role (b) for now:

- It already lives behind `API_func.ask_gpt` with vision support.
- Its labels feed `trainer/SFT` directly without format conversion.
- It is *never* deployed at inference; the Qwen3-VL-8B student in
  `decision_agents/grpo/` takes over after Stage A SFT + GRPO.

Once Qwen3-VL-235B-A22B is wired in (per `vlm_wrapper/README.md`'s
"two-stage SFT pipeline" table), this collector will keep its
interface — only the model id and the OpenAI client setup will swap.
