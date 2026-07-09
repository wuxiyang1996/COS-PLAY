# Fitting Intention-Based Skill Extraction into SkillBankAgent

## The Root Problem

There is a **type mismatch** between what the intention-extracted contracts expect
and what the predicate system actually produces at runtime.

Contracts (from intention-based extraction in `label_and_extract_skills_gpt54.py`) store:

```
eff_add = {"clear_completed"}
eff_event = {"tag_clear"}
```

But `_extract_predicates` in `infer_segmentation/episode_adapter.py` returns:

```python
{"intention": "[CLEAR] Push box onto target", "done": False}
```

So `_effects_compat_score` (in `skill_agents/skill_bank/bank.py`) can never match
`"clear_completed"` against the predicate dict — every effect clause hits
`missing_penalty = -0.5`, and the contracts are dead weight.

---

## Strategy A: Seed the Bank (Warm Start) — ~20 lines

**Idea:** Load the 48 intention-extracted skills into the bank, then patch the
consumer side so Stage 2 can actually score segments against them.

### What exists today

- `label_and_extract_skills_gpt54.py` already calls `agent.bank.add_or_update(contract)`
  for the fallback skills. The 48 contracts sit in `skill_bank.jsonl`.
- Stage 2 defaults `skill_names = list(self.bank.skill_ids)` when the bank is
  non-empty, so seeded skills would automatically enter the decode label set.

### What needs to change

Patch `_extract_predicates` in `skill_agents/infer_segmentation/episode_adapter.py`
to parse the `[TAG]` from intentions into keys the contracts understand:

```python
# Before (raw string, useless for matching):
preds["intention"] = exp.intentions

# After (keys that match contract effects):
import re
tag_m = re.match(r"\[(\w+)\]", (exp.intentions or "").strip())
if tag_m:
    tag = tag_m.group(1).lower()
    preds[f"tag_{tag}"] = 1.0
    preds[f"{tag}_completed"] = float(getattr(exp, "done", False))
```

### Scope & limitations

- Only fixes Stage 2 scoring. Stage 1 boundary proposals still rely on embedding
  changepoints + reward spikes; the tag is just one more signal.
- Stage 3 contract learning still sees sparse predicate dicts.
- Only works on LLM-labeled episodes (those with `intentions`).

---

## Strategy B: Replace the Predicate Extractor with Intention Tags — ~80 lines

**Idea:** The `[TAG]` in intentions is effectively a high-quality, LLM-generated
predicate. Wire it directly into the predicate system via a new
`SignalExtractorBase` subclass.

### Key abstraction (already exists)

```
skill_agents/boundary_proposal/signal_extractors.py

class SignalExtractorBase(ABC):
    extract_predicates(experiences) -> List[Optional[dict]]
    extract_event_times(experiences) -> List[int]
```

Factory: `get_signal_extractor(env_name, **kwargs)` — supports `"overcooked"`,
`"avalon"`, `"diplomacy"`, `"generic"`, `"llm"`, `"llm+overcooked"`, etc.

### What to build

```python
class IntentionSignalExtractor(SignalExtractorBase):
    def extract_predicates(self, experiences):
        predicates = []
        for exp in experiences:
            intent = getattr(exp, "intentions", "") or ""
            m = re.match(r"\[(\w+)\]", intent.strip())
            tag = m.group(1).upper() if m else "UNKNOWN"
            preds = {f"tag_{t.lower()}": float(t == tag) for t in SUBGOAL_TAGS}
            predicates.append(preds)
        return predicates

    def extract_event_times(self, experiences):
        events = []
        for t, exp in enumerate(experiences):
            if getattr(exp, "done", False):
                events.append(t)
        return events
```

Register as `"intention"` in `get_signal_extractor`.

### What improves (vs Strategy A)

| Stage | Improvement |
|-------|-------------|
| Stage 1 (boundary proposal) | Tag changes become explicit boundary candidates (not just embedding changepoints) |
| Stage 2 (segmentation) | `{tag_clear: 1.0, tag_attack: 0.0, ...}` → contracts have real compat scores |
| Stage 3 (contract learning) | Can learn/verify patterns like "skill_X starts with tag_setup=1, ends with tag_clear=1" |
| Materialization | Effect signatures are meaningful → clustering works |

### Limitations

- Only works on episodes that have `intentions` (LLM-labeled data).
- Raw rollouts from the coevolution loop (`RolloutStep`) don't have intentions.

---

## Strategy C: Full Closed-Loop (Longer Term)

**Idea:** Use A or B as Gen 0, then iterate the coevolution loop that already
exists in `trainer/launch_coevolution.py` and `trainer/skillbank/em_trainer.py`.

### Existing loop structure

```
launch_coevolution.py:
  while episode_count < total_episodes:
    1. collect_batch (with current bank)
    2. grpo_trainer.train_step
    3. periodic eval
    4. em_trainer.run(trajectories)   ← Stage 0→4 on real rollouts
       → commit or rollback bank

em_trainer.py EM loop:
    1. enrich_trajectory_predicates (Stage 0)
    2. For each iteration:
       a. propose_cuts        (Stage 1)
       b. decode_batch         (Stage 2)
       c. learn_contracts      (Stage 3)
       d. run_update           (Stage 4: refine, materialize, merge, split)
    3. SkillEval gating on holdout
    4. Commit or rollback
```

### The gap

The trainer EM path (`stage0_predicates.py`) uses
`extract_predicates_from_text(observation_text, predicate_vocabulary)` — keyword/regex
matching on raw observations. It does **not** have access to intentions because
`RolloutStep` doesn't store them.

### What needs to change for C

Option 1: Add `intentions` to `RolloutStep`
- Populate from the decision agent's `get_intention` tool during rollout collection
- Extend `ingest_rollouts` and `enrich_trajectory_predicates` to map intentions → predicates

Option 2: Lightweight intention tagger
- Small LLM call or classifier in `stage0_predicates` that produces `[TAG]` from raw obs
- Avoids changing the rollout collection path

### Generation flow

```
Gen 0: Intention-based extraction → 48 skills (what we have now)
    ↓ seed bank
Gen 1: Run new episodes with skill-conditioned policy
    ↓ SkillBankAgent segments with seeded bank
    ↓ Stage 3 refines contracts from real execution data
    ↓ Bank maintenance: split/merge/refine
Gen 2+: Repeat with growing bank
```

---

## Comparison

| | A: Seed | B: Replace Extractor | C: Full Loop |
|---|---------|---------------------|--------------|
| **What adapts** | Consumer (`_effects_compat_score` or `_extract_predicates`) | Producer (new `SignalExtractorBase`) | Infrastructure (`RolloutStep` + EM pipeline) |
| **Where the fix lives** | `episode_adapter.py` (one function) | `signal_extractors.py` (new class) + factory | `rollout_collector.py` + `ingest_rollouts.py` + `stage0_predicates.py` |
| **Stages improved** | Stage 2 only | Stage 1 + 2 + 3 + materialization | All stages, iteratively |
| **Works on** | LLM-labeled episodes | LLM-labeled episodes | Live rollouts (needs intentions in rollout path) |
| **Effort** | ~20 lines | ~80 lines + wiring | Moderate refactor across trainer path |

---

## Recommended Path

**Start with B**, then build toward C:

1. **B now** — lights up the full pipeline for labeled data. Not much more work
   than A, but Stage 1 boundaries become much cleaner (tag changes are explicit
   boundary proposals instead of just embedding changepoints).

2. **C next** — once B validates that intention-based predicates make the pipeline
   work end-to-end, plumb intentions into the rollout collection path so the
   coevolution loop can self-improve the skill bank on live data.

---

## Key Files

| File | Role |
|------|------|
| `labeling/label_and_extract_skills_gpt54.py` | Intention-based extraction + fallback segmentation (GPT-5.4) |
| `scripts/extract_skills_qwen.py` | Qwen3-8B skill extraction via vLLM (Strategy B) |
| `skill_agents/infer_segmentation/episode_adapter.py` | `_extract_predicates` (Stage 2 predicate producer) |
| `skill_agents/skill_bank/bank.py` | `_effects_compat_score` (contract matching) |
| `skill_agents/boundary_proposal/signal_extractors.py` | `SignalExtractorBase` + factory + `IntentionSignalExtractor` |
| `skill_agents/pipeline.py` | `SkillBankAgent` orchestrator |
| `trainer/launch_coevolution.py` | Coevolution loop |
| `trainer/skillbank/em_trainer.py` | EM pipeline (Stage 0→4) |
| `trainer/skillbank/stages/stage0_predicates.py` | Predicate enrichment for trainer path |

---

## Implementation Status

### Strategy B: DONE

Implemented across these files:

| File | Change |
|------|--------|
| `skill_agents/boundary_proposal/signal_extractors.py` | Added `IntentionSignalExtractor`, `parse_intention_tag()`, registered `"intention"` and `"intention+<env>"` in factory |
| `skill_agents/boundary_proposal/__init__.py` | Exported `IntentionSignalExtractor`, `parse_intention_tag` |
| `skill_agents/infer_segmentation/episode_adapter.py` | Patched `_extract_predicates` to decompose `[TAG]` → `tag_<tag>` + `<tag>_completed` keys |
| `skill_agents/pipeline.py` | Added `max_concurrent_llm_calls` to `PipelineConfig` (for local GPU inference) |
| `API_func.py` | Added `_strip_think_tags` in `ask_vllm` (Qwen3 `<think>` block removal) |
| `scripts/extract_skills_qwen.py` | Standalone Qwen3-8B skill extraction script using Strategy B |

Verified: `_effects_compat_score` now scores +1.0 for matching intention contracts
(was stuck at -0.5 before). Tag transitions are explicit boundary proposals.

### Strategy C: Training Integration — NOT YET DONE

Audit completed. Here is what needs to change and where:

**Architecture context:**
- The training co-evolution loop uses `EMTrainer` (Hard-EM), NOT `SkillBankAgent`.
- `EMTrainer` uses `MultiLoraSkillBankLLM` (local HF, Qwen3-8B + LoRA adapters).
- `SkillBankAgent` is used in labeling/offline extraction scripts only.
- Both write the same `SkillBankMVP` bank format (`skill_bank.jsonl`), so banks
  produced by `extract_skills_qwen.py` can seed the EM trainer directly.

**Changes needed (~40 lines across 4 files):**

| Change | File | Lines | Blocked on |
|--------|------|-------|------------|
| Add `intentions: Optional[str] = None` | `trainer/common/metrics.py` (`RolloutStep`) | ~2 | Nothing |
| Populate intentions during rollout | rollout collector / decision agent | ~10 | Decision agent producing intentions |
| Map `intentions` into `TrajectoryFrame` | `trainer/skillbank/ingest_rollouts.py` | ~5 | `RolloutStep.intentions` |
| Parse `[TAG]` → `tag_*` predicates in Stage 0 | `trainer/skillbank/stages/stage0_predicates.py` | ~20 | `TrajectoryFrame.intentions` |

**Model swap (config only, no code):**

| Config | Current | Change to |
|--------|---------|-----------|
| `coevolution_train.sh` `SkillBank_base_model` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| `skillbank_agent_train.sh` `SKILLBANK_BASE_MODEL` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| `trainer/common/configs/skillbank_em.yaml` `lora.base_model_name_or_path` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |

**LoRA adapters:** Need retraining for Qwen3-8B base (boundary, segment,
contract, retrieval adapters). This is a training run, not a code change.

**HF vs vLLM for EM:** `MultiLoraSkillBankLLM` uses HuggingFace `from_pretrained`,
not vLLM. Two options:
- **Option A (recommended):** Keep HF for EM (LoRA adapters on local GPU), use vLLM
  only for decision agent rollout. Just swap base model to Qwen3-8B in config.
- **Option B:** Add vLLM backend to `MultiLoraSkillBankLLM` so EM stages call
  `ask_vllm`. More work, only needed if LoRA adapters are not used.
