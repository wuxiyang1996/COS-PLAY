# Stage 2: InferSegmentation — Preference-Learned Skill-Sequence Decoding

**Location:** `skill_agents/infer_segmentation/`

## Overview

InferSegmentation is Stage 2 of the trajectory segmentation pipeline. Given candidate boundaries **C** from [Stage 1 (boundary_proposal)](../boundary_proposal/README.md), it finds the optimal segmentation and skill labeling.

**Key principle: the LLM provides preferences, not scores.**

The LLM acts as a **preference teacher** — it ranks skills for each segment ("move fits better than attack for this segment"). A `PreferenceScorer` is *trained* from those rankings (Bradley-Terry), and the trained scorer provides the numeric values used by DP/beam decoders.

## Pipeline

```
Episode
  │
  ▼
Stage 1: boundary_proposal → candidate boundaries C
  │
  ▼
LLM Teacher: rank skills for each segment → pairwise preferences (k+ ≻ k-)
  │
  ▼
Train PreferenceScorer from preferences (Bradley-Terry)
  │
  ▼
Decode with trained scorer (Viterbi DP or beam) → skill sequence + diagnostics
  │
  ├──→ SubTask_Experience segments (with learned skill labels)
  │
  └──→ Uncertain segments (low margin) → query LLM for more preferences
                                              │
                                              ▼
                                     Retrain scorer → re-decode (iterate)
```

## How the LLM Teacher Works

The LLM **never produces numeric scores**. It provides:

### 1. Segment skill rankings (cold-start)

Prompt: "Here's a segment with these observations and actions. Rank the candidate skills from best fit to worst."

Response: `{"ranking": ["move", "gather", "attack"], "reasoning": "..."}`

This ranking is converted to pairwise preferences: move≻gather, move≻attack, gather≻attack.

### 2. Transition rankings

Prompt: "After skill 'move', rank which skills are most likely to follow."

Response: `{"ranking": ["attack", "move", "gather"], "reasoning": "..."}`

### 3. Active learning (uncertain segments)

Prompt: "Which skill better explains this segment, A or B?"

Response: `{"choice": "A", "evidence": "..."}`

Only queried for segments where the scorer is uncertain (low margin between top-2 candidates).

## Trained Scorer

The `PreferenceScorer` is trained from collected preferences:

- **Behavior fit**: learned per-skill affinity (how well each skill matches segments)
- **Transition prior**: learned per-transition affinity (how likely each skill follows another)
- Training uses **Bradley-Terry** log-likelihood on pairwise preferences (batch update by default: accumulate gradients over all preferences then apply once per parameter)
- The LLM's ranking implicitly considers duration and state-change consistency, so these are folded into the learned behavior_fit

## How the score is computed

The decoder assigns one **segment score** per (segment, skill, prev_skill). That score is a weighted sum of five terms, computed in **`SegmentScorer`** (`scorer.py`):

```
segment_score = w_bf * behavior_fit + w_dp * duration_prior + w_tp * transition_prior
              + w_cc * contract_compat + w_bq * boundary_quality
```

Defaults: `ScorerWeights(behavior_fit=1.0, duration_prior=0.3, transition_prior=1.0, contract_compat=0.0, boundary_quality=0.0)`.

When contract feedback is enabled (via `ContractFeedbackConfig`), `w_cc` is set automatically: 0.3 for `"weak"`, 1.0 for `"strong"`. When a `BoundaryPreferenceScorer` is plugged in, `w_bq` controls its influence.

### 1. Behavior fit

- **Source:** `PreferenceScorer.behavior_fit` (trained on LLM rankings), plugged into `SegmentScorer` as `behavior_fit_fn`.
- **Formula (PreferenceScorer):**  
  `(global_score + seg_win_rate * 5.0) * len(observations)`
  - **global_score:** Bradley–Terry score for that skill from segment preferences (skill_win ≻ skill_lose). Learned by gradient updates on pairwise data.
  - **seg_win_rate:** For the segment `[seg_start, seg_end]`, win rate of this skill among stored preferences whose segment overlaps `[seg_start, seg_end]`. Value in [-1, +1]; +1 = always preferred there, -1 = always losing. Multiplied by 5.0 then added to global_score so segment-specific signal can override the global prior.
  - **× len(observations):** Makes the term scale with segment length (longer segment = more “evidence”).
- **NEW_SKILL:** `background_log_prob * len(observations)` (e.g. -3.0 per step from `NewSkillConfig`).

So: behavior_fit is **learned from preferences**, with a global per-skill term plus a segment-aware win rate over overlapping preferences.

### 2. Duration prior

- **Source:** `SegmentScorer.duration_prior` → `gaussian_duration_log_prob(length, skill)` in `scorer.py`.
- **Formula:** Log of a truncated Gaussian on segment length `l = j - i + 1`:
  - `log p(l | k)` with mean/std from `DurationPriorConfig` (default mean=20, std=10), or from optional per-skill stats. If `l` is outside `[min_length, max_length]`, returns -∞.
- **NEW_SKILL:** 0.0.

So: short or very long segments get penalized unless the skill’s duration stats say otherwise.

### 3. Transition prior

**How the score is obtained:**

- **At decode time:** `SegmentScorer.transition_prior(skill, prev_skill)` is called. If a `transition_fn` was provided (the trained `PreferenceScorer.transition_prior`), it does:
  - `key = f"{prev_skill}->{skill}"` (e.g. `"move->attack"`)
  - `return _transition_scores.get(key, 0.0)`  
  So the **score is a single lookup** in a learned dict. No data for that pair → **0.0**.

- **Where the dict comes from:** Transition preferences are collected by **`collect_transition_preferences(skill_names, config)`** in `llm_teacher.py`. For each `prev_skill`, the LLM ranks “which skill is most likely to follow.” That ranking is turned into pairwise preferences where:
  - `segment_start = segment_end = -1` (marks them as transition prefs)
  - `skill_win` / `skill_lose` are **full transition strings**, e.g. `"move->attack"` and `"move->gather"` (so “attack follows move” is preferred over “gather follows move”).

- **Training:** `PreferenceScorer.train(store)` runs Bradley–Terry on all preferences. For each preference with `segment_start == -1` and `segment_end == -1`, it calls `_bt_update(self._transition_scores, pref.skill_win, pref.skill_lose)`. So the **keys** in `_transition_scores` are exactly those strings (e.g. `"move->attack"`). Each key gets a real-valued score; higher = that transition was preferred more often in the data. No hand-coded matrix—only what was learned from LLM transition rankings.

- **NEW_SKILL:** Handled in `SegmentScorer`, not in PreferenceScorer: if current skill is NEW_SKILL, return `-penalty` (e.g. -5.0); if prev_skill is NEW_SKILL, 0.0.

### 4. Contract compatibility (Stage 3 → Stage 2 closed loop)

**How the score is obtained:**

- **At decode time:** `SegmentScorer.contract_compat(skill, predicates_start, predicates_end)` is called. In code (`scorer.py`):
  - If `self._compat_fn` is **None** (the default), it **always returns 0.0**. So by default this term contributes **nothing** to the segment score.
  - If you constructed `SegmentScorer(..., compat_fn=my_fn)`, then for each skill (except NEW_SKILL) it returns `my_fn(skill, predicates_start, predicates_end)`.

- **Closed loop via `SkillBankMVP.compat_fn`:** When `contract_feedback_mode` is `"weak"` or `"strong"` in `PipelineConfig`, the pipeline passes `bank.compat_fn` to the `SegmentScorer`. This creates a Stage 3 → Stage 2 feedback loop:
  1. Stage 3 learns effects contracts from segments.
  2. Next iteration, Stage 2 uses those contracts as a soft bias: skills whose contracts match the observed predicate changes get a bonus; skills with missing or contradictory effects are penalized.
  3. This guides segmentation toward assigning skills where their contracts are a good fit.

- **`ContractFeedbackConfig`** (in `config.py`) controls the loop:

  | Field | Default | Meaning |
  |-------|---------|---------|
  | `mode` | `"off"` | `"off"` = no feedback; `"weak"` = soft bias (weight 0.3); `"strong"` = dominant (weight 1.0) |
  | `strength` | `0.3` | Manual weight override (used when mode is set) |
  | `p_thresh` | `0.5` | Predicate probability threshold for binary effect evaluation |
  | `missing_penalty` | `-0.5` | Score penalty when contract expects an effect not in predicates |
  | `contradiction_penalty` | `-1.0` | Score penalty when observed change contradicts the contract |

### 5. Boundary quality

**How the score is obtained:**

- **At decode time:** `SegmentScorer.boundary_quality(seg_start, seg_end)` is called. If a `BoundaryPreferenceScorer` is plugged in, it returns the scorer's `decoding_bonus(seg_start, seg_end)` which considers:
  - **Signal strength:** How many Stage 1 signals support the boundary (predicate flip, reward spike, done flag, etc.)
  - **Predicate discontinuity:** How much the predicate state changes at the boundary
  - **Learned preference:** Pairwise preference data from human or LLM feedback on boundary quality
- If no boundary scorer is set, returns 0.0.
- Controlled by `ScorerWeights.boundary_quality` (default 0.0).

### Where it’s used

- **Decoders** (`dp_decoder`, `beam_decoder`) call `SegmentScorer.score_breakdown(i, j, skill, prev_skill, observations, actions, predicates_start, predicates_end)` for each segment and each candidate skill. They use the returned `"total"` as the segment score and put the per-term breakdown into `SkillCandidate.breakdown` for diagnostics.
- **Total path score** = sum of segment scores along the chosen path. Viterbi/beam maximize this sum over paths that respect candidate boundaries.

### Getting a skill ranking for a sub-trajectory (from preferences only)

We only store **pairwise** preferences (A ≻ B). To get a **full ranking** of skills for an extracted sub-trajectory you have two options:

1. **From the LLM (cold-start):** Ask the LLM once to rank all skills for that segment; it returns `{"ranking": ["best", "second", ...]}`. We then convert that to pairwise preferences via `ranking_to_pairwise`. So the ranking is produced by the LLM, not inferred from stored pairs.

2. **From the trained scorer (no LLM):** Score every skill on that segment with the same formula the decoder uses (behavior_fit + duration_prior + transition_prior + contract_compat), then **sort by total score descending**. That ordered list is the ranking. Use **`SegmentScorer.rank_skills_for_segment`**:

   ```python
   # After training (SegmentScorer built from PreferenceScorer)
   ranking = scorer.rank_skills_for_segment(
       i, j,
       observations[i:j+1], actions[i:j+1],
       predicates_start=predicates[i], predicates_end=predicates[j],
       prev_skill=None,  # or previous segment's skill
       include_breakdown=True,
   )
   # ranking is [(skill, total_score, breakdown), ...] best-first
   skill_order = [r[0] for r in ranking]
   ```

So: pairwise preferences train the scorer; the **ranking** for a segment is obtained by scoring all skills for that segment and sorting.

## Modes: Inference (with LLM) vs Offline (no LLM)

There are two ways to run segmentation:

| Mode | Function | When to use |
|------|----------|-------------|
| **Inference with LLM** | `infer_and_segment` | You want to use an LLM (e.g. GPT-5) at inference time: the LLM is prompted to rank skills for segments and transitions, preferences are collected, the PreferenceScorer is trained, and decoding runs. Use this when you want to interface with GPT or another API model. |
| **Offline (no LLM)** | `infer_and_segment_offline` | No LLM calls. You supply pre-trained `behavior_fit_fn` and `transition_fn` (e.g. from a saved PreferenceScorer or mock). Use for (1) **training** runs where preferences were collected elsewhere (e.g. human labels or a separate batch), or (2) **testing** without API keys, or (3) inference when you already have a trained scorer and do not want to call an API. |

So: **inference mode** = segmentation that can call GPT-5 (or another LLM); **offline** = segmentation with no LLM, using only the scoring functions you provide (trained or hand-written).

## Quick Start

### Inference with LLM (e.g. GPT-5)

```python
from skill_agents.infer_segmentation import infer_and_segment

result, sub_episodes, store = infer_and_segment(
    episode,
    skill_names=["move", "attack", "gather", "craft"],
    env_name="overcooked",
)

# Inspect results
print(result.skill_sequence)
for seg in result.segments:
    print(f"[{seg.start}-{seg.end}] {seg.assigned_skill} margin={seg.margin:.2f}")

# Save preferences for future use
store.save("preferences.json")
```

### Offline (no LLM) — pre-trained scorer or testing

```python
from skill_agents.infer_segmentation import infer_and_segment_offline

result, sub_episodes = infer_and_segment_offline(
    episode,
    skill_names=["move", "attack", "gather"],
    behavior_fit_fn=my_custom_scorer,
    transition_fn=my_transition_fn,
)
```

### Manual preference training

```python
from skill_agents.infer_segmentation import (
    PreferenceStore, PreferenceScorer, SegmentScorer,
    ranking_to_pairwise, viterbi_decode,
)

# Collect preferences (from LLM or manual)
store = PreferenceStore()
store.add_batch(ranking_to_pairwise(["move", "attack", "gather"], 0, 9))

# Train
scorer = PreferenceScorer(["move", "attack", "gather"])
scorer.train(store, epochs=20)

# Decode
seg_scorer = SegmentScorer(
    skill_names=["move", "attack", "gather"],
    behavior_fit_fn=scorer.behavior_fit,
    transition_fn=scorer.transition_prior,
)
result = viterbi_decode(candidates, T, seg_scorer, obs, acts)
```

## Decoders

### Viterbi DP (HSMM-style)
Globally optimal. Complexity O(|C|² · K²).

### Beam Search (agent-friendly)
Keeps top-B partial segmentations. Supports early stopping.

## Diagnostics

The `diagnostics` module (`diagnostics.py`) holds everything the decoders return beyond “which skill per segment”: per-segment alternatives, score breakdowns, and uncertainty so you can drive **active learning** and inspect decisions.

### What gets recorded

- **Per segment:** top-K skill candidates with total score and per-term breakdown (behavior_fit, duration_prior, transition_prior, contract_compat, boundary_quality).
- **Margin** = score(rank-1) − score(rank-2). Small margin → decoder is uncertain which skill fits.
- **Per boundary (optional):** score if we cut there vs not, so you can see boundary confidence.

Uncertain segments (low margin) are the best targets for asking the LLM for more preferences.

### The four classes

**1. `SkillCandidate`** — One possible skill label for a segment.

| Field | Meaning |
|-------|--------|
| `skill` | Skill name (e.g. `"move"`, `"attack"`, or `NEW_SKILL`) |
| `total_score` | Weighted sum of the four score terms for this (segment, skill) pair |
| `breakdown` | Dict of per-term scores, e.g. `{"behavior_fit": 2.1, "duration_prior": -0.3, "transition_prior": 0.5, "contract_compat": 0.0, "boundary_quality": 0.0}` |

The decoder keeps the top-K candidates per segment (K from `DecoderConfig.top_k_diagnostics`). The first candidate is the chosen skill; the rest explain why the segment was or wasn’t assigned something else.

**2. `SegmentDiagnostic`** — One decoded segment and its alternatives.

| Field / property | Meaning |
|------------------|--------|
| `start`, `end` | Segment time range (inclusive indices) |
| `assigned_skill` | The skill picked for this segment (same as `candidates[0].skill`) |
| `candidates` | List of `SkillCandidate`, sorted by `total_score` descending (best first) |
| `margin` | `candidates[0].total_score - candidates[1].total_score` if ≥2 candidates; else `float("inf")` |
| `is_uncertain` | `True` when `margin < 1.0` (heuristic: worth querying the LLM) |

So for each segment you see the winning skill, the runner-up(s), and how close the race was. Low margin means the model is unsure and preference labels there are most informative.

**3. `BoundaryDiagnostic`** — How confident the decoder is about a cut.

| Field / property | Meaning |
|------------------|--------|
| `time` | Timestep where the cut is considered |
| `score_with_cut` | Score of the best path that places a segment boundary at `time` |
| `score_without_cut` | Score of the best path that does *not* cut at `time` |
| `confidence` | `score_with_cut - score_without_cut`; positive means cutting here is better |

Not all decoders populate `boundaries`; when they do, you can see which boundaries were clear vs borderline.

**4. `SegmentationResult`** — Full decoder output.

| Field / property | Meaning |
|------------------|--------|
| `segments` | List of `SegmentDiagnostic`, one per segment in the best path |
| `boundaries` | Optional list of `BoundaryDiagnostic` |
| `total_score` | Sum of segment scores along the best path |
| `skill_sequence` | `[s.assigned_skill for s in segments]` — the skill string per segment in order |
| `cut_points` | `[s.start for s in segments]` — start index of each segment |
| `uncertain_segments(margin_threshold=1.0)` | Segments with `margin < margin_threshold`, sorted by margin (smallest first). Used by active learning to choose which segments to ask the LLM about. |
| `to_dict()` | Serializable dict of all of the above (for logging or APIs) |

### How it’s used in the pipeline

- **DP and beam decoders** build `SegmentationResult` by assigning each segment in the best path a `SegmentDiagnostic` with top-K `SkillCandidate`s (and optional `BoundaryDiagnostic`s). They get the breakdown from `SegmentScorer.score_breakdown(...)`.
- **Active learning:** `generate_preference_queries(result, margin_threshold, max_queries)` in `preference.py` calls `result.uncertain_segments(margin_threshold)`, then builds A/B preference queries for the top-2 skills on those segments. `collect_uncertain_preferences` in `llm_teacher.py` uses the same idea to decide which segments to send to the LLM.
- **Inspection:** You can log or display `result.to_dict()`, or iterate `result.segments` and use `seg.margin`, `seg.candidates`, and `seg.breakdown` to explain why a segment got a given skill.

## NEW Skill Channel

A special `__NEW__` label handles segments where no existing skill fits:

- Penalized by −α to prevent over-creation
- Segments labeled `__NEW__` go to a "new skill candidate pool"

## Configuration

```python
from skill_agents.infer_segmentation import (
    SegmentationConfig, LLMTeacherConfig, PreferenceLearningConfig,
)

config = SegmentationConfig(
    llm_teacher=LLMTeacherConfig(
        model="gpt-4o",       # or None for default
        temperature=0.3,
        max_workers=8,        # worker threads (None or 1 = sequential)
        max_concurrent_llm_calls=None,  # cap concurrent inference (e.g. 1 for local GPU)
    ),
    preference=PreferenceLearningConfig(
        num_iterations=3,      # active learning rounds
        margin_threshold=1.0,  # query segments below this margin
        max_queries_per_iter=5,
        training_epochs=20,
        learning_rate=0.1,
    ),
)
```

**Local models (skill bank / GPU):** Keep parallel task flow but avoid GPU OOM by capping concurrent inference: set `max_concurrent_llm_calls=1` (or 2 if your GPU can handle it). Worker threads still run in parallel for prompt building and result parsing; only the actual model call is serialized.

## Preference Data Schema

```json
{
    "segment_start": 10,
    "segment_end": 19,
    "skill_win": "attack",
    "skill_lose": "move",
    "evidence": "strike actions clearly match attack skill",
    "source": "llm",
    "timestamp": 1708000000.0
}
```

## File Structure

```
infer_segmentation/
├── __init__.py          # Public API exports
├── README.md            # This file
├── config.py            # SegmentationConfig, LLMTeacherConfig, etc.
├── llm_teacher.py       # LLM preference teacher (rankings → preferences)
├── preference.py        # PreferenceStore, PreferenceScorer (Bradley-Terry)
├── scorer.py            # SegmentScorer (uses trained PreferenceScorer)
├── dp_decoder.py        # Viterbi DP (HSMM-style)
├── beam_decoder.py      # Beam search decoder
├── diagnostics.py       # SegmentationResult, margins, breakdowns
├── episode_adapter.py   # Full pipeline: Stage 1 → prefs → train → decode
├── requirements.txt     # Dependencies
└── example_toy.py       # Example with simulated preferences (no LLM needed)
```

---

## Functions and Purposes (Reference)

This section is a quick reference for what each file and its main components do.

### `config.py` — Configuration

| Class / field | Purpose |
|---------------|--------|
| **ScorerWeights** | Weights for the five score terms: `behavior_fit`, `duration_prior`, `transition_prior`, `contract_compat`, `boundary_quality`. |
| **ContractFeedbackConfig** | Stage 3 → Stage 2 closed-loop control: `mode` (`"off"`/`"weak"`/`"strong"`), `strength`, `p_thresh`, penalty values. |
| **DurationPriorConfig** | Gaussian duration prior: `default_mean`, `default_std`, `min_length`, `max_length`. |
| **NewSkillConfig** | Special `__NEW__` skill: `enabled`, `penalty`, `background_log_prob`. |
| **LLMTeacherConfig** | LLM calls: `model`, `temperature`, `max_tokens`, `max_workers`, `max_concurrent_llm_calls` (set to 1 for local GPU to avoid OOM). |
| **PreferenceLearningConfig** | Preference loop: `num_iterations`, `margin_threshold`, `max_queries_per_iter`, `training_epochs`, `learning_rate`, `collect_transitions`. |
| **DecoderConfig** | Decoder: `top_m_skills`, `beam_width`, `beam_max_segments`, `top_k_diagnostics`. |
| **SegmentationConfig** | Top-level config aggregating all of the above; `method` = `"dp"` or `"beam"`. |

### `scorer.py` — Segment scoring

Defines how each segment–skill pair is scored (weighted sum of four terms).

| Function / class | Purpose |
|------------------|--------|
| **SegmentScorer** | Composite scorer: behavior_fit + duration_prior + transition_prior + contract_compat + boundary_quality. Accepts pluggable `behavior_fit_fn`, `transition_fn`, `compat_fn`, and `boundary_scorer`. |
| **behavior_fit**, **duration_prior**, **transition_prior**, **contract_compat**, **boundary_quality** | The five terms; behavior_fit and transition_prior from PreferenceScorer; contract_compat from `SkillBankMVP.compat_fn`; boundary_quality from `BoundaryPreferenceScorer`. |
| **score**, **score_breakdown** | Full segment score and per-term breakdown (for diagnostics). |
| **NEW_SKILL** | Sentinel for “none of the known skills”; gets a penalty. |

### `diagnostics.py` — Decoder output and uncertainty

| Class / method | Purpose |
|----------------|--------|
| **SkillCandidate** | One skill option for a segment: `skill`, `total_score`, `breakdown`. |
| **SegmentDiagnostic** | One segment: `start`, `end`, `assigned_skill`, `candidates`. `margin` = score(top1) − score(top2); `is_uncertain` = margin < 1.0. |
| **BoundaryDiagnostic** | For a cut: `time`, `score_with_cut`, `score_without_cut`, `confidence`. |
| **SegmentationResult** | Full result: `segments`, `boundaries`, `total_score`, `skill_sequence`, `cut_points`, `uncertain_segments(margin_threshold)`, `to_dict()`. |

### `preference.py` — Preferences and learned scorer

| Class / function | Purpose |
|------------------|--------|
| **PreferenceExample** | One preference: “skill_win ≻ skill_lose” for a segment; `is_transition_pref` when segment is (-1,-1). |
| **PreferenceQuery** | Template for a single uncertain-segment query (top-2 candidates, scores, margin). |
| **PreferenceStore** | List of PreferenceExamples; `add`, `add_batch`, `segment_preferences`, `transition_preferences`, `save`, `load`. |
| **generate_preference_queries(result, margin_threshold, max_queries)** | Build PreferenceQueries for segments with margin below threshold (active learning). |
| **PreferenceScorer** | Trained from preferences. **behavior_fit** and **transition_prior** plug into SegmentScorer; **train(store, epochs)** runs Bradley–Terry. **behavior_fit_batch** enables batched preference scoring at inference (decoders use **score_breakdown_batch** when available). |

### `llm_teacher.py` — LLM as preference teacher

| Function | Purpose |
|----------|--------|
| **ranking_to_pairwise(ranking, segment_start, segment_end, ...)** | Turn a ranking [best, …, worst] into preference pairs. |
| **collect_segment_preferences(...)** | Cold-start: for each segment, LLM ranks skills → PreferenceExamples. |
| **collect_transition_preferences(skill_names, config)** | LLM ranks “which skill follows” per prev_skill → transition preferences. |
| **collect_uncertain_preferences(result, ...)** | Active learning: ask LLM “A or B?” for uncertain segments. |

### `dp_decoder.py` — Viterbi DP

| Function | Purpose |
|----------|--------|
| **viterbi_decode(candidates, T, scorer, observations, actions, predicates, config)** | Globally optimal segmentation over candidate boundaries. Returns **SegmentationResult**. Complexity O(\|C\|² × K²). |

### `beam_decoder.py` — Beam search

| Function | Purpose |
|----------|--------|
| **beam_decode(candidates, T, scorer, observations, actions, predicates, config)** | Approximate decoding with fixed beam width; uses `beam_width`, `top_m_skills`, `beam_max_segments`, `top_k_diagnostics`. Returns **SegmentationResult**. |

### `episode_adapter.py` — End-to-end pipeline

| Function | Purpose |
|----------|--------|
| **infer_segmentation(candidates, T, skill_names, ...)** | Low-level: run decoder with a given or built SegmentScorer. Returns **SegmentationResult**. |
| **infer_and_segment(episode, skill_names, env_name, ...)** | Inference with LLM (e.g. GPT-5): Stage 1 → collect prefs → train → decode → optional active-learning loop. Returns **(SegmentationResult, list[SubTask_Experience], PreferenceStore)**. |
| **infer_and_segment_offline(episode, skill_names, ...)** | Offline (no LLM): Stage 1 → decode with provided scoring functions. For training from pre-collected prefs or testing. |

### `__init__.py` — Public API

Re-exports configs, SegmentScorer, diagnostics, decoders, preference types, LLM teacher functions, and the three pipeline functions.

### `example_toy.py` — Example script

Runs the pipeline without the LLM: simulated rankings → preferences → train PreferenceScorer → Viterbi and beam → show diagnostics and preference queries.

### Data flow (concise)

1. Episode → **infer_and_segment** (or offline variant).
2. Stage 1 → **candidate boundaries C** and segment list.
3. **LLM teacher** → rankings → **ranking_to_pairwise** → **PreferenceExample**s → **PreferenceStore**.
4. **PreferenceScorer.train(store)** → learned behavior_fit and transition_prior.
5. **SegmentScorer** uses PreferenceScorer (+ duration prior) → **viterbi_decode** or **beam_decode** → **SegmentationResult**.
6. **uncertain_segments** → **generate_preference_queries** → (optional) more LLM calls → retrain → re-decode.
7. **SegmentationResult.segments** → mapped to **SubTask_Experience** with skill labels.

### Main concepts

- **Preference, not score:** The LLM only outputs rankings (or A/B choices); all numeric scores come from the trained PreferenceScorer.
- **Segment score** = behavior_fit + duration_prior + transition_prior + contract_compat + boundary_quality (weights in ScorerWeights).
- **Closed loop:** When contract feedback is enabled, Stage 3 contracts bias Stage 2 segmentation via `compat_fn` (see §4 above).
- **Boundary quality:** When a `BoundaryPreferenceScorer` is plugged in, segment boundaries are scored for plausibility (see §5 above).
- **Uncertainty** = low margin between top-two skills on a segment; those segments are used for active learning.
- **Decoding** is either exact (Viterbi DP) or approximate (beam) over the same candidate boundaries and the same SegmentScorer interface.
