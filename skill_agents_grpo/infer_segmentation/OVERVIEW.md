# InferSegmentation — Functions and Purposes

This folder implements **Stage 2** of trajectory segmentation: given candidate cut points from Stage 1 (`boundary_proposal`), it finds the **optimal skill label for each segment** and returns a full segmentation with diagnostics. The LLM acts as a **preference teacher** (rankings only); a **PreferenceScorer** is trained from those preferences and used for decoding.

---

## 1. Overall pipeline

```
Episode → Stage 1 (boundary_proposal) → candidate boundaries C
       → LLM teacher ranks skills per segment → pairwise preferences
       → Train PreferenceScorer (Bradley-Terry)
       → Decode (DP or beam) with trained scorer → skill sequence + diagnostics
       → Optionally: query LLM on uncertain segments → more prefs → retrain → re-decode
       → SubTask_Experience segments with skill labels
```

---

## 2. File-by-file summary

### `config.py` — Configuration

| Class / field | Purpose |
|---------------|--------|
| **ScorerWeights** | Weights for the four score terms: `behavior_fit`, `duration_prior`, `transition_prior`, `contract_compat`. |
| **DurationPriorConfig** | Gaussian duration prior: `default_mean`, `default_std`, `min_length`, `max_length`. |
| **NewSkillConfig** | Special `__NEW__` skill: `enabled`, `penalty`, `background_log_prob`. |
| **LLMTeacherConfig** | LLM calls: `model`, `temperature`, `max_tokens`. |
| **PreferenceLearningConfig** | Preference loop: `num_iterations`, `margin_threshold`, `max_queries_per_iter`, `training_epochs`, `learning_rate`, `collect_transitions`. |
| **DecoderConfig** | Decoder: `top_m_skills`, `beam_width`, `beam_max_segments`, `top_k_diagnostics`. |
| **SegmentationConfig** | Top-level config aggregating all of the above; `method` = `"dp"` or `"beam"`. |

---

### `scorer.py` — Segment scoring

**Purpose:** Define how each segment–skill pair is scored. The score is a weighted sum of four terms.

| Function / class | Purpose |
|------------------|--------|
| **SegmentScorer** | Composite scorer. Combines: (1) behavior_fit, (2) duration_prior, (3) transition_prior, (4) contract_compat. Accepts pluggable `behavior_fit_fn` and `transition_fn` (e.g. from PreferenceScorer). |
| **behavior_fit(obs, actions, skill, seg_start, seg_end)** | “How well do actions match this skill?” Usually from PreferenceScorer; supports segment range for segment-aware scoring. |
| **duration_prior(length, skill)** | “Is this segment length plausible for this skill?” Gaussian in length (configurable). |
| **transition_prior(skill, prev_skill)** | “How likely is this skill after prev_skill?” Usually from PreferenceScorer. |
| **contract_compat(skill, preds_start, preds_end)** | Optional; default 0 (folded into LLM ranking). |
| **score(i, j, k, prev_skill, ...)** | Full segment score = weighted sum of the four terms. |
| **score_breakdown(...)** | Same as `score` but returns per-term values (for diagnostics). |
| **uniform_behavior_fit**, **gaussian_duration_log_prob** | Fallback / helper implementations. |
| **NEW_SKILL** | Sentinel string for “none of the known skills”; gets a penalty. |

---

### `diagnostics.py` — Decoder output and uncertainty

**Purpose:** Represent the decoder result and per-segment uncertainty for active learning.

| Class / method | Purpose |
|----------------|--------|
| **SkillCandidate** | One skill option for a segment: `skill`, `total_score`, `breakdown` (per-term scores). |
| **SegmentDiagnostic** | One segment in the result: `start`, `end`, `assigned_skill`, `candidates` (top-K skills). `margin` = score(top1) − score(top2); `is_uncertain` = margin < 1.0. |
| **BoundaryDiagnostic** | For a cut: `time`, `score_with_cut`, `score_without_cut`, `confidence`. |
| **SegmentationResult** | Full result: `segments`, `boundaries`, `total_score`. `skill_sequence`, `cut_points`, `uncertain_segments(margin_threshold)`, `to_dict()`. |

Small margin → uncertain segment → good target for asking the LLM for more preferences.

---

### `preference.py` — Preferences and learned scorer

**Purpose:** Store pairwise preferences, train a scorer from them, and expose it for decoding.

| Class / function | Purpose |
|------------------|--------|
| **PreferenceExample** | One preference: “skill_win ≻ skill_lose” for a segment (`segment_start`, `segment_end`), plus `evidence`, `source`. `is_transition_pref` when segment is (-1,-1). |
| **PreferenceQuery** | Template for a single uncertain-segment query: top-2 candidates, scores, margin, breakdowns. |
| **PreferenceStore** | List of PreferenceExamples. `add`, `add_batch`, `examples`, `segment_preferences`, `transition_preferences`, `save`, `load`. |
| **generate_preference_queries(result, margin_threshold, max_queries)** | From a SegmentationResult, build PreferenceQueries for segments with margin below threshold (for active learning). |
| **PreferenceScorer** | Scorer trained from preferences. **behavior_fit(obs, actions, skill, _seg_start, _seg_end)**: combines global Bradley–Terry skill scores and segment-specific win rate over overlapping preferences. **transition_prior(skill, prev_skill)**: learned transition scores. **train(store, epochs)**: Bradley–Terry updates; stores segment prefs for segment-aware behavior_fit. |

So: preferences → train PreferenceScorer → use as `behavior_fit_fn` and `transition_fn` in SegmentScorer.

---

### `llm_teacher.py` — LLM as preference teacher

**Purpose:** Call the LLM to get **rankings** (not numeric scores); convert rankings to pairwise preferences.

| Function | Purpose |
|----------|--------|
| **ranking_to_pairwise(ranking, segment_start, segment_end, source, evidence)** | Turn a ranking [best, …, worst] into preference pairs (e.g. best≻second, best≻third, …). |
| **collect_segment_preferences(segments, observations, actions, skill_names, predicates, config)** | **Cold-start.** For each segment, prompt LLM to rank all skills; parse JSON `{ranking, reasoning}`; convert to PreferenceExamples and return. |
| **collect_transition_preferences(skill_names, config)** | For each skill as prev_skill, prompt LLM to rank “which skill follows next”; convert to transition preferences (segment -1,-1; skill_win/lose like `"move->attack"`). |
| **collect_uncertain_preferences(result, observations, actions, margin_threshold, max_queries, config)** | **Active learning.** Take uncertain segments from result; for each, ask LLM “A or B?” for top-2 skills; append one PreferenceExample per answer. |

Internal helpers: `_build_segment_ranking_prompt`, `_build_transition_ranking_prompt`, `_build_pairwise_prompt`, `_parse_json_from_response`, `_get_ask_model` (lazy API import).

---

### `dp_decoder.py` — Viterbi DP

**Purpose:** Find the **globally optimal** segmentation and skill sequence over candidate boundaries.

| Function | Purpose |
|----------|--------|
| **viterbi_decode(candidates, T, scorer, observations, actions, predicates, config)** | DP over boundaries C ∪ {0, T-1}. `dp[b_idx][k]` = best score ending at boundary b with last skill k. Recurrence: max over previous boundary and previous skill of (dp + segment score). Backtrack to get best path; attach top-K skill candidates per segment for diagnostics. Returns **SegmentationResult**. |

Only boundaries in the candidate set are used; complexity is O(|C|² × K²) scorer calls.

---

### `beam_decoder.py` — Beam search

**Purpose:** Approximate decoding with a fixed beam width (faster, can stop early).

| Function | Purpose |
|----------|--------|
| **beam_decode(candidates, T, scorer, observations, actions, predicates, config)** | State = (last_cut, last_skill, total_score, path). Expand by choosing next boundary and next skill; score segment; keep top-B entries per round. When a path reaches the end, put it in `completed`. Return best completed path as **SegmentationResult**. |

Uses `DecoderConfig.beam_width`, `top_m_skills`, `beam_max_segments`, `top_k_diagnostics`.

---

### `episode_adapter.py` — End-to-end pipeline

**Purpose:** Wire Stage 1, LLM teacher, preference training, and decoding into one or two entry points.

| Function | Purpose |
|----------|--------|
| **_extract_obs_actions(experiences)** | From an Episode’s experiences, get lists of observations and actions (for decoder and LLM). |
| **_extract_predicates(experiences)** | Build per-timestep predicate dicts (e.g. sub_task, intention, done) from experiences. |
| **_build_scorer_from_preferences(skill_names, store, config)** | Build PreferenceScorer, train on store, wrap as SegmentScorer (behavior_fit + transition_prior). |
| **_decode(candidates, T, scorer, observations, actions, predicates, config)** | Call `viterbi_decode` or `beam_decode` according to config. |
| **infer_segmentation(candidates, T, skill_names, observations, actions, predicates, config, scorer, …)** | **Low-level API.** Run decoder with a given (or newly built) SegmentScorer. If no scorer given, builds one from optional behavior_fit_fn / transition_fn. Returns **SegmentationResult**. |
| **infer_and_segment(episode, skill_names, env_name, config, preference_store, …)** | **Inference with LLM (e.g. GPT-5).** (1) Stage 1 → boundaries C. (2) If store empty: collect segment (+ optional transition) preferences via LLM. (3) Train scorer, decode. (4) For several iterations: collect uncertain preferences, retrain, re-decode. (5) Convert result to SubTask_Experience list. Use when you want to interface with an LLM at inference time. Returns **(SegmentationResult, list[SubTask_Experience], PreferenceStore)**. |
| **infer_and_segment_offline(episode, skill_names, env_name, …, behavior_fit_fn, transition_fn, …)** | **Offline (no LLM).** Stage 1 → decode with provided scoring functions only. Use for training from pre-collected preferences, or testing without API. Returns **(SegmentationResult, list[SubTask_Experience])**. |

---

### `__init__.py` — Public API

Re-exports configs, SegmentScorer, diagnostics, viterbi_decode, beam_decode, preference types, LLM teacher functions, and the three pipeline functions so callers can use the package from a single import surface.

---

### `example_toy.py` — Example script

**Purpose:** Demonstrate the pipeline **without** calling the LLM: simulate LLM rankings → build preferences → train PreferenceScorer → run Viterbi and beam → show diagnostics and preference queries. Useful to validate behavior and data flow.

---

### `README.md` — User-facing docs

Describes the preference-learning design, pipeline, LLM teacher, configuration, and file layout.

---

### `requirements.txt` — Dependencies

Declares runtime deps (e.g. `numpy`); LLM calls go through `API_func.ask_model` (project-level).

---

## 3. Data flow (concise)

1. **Episode** → `infer_and_segment` (or offline variant).
2. **Stage 1** (boundary_proposal) → **candidate boundaries C** and segment list.
3. **LLM teacher** (llm_teacher) → **rankings** → **ranking_to_pairwise** → **PreferenceExample**s → **PreferenceStore**.
4. **PreferenceScorer** (preference) **.train(store)** → learned behavior_fit and transition_prior.
5. **SegmentScorer** (scorer) uses PreferenceScorer as behavior_fit_fn and transition_fn (+ duration prior).
6. **viterbi_decode** or **beam_decode** (dp_decoder / beam_decoder) → **SegmentationResult** (segments + diagnostics).
7. **uncertain_segments** → **generate_preference_queries** → (optional) more LLM calls → store.add_batch → retrain → re-decode.
8. **SegmentationResult.segments** → mapped to **SubTask_Experience** with skill labels.

---

## 4. Main concepts

- **Preference, not score:** The LLM only outputs rankings (or A/B choices); all numeric scores come from the trained PreferenceScorer.
- **Segment score = behavior_fit + duration_prior + transition_prior + contract_compat** (weights in ScorerWeights).
- **Uncertainty = low margin** between top-two skills on a segment; those segments are used for active learning.
- **Decoding** is either exact (Viterbi DP) or approximate (beam) over the same candidate boundaries and the same SegmentScorer interface.
