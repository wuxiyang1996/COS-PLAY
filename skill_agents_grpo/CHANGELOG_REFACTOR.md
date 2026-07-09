# Skill Bank Agent — Refactoring Changelog

## Overview

Four modular improvements to the skill bank agent, preserving the existing architecture:

1. **Stage 3 → Stage 2 closed-loop feedback** (Priority 1)
2. **Improved NEW materialization** (Priority 2)
3. **Boundary preference learning** (Priority 3)
4. **Skill selection policy** (Priority 4)

---

## Priority 1: Contract Feedback Loop (Stage 3 → Stage 2)

### Where the feedback loop is connected

The loop flows:

```
Stage 3 (contract_learn → contract_verify → contract_refine)
   ↓  produces SkillEffectsContract per skill in SkillBankMVP
   ↓
SkillBankMVP.compat_fn(skill, P_start, P_end)
   ↓  called during Stage 2 decoding via SegmentScorer.contract_compat()
   ↓
Stage 2 DP/Beam decoder — contract_compat term biases skill assignment
```

Key integration points:
- `skill_bank/bank.py`: `SkillBankMVP.compat_fn()` + `_effects_compat_score()` helper
- `infer_segmentation/scorer.py`: `SegmentScorer.__init__(compat_fn=...)` and `contract_compat()` method
- `infer_segmentation/episode_adapter.py`: `_build_scorer_from_preferences(compat_fn=...)`
- `pipeline.py`: `segment_episode()` passes `bank.compat_fn` when feedback is enabled

### How the compatibility score is computed

`_effects_compat_score(contract, P_start, P_end)`:

| Condition | Score |
|-----------|-------|
| `eff_add` literal observed true at end | +1.0 |
| `eff_add` literal observed false at end | -1.0 (contradiction) |
| `eff_add` literal not in predicates | -0.5 (missing) |
| `eff_del` literal observed false at end | +1.0 (correctly deleted) |
| `eff_del` literal still true at end | -1.0 (contradiction) |
| `eff_event` literal observed | +1.0 |

Normalised by number of contract literals → range approx. [-1, +1].

### Config knobs added

In `infer_segmentation/config.py`:

```python
@dataclass
class ContractFeedbackConfig:
    mode: str = "off"      # "off" | "weak" | "strong"
    strength: float = 0.3  # effective weight when mode != "off"
    p_thresh: float = 0.5
    missing_penalty: float = -0.5
    contradiction_penalty: float = -1.0
```

In `pipeline.py` `PipelineConfig`:
```python
contract_feedback_mode: str = "off"
contract_feedback_strength: float = 0.3
```

**Ablation modes:**
- `"off"`: `contract_compat` weight stays at 0.0 (default, no regression)
- `"weak"`: weight set to `strength` (0.3) — soft bias
- `"strong"`: weight set to `max(strength, 1.0)` — heavy reliance

---

## Priority 2: Improved NEW Materialization

### What changed

New module: `skill_bank/new_pool.py` — `NewPoolManager`

**Before:** `__NEW__` segments were grouped by exact `effect_signature()` string
(e.g. `A:p1,p2|D:q1`).  This is brittle — any minor variation creates a
separate bucket.

**After:** `NewPoolManager` provides:
- Rich per-candidate metadata: effect signature, duration, predecessor/successor context
- Jaccard-based agglomerative clustering (not exact string match)
- Cluster quality metrics: consistency, duration stats, context distributions
- Promotion criteria: support (cluster size) + consistency + separability (Jaccard distance from existing skills)
- Modular API: `add()`, `cluster()`, `get_candidates()`, `promote()`

### Promotion criteria

A cluster is promoted to a real skill when:
1. `size >= min_cluster_size` (default 5)
2. `consistency >= min_consistency` (default 0.5) — fraction sharing majority effect pattern
3. `distinctiveness >= min_distinctiveness` (default 0.25) — Jaccard distance from nearest existing skill
4. Stage 3 verification `pass_rate >= min_pass_rate` (default 0.7)

### Config knobs

In `pipeline.py` `PipelineConfig`:
```python
new_pool_min_cluster_size: int = 5
new_pool_min_consistency: float = 0.5
new_pool_min_distinctiveness: float = 0.25
```

In `skill_bank/new_pool.py` `NewPoolConfig`:
```python
min_cluster_size: int = 5
min_pass_rate: float = 0.7
min_distinctiveness: float = 0.25
min_consistency: float = 0.5
cluster_similarity_thresh: float = 0.4
max_pool_size: int = 500
max_promotions_per_call: int = 10
```

---

## Priority 3: Boundary Preference Learning

### New module

`boundary_proposal/boundary_preference.py` — `BoundaryPreferenceScorer`

### Plausibility features

1. **Signal strength**: how many independent signal sources support this boundary (predicate flips, surprisal spikes, change-points, hard events). Hard sources get a bonus.
2. **Predicate discontinuity**: magnitude of predicate state change across the boundary (windowed).
3. **Learned preference** (optional): pairwise Bradley-Terry scoring from LLM/human feedback about boundary quality.

### Integration points

**Stage 1 filtering:**
```python
bp = BoundaryPreferenceScorer(config=BoundaryPreferenceConfig(enabled=True))
bp.set_candidates(candidates)
bp.set_predicates(predicates)
filtered = bp.filter_candidates(candidates, top_frac=0.7)
```

**Stage 2 decoding:**
```python
scorer = SegmentScorer(..., boundary_scorer=bp.decoding_bonus)
# boundary_quality(seg_start, seg_end) is automatically added to segment score
```

### Config knobs

```python
@dataclass
class BoundaryPreferenceConfig:
    enabled: bool = False           # disabled by default
    w_signal_strength: float = 1.0
    w_predicate_discontinuity: float = 1.0
    w_effect_contrast: float = 0.5
    min_plausibility: float = 0.1
    pred_window: int = 3
    hard_source_bonus: float = 0.5
    use_in_decoding: bool = True
    decoding_weight: float = 0.2
```

---

## Priority 4: Skill Selection Policy

### What changed

`query.py` upgraded from a pure retriever to a structured skill selection policy.

**Before:** `query()` returned `[{skill_id, score, contract, micro_plan}]` —
a flat relevance ranking.

**After:** New `select()` method returns `SkillSelectionResult` with separate dimensions:

| Field | Meaning |
|-------|---------|
| `relevance` | Retrieval match score (embedding + keyword Jaccard) |
| `applicability` | Contract-based execution compatibility with current state |
| `confidence` | Blended score (0.4 × relevance + 0.35 × applicability + 0.25 × pass_rate) |
| `contract_match_score` | Raw compat score from effects contract |
| `pass_rate` | Historical verification pass rate |
| `matched_effects` | Which contract effects match the query/state |
| `missing_effects` | Expected effects not present in current state |
| `micro_plan` | Action plan from contract effects |

### Backward compatibility

- `query()` preserved unchanged
- `query_by_effects()` preserved unchanged
- `query_for_decision_agent()` enhanced: when `current_state` is provided, delegates to `select()`; otherwise falls back to `query()`
- `SkillBankAgent.select_skill()` added as the preferred pipeline-level API

---

## Files Changed

| File | Change |
|------|--------|
| `skill_bank/bank.py` | Added `compat_fn()`, `get_skill_names()`, `_effects_compat_score()` |
| `skill_bank/new_pool.py` | **NEW** — `NewPoolManager`, `NewPoolConfig`, `NewCandidate`, `ClusterSummary` |
| `infer_segmentation/config.py` | Added `ContractFeedbackConfig`, `contract_feedback` field in `SegmentationConfig`, `__post_init__` |
| `infer_segmentation/scorer.py` | Updated docstrings, added `boundary_scorer` param, `boundary_quality()`, boundary term in score/breakdown |
| `infer_segmentation/episode_adapter.py` | Thread `compat_fn` through `_build_scorer_from_preferences`, `infer_and_segment`, `infer_and_segment_offline` |
| `boundary_proposal/boundary_preference.py` | **NEW** — `BoundaryPreferenceScorer`, `BoundaryPreferenceConfig`, `BoundaryScore` |
| `boundary_proposal/__init__.py` | Export `BoundaryPreferenceScorer`, `BoundaryPreferenceConfig` |
| `query.py` | Added `SkillSelectionResult`, `select()`, `_compute_relevance()`, `_compute_applicability()`, `_compute_confidence()`, enhanced `query_for_decision_agent()` |
| `pipeline.py` | Added `contract_feedback_*`, `new_pool_*` config, `NewPoolManager` integration, `select_skill()`, `_materialize_legacy()` |
| `__init__.py` | Export `SkillSelectionResult`, `NewPoolManager`, `NewPoolConfig` |

## New Configs / Flags Summary

| Config | Location | Default | Purpose |
|--------|----------|---------|---------|
| `contract_feedback_mode` | `PipelineConfig` | `"off"` | Enable/disable contract feedback |
| `contract_feedback_strength` | `PipelineConfig` | `0.3` | Weight of contract compat term |
| `ContractFeedbackConfig.*` | `infer_segmentation/config.py` | various | Fine-grained feedback control |
| `new_pool_min_cluster_size` | `PipelineConfig` | `5` | Min cluster for NEW promotion |
| `new_pool_min_consistency` | `PipelineConfig` | `0.5` | Min pattern consistency |
| `new_pool_min_distinctiveness` | `PipelineConfig` | `0.25` | Min Jaccard distance from existing |
| `NewPoolConfig.*` | `skill_bank/new_pool.py` | various | Pool management thresholds |
| `BoundaryPreferenceConfig.*` | `boundary_proposal/boundary_preference.py` | `enabled=False` | Boundary scoring control |

## Why the Changes Match the Current Architecture

1. All new code follows the existing module boundaries (boundary_proposal, infer_segmentation, skill_bank, query).
2. No existing APIs were broken — all changes are additive or backward compatible.
3. New features are disabled by default (contract feedback = "off", boundary preference = disabled).
4. Config is via dataclasses consistent with the existing pattern.
5. The `_effects_compat_score` function reuses the same `SkillEffectsContract` schema from Stage 3 MVP.
6. `NewPoolManager` reuses Stage 3 `run_stage3_mvp` for verification, same as the old pipeline.

## TODOs for Future Research Extensions

- [ ] Extend `_effects_compat_score` to include precondition checking (when `SkillContract.pre` is available from full Stage 3)
- [ ] Extend `_effects_compat_score` to include invariant checking
- [ ] Add embedding-based clustering to `NewPoolManager` (hybrid effect + embedding features)
- [ ] Add LLM-based boundary preference collection (integrate with `llm_teacher.py`)
- [ ] Add boundary preference as a trainable component (neural boundary scorer)
- [ ] Add temporal context features to `BoundaryPreferenceScorer` (e.g., duration since last boundary)
- [ ] Evolve `SkillSelectionResult` to include expected duration, risk, and alternative skills
- [ ] Add online learning for the skill selection policy (update weights from execution outcomes)
- [ ] Persist `NewPoolManager` state across pipeline runs for incremental discovery
- [ ] Add A/B testing support: run with/without contract feedback and compare segmentation quality
