# Stage 1: Boundary Proposal (High-Recall Candidate Cut Points)

## Why Stage 1?

**Stage 1 is a cheap, high-recall boundary proposal step.** It does **not** decide the final segmentation. It only produces a set of **candidate cut points** \( C \) so that later DP/HSMM/beam search considers cuts **only** at those times (or in small windows around them).

### Complexity

- If boundaries are allowed at **every** timestep, segmentation search is **O(T²)** and blows up for long trajectories.
- If boundaries are restricted to **C** with **|C| ≪ T**, search becomes **O(|C|²)** and is tractable—as long as **C** contains (almost) all true boundaries.

So Stage 1 aims for **high recall**: miss as few true boundaries as possible, while keeping **|C|** small enough for downstream search.

---

## Integration with the Agentic Framework

This module plugs directly into the framework's data structures:

| Framework concept | Boundary proposal role |
|---|---|
| `Episode` | Input trajectory to segment |
| `Experience` | Per-timestep observation; signals are extracted from `.state`, `.reward`, `.done` |
| `SubTask_Experience` | Output segments produced by `segment_episode()` |
| `Episode.separate_into_sub_episodes()` | Can be augmented via `annotate_episode_boundaries()` which marks cuts on Experience objects |
| RAG `TextEmbedder` | Optional: used to compute embedding change-point scores from experience summaries |
| `ask_model()` | LLM-based predicate extraction (general, adaptive) |
| Environment wrappers | Per-env signal extractors know how to read Overcooked/Avalon/Diplomacy state dicts |

### Where it fits in the pipeline

```
Raw Episode (from rollout or external demo)
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: Boundary Proposal     │  ◄── this module
│  (extract signals → propose C)  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 2: Segmentation          │  DP / HSMM / beam search
│  (only place cuts at C)         │  (downstream; not in this module)
└───────────────┬─────────────────┘
                │
                ▼
    list[SubTask_Experience]
    (labeled sub-trajectories for training / RAG)
```

---

## Signal Extraction: Three Strategies

The `env_name` parameter controls how predicates are extracted from Experience states.

### 1. Rule-based (legacy, per-env)

```python
env_name="overcooked"   # or "avalon", "diplomacy", "generic"
```

Fast (zero LLM cost), but uses hardcoded keyword matching on NL state strings. Brittle — breaks if wrapper phrasing changes. Good only when states are structured dicts with known keys.

### 2. LLM-based (fully general) — recommended for new environments

```python
env_name="llm"
```

Uses `ask_model()` to extract structured predicates from NL state descriptions. **Environment-agnostic** — works with any game, any wrapper, without writing per-env code.

How it works:
- States are batched into chunks (default 30 states per LLM call)
- Each call asks the LLM to return a JSON array of predicate dicts
- The LLM identifies what's important: location, inventory, phase, objectives, etc.
- Predicate keys are normalized across the trajectory for consistent flip detection
- A T=1000 trajectory needs ~33 cheap LLM calls, not 1000

### 3. Hybrid (recommended for production)

```python
env_name="llm+overcooked"   # or "llm+avalon", "llm+diplomacy"
```

Combines the best of both:
- **Predicates** from LLM (general, adaptive, no brittle keywords)
- **Hard events** from rule-based per-env extractor (free, reliable: done flags, phase transitions, reward spikes)

This is the recommended strategy because hard events are structural (no interpretation needed) while predicates require understanding NL state descriptions.

### Comparison

| Strategy | Predicates | Hard events | Cost | Generality |
|---|---|---|---|---|
| Rule-based | Keyword matching | Per-env rules | Free | Per-env only |
| LLM | LLM extraction | Reward spikes + done | ~33 LLM calls/1000 steps | Any environment |
| Hybrid | LLM extraction | Per-env rules | ~33 LLM calls/1000 steps | Any env + env-specific events |

---

## Quick Start (Framework Usage)

### 1. LLM-based segmentation (general, any environment)

```python
from skill_agents.boundary_proposal import segment_episode, ProposalConfig

sub_episodes = segment_episode(
    episode,
    env_name="llm",
    config=ProposalConfig(merge_radius=5),
    extractor_kwargs={
        "model": "gpt-4o-mini",       # cheap model for Stage 1
        "chunk_size": 30,              # states per LLM call
        "filter_significance": True,   # 2nd pass: filter noisy changes
    },
)
```

### 2. Hybrid segmentation (recommended for known environments)

```python
sub_episodes = segment_episode(
    episode,
    env_name="llm+overcooked",
    config=ProposalConfig(merge_radius=5),
    extractor_kwargs={"model": "gemini-2.5-flash"},
)
```

### 3. Rule-based segmentation (fast, no LLM cost)

```python
sub_episodes = segment_episode(
    episode,
    env_name="overcooked",
    config=ProposalConfig(merge_radius=5),
)
```

### 4. Annotate boundaries on existing Episodes

If you prefer using `Episode.separate_into_sub_episodes()`, call `annotate_episode_boundaries()` first. It sets `sub_tasks` labels and `sub_task_done` flags on each `Experience` so the existing method segments correctly:

```python
from skill_agents.boundary_proposal import annotate_episode_boundaries

candidates = annotate_episode_boundaries(
    episode,
    env_name="llm+avalon",
    extractor_kwargs={"model": "gpt-4o-mini"},
)
sub_episodes = episode.separate_into_sub_episodes(outcome_length=5)
```

### 5. Use with RAG embeddings (change-point detection)

If experience summaries exist, pass the RAG `TextEmbedder` to add embedding-based change-point scores:

```python
from rag.embedding.text_embedder import get_text_embedder
from skill_agents.boundary_proposal import segment_episode

embedder = get_text_embedder()  # Qwen3-Embedding-0.6B

sub_episodes = segment_episode(
    episode,
    env_name="llm+overcooked",
    embedder=embedder,
    changepoint_method="cusum",
    extractor_kwargs={"model": "gpt-4o-mini"},
)
```

### 6. Use with action surprisal

```python
import numpy as np
from skill_agents.boundary_proposal import segment_episode

surprisal = np.array([...])  # -log p(a_t | x_t) from your behavior model

sub_episodes = segment_episode(
    episode,
    env_name="llm+diplomacy",
    surprisal=surprisal,
    extractor_kwargs={"model": "gpt-4o-mini", "controlled_power": "FRANCE"},
)
```

### 7. Low-level: just get boundary candidates

```python
from skill_agents.boundary_proposal import propose_from_episode, candidate_centers_only

candidates = propose_from_episode(
    episode,
    env_name="llm",
    extractor_kwargs={"model": "gpt-4o-mini"},
)
centers = candidate_centers_only(candidates)
print(f"|C| = {len(centers)} out of T = {len(episode.experiences)}")
```

---

## LLM Extractor Details

### How batched predicate extraction works

```
Trajectory: [exp_0, exp_1, ..., exp_999]
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Chunk 0      Chunk 1     Chunk 33
    [t=0..29]    [t=30..59]  [t=970..999]
          │           │           │
     LLM call     LLM call    LLM call
     "Extract     "Extract    "Extract
      predicates"  predicates"  predicates"
          │           │           │
     JSON array   JSON array  JSON array
     of 30 dicts  of 30 dicts of 30 dicts
          │           │           │
          └───────────┼───────────┘
                      ▼
          Normalize keys across T
                      │
                      ▼
          list[dict] predicates (T=1000)
```

### The LLM prompt asks for

- Location / area the agent is in
- Items held, inventory changes
- Game phase, menu/UI mode
- Objectives completed or active
- Interaction targets (NPCs, objects)
- Agent status (alive, health, role)
- Team composition or alliances

### Optional: significance filtering (2nd pass)

When `filter_significance=True`, a second LLM call examines each predicate *change* and judges whether it's boundary-significant. This reduces false positives (e.g., routine score increments) at the cost of one extra LLM call.

### Fallback behavior

If an LLM call fails (API error, malformed JSON), that chunk gets empty predicates. Hard events (done, reward spikes) still fire from the rule-based layer, so the pipeline degrades gracefully.

---

## Configuration

```python
from skill_agents.boundary_proposal import ProposalConfig

config = ProposalConfig(
    # Merge and window
    merge_radius=5,              # merge candidates within this many timesteps
    window_half_width=2,         # each candidate spans [center-2, center+2]

    # Surprisal
    surprisal_std_factor=2.0,    # threshold = mean + 2*std
    surprisal_local_radius=3,    # local max search radius
    surprisal_delta_threshold=None,  # optional: sharp delta threshold

    # Change-point (embedding)
    changepoint_threshold=None,  # score threshold (None = all local maxima)
    changepoint_top_k_per_minute=None,  # top-K per minute cap
    changepoint_local_radius=5,
    steps_per_minute=60,

    # Density control
    soft_max_per_minute=20,      # cap soft signals (surprisal, changepoint) per minute
)
```

### LLM extractor kwargs

Passed via `extractor_kwargs` in `segment_episode()` etc:

| Kwarg | Default | Description |
|---|---|---|
| `model` | `"gpt-4o"` | LLM model name (cheap models like `gpt-4o-mini` or `gemini-2.5-flash` recommended) |
| `chunk_size` | `30` | States per LLM call |
| `temperature` | `0.2` | Low for structured JSON output |
| `filter_significance` | `False` | Enable 2nd-pass significance filtering |
| `max_state_chars` | `500` | Truncate long state strings |
| `ask_model_fn` | `ask_model` from `API_func` | Custom LLM call function |

---

## Boundary Preference Learning

The `BoundaryPreferenceScorer` provides lightweight plausibility evaluation for candidate cut points. It can be used in two ways:

1. **Stage 1 filtering:** Remove low-quality candidates before decoding.
2. **Stage 2 scoring term:** Add a `boundary_quality` bonus/penalty in the `SegmentScorer`.

### How it works

The scorer combines three signals for each candidate timestep:

| Signal | Source | Description |
|--------|--------|-------------|
| **Signal strength** | Stage 1 candidates | How many distinct signals support the boundary (predicate flip, reward spike, done flag, change-point, surprisal) |
| **Predicate discontinuity** | Predicate data | Fraction of predicates that change value at the boundary |
| **Learned preference** | Pairwise feedback | Bradley-Terry score from human or LLM pairwise preferences on boundary quality |

The composite plausibility score is a weighted combination of these three signals.

### Usage

```python
from skill_agents.boundary_proposal import BoundaryPreferenceScorer, BoundaryPreferenceConfig

scorer = BoundaryPreferenceScorer(config=BoundaryPreferenceConfig(
    signal_strength_weight=1.0,
    predicate_weight=1.0,
    learned_weight=0.5,
    min_plausibility=0.2,
))

# Register Stage 1 candidates and predicate data
scorer.set_candidates(candidates)
scorer.set_predicates(predicates)

# Optional: add pairwise preferences (from LLM or human)
scorer.add_preference(t_win=42, t_lose=38)

# Filter candidates (Stage 1 integration)
filtered = scorer.filter_candidates(candidates, top_frac=0.8)

# Score a segment's boundaries (Stage 2 integration)
bonus = scorer.decoding_bonus(seg_start=10, seg_end=25)
```

### Configuration

```python
from skill_agents.boundary_proposal import BoundaryPreferenceConfig

config = BoundaryPreferenceConfig(
    signal_strength_weight=1.0,   # Weight for number of supporting signals
    predicate_weight=1.0,         # Weight for predicate discontinuity
    learned_weight=0.5,           # Weight for learned preference (Bradley-Terry)
    learning_rate=0.1,            # LR for preference learning
    min_plausibility=0.2,         # Floor for filter_candidates
)
```

---

## Module layout

```
skill_agents/boundary_proposal/
├── README.md              # This file
├── __init__.py            # Public API (framework-integrated + low-level)
├── proposal.py            # Core trigger generation, merge/window, density control
├── changepoint.py         # CUSUM / sliding-window change-point from embeddings
├── signal_extractors.py   # Factory + rule-based extractors + HybridSignalExtractor
├── llm_extractor.py       # LLM-based predicate extraction (batched, JSON output)
├── episode_adapter.py     # Episode → signals → candidates → SubTask_Experience
├── boundary_preference.py # BoundaryPreferenceScorer: plausibility scoring for cut points
├── example_toy.py         # Standalone toy example (no LLM needed)
└── requirements.txt       # numpy
```

### Key functions

| Function / Class | Module | Purpose |
|---|---|---|
| `segment_episode()` | `episode_adapter` | Full pipeline: Episode → SubTask_Experience list |
| `propose_from_episode()` | `episode_adapter` | Episode → BoundaryCandidate list |
| `annotate_episode_boundaries()` | `episode_adapter` | Mark boundaries on Experiences for `separate_into_sub_episodes()` |
| `extract_signals()` | `episode_adapter` | Episode → predicates, events, changepoint_scores |
| `propose_boundary_candidates()` | `proposal` | Raw arrays → BoundaryCandidate list |
| `compute_changepoint_scores()` | `changepoint` | Embeddings → change-point score array |
| `get_signal_extractor()` | `signal_extractors` | Factory: rule-based / LLM / hybrid |
| `BoundaryPreferenceScorer` | `boundary_preference` | Plausibility scoring for cut points (signal strength + predicate discontinuity + learned preference) |
| `BoundaryPreferenceConfig` | `boundary_preference` | Configuration for boundary preference scoring |

---

## Toy Example (standalone, no LLM)

```bash
python -m skill_agents.boundary_proposal.example_toy
```

Runs on synthetic data with predicate flips, surprisal spikes, change-point peaks, and hard events.
