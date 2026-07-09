"""
Toy example: InferSegmentation with preference learning.

Demonstrates:
  1. Simulating LLM teacher rankings → pairwise preferences
  2. Training a PreferenceScorer from those preferences
  3. Decoding with the trained scorer (DP and beam)
  4. Inspecting diagnostics, uncertainty, and preference queries

No LLM or GPU required — uses simulated preferences.
"""

from skill_agents_grpo.infer_segmentation import (
    SegmentationConfig,
    ScorerWeights,
    DecoderConfig,
    NewSkillConfig,
    SegmentScorer,
    PreferenceStore,
    PreferenceScorer,
    PreferenceExample,
    viterbi_decode,
    beam_decode,
    generate_preference_queries,
    ranking_to_pairwise,
)


def make_toy_data():
    """Create a synthetic trajectory with 3 clear skill phases."""
    T = 30
    observations = [f"obs_{t}" for t in range(T)]

    # Phase 0-9: "move", Phase 10-19: "attack", Phase 20-29: "gather"
    actions = ["walk"] * 10 + ["strike"] * 10 + ["pick_up"] * 10

    candidates = [5, 9, 10, 15, 19, 20, 25]
    skill_names = ["move", "attack", "gather"]

    return T, observations, actions, candidates, skill_names


def simulate_teacher_rankings(segments, skill_names):
    """
    Simulate what the LLM teacher would return: skill rankings per segment.

    In the real pipeline, the LLM is prompted with observations/actions
    and returns a ranking like ["move", "attack", "gather"].
    """
    rankings = {
        (0, 9): ["move", "attack", "gather"],
        (0, 4): ["move", "gather", "attack"],
        (5, 9): ["move", "gather", "attack"],
        (10, 19): ["attack", "move", "gather"],
        (10, 14): ["attack", "move", "gather"],
        (15, 19): ["attack", "gather", "move"],
        (20, 29): ["gather", "move", "attack"],
        (20, 24): ["gather", "attack", "move"],
        (25, 29): ["gather", "move", "attack"],
        (5, 10): ["move", "attack", "gather"],
        (9, 10): ["move", "attack", "gather"],
    }
    return rankings


def simulate_transition_rankings(skill_names):
    """
    Simulate LLM transition rankings: after each skill, rank likely next skills.
    """
    return {
        "move": ["attack", "gather", "move"],
        "attack": ["gather", "move", "attack"],
        "gather": ["move", "attack", "gather"],
    }


def main():
    T, observations, actions, candidates, skill_names = make_toy_data()

    print("=" * 60)
    print("Step 1: Simulate LLM teacher rankings -> pairwise preferences")
    print("=" * 60)

    # Build segment list from candidates
    cut_indices = sorted(set([0] + candidates + [T - 1]))
    segments = [(cut_indices[i], cut_indices[i + 1]) for i in range(len(cut_indices) - 1)]

    # Simulate LLM rankings
    seg_rankings = simulate_teacher_rankings(segments, skill_names)
    trans_rankings = simulate_transition_rankings(skill_names)

    store = PreferenceStore()

    # Convert segment rankings to pairwise preferences
    for (start, end), ranking in seg_rankings.items():
        valid = [s for s in ranking if s in skill_names]
        if len(valid) >= 2:
            pairs = ranking_to_pairwise(valid, start, end, source="llm_sim")
            store.add_batch(pairs)

    # Convert transition rankings to pairwise preferences
    for prev_skill, ranking in trans_rankings.items():
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                store.add(PreferenceExample(
                    segment_start=-1, segment_end=-1,
                    skill_win=f"{prev_skill}->{ranking[i]}",
                    skill_lose=f"{prev_skill}->{ranking[j]}",
                    source="llm_sim",
                ))

    print(f"Collected {len(store)} pairwise preferences")
    print(f"  Segment prefs: {len(store.segment_preferences)}")
    print(f"  Transition prefs: {len(store.transition_preferences)}")

    # ── Step 2: Train scorer ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Train PreferenceScorer from preferences")
    print("=" * 60)

    pref_scorer = PreferenceScorer(skill_names, lr=0.1)
    losses = pref_scorer.train(store, epochs=30)
    print(f"Training losses: {[f'{l:.3f}' for l in losses[:5]]} ... {[f'{l:.3f}' for l in losses[-3:]]}")
    print(f"Learned skill scores: {pref_scorer.skill_scores}")
    print(f"Learned transition scores (sample): ", end="")
    trans = pref_scorer.transition_scores
    sample_keys = list(trans.keys())[:6]
    print({k: f"{trans[k]:.2f}" for k in sample_keys})

    # ── Step 3: Decode with trained scorer ──────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Viterbi DP decoding with preference-trained scorer")
    print("=" * 60)

    config = SegmentationConfig(
        weights=ScorerWeights(
            behavior_fit=1.0,
            duration_prior=0.3,
            transition_prior=1.0,
            contract_compat=0.0,
        ),
        new_skill=NewSkillConfig(enabled=True, penalty=8.0),
        decoder=DecoderConfig(top_k_diagnostics=3, beam_width=8),
        method="dp",
    )

    scorer = SegmentScorer(
        skill_names=skill_names,
        config=config,
        behavior_fit_fn=pref_scorer.behavior_fit,
        transition_fn=pref_scorer.transition_prior,
    )

    result = viterbi_decode(
        candidates, T, scorer, observations, actions, config=config,
    )

    print(f"Total score: {result.total_score:.2f}")
    print(f"Skill sequence: {result.skill_sequence}")
    for seg in result.segments:
        print(f"  [{seg.start:3d} - {seg.end:3d}]  skill={seg.assigned_skill:10s}  "
              f"margin={seg.margin:.2f}  uncertain={seg.is_uncertain}")

    # ── Step 4: Beam search ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Beam search decoding")
    print("=" * 60)

    result_beam = beam_decode(
        candidates, T, scorer, observations, actions, config=config,
    )

    print(f"Total score: {result_beam.total_score:.2f}")
    print(f"Skill sequence: {result_beam.skill_sequence}")

    # ── Step 5: Preference queries ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Uncertain segments for active learning")
    print("=" * 60)

    queries = generate_preference_queries(result, margin_threshold=5.0)
    if queries:
        for q in queries:
            print(f"  [{q.segment_start}-{q.segment_end}]: "
                  f"{q.candidate_a} vs {q.candidate_b}  margin={q.margin:.2f}")
    else:
        print("  No uncertain segments (all margins are high)")

    print("\nDone.")


if __name__ == "__main__":
    main()
