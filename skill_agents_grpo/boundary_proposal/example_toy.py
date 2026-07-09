"""
Toy example: Stage 1 boundary proposal with synthetic signals.

Run from repo root: python -m skill_agents.boundary_proposal.example_toy
"""

import numpy as np
from skill_agents_grpo.boundary_proposal import (
    propose_boundary_candidates,
    ProposalConfig,
    candidate_centers_only,
    candidate_windows,
)


def main():
    T = 1000

    # Predicates: flip at 310 (room) and 530 (menu)
    predicates = []
    for t in range(T):
        predicates.append({
            "at_room_A": t < 310,
            "at_room_B": t >= 310,
            "menu_open": 530 <= t < 600,
        })

    # Surprisal: spike around 525
    np.random.seed(42)
    surprisal = np.abs(np.random.randn(T).cumsum())
    surprisal[520:535] += 5.0

    # Change-point: peak around 900
    changepoint_scores = np.random.rand(T) * 0.3
    changepoint_scores[895:910] = 0.8 + np.random.rand(15) * 0.2

    # Hard events
    event_times = [100, 905]

    config = ProposalConfig(
        merge_radius=5,
        window_half_width=2,
        surprisal_std_factor=2.0,
        changepoint_threshold=0.5,
        soft_max_per_minute=20,
    )

    candidates = propose_boundary_candidates(
        T,
        predicates=predicates,
        surprisal=surprisal,
        changepoint_scores=changepoint_scores,
        event_times=event_times,
        config=config,
        event_window=1,
    )

    print("Candidate cut points (center, half_window, source):")
    for c in candidates:
        print(f"  {c.center:4d}  ±{c.half_window}  {c.source}")

    centers = candidate_centers_only(candidates)
    print(f"\nCenters only (|C| = {len(centers)}): {centers}")

    windows = candidate_windows(candidates)
    print("\nWindows [start, end]:")
    for (a, b) in windows:
        print(f"  [{a}, {b}]")


if __name__ == "__main__":
    main()
