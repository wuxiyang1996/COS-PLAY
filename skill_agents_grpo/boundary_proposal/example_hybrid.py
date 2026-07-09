"""
Walkthrough: how the hybrid (LLM + rule-based) extractor works.

This example simulates a 12-step Overcooked trajectory and shows exactly
what each layer (LLM predicates vs rule-based events) contributes to the
final boundary candidates.

Run from repo root:
    python -m skill_agents.boundary_proposal.example_hybrid

No real LLM call is made — we mock ask_model to show the data flow.
"""

import json
import numpy as np

# -- 1.  Simulate an Overcooked trajectory -----------------------------------
#
# 12 timesteps of natural-language state descriptions, rewards, and done flags.
# We use simple objects with .state, .action, .reward, .next_state, .done
# to avoid importing the full data_structure module.


class FakeExperience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.summary = None
        self.summary_state = None
        self.sub_tasks = None
        self.sub_task_done = None
        self.idx = None


class FakeEpisode:
    def __init__(self, experiences, task):
        self.experiences = experiences
        self.task = task


# A realistic Overcooked trajectory
experiences = [
    FakeExperience(
        "Agent 0 at (2,1) facing east, holding nothing. Pot at (3,0) is empty. No orders pending.",
        "move east", 0.0,
        "Agent 0 at (3,1) facing east, holding nothing. Pot at (3,0) is empty.", False),
    FakeExperience(
        "Agent 0 at (3,1) facing east, holding nothing. Pot at (3,0) is empty.",
        "move north", 0.0,
        "Agent 0 at (3,0) facing north, holding nothing. Near pot.", False),
    FakeExperience(
        "Agent 0 at (3,0) facing north, holding nothing. Near onion dispenser.",
        "interact", 0.0,
        "Agent 0 at (3,0) facing north, holding onion.", False),
    # --- t=3: picked up onion (inventory change) ---
    FakeExperience(
        "Agent 0 at (3,0) facing north, holding onion. Pot is empty.",
        "move south", 0.0,
        "Agent 0 at (3,1) facing south, holding onion.", False),
    FakeExperience(
        "Agent 0 at (3,1) facing south, holding onion. Near pot.",
        "interact", 0.0,
        "Agent 0 at (3,1) facing south, holding nothing. Pot has 1 onion.", False),
    # --- t=5: dropped onion in pot ---
    FakeExperience(
        "Agent 0 at (3,1) facing south, holding nothing. Pot has 1 onion.",
        "move west", 0.0,
        "Agent 0 at (2,1) facing west. Pot has 1 onion.", False),
    FakeExperience(
        "Agent 0 at (2,1) facing west. Pot has 3 onions, cooking.",
        "stay", 0.0,
        "Agent 0 at (2,1). Pot has 3 onions, cooking.", False),
    FakeExperience(
        "Agent 0 at (2,1). Pot has 3 onions, soup is ready!",
        "move east", 0.0,
        "Agent 0 at (3,1). Soup is ready in pot.", False),
    # --- t=8: soup became ready ---
    FakeExperience(
        "Agent 0 at (3,1). Soup is ready in pot.",
        "interact", 0.0,
        "Agent 0 at (3,1), holding soup. Pot is empty.", False),
    FakeExperience(
        "Agent 0 at (3,1), holding soup. Serving counter at (1,3).",
        "move south", 0.0,
        "Agent 0 at (3,2), holding soup.", False),
    FakeExperience(
        "Agent 0 at (3,2), holding soup. Near serving counter.",
        "interact", 20.0,
        "Agent 0 at (3,2), holding nothing. Soup delivered! +20 reward.", False),
    # --- t=11: soup delivered, +20 reward spike ---
    FakeExperience(
        "Agent 0 at (3,2), holding nothing. Order complete.",
        "stay", 0.0,
        "Agent 0 at (3,2), holding nothing.", True),
]

episode = FakeEpisode(experiences, task="deliver onion soup")


# -- 2.  Mock the LLM call ---------------------------------------------------
#
# In production, ask_model sends the state strings to GPT/Claude/Gemini.
# Here we return what the LLM *would* return: structured predicates.

def mock_ask_model(prompt, **kwargs):
    """Simulate what the LLM returns for predicate extraction."""
    # The LLM reads the NL states and extracts structured facts
    # A well-prompted LLM focuses on SEMANTICALLY MEANINGFUL facts:
    #   - holding: what the agent carries (changes = inventory event)
    #   - pot_status: cooking progress (changes = workflow milestone)
    #   - objective: current sub-goal (changes = task switch)
    # It does NOT include trivial per-step changes like grid position.
    mock_predicates = [
        {"holding": "nothing", "pot_status": "empty",   "objective": "get_ingredient"},
        {"holding": "nothing", "pot_status": "empty",   "objective": "get_ingredient"},
        {"holding": "nothing", "pot_status": "empty",   "objective": "get_ingredient"},
        {"holding": "onion",   "pot_status": "empty",   "objective": "deliver_to_pot"},    # <- flip: picked up onion
        {"holding": "onion",   "pot_status": "empty",   "objective": "deliver_to_pot"},
        {"holding": "nothing", "pot_status": "has_onion","objective": "fill_pot"},          # <- flip: put onion in pot
        {"holding": "nothing", "pot_status": "has_onion","objective": "fill_pot"},
        {"holding": "nothing", "pot_status": "cooking", "objective": "wait_for_soup"},     # <- flip: pot started cooking
        {"holding": "nothing", "pot_status": "ready",   "objective": "pick_up_soup"},      # <- flip: soup ready
        {"holding": "soup",    "pot_status": "empty",   "objective": "deliver_soup"},      # <- flip: grabbed soup
        {"holding": "soup",    "pot_status": "empty",   "objective": "deliver_soup"},
        {"holding": "nothing", "pot_status": "empty",   "objective": "order_complete"},    # <- flip: delivered soup
    ]
    return json.dumps(mock_predicates)


# -- 3.  Run the hybrid extractor ---------------------------------------------

from skill_agents_grpo.boundary_proposal.signal_extractors import get_signal_extractor

hybrid = get_signal_extractor(
    "llm+overcooked",
    ask_model_fn=mock_ask_model,
    model="mock",
    chunk_size=30,  # all 12 states fit in one chunk
)

# LAYER 1: LLM extracts predicates
predicates = hybrid.extract_predicates(experiences)

# LAYER 2: Rule-based extracts hard events
event_times = hybrid.extract_event_times(experiences)

print("=" * 70)
print("HYBRID EXTRACTOR WALKTHROUGH: Overcooked")
print("=" * 70)

print("\n-- Layer 1: LLM-extracted predicates --")
print("(LLM reads NL states, returns structured facts)")
for t, p in enumerate(predicates):
    print(f"  t={t:2d}: {p}")

print("\n-- Layer 1: Predicate FLIPS detected --")
print("(Flip = any predicate key changed from t-1 to t)")
for t in range(1, len(predicates)):
    prev, curr = predicates[t - 1], predicates[t]
    changes = {k: (prev.get(k), curr.get(k)) for k in set(prev) | set(curr) if prev.get(k) != curr.get(k)}
    if changes:
        print(f"  t={t:2d}: {changes}")

print("\n-- Layer 2: Rule-based hard events --")
print("(Overcooked rules: done flags + any positive reward)")
for t in event_times:
    reason = []
    if experiences[t].done:
        reason.append("done=True")
    r = experiences[t].reward
    if isinstance(r, (int, float)) and r > 0:
        reason.append(f"reward={r}")
    print(f"  t={t:2d}: {', '.join(reason)}")


# -- 4.  Run full boundary proposal -------------------------------------------

from skill_agents_grpo.boundary_proposal import ProposalConfig
from skill_agents_grpo.boundary_proposal.proposal import propose_boundary_candidates, candidate_centers_only

config = ProposalConfig(merge_radius=1, window_half_width=1)

candidates = propose_boundary_candidates(
    T=len(experiences),
    predicates=predicates,
    event_times=event_times,
    config=config,
)

print("\n-- Combined: Boundary candidates --")
print("(Predicate flips + hard events, merged within radius=2)")
for c in candidates:
    print(f"  center={c.center:2d}  ±{c.half_window}  source={c.source}")

centers = candidate_centers_only(candidates)
print(f"\n  Final C = {centers}  (|C|={len(centers)} out of T={len(experiences)})")


# -- 5.  Show what each source contributed ------------------------------------

print("\n-- Attribution: what caught each boundary --")
for c in candidates:
    t = c.center
    reasons = []
    if "predicate" in c.source:
        # Show nearby predicate flips (within the merge window)
        for tt in range(max(0, t - 1), min(len(predicates), t + 2)):
            if tt > 0:
                prev_p = predicates[tt - 1]
                curr_p = predicates[tt]
                changes = {k: f"{prev_p.get(k)} -> {curr_p.get(k)}"
                           for k in set(prev_p) | set(curr_p) if prev_p.get(k) != curr_p.get(k)}
                if changes:
                    reasons.append(f"LLM flip at t={tt}: {changes}")
    if "event" in c.source:
        reasons.append(f"Rule-based event (reward={experiences[t].reward}, done={experiences[t].done})")
    for r in reasons:
        print(f"  center={t}: {r}")


print("\n-- Interpretation --")
print("""
  t=3:  Picked up onion           (LLM: holding nothing -> onion, objective -> deliver_to_pot)
  t=5:  Put onion in pot          (LLM: holding onion -> nothing, pot empty -> has_onion)
  t=7:  Pot started cooking       (LLM: pot has_onion -> cooking, objective -> wait_for_soup)
  t=8:  Soup became ready         (LLM: pot cooking -> ready, objective -> pick_up_soup)
  t=9:  Grabbed soup from pot     (LLM: holding nothing -> soup, pot ready -> empty)
  t=10: Delivered soup! +20       (Rule: reward=20.0)
  t=11: Episode ends              (LLM: holding soup -> nothing, objective -> order_complete)
                                  (Rule: done=True)

  The RULE-BASED layer catches t=10 and t=11 for FREE (reward spike + done).
  The LLM layer catches SEMANTIC transitions (t=3,5,7,8,9) that no
  keyword list could find without per-env hardcoding.
  Together they give complete boundary coverage.
""")
