"""Phase-0 cross-eligibility probe (twenty_forty_eight ↔ tetris).

The dump driver feeds the actor's *retrieved* skill IDs as candidates. The
cold-start actor RAG was trained per-game, so cross-game skills never appear
in `retrieved_skill_ids` — meaning the dump driver cannot exhibit §22's
"filter admits cross-game skill" failure mode by itself.

This probe bypasses the RAG and feeds ALL skills in the fused bank as
candidates at every step, then tallies how many cross-game skills get
through the EligibilityFilter. It answers harness/README.md §22's
hypothesis empirically: with a single `applicable_domains=["gymv"]`
bucket, does the filter narrow cross-game?
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent))

from common.enums import SkillStatus  # noqa: E402
from harness import SkillHarness  # noqa: E402
from harness.adapter_registry import AdapterRegistry  # noqa: E402
from harness.adapters import GymvAdapter  # noqa: E402
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore  # noqa: E402
from skill_bank.stores import StoreName  # noqa: E402
from labeling_supplement._harness_io_helpers import (  # noqa: E402
    load_bank_records,
    parse_online_step,
    seed_lifecycle,
)

BANK_RUN = REPO / "labeling/skill_bank_out/run_20260430_030637/env_wrappers"
ACTIONS_RUN = REPO / "labeling/skill_actions_out/run_20260430_064325/env_wrappers"
GAMES = ("twenty_forty_eight", "tetris")
MAX_EPISODES = 3
MAX_STEPS = 50


def main() -> int:
    import tempfile

    temp_root = Path(tempfile.mkdtemp(prefix="phase0_xprobe_"))
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(temp_root / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(temp_root / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(temp_root / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(temp_root / "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)

    # Load + seed both games' banks into a single repo.
    # NOTE: cold-start `skill_id`s collide across games (e.g.
    # `INSPECT/SETUP` is in both tetris and super_mario). After
    # `safe_skill_id` (`/`→`__`) the IDs are still equal as strings, so
    # `skill_to_game[r.skill_id] = game` would silently overwrite. We
    # instead key the attribution table on `(skill_id, game)` and refuse
    # to seed a duplicate id from a second game (the lifecycle would
    # also raise, but raising late hides the cause).
    skill_to_game: Dict[str, str] = {}
    duplicate_ids: List[str] = []
    all_records: List[Any] = []
    for game in GAMES:
        recs = load_bank_records(BANK_RUN / game / "skill_bank.jsonl", default_domain="gymv")
        for r in recs:
            if r.skill_id in skill_to_game:
                duplicate_ids.append(f"{r.skill_id} [{skill_to_game[r.skill_id]} vs {game}]")
                continue
            skill_to_game[r.skill_id] = game
            all_records.append(r)
    if duplicate_ids:
        print(f"WARN: {len(duplicate_ids)} skill_id collision(s) across games; "
              f"second occurrence dropped: {duplicate_ids}")
    n_seeded, n_skipped = seed_lifecycle(lifecycle, all_records, promote_to=SkillStatus.PROVISIONAL)
    print(f"seeded {n_seeded} skills (skipped {n_skipped}); total in fused bank: {len(all_records)}")
    for r in all_records:
        print(f"  {r.skill_id:30s} game={skill_to_game[r.skill_id]:20s} type={r.skill_type.value:10s} status={r.status.value}")

    # Re-fetch from repo (status may have changed)
    skills_by_id: Dict[str, Any] = {}
    for r in all_records:
        cur = repo.get(r.skill_id)
        if cur is not None:
            skills_by_id[r.skill_id] = cur
    all_candidates = list(skills_by_id.values())

    registry = AdapterRegistry()
    registry.register(GymvAdapter())
    harness = SkillHarness(registry=registry)

    # For each game's actor episodes, run the filter with ALL skills as candidates
    results: Dict[str, Any] = {}
    for game in GAMES:
        actions_dir = ACTIONS_RUN / game
        eps = sorted(actions_dir.glob("episode_*.json"))[:MAX_EPISODES]
        same_game_eligible = Counter()
        cross_game_eligible = Counter()
        per_skill_eligibility: Counter = Counter()
        n_steps = 0
        for ep in eps:
            data = json.loads(ep.read_text())
            for step in (data.get("experiences") or [])[:MAX_STEPS]:
                try:
                    inputs = parse_online_step(step, fallback_domain="gymv")
                except Exception:
                    continue
                eligible = harness.select_eligible_skills(
                    candidates=all_candidates,
                    state=inputs.state,
                )
                same = sum(1 for es in eligible if skill_to_game[es.skill.skill_id] == game)
                cross = sum(1 for es in eligible if skill_to_game[es.skill.skill_id] != game)
                same_game_eligible[same] += 1
                cross_game_eligible[cross] += 1
                for es in eligible:
                    per_skill_eligibility[es.skill.skill_id] += 1
                n_steps += 1
        results[game] = {
            "n_steps": n_steps,
            "same_game_eligible_per_step_hist": dict(same_game_eligible),
            "cross_game_eligible_per_step_hist": dict(cross_game_eligible),
            "per_skill_eligibility_count": dict(per_skill_eligibility),
        }

    print("\n=== CROSS-ELIGIBILITY PROBE RESULT ===")
    for game, r in results.items():
        print(f"\n[{game}] {r['n_steps']} steps")
        print(f"  same-game eligible-count histogram : {r['same_game_eligible_per_step_hist']}")
        print(f"  CROSS-game eligible-count histogram: {r['cross_game_eligible_per_step_hist']}")
        print(f"  per-skill eligibility tallies:")
        for sid, n in sorted(r["per_skill_eligibility_count"].items(), key=lambda kv: -kv[1]):
            tag = "SAME" if skill_to_game[sid] == game else "CROSS"
            print(f"    [{tag:5s}] {sid:30s} eligible {n} steps")

    out_path = REPO / "labeling_supplement/harness_io_out/_phase0_cross_eligibility_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"games": GAMES, "results": results}, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
