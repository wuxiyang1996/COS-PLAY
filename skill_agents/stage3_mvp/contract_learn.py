"""
Step 3 — Learn initial effects contract per skill.

Aggregates per-instance eff_add / eff_del / eff_event across all segment
instances for a given skill and keeps only predicates that appear with
frequency >= ``eff_freq``.  Caps contract size at ``max_effects_per_skill``.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from skill_agents.stage3_mvp.config import Stage3MVPConfig
from skill_agents.stage3_mvp.schemas import SegmentRecord, SkillEffectsContract


def _frequent_literals(
    counter: Counter,
    n_instances: int,
    freq_thresh: float,
    budget: int,
) -> Tuple[Set[str], Dict[str, int]]:
    """Keep literals with frequency >= *freq_thresh*, capped by *budget*.

    Returns the kept set and the support dict (count per literal).
    """
    qualifying = {
        lit: cnt
        for lit, cnt in counter.items()
        if cnt / n_instances >= freq_thresh
    }
    # Cap by highest support
    if len(qualifying) > budget:
        top = sorted(qualifying.items(), key=lambda x: -x[1])[:budget]
        qualifying = dict(top)
    return set(qualifying.keys()), qualifying


def learn_effects_contract(
    skill_id: str,
    instances: List[SegmentRecord],
    config: Stage3MVPConfig,
    prev_version: int = 0,
) -> SkillEffectsContract:
    """Build an initial ``SkillEffectsContract`` from segment instances.

    Parameters
    ----------
    skill_id : str
        The skill being contracted.
    instances : list[SegmentRecord]
        All segment records labelled with this skill (effects already computed).
    config : Stage3MVPConfig
    prev_version : int
        Previous contract version (new contract = prev_version + 1).

    Returns
    -------
    SkillEffectsContract
    """
    n = len(instances)

    add_counts: Counter = Counter()
    del_counts: Counter = Counter()
    event_counts: Counter = Counter()

    for rec in instances:
        add_counts.update(rec.eff_add)
        del_counts.update(rec.eff_del)
        event_counts.update(rec.eff_event)

    # Total budget is split across all three categories
    budget = config.max_effects_per_skill

    eff_add, sup_add = _frequent_literals(add_counts, n, config.eff_freq, budget)
    eff_del, sup_del = _frequent_literals(del_counts, n, config.eff_freq, budget)
    eff_event, sup_evt = _frequent_literals(event_counts, n, config.eff_freq, budget)

    # Enforce global cap
    total = len(eff_add) + len(eff_del) + len(eff_event)
    if total > budget:
        all_items = (
            [(lit, sup_add[lit], "add") for lit in eff_add]
            + [(lit, sup_del[lit], "del") for lit in eff_del]
            + [(lit, sup_evt[lit], "evt") for lit in eff_event]
        )
        all_items.sort(key=lambda x: -x[1])
        kept = all_items[:budget]
        eff_add = {lit for lit, _, cat in kept if cat == "add"}
        eff_del = {lit for lit, _, cat in kept if cat == "del"}
        eff_event = {lit for lit, _, cat in kept if cat == "evt"}
        sup_add = {k: v for k, v in sup_add.items() if k in eff_add}
        sup_del = {k: v for k, v in sup_del.items() if k in eff_del}
        sup_evt = {k: v for k, v in sup_evt.items() if k in eff_event}

    support = {}
    support.update(sup_add)
    support.update(sup_del)
    support.update(sup_evt)

    return SkillEffectsContract(
        skill_id=skill_id,
        version=prev_version + 1,
        eff_add=eff_add,
        eff_del=eff_del,
        eff_event=eff_event,
        support=support,
        n_instances=n,
    )
