"""
Step 5 — Refine an effects contract by removing unreliable literals.

Simple deterministic rule: drop any literal whose per-instance success rate
is below ``config.eff_freq``.  This is the MVP refinement — no splitting,
no precondition learning.
"""

from __future__ import annotations

from skill_agents.stage3_mvp.config import Stage3MVPConfig
from skill_agents.stage3_mvp.schemas import SkillEffectsContract, VerificationReport


def refine_effects_contract(
    contract: SkillEffectsContract,
    report: VerificationReport,
    config: Stage3MVPConfig,
) -> SkillEffectsContract:
    """Drop unreliable literals and return a new, refined contract.

    Parameters
    ----------
    contract : SkillEffectsContract
        The current contract (not mutated).
    report : VerificationReport
        Verification results from ``verify_effects_contract``.
    config : Stage3MVPConfig

    Returns
    -------
    SkillEffectsContract
        A new contract with version bumped, unreliable literals removed.
    """
    thresh = config.eff_freq

    refined_add = {
        p for p in contract.eff_add
        if report.eff_add_success_rate.get(p, 0.0) >= thresh
    }
    refined_del = {
        p for p in contract.eff_del
        if report.eff_del_success_rate.get(p, 0.0) >= thresh
    }
    refined_event = {
        e for e in contract.eff_event
        if report.eff_event_rate.get(e, 0.0) >= thresh
    }

    refined_support = {
        k: v for k, v in contract.support.items()
        if k in refined_add or k in refined_del or k in refined_event
    }

    return SkillEffectsContract(
        skill_id=contract.skill_id,
        version=contract.version + 1,
        name=contract.name,
        description=contract.description,
        eff_add=refined_add,
        eff_del=refined_del,
        eff_event=refined_event,
        support=refined_support,
        n_instances=contract.n_instances,
    )
