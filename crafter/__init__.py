"""Skill Crafter — slow-timescale proposal layer.

Spec: PLAN-SKILL-CRAFTER, PLAN-COMPONENTS-IMPLEMENTATION §4 (Phase C),
``implementation_notes/legacy/crafter-harness-orchestrator-roles.md`` §"Two-tier
trigger model".

The crafter is the *creative* layer of the system: it proposes new skills
(compositions, generalizations, novel hypotheses) and patches to existing
ones (failure repairs, retirements). Every proposal is typed
(`BankMutationProposal`) and is *only* a proposal — it lands in the
draft store and must pass the unified gate to be promoted.

Trigger surfaces (PLAN-SKILL-CRAFTER §6.4 + implementation note):

  * :meth:`SkillCrafterService.reflect_on_episode` — per-episode reactive
    pass, fired immediately after the Skill Bank Agent finishes one
    episode. Runs Failure-Reflector (threshold=1), per-episode
    Hypothesizer fall-through, and subsumption-retire detection over
    the freshly-minted candidate skills the Bank Agent just produced.
  * :meth:`SkillCrafterService.cycle` — per-batch reflective pass,
    fired every K episodes by the orchestrator. Runs the same dispatch
    with the configured ``hot_pattern_threshold`` (default 3).
    Composer / Generalizer belong here (they require multi-episode
    statistics).

Architectural rules (mechanically enforced; see invariant tests):
  * The crafter never imports `skill_bank.stores` directly.
  * The crafter never holds a `SkillLifecycleManager` reference outside
    `crafter.service.SkillCrafterService`.
  * The crafter writes proposals via `ArtifactStore.put_proposal`
    and ingests draft records via `SkillLifecycleManager.ingest_draft`
    *only* through `crafter.service.SkillCrafterService`.
  * Component proposers (Composer / Generalizer / Hypothesizer /
    Repairer) never re-fetch the bank — when they need cross-store
    visibility, the service builds a frozen :class:`BankView` and hands
    it in as a parameter.

Public surface:

    from crafter import (
        SkillCrafterService,
        BankView,
        Composer,
        Generalizer,
        Hypothesizer,
        Repairer,
        FailureDiagnoser,
        FailureMemory,
        FailurePattern,
    )

    # Per-episode bundle accepted by `SkillCrafterService.reflect_on_episode`:
    from data_structure.extensions import EpisodeReflection
"""

from crafter._bank_view import BankView
from crafter.composer import Composer
from crafter.failure_diagnoser import FailureDiagnoser
from crafter.failure_memory import FailureMemory, FailurePattern
from crafter.generalizer import Generalizer
from crafter.hypothesizer import Hypothesizer
from crafter.repairer import Repairer
from crafter.service import CrafterCycleResult, SkillCrafterService

__all__ = [
    "BankView",
    "Composer",
    "CrafterCycleResult",
    "FailureDiagnoser",
    "FailureMemory",
    "FailurePattern",
    "Generalizer",
    "Hypothesizer",
    "Repairer",
    "SkillCrafterService",
]
