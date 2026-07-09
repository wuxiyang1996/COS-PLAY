"""P0 extension records used by Harness, Orchestrator, and Crafter.

Per `plans/00-overview/PLAN-EXTENSION.md` and PLAN-COMPONENTS-IMPLEMENTATION
§8 (P0 step), these records *extend* the existing
`data_structure/experience.py` types — they do NOT replace them.

Existing `Experience` / `Episode` / `SubTask_Experience` continue to be
the per-tick / per-rollout / per-segment ground truth. The records below
are higher-level artifacts that close the loop across components:

  - `SkillEpisode`           : one full skill invocation by the Harness
  - `SkillRecord`            : the bank entry being evaluated / promoted
  - `SkillEvaluationRecord`  : the gate's signed verdict
  - `GateVerdictPayload`     : per-stage breakdown attached to the eval
  - `BankMutationProposal`   : a typed crafter proposal (compose / hyp / ...)
  - `FailureTrace`           : a single observed failure of a skill / step
  - `RunRelease`             : a frozen snapshot of (bank ⊕ adapter) ⊕ config
"""

from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    MergeProposal,
    PatchProposal,
    RetireProposal,
    RewriteProposal,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace
from data_structure.extensions.gate_verdict import (
    GateVerdictPayload,
    StageVerdict,
)
from data_structure.extensions.run_release import RunRelease
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillRecord, SkillContract

__all__ = [
    "BankMutationProposal",
    "ComposeProposal",
    "EpisodeReflection",
    "FailureDiagnosis",
    "FailureTrace",
    "GateVerdictPayload",
    "GeneralizeProposal",
    "HypothesisProposal",
    "MergeProposal",
    "PatchProposal",
    "RetireProposal",
    "RewriteProposal",
    "RunRelease",
    "SkillContract",
    "SkillEpisode",
    "SkillEpisodeOutcome",
    "SkillEpisodeStep",
    "SkillEvaluationRecord",
    "SkillRecord",
    "StageVerdict",
]
