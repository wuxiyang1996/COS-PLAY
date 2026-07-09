"""Pipeline Orchestrator — single top-level runner.

Spec: PLAN-PIPELINE-ORCHESTRATOR.

The orchestrator is the system control plane:
  - Drives outer-env episodes (`runner.EpisodeRunner`).
  - Persists artifacts via `artifact_store.ArtifactStore`.
  - Owns the gate via `gate_service.GateService` (composes the seven
    canonical stages).
  - Performs *atomic* promotion / rollback via `promotion_orchestrator
    .PromotionOrchestrator` (snapshots through `snapshot_manager`).
  - Caps resource usage per episode via `budget.BudgetController`.
  - Loads run-time configuration from `config.OrchestratorConfig`.

Public surface:

    from orchestrator import (
        ArtifactStore,
        BudgetController,
        EpisodeRunner,
        GateService,
        OrchestratorConfig,
        PromotionOrchestrator,
        SnapshotManager,
    )
"""

from orchestrator.artifact_store import ArtifactStore
from orchestrator.budget import BudgetController, BudgetExceeded
from orchestrator.config import (
    BudgetLimits,
    FewShotConfig,
    GateThresholds,
    JudgeConfig,
    OrchestratorConfig,
    TeacherConfig,
)
from orchestrator.eval_suite import (
    EvalSuite,
    EvalSuiteLoader,
    EvalSuiteSpec,
    Scoreboard,
    default_suites_root,
    load_eval_suite,
    load_eval_suite_spec,
    load_scoreboard,
)
from orchestrator.gate_service import GateService, NonRegressionResult
from orchestrator.promotion_orchestrator import (
    PromotionOrchestrator,
    PromotionPlan,
    PromotionResult,
)
from orchestrator.runner import EpisodeRunner, EpisodeResult
from orchestrator.snapshot_manager import SnapshotManager

__all__ = [
    "ArtifactStore",
    "BudgetController",
    "BudgetExceeded",
    "BudgetLimits",
    "EpisodeResult",
    "EpisodeRunner",
    "EvalSuite",
    "EvalSuiteLoader",
    "EvalSuiteSpec",
    "FewShotConfig",
    "GateService",
    "GateThresholds",
    "JudgeConfig",
    "NonRegressionResult",
    "OrchestratorConfig",
    "PromotionOrchestrator",
    "Scoreboard",
    "TeacherConfig",
    "PromotionPlan",
    "PromotionResult",
    "SnapshotManager",
    "default_suites_root",
    "load_eval_suite",
    "load_eval_suite_spec",
    "load_scoreboard",
]
