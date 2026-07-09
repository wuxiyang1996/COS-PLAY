"""Single source of truth for the project's backbone LLM models.

Decision (2026-04-28, judge revised 2026-05-03): the project ships a
**three-tier backbone stack** with judge consolidated onto the local
35B-A3B teacher.

* **Actor + Skill-Bank** — ``Qwen/Qwen3.5-9B`` (LoRA-trained policy + the
  GRPO-trained ``segment`` / ``contract`` / ``curator`` skill-bank
  adapters).  Lives behind ``BACKBONE_MODEL``.
* **Crafter / Harness / Orchestrator + LLM-as-judge** —
  ``Qwen/Qwen3.5-35B-A3B`` (frozen 35B-total / 3B-active MoE
  control-plane backbone, served separately via
  ``inference/serve_qwen35_35b_a3b.sh``). Lives behind
  ``BACKBONE_TEACHER_MODEL`` (control-plane role) and
  ``BACKBONE_JUDGE_MODEL`` (eval-driver / promotion-gate role).
  Same weights, two roles — saves a GPU group + eliminates judge API
  spend at the cost of a within-Qwen-family bias the spot-check
  protocol below covers.
* **SFT data generation** — ``gpt-5.5`` (frontier teacher used to label
  cold-start data consumed by the SFT trainer). Lives behind
  ``BACKBONE_SFT_TEACHER_MODEL``. Kept on the frontier model because
  cold-start labels are baked once into the SFT adapters and never
  re-run during training, so paying API cost once for a stronger
  teacher is the right trade.

Mapping summary
---------------

============================ =========================== =====================================
Symbol                       Default                     Used by
============================ =========================== =====================================
``BACKBONE_MODEL``           ``Qwen/Qwen3.5-9B``         Actor (``decision_agents``) +
                                                         Skill-Bank (``skill_agents``) — the
                                                         LoRA-trained policy.
``BACKBONE_TEACHER_MODEL``   ``Qwen/Qwen3.5-35B-A3B``    Skill Crafter teacher (``crafter``),
                                                         Skill Harness control logic
                                                         (``harness``), Pipeline Orchestrator
                                                         (``orchestrator``).
``BACKBONE_JUDGE_MODEL``     ``Qwen/Qwen3.5-35B-A3B``    LLM-as-judge for skill evaluation
                                                         (``skill_evaluation``) and the eval
                                                         driver promotion gates (E0–E2).
                                                         Shares weights with
                                                         ``BACKBONE_TEACHER_MODEL`` — same
                                                         vLLM server, different role.
``BACKBONE_SFT_TEACHER_MODEL`` ``gpt-5.5``               SFT cold-start data generation
                                                         (``cold_start``, ``labeling``)
                                                         that feeds ``trainer/SFT/``.
============================ =========================== =====================================

Judge family-bias spot-check (when paper-grade rigour is required)
-------------------------------------------------------------------

Because the default judge now shares the Qwen3.5 pretraining family
with the actor, naive use creates a within-family self-preference risk.
For formal eval / paper runs, periodically re-judge a 5% random sample
with an off-distribution oracle (gpt-5.5) and log the disagreement
rate. To run a one-shot judge override::

    VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5  python -m ...

See ``implementation_notes/coevolution-cross-domain-integration.md``
§"Judge family bias" for the full protocol.

Phase-F frozen Qwen3-VL teachers
--------------------------------

The Skill Crafter is wired to optionally swap its frozen teacher to a
larger Qwen3-VL backbone (``Qwen/Qwen3-VL-32B`` or the MoE
``Qwen/Qwen3-VL-235B-A22B``).  Those identifiers stay registered under
:data:`QWEN3_VL_TEACHERS` and are reachable via
``SkillCrafterService(teacher_model=qwen3_vl_teacher(...))``.  They are
*opt-in*; the project-wide default for the crafter teacher is
``BACKBONE_TEACHER_MODEL = Qwen/Qwen3.5-35B-A3B``.

Two opt-in env-vars activate the Phase-F teacher without code edits:

* ``VLM_AGENT_BACKBONE_TEACHER_MODEL=Qwen/Qwen3-VL-32B`` — full override
  (works for any phase / call site).
* ``VLM_AGENT_PHASE_F_TEACHER=qwen3-vl-32b`` (or ``qwen3-vl-235b-a22b``)
  — Phase-F flag, read by ``crafter.SkillCrafterService.from_phase_f``.
  No implicit mutation of ``BACKBONE_TEACHER_MODEL``.

Rules
-----

1. Every new module in ``harness/``, ``orchestrator/``, ``crafter/``,
   ``skill_bank/``, and ``evaluation/`` MUST import the relevant constant
   from here when picking a default model.  Do NOT hardcode model
   strings.
2. Existing modules (``decision_agents/``, ``skill_agents/``,
   ``vlm_wrapper/``, ``inference/``) have been migrated where it changes
   a *runtime* default.  Documentation strings retaining historical
   model names are kept as historical context.
3. Routing is unchanged: the central LLM caller ``API_func.ask_model``
   already routes ``"gpt-*"`` → OpenAI/OpenRouter, ``"Qwen/..."`` → vLLM.

Switching the backbone in the future
------------------------------------

Set the following environment variables to override defaults at process
start.  Programmatic overrides should be passed as explicit ``model=``
arguments and never mutate the module-level constants.

* ``VLM_AGENT_BACKBONE_MODEL`` — actor / skill-bank policy.
* ``VLM_AGENT_BACKBONE_TEACHER_MODEL`` — crafter / harness / orchestrator.
* ``VLM_AGENT_BACKBONE_JUDGE_MODEL`` — validation / eval driver.
* ``VLM_AGENT_BACKBONE_SFT_TEACHER_MODEL`` — SFT cold-start data
  generation.
"""

from __future__ import annotations

import os
from typing import Mapping

# ---- canonical defaults ------------------------------------------------

#: Actor (decision_agents) and Skill-Bank (skill_agents) policy backbone.
#: This is the LoRA-trained model — Qwen3.5-9B dense decoder shared by
#: the ``skill_selection`` / ``action_taking`` decision adapters and the
#: ``segment`` / ``contract`` / ``curator`` skill-bank adapters.
BACKBONE_MODEL: str = os.environ.get(
    "VLM_AGENT_BACKBONE_MODEL", "Qwen/Qwen3.5-9B"
)

#: Crafter / Harness / Orchestrator control-plane backbone.  Frozen MoE
#: 35B-total / 3B-active served via ``inference/serve_qwen35_35b_a3b.sh``.
#: Used by ``crafter.SkillCrafterService`` (teacher), the harness control
#: logic, and the orchestrator's ``TeacherConfig``.  No fine-tuning runs
#: against this backbone inside the loop (PLAN-SKILL-CRAFTER §3
#: "Frozen-first design").
BACKBONE_TEACHER_MODEL: str = os.environ.get(
    "VLM_AGENT_BACKBONE_TEACHER_MODEL", "Qwen/Qwen3.5-35B-A3B"
)

#: LLM-as-judge for skill evaluation + the eval driver promotion gates
#: (E0 / E1 / E2 + replay validation).  Defaults to the local 35B-A3B
#: teacher backbone (same weights as ``BACKBONE_TEACHER_MODEL``,
#: different role) so judge calls hit the local vLLM server with no
#: API spend.  Override to ``gpt-5.5`` (or another off-distribution
#: oracle) when running formal eval where within-Qwen-family bias must
#: be controlled — see the "Judge family-bias spot-check" section in
#: this module's docstring.
BACKBONE_JUDGE_MODEL: str = os.environ.get(
    "VLM_AGENT_BACKBONE_JUDGE_MODEL", "Qwen/Qwen3.5-35B-A3B"
)

#: SFT cold-start data generation teacher.  Used by ``cold_start/`` and
#: ``labeling/`` to produce the labeled trajectories that
#: ``trainer/SFT/`` then trains the actor + skill-bank adapters on.
#: Distinct from ``BACKBONE_TEACHER_MODEL`` (which is the *runtime*
#: crafter teacher) so the SFT data pipeline can be retargeted
#: independently of the live control plane.
BACKBONE_SFT_TEACHER_MODEL: str = os.environ.get(
    "VLM_AGENT_BACKBONE_SFT_TEACHER_MODEL", "gpt-5.5"
)


# ---- Phase-F frozen Qwen3-VL teacher registry --------------------------

#: Frozen Qwen3-VL teacher model identifiers (PLAN-SKILL-CRAFTER §2,
#: "Frozen-first design").  Map of ``size_key`` → canonical HF model id.
#: New entries are additive; nothing here is a *default*.
QWEN3_VL_TEACHERS: Mapping[str, str] = {
    "32b": "Qwen/Qwen3-VL-32B",
    "235b-a22b": "Qwen/Qwen3-VL-235B-A22B",
}


def qwen3_vl_teacher(size: str = "32b") -> str:
    """Return the canonical HF id for a frozen Qwen3-VL teacher.

    ``size`` is matched case-insensitively against the keys of
    :data:`QWEN3_VL_TEACHERS`.  Raises :class:`ValueError` for unknown
    sizes so a typo can't silently fall back to a different model.
    """
    key = size.lower().strip()
    if key not in QWEN3_VL_TEACHERS:
        raise ValueError(
            f"Unknown Qwen3-VL teacher size {size!r}; "
            f"valid options: {sorted(QWEN3_VL_TEACHERS)}."
        )
    return QWEN3_VL_TEACHERS[key]


def phase_f_teacher_from_env() -> str | None:
    """Resolve the Phase-F teacher from ``VLM_AGENT_PHASE_F_TEACHER``.

    Returns the canonical HF model id when the env-var is set to a
    recognized size, or ``None`` when unset.  Unknown sizes raise so
    misconfiguration surfaces at the entry point, not silently as a
    fallback to the default teacher.
    """
    raw = os.environ.get("VLM_AGENT_PHASE_F_TEACHER")
    if not raw:
        return None
    return qwen3_vl_teacher(raw)


# ---- deferred-track registry ------------------------------------------

#: Canonical model names that are *deferred* — mentioned in plans but
#: not part of the current default surface.  Tests check that no live
#: code path defaults to one of these.  The Qwen3-VL Phase-F teachers
#: are deferred-by-default too: they're opt-in via
#: ``SkillCrafterService(teacher_model=qwen3_vl_teacher(...))``.
DEFERRED_MODELS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-VL-72B",
        *QWEN3_VL_TEACHERS.values(),
    }
)


def is_deferred(model: str) -> bool:
    """True if ``model`` is one of the deferred Qwen tracks."""
    return model in DEFERRED_MODELS


def is_frozen_qwen_teacher(model: str) -> bool:
    """True if ``model`` is one of the Phase-F frozen Qwen3-VL teachers."""
    return model in set(QWEN3_VL_TEACHERS.values())


def assert_default_backbone() -> None:
    """Used by tests; raises if the actor backbone has been silently changed.

    Pinned to ``Qwen/Qwen3.5-9B`` for the current phase.  Set
    ``VLM_AGENT_BACKBONE_MODEL`` to override only when explicitly
    enabling a different actor backbone.
    """
    if BACKBONE_MODEL != "Qwen/Qwen3.5-9B":
        raise AssertionError(
            f"BACKBONE_MODEL must be 'Qwen/Qwen3.5-9B' for the current "
            f"phase; got {BACKBONE_MODEL!r}. Set VLM_AGENT_BACKBONE_MODEL "
            f"to override only when explicitly enabling a different "
            f"actor backbone."
        )


# Backward-compatible alias kept for callers / tests that still import
# the older name.  New code should prefer :func:`assert_default_backbone`.
def assert_default_is_gpt4o() -> None:  # pragma: no cover — legacy shim
    """Deprecated: use :func:`assert_default_backbone` instead.

    The historical ``gpt-4o`` pin was retired in 2026-04 when the actor
    backbone moved to ``Qwen/Qwen3.5-9B``.  This shim now delegates to
    :func:`assert_default_backbone` so legacy callers keep working.
    """
    assert_default_backbone()


__all__ = [
    "BACKBONE_JUDGE_MODEL",
    "BACKBONE_MODEL",
    "BACKBONE_SFT_TEACHER_MODEL",
    "BACKBONE_TEACHER_MODEL",
    "DEFERRED_MODELS",
    "QWEN3_VL_TEACHERS",
    "assert_default_backbone",
    "assert_default_is_gpt4o",
    "is_deferred",
    "is_frozen_qwen_teacher",
    "phase_f_teacher_from_env",
    "qwen3_vl_teacher",
]
