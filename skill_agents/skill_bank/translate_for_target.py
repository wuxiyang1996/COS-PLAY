"""Cross-game skill translator (shared-bank lifelong-learning mode).

At each curriculum phase boundary, this module re-grounds skills mined
on the previous source game so they're admissible on the next target
game *without* triggering the §22 "100 % cross-contamination"
pathology measured in
``labeling_supplement/_phase0_cross_eligibility_probe.py``.

The contract surface
--------------------
Given (source skill, target game, target action vocabulary, target
schema sample) the translator returns a *new* :class:`Skill` record:

* ``skill_id``      = ``"<source_id>__translated_to__<target_game>"``
* ``derived_from``  = source skill_id (lineage for retirement / curator)
* ``confidence_tag``= ``"translated"`` (soft-promote; gates re-run on target)
* ``feasible_tasks``= ``[<target_game>]``  ← the load-bearing invariant
* ``verified_tasks``= ``[]``  (earns through gates)
* ``protocol``      = LLM-rewritten action steps in the target action vocabulary
* ``contract``      = predicate-rewritten via :mod:`harness.predicate_translator`

Hard invariants
---------------
1. ``len(out.feasible_tasks) == 1`` — exactly the target game. The
   translator never widens eligibility on its own; widening only
   happens when the post-translation skill earns a verified task slot
   through E0/E1/E2 gates during the target phase.
2. ``out.derived_from is not None`` — provenance must survive.
3. ``out.confidence_tag == "translated"`` — disambiguates from
   foundry-mined records on retirement and skill_selection display.
4. Every action in ``out.protocol`` parses to a token in
   ``target_actions`` (validated post-LLM; failed translations are
   dropped with a debug log line, not silently retained).

These are enforced at the public entrypoint
:func:`translate_skill_for_target`. A failed invariant is treated as
"this skill is not transferable to the target" and the translator
returns ``None`` for that skill — it does NOT raise. The downstream
caller (``scripts/run_phase1_curriculum.sh`` or
``skill_agents.skill_bank.translate_for_target.__main__``) decides
whether to log + continue or abort the phase boundary.

CLI
---
::

    python -m skill_agents.skill_bank.translate_for_target \\
        --source-bank PATH/skill_bank.jsonl \\
        --target-game gymv_streets_of_rage_2 \\
        --target-actions "B,A,UP,DOWN,LEFT,RIGHT,C,Y,X,Z,MODE,START" \\
        --output PATH/translated_skill_bank.jsonl \\
        --judge-model Qwen/Qwen3.5-35B-A3B

When ``--judge-model`` is omitted the value of
``common.models.BACKBONE_JUDGE_MODEL`` is used (35B by default since
2026-05-03 — see ``common/models.py``).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger("skill_bank.translate_for_target")


# ---------------------------------------------------------------------------
# Constants — invariants and prompt scaffolding
# ---------------------------------------------------------------------------

CONFIDENCE_TAG_TRANSLATED = "translated"
"""``Skill.confidence_tag`` value for records produced by this module."""


_SYSTEM_PROMPT = """You are a cross-game skill translator. Your job is to
rewrite a *source* skill — mined and verified on game A — so it can be
attempted on a *target* game B without changing its strategic intent.

You must respect three hard rules:

1. The translated `protocol.steps` may use ONLY actions from the
   provided `target_actions` list. If the source action sequence has
   no plausible target-action analogue, output `transferable=false`.
2. The translated `contract.eff_add` and `contract.eff_del` predicates
   must reference predicates that the *target* game's success_fn can
   evaluate. Reuse the source predicates when they're shared
   vocabulary (e.g. `cumulative_reward_increased`,
   `entity_value_increased`, `phase_transitioned`); replace
   game-specific ones with the closest target-grounded analogue.
3. The translated skill remains semantically faithful to the source's
   strategic_description. If the strategy makes no sense in the
   target's mechanics (e.g. "match-3 piece swap" → SoR2 brawler),
   output `transferable=false` rather than inventing a fake mapping.

Output strict JSON (no prose, no code fences) with the schema:

{
  "transferable": true | false,
  "name": "<keep or rewrite>",
  "strategic_description": "<rewritten for target context, ~1-2 sentences>",
  "protocol": {
    "steps": ["<target-action token>", "<target-action token>", ...]
  },
  "contract": {
    "eff_add":   ["<predicate>", ...],
    "eff_del":   ["<predicate>", ...],
    "eff_event": ["<predicate>", ...]
  },
  "rationale": "<1-2 sentence justification>"
}

When `transferable=false`, `name`, `strategic_description`, `protocol`,
and `contract` may be empty; only `rationale` (the reason) is needed.
"""


# Predicate vocabulary that's safe to reuse identity-style across all
# gymv games (their success_fn families share these). Anything else is
# considered game-specific and must be remapped or dropped.
_SHARED_GYMV_PREDICATES = frozenset({
    "cumulative_reward_increased",
    "entity_value_increased",
    "entity_value_decreased",
    "entity_appeared",
    "entity_disappeared",
    "entity_count_changed",
    "attribute_changed",
    "phase_transitioned",
})


# ---------------------------------------------------------------------------
# Data plumbing — read source bank, derive translation inputs
# ---------------------------------------------------------------------------

def load_source_skills(bank_path: Path) -> List["Skill"]:                  # noqa: F821
    """Load every skill from a legacy ``skill_bank.jsonl`` file.

    Uses :class:`SkillBankMVP` so the same code path as the trainer
    handles the legacy / new envelope distinction. Returns an empty
    list (with a warning) if the path is missing rather than raising,
    so the curriculum boundary script can survive a fresh first
    phase that hasn't written any skills yet.
    """
    from skill_agents.skill_bank.bank import SkillBankMVP

    if not bank_path.is_file():
        logger.warning(
            "translate_for_target: source bank not found at %s — returning empty list",
            bank_path,
        )
        return []
    bank = SkillBankMVP(str(bank_path))
    bank.load(str(bank_path))
    return [bank.get_skill(sid) for sid in bank.skill_ids if bank.get_skill(sid) is not None]


def _build_user_prompt(
    *,
    source_skill: "Skill",                                                  # noqa: F821
    source_game: str,
    target_game: str,
    target_actions: Sequence[str],
    target_schema_sample: Optional[str],
) -> str:
    """Assemble the user-side prompt for one skill translation."""
    parts: List[str] = []
    parts.append(f"source_game: {source_game}")
    parts.append(f"target_game: {target_game}")
    parts.append(f"target_actions: {list(target_actions)}")

    src_dict: Dict[str, Any] = {
        "skill_id": source_skill.skill_id,
        "name": source_skill.name,
        "strategic_description": source_skill.strategic_description,
        "tags": list(source_skill.tags),
    }
    if source_skill.protocol is not None:
        src_dict["protocol"] = source_skill.protocol.to_dict()
    if source_skill.contract is not None:
        src_dict["contract"] = source_skill.contract.to_dict()
    parts.append("source_skill: " + json.dumps(src_dict, default=str))

    if target_schema_sample:
        parts.append("target_schema_sample: " + target_schema_sample)
    parts.append(
        "Output ONLY the strict JSON described in the system message. "
        "Do not include code fences."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call — defers to API_func.ask_vllm for OpenAI-compatible vLLM routing
# ---------------------------------------------------------------------------

def _call_judge(
    *,
    system_prompt: str,
    user_prompt: str,
    judge_model: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """Invoke the judge model. Returns the raw completion text or
    ``None`` on transport / parse failure (the caller decides whether
    to retry or skip).

    Defers to :func:`API_func.ask_vllm` so per-model URL routing
    (``VLLM_BASE_URL_MAP``) works for the 35B judge endpoint without
    extra plumbing here.
    """
    try:
        from API_func import ask_vllm
    except Exception as exc:                                              # noqa: BLE001
        logger.error("translate_for_target: failed to import API_func: %s", exc)
        return None

    full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
    try:
        return ask_vllm(
            full_prompt,
            model=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:                                              # noqa: BLE001
        logger.warning(
            "translate_for_target: judge call failed for model %s: %s",
            judge_model, exc,
        )
        return None


def _strip_code_fences(text: str) -> str:
    """Be lenient with judge output: tolerate ```json``` fences."""
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _parse_judge_output(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's strict-JSON reply. ``None`` on malformed output."""
    if not raw:
        return None
    cleaned = _strip_code_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.debug(
            "translate_for_target: parse failed (%s); raw=%s",
            exc, cleaned[:512],
        )
        return None
    if not isinstance(obj, dict):
        return None
    return obj


# ---------------------------------------------------------------------------
# Predicate filtering (Layer C identity for gymv→gymv; drop unknowns)
# ---------------------------------------------------------------------------

def _filter_target_predicates(
    predicates: Iterable[str],
    *,
    extra_allowed: Iterable[str] = (),
) -> List[str]:
    """Keep predicates that are in the shared gymv vocabulary OR
    explicitly allowed (per-game extension hook). Drops everything
    else with a debug log line.
    """
    allowed = _SHARED_GYMV_PREDICATES | set(extra_allowed)
    out: List[str] = []
    for p in predicates:
        if not isinstance(p, str) or not p:
            continue
        if p in allowed:
            out.append(p)
        else:
            logger.debug(
                "translate_for_target: dropping unknown predicate %r", p,
            )
    return out


# ---------------------------------------------------------------------------
# Public entrypoint — single-skill translation
# ---------------------------------------------------------------------------

def translate_skill_for_target(
    source_skill: "Skill",                                                  # noqa: F821
    *,
    source_game: str,
    target_game: str,
    target_actions: Sequence[str],
    target_schema_sample: Optional[str] = None,
    judge_model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    extra_allowed_predicates: Iterable[str] = (),
) -> Optional["Skill"]:                                                     # noqa: F821
    """Translate one source skill into a target-game-grounded
    derivative. Returns ``None`` when the skill is not transferable
    or any hard invariant fails.

    Hard invariants enforced before returning a non-``None`` result:

    * ``out.feasible_tasks == [target_game]`` (single-element list)
    * ``out.derived_from == source_skill.skill_id``
    * ``out.confidence_tag == CONFIDENCE_TAG_TRANSLATED``
    * Every action in ``out.protocol.steps`` parses to a token in
      ``target_actions``.

    On any invariant failure (or judge error / parse failure) we log
    at WARNING and return ``None``.
    """
    from skill_agents.stage3_mvp.schemas import (
        Protocol,
        Skill,
        SkillEffectsContract,
    )

    if judge_model is None:
        try:
            from common.models import BACKBONE_JUDGE_MODEL
            judge_model = BACKBONE_JUDGE_MODEL
        except Exception:
            judge_model = "Qwen/Qwen3.5-35B-A3B"

    target_action_set = set(target_actions)
    if not target_action_set:
        logger.warning(
            "translate_for_target: empty target_actions for %s — refusing",
            target_game,
        )
        return None

    user_prompt = _build_user_prompt(
        source_skill=source_skill,
        source_game=source_game,
        target_game=target_game,
        target_actions=target_actions,
        target_schema_sample=target_schema_sample,
    )

    raw = _call_judge(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        judge_model=judge_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if raw is None:
        return None

    parsed = _parse_judge_output(raw)
    if parsed is None or not parsed.get("transferable"):
        if parsed is not None:
            logger.info(
                "translate_for_target: %s rejected by judge (rationale=%s)",
                source_skill.skill_id, parsed.get("rationale", "<none>")[:200],
            )
        return None

    raw_steps = (
        ((parsed.get("protocol") or {}).get("steps") or [])
        if isinstance(parsed.get("protocol"), Mapping)
        else []
    )
    valid_steps: List[str] = []
    for step in raw_steps:
        if not isinstance(step, str):
            continue
        token = step.strip()
        if not token:
            continue
        if token not in target_action_set:
            logger.debug(
                "translate_for_target: %s step %r not in target_actions; dropping",
                source_skill.skill_id, token,
            )
            continue
        valid_steps.append(token)
    if not valid_steps:
        logger.info(
            "translate_for_target: %s produced no valid target-action steps",
            source_skill.skill_id,
        )
        return None

    raw_contract = parsed.get("contract") or {}
    eff_add = _filter_target_predicates(
        raw_contract.get("eff_add") or [],
        extra_allowed=extra_allowed_predicates,
    )
    eff_del = _filter_target_predicates(
        raw_contract.get("eff_del") or [],
        extra_allowed=extra_allowed_predicates,
    )
    eff_event = _filter_target_predicates(
        raw_contract.get("eff_event") or [],
        extra_allowed=extra_allowed_predicates,
    )

    base_contract = source_skill.contract
    if base_contract is None:
        # Build a minimal contract carrying only the LLM-validated
        # predicates so downstream gates have a non-empty surface.
        from skill_agents.stage3_mvp.schemas import SkillEffectsContract as _SEC
        translated_contract = _SEC(
            skill_id=f"{source_skill.skill_id}__translated_to__{target_game}",
            version=1,
            name=parsed.get("name") or source_skill.name,
            description=parsed.get("strategic_description") or "",
            eff_add=set(eff_add),
            eff_del=set(eff_del),
            eff_event=set(eff_event),
        )
    else:
        translated_contract = copy.deepcopy(base_contract)
        translated_contract.skill_id = f"{source_skill.skill_id}__translated_to__{target_game}"
        translated_contract.name = parsed.get("name") or translated_contract.name
        translated_contract.description = (
            parsed.get("strategic_description") or translated_contract.description
        )
        translated_contract.eff_add = set(eff_add)
        translated_contract.eff_del = set(eff_del)
        translated_contract.eff_event = set(eff_event)
        translated_contract.updated_at = time.time()

    translated_protocol = Protocol(steps=list(valid_steps))

    out = Skill(
        skill_id=f"{source_skill.skill_id}__translated_to__{target_game}",
        version=1,
        name=parsed.get("name") or source_skill.name,
        strategic_description=(
            parsed.get("strategic_description") or source_skill.strategic_description
        ),
        tags=list(source_skill.tags) + ["translated", f"target:{target_game}"],
        protocol=translated_protocol,
        contract=translated_contract,
        sub_episodes=[],
        expected_tag_pattern=list(source_skill.expected_tag_pattern),
        execution_hint=None,
        protocol_history=[],
        n_instances=0,
        retired=False,
        feasible_tasks=[target_game],          # invariant: single-element
        verified_tasks=[],                      # earns through gates
        derived_from=source_skill.skill_id,    # provenance
        confidence_tag=CONFIDENCE_TAG_TRANSLATED,
    )

    # ── Hard-invariant assertions ──────────────────────────────────
    if out.feasible_tasks != [target_game]:
        logger.error(
            "translate_for_target: invariant violated — feasible_tasks=%s != [%s]; refusing",
            out.feasible_tasks, target_game,
        )
        return None
    if not out.derived_from:
        logger.error(
            "translate_for_target: invariant violated — derived_from is empty for %s",
            source_skill.skill_id,
        )
        return None
    if out.confidence_tag != CONFIDENCE_TAG_TRANSLATED:
        logger.error(
            "translate_for_target: invariant violated — confidence_tag=%r != %r",
            out.confidence_tag, CONFIDENCE_TAG_TRANSLATED,
        )
        return None
    return out


# ---------------------------------------------------------------------------
# Bank-level helper — translate a whole bank in one call
# ---------------------------------------------------------------------------

def translate_bank_for_target(
    *,
    source_bank_path: Path,
    target_game: str,
    target_actions: Sequence[str],
    output_bank_path: Path,
    source_game: Optional[str] = None,
    target_schema_sample: Optional[str] = None,
    judge_model: Optional[str] = None,
    extra_allowed_predicates: Iterable[str] = (),
    seed_with_source: bool = True,
) -> Dict[str, Any]:
    """Translate every skill in *source_bank_path* and write the
    derivatives into *output_bank_path*.

    When *seed_with_source* is True (default), the source skills are
    *also* copied into the output bank with their original
    ``feasible_tasks=[<source_game>]`` (so the harness still admits
    them on the source game during cross-phase evaluation). This lets
    the shared bank accumulate provenance: source + translated copies
    coexist with disjoint ``feasible_tasks``.

    Returns a summary dict with ``n_source``, ``n_translated``,
    ``n_rejected`` for logging.
    """
    from skill_agents.skill_bank.bank import SkillBankMVP

    sources = load_source_skills(source_bank_path)
    output_bank_path.parent.mkdir(parents=True, exist_ok=True)
    out_bank = SkillBankMVP(str(output_bank_path))

    if seed_with_source and sources:
        for skill in sources:
            preserved = copy.deepcopy(skill)
            if not preserved.feasible_tasks and source_game:
                preserved.feasible_tasks = [source_game]
            out_bank.add_or_update_skill(preserved)

    n_translated = 0
    n_rejected = 0
    for source_skill in sources:
        actual_source_game = (
            source_game
            or (source_skill.feasible_tasks[0] if source_skill.feasible_tasks else "unknown")
        )
        translated = translate_skill_for_target(
            source_skill,
            source_game=actual_source_game,
            target_game=target_game,
            target_actions=target_actions,
            target_schema_sample=target_schema_sample,
            judge_model=judge_model,
            extra_allowed_predicates=extra_allowed_predicates,
        )
        if translated is None:
            n_rejected += 1
            continue
        out_bank.add_or_update_skill(translated)
        n_translated += 1

    out_bank.save()
    summary = {
        "n_source": len(sources),
        "n_translated": n_translated,
        "n_rejected": n_rejected,
        "source_bank": str(source_bank_path),
        "target_bank": str(output_bank_path),
        "target_game": target_game,
    }
    logger.info(
        "translate_bank_for_target: %d source → %d translated (%d rejected) → %s",
        summary["n_source"], summary["n_translated"],
        summary["n_rejected"], output_bank_path,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_actions_arg(raw: str) -> List[str]:
    """Accept either a comma-separated list or a JSON list."""
    raw = raw.strip()
    if raw.startswith("["):
        return [str(x) for x in json.loads(raw)]
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skill_agents.skill_bank.translate_for_target",
        description=(
            "Cross-game skill translator: re-grounds skills mined on a "
            "source game so they're admissible on a target game in the "
            "shared-bank lifelong-learning mode (config.bank_mode='shared')."
        ),
    )
    p.add_argument(
        "--source-bank", type=str, required=True,
        help="Path to the source skill_bank.jsonl (output of the previous phase).",
    )
    p.add_argument(
        "--target-game", type=str, required=True,
        help="Target game slug (e.g. 'gymv_streets_of_rage_2').",
    )
    p.add_argument(
        "--target-actions", type=str, required=True,
        help=(
            "Comma-separated or JSON-list of action tokens valid in the "
            "target game. E.g. 'B,A,UP,DOWN,LEFT,RIGHT,C,Y,X,Z,MODE,START'."
        ),
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output skill_bank.jsonl path (will be overwritten).",
    )
    p.add_argument(
        "--source-game", type=str, default=None,
        help="Optional explicit source-game slug; defaults to the first "
             "feasible_task on each source skill.",
    )
    p.add_argument(
        "--target-schema-sample", type=str, default=None,
        help="Optional path to a single schema_gen sample for the target "
             "game (raw text). Helps the judge see target ontology examples.",
    )
    p.add_argument(
        "--judge-model", type=str, default=None,
        help="Judge model slug (defaults to common.models.BACKBONE_JUDGE_MODEL).",
    )
    p.add_argument(
        "--extra-allowed-predicates", type=str, default="",
        help="Comma-separated extra predicate names to keep through the "
             "filter (target-game-specific extensions beyond the shared "
             "gymv vocabulary).",
    )
    p.add_argument(
        "--no-seed-with-source", action="store_true",
        help="Don't copy source skills into the output bank. Default "
             "behaviour seeds the source skills (with their original "
             "feasible_tasks) so the shared bank accumulates lineage.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v: INFO, -vv: DEBUG).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    actions = _parse_actions_arg(args.target_actions)
    extra_preds = _parse_actions_arg(args.extra_allowed_predicates) if args.extra_allowed_predicates else []

    schema_sample: Optional[str] = None
    if args.target_schema_sample:
        sp = Path(args.target_schema_sample)
        if sp.is_file():
            schema_sample = sp.read_text(encoding="utf-8")[:4000]
        else:
            schema_sample = args.target_schema_sample[:4000]

    summary = translate_bank_for_target(
        source_bank_path=Path(args.source_bank),
        target_game=args.target_game,
        target_actions=actions,
        output_bank_path=Path(args.output),
        source_game=args.source_game,
        target_schema_sample=schema_sample,
        judge_model=args.judge_model,
        extra_allowed_predicates=extra_preds,
        seed_with_source=not args.no_seed_with_source,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
