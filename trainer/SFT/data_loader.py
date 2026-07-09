"""Cold-start data loaders for all 5 LoRA adapters.

Data sources
~~~~~~~~~~~~
**Decision adapters** (``skill_selection``, ``action_taking``):
  ``labeling/output/gpt54_skill_labeled/grpo_coldstart/<game>/<adapter>.jsonl``
  Each line: ``{"prompt": ..., "completion": ..., "intention": ..., ...}``

  ``action_taking`` cold-start data is generated without the SUBGOAL line
  that the co-evolution episode runner uses.  This loader transforms the
  prompt and completion to include SUBGOAL, using the ``intention``
  metadata field, so the SFT-trained adapter matches the co-evolution
  inference format.

**Skill-bank adapters**:

``segment``:
  ``skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo/<game>/teacher_io_coldstart.jsonl``
  Functions: ``segment_ranking``, ``transition_ranking``, ``pairwise_choice``
  Each line: ``{"prompt": ..., "response": ..., ...}``

``contract``:
  ``…/<game>/coldstart_io_all.jsonl`` filtered to
  modules ``boundary_proposal`` and ``pipeline``.
  NOTE: the cold-start extraction did not run Stage 3 (contract learning),
  so there are no exact ``stage3_contract`` records.  The mapped modules
  provide domain-proximate data (predicate analysis, protocol synthesis)
  that GRPO refines to the actual contract summarisation task.

``curator``:
  ``…/<game>/coldstart_io_all.jsonl`` filtered to module ``skill_naming``.
  NOTE: the cold-start extraction did not run bank maintenance, so there
  are no exact ``bank_curator`` records.  Skill naming data provides
  approximate domain overlap that GRPO refines to the approve/veto/defer
  curation task.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.SFT.config import (
    COLDSTART_IO_MODULE_MAP,
    DECISION_ADAPTERS,
    SFTConfig,
)

logger = logging.getLogger(__name__)

SUBGOAL_TAGS = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)
_SUBGOAL_TAGS_STR = "|".join(SUBGOAL_TAGS)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalise_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Extract a (prompt, completion) pair from a data row.

    Decision data uses ``completion``; skill-bank data uses ``response``.
    """
    prompt = row.get("prompt", "")
    completion = row.get("completion") or row.get("response", "")
    return {"prompt": prompt, "completion": completion}


def _upgrade_skill_guidance_block(
    prompt: str,
    skill_bank: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Upgrade old-format ``--- Active Skill ---`` blocks to match GRPO.

    Matches the exact format produced by
    ``episode_runner._format_skill_guidance_for_prompt``:
      - Strategy (from ``execution_hint`` / ``strategic_description``)
      - Progress line
      - Plan with numbered steps and ``>>`` marker
      - Preconditions
      - Done when / Abort if
    """
    start = prompt.find("--- Active Skill:")
    if start < 0:
        return prompt
    end = prompt.find("--- end skill ---")
    if end < 0:
        return prompt
    end += len("--- end skill ---")

    old_block = prompt[start:end]

    skill_name_match = re.search(r"--- Active Skill:\s*(.+?)\s*---", old_block)
    skill_name = skill_name_match.group(1) if skill_name_match else "Unknown"

    skill = None
    if skill_bank:
        for sid, s in skill_bank.items():
            if s.get("name", "") == skill_name or sid == skill_name:
                skill = s
                break

    parts = [f"\n--- Active Skill: {skill_name} ---"]

    if skill:
        strat = skill.get("strategic_description", "")
        if strat:
            parts.append(f"  Strategy: {strat[:200]}")

        protocol = skill.get("protocol", {})
        if isinstance(protocol, dict):
            parts.append("  Progress: Starting step 1.")

            steps = protocol.get("steps", [])
            if steps:
                parts.append(f"  Plan ({len(steps)} steps):")
                for i, step in enumerate(steps[:7], 1):
                    marker = ">>" if i == 1 else "  "
                    parts.append(f"  {marker} {i}. {step}")

            preconditions = protocol.get("preconditions", [])
            if preconditions:
                parts.append(f"  Preconditions: {'; '.join(preconditions[:3])}")

            success = protocol.get("success_criteria", [])
            if success:
                parts.append(f"  Done when: {'; '.join(success[:2])}")

            abort = protocol.get("abort_criteria", [])
            if abort:
                parts.append(f"  Abort if: {'; '.join(abort[:2])}")
    else:
        strat_match = re.search(r"Strategy:\s*(.+?)(?:\n|$)", old_block)
        if strat_match:
            raw_strat = strat_match.group(1).strip()
            if "|" not in raw_strat:
                parts.append(f"  Strategy: {raw_strat[:200]}")

    parts.append("--- end skill ---\n")
    new_block = "\n".join(parts)

    return prompt[:start] + new_block + prompt[end:]


def _inject_context_lines(
    prompt: str,
    row: Dict[str, Any],
    skill_bank: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Inject GRPO context lines between ``Game state:`` and ``Available actions``.

    GRPO episode_runner inserts (in order):
      - ``Assigned subgoal: ...`` (from base-model intention call)
      - ``URGENCY: ...`` (conditional, game-specific)
      - ``Active skill: {name} — {hint}`` (conditional)
      - ``Recent actions and rewards: ...`` (conditional)

    We inject what we can reconstruct from cold-start metadata.
    """
    actions_idx = prompt.find("Available actions (pick ONE by number):")
    if actions_idx < 0:
        return prompt

    lines: List[str] = []

    intention = row.get("intention", "")
    if intention:
        lines.append(f"Assigned subgoal: {intention}")

    active_skill = row.get("active_skill", "")
    if active_skill and skill_bank:
        skill = skill_bank.get(active_skill)
        if skill:
            sk_name = skill.get("name", active_skill)
            sk_hint = skill.get("strategic_description", "")
            line = f"Active skill: {sk_name}"
            if sk_hint:
                line += f" \u2014 {sk_hint[:100]}"
            lines.append(line)
        else:
            lines.append(f"Active skill: {active_skill}")
    elif active_skill:
        lines.append(f"Active skill: {active_skill}")

    if not lines:
        return prompt

    insert_text = "\n".join(lines) + "\n"
    return prompt[:actions_idx] + insert_text + prompt[actions_idx:]


def _align_action_taking_to_coevolution(
    row: Dict[str, Any],
    skill_bank: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Transform an action_taking cold-start row to match the co-evolution
    episode_runner prompt/completion format exactly.

    Alignment with ``episode_runner.py`` SYSTEM_PROMPT + action_user:
      1. Rule line 3: add ``— try something different`` suffix.
      2. Output format: add SUBGOAL line before REASONING.
      3. Skill guidance block: upgrade to full protocol format with
         Preconditions, Progress, Plan steps, Done/Abort.
      4. Context lines: inject Assigned subgoal, Active skill one-liner
         between Game state and Available actions.
      5. Closing: Subgoal tags + "Use the assigned subgoal tag" instruction.
      6. Completion: prepend ``SUBGOAL: <intention>`` from metadata.
    """
    prompt = row.get("prompt", "")
    completion = row.get("completion", "")
    intention = row.get("intention", "")

    # Fix 1: Rule line 3 wording
    prompt = prompt.replace(
        "NEVER repeat the same action more than 2 times in a row.",
        "NEVER repeat the same action more than 2 times in a row "
        "\u2014 try something different.",
    )

    # Fix 2: Output format — add SUBGOAL
    prompt = prompt.replace(
        "Output format (strict):\nREASONING: <1-2 sentences>\nACTION: <number>",
        "Output format (strict):\n"
        "SUBGOAL: [TAG] <your immediate objective in \u226415 words>\n"
        "REASONING: <1-2 sentences>\n"
        "ACTION: <number>",
    )

    # Fix 3: Skill guidance block — upgrade to new protocol format
    prompt = _upgrade_skill_guidance_block(prompt, skill_bank)

    # Fix 4: Inject context lines (Previous subgoal, Active skill)
    prompt = _inject_context_lines(prompt, row, skill_bank)

    # Fix 5: Closing instructions — add subgoal tags + assigned subgoal
    prompt = re.sub(
        r"Choose the best action\.\s*Output REASONING then ACTION number\.",
        f"Subgoal tags: {_SUBGOAL_TAGS_STR}\n"
        "Use the assigned subgoal tag in your SUBGOAL line, then choose the best action.\n"
        "Output SUBGOAL: [TAG] objective, REASONING, then ACTION number.",
        prompt,
    )

    # Fix 6: Completion — prepend SUBGOAL
    if intention:
        completion = f"SUBGOAL: {intention}\n{completion}"

    return {"prompt": prompt, "completion": completion}


def _load_skill_bank(bank_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a skill bank JSONL and return ``{skill_id: skill_dict}``."""
    bank: Dict[str, Dict[str, Any]] = {}
    if not bank_path.exists():
        return bank
    for row in _read_jsonl(bank_path):
        skill = row.get("skill", row)
        sid = skill.get("skill_id", "")
        if sid:
            bank[sid] = skill
    return bank


def _format_skill_line(skill: Dict[str, Any], number: int) -> str:
    """Format a skill for the selection prompt, matching co-evolution
    ``_format_candidates_for_selection`` output exactly.

    GRPO format (episode_runner.py):
      {N}. {name}
         Strategy: {execution_hint[:150]}
         Plan: step1 -> step2 -> ...
         Confidence: {confidence:.2f}
    """
    name = skill.get("name") or skill.get("skill_id", f"strategy_{number}")
    lines = [f"  {number}. {name}"]
    hint = skill.get("strategic_description") or ""
    if isinstance(hint, str) and hint:
        lines.append(f"     Strategy: {hint[:150]}")
    protocol = skill.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        step_text = " -> ".join(steps[:4])
        if len(steps) > 4:
            step_text += " -> ..."
        lines.append(f"     Plan: {step_text}")
    confidence = skill.get("success_rate") or skill.get("pass_rate")
    if confidence is None:
        sub_eps = skill.get("sub_episodes", [])
        if sub_eps:
            scores = [se.get("quality_score", 0) for se in sub_eps]
            if scores:
                confidence = sum(1 for s in scores if s >= 0.5) / len(scores)
    if confidence is not None:
        try:
            lines.append(f"     Confidence: {float(confidence):.2f}")
        except (ValueError, TypeError):
            pass
    return "\n".join(lines)


def _align_skill_selection_system_prompt(prompt: str) -> str:
    """Align skill_selection system prompt wording with GRPO runtime.

    GRPO (episode_runner.py SKILL_SELECTION_SYSTEM_PROMPT) uses:
      ``and a set of candidate strategies``
      ``fits the current state``
    Cold-start uses slightly different wording.
    """
    prompt = prompt.replace(
        "and candidate strategies,",
        "and a set of candidate strategies,",
    )
    prompt = prompt.replace(
        "why this strategy fits>",
        "why this strategy fits the current state>",
    )
    return prompt


def _enrich_skill_selection_prompt(
    prompt: str, bank: Dict[str, Dict[str, Any]],
) -> str:
    """Replace bare skill IDs/tags in the 'Available strategies' block with
    rich descriptions from the skill bank (name + strategy + plan + confidence),
    and align system prompt wording with GRPO runtime."""
    prompt = _align_skill_selection_system_prompt(prompt)

    strat_start = prompt.find("Available strategies (pick ONE by number):")
    if strat_start < 0:
        return prompt
    choose_start = prompt.find("Choose the best strategy.", strat_start)
    if choose_start < 0:
        choose_start = len(prompt)
    before = prompt[:strat_start]
    after = prompt[choose_start:]
    block = prompt[strat_start:choose_start]

    new_lines = ["Available strategies (pick ONE by number):"]
    for m in re.finditer(r"^\s+(\d+)\.\s+(.+)$", block, re.MULTILINE):
        num = int(m.group(1))
        raw_id = m.group(2).strip()
        skill = bank.get(raw_id)
        if not skill:
            tag = raw_id.split("_")[-2].upper() if "_" in raw_id else raw_id
            skill = bank.get(tag)
        if skill:
            new_lines.append(_format_skill_line(skill, num))
        else:
            new_lines.append(f"  {num}. {raw_id}")
    new_lines.append("")

    return before + "\n".join(new_lines) + after


def load_decision_adapter_data(
    adapter_name: str,
    data_dir: str,
    games: List[str],
    skillbank_data_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load cold-start data for a decision adapter across all games.

    For ``skill_selection``, enriches prompts with skill bank descriptions
    (name + strategy + plan) so the SFT format matches co-evolution.

    For ``action_taking``, upgrades old-format ``--- Active Skill ---``
    blocks to include protocol steps, progress, and success/abort criteria
    from the skill bank — matching the new co-evolution format.

    Returns list of ``{"prompt": ..., "completion": ...}`` dicts.
    """
    assert adapter_name in DECISION_ADAPTERS
    examples: List[Dict[str, str]] = []
    base = Path(data_dir)
    for game in games:
        bank: Dict[str, Dict[str, Any]] = {}
        if skillbank_data_dir:
            bank = _load_skill_bank(Path(skillbank_data_dir) / game / "skill_bank.jsonl")
            if bank:
                logger.info("[%s] %s: loaded %d skills from bank", adapter_name, game, len(bank))

        path = base / game / f"{adapter_name}.jsonl"
        rows = _read_jsonl(path)
        for row in rows:
            if adapter_name == "action_taking":
                ex = _align_action_taking_to_coevolution(row, skill_bank=bank or None)
            elif adapter_name == "skill_selection" and bank:
                ex = _normalise_example(row)
                ex["prompt"] = _enrich_skill_selection_prompt(ex["prompt"], bank)
            else:
                ex = _normalise_example(row)
            if ex["prompt"] and ex["completion"]:
                examples.append(ex)
        if rows:
            logger.info(
                "[%s] %s: %d examples from %s",
                adapter_name, game, len(rows), path,
            )
    logger.info("[%s] Total: %d examples across %d games", adapter_name, len(examples), len(games))
    return examples


def _build_skill_descriptions(bank: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Build ``{skill_id: "Name — strategic description"}`` from a skill bank."""
    descs: Dict[str, str] = {}
    for sid, skill in bank.items():
        name = skill.get("name", "")
        strat = skill.get("strategic_description", "")
        if name and strat:
            descs[sid] = f"{name} — {strat}"
        elif name:
            descs[sid] = name
        elif strat:
            descs[sid] = strat
    return descs


def _enrich_segment_prompt(
    prompt: str, descs: Dict[str, str],
) -> str:
    """Replace bare skill ID lists in segment prompts with descriptions.

    Handles two formats:
      ``Candidate skills: ["id1", "id2", ...]``
      ``Candidate next skills: ["id1", "id2", ...]``
    """
    for label in ("Candidate skills:", "Candidate next skills:"):
        idx = prompt.find(label)
        if idx < 0:
            continue
        bracket_start = prompt.find("[", idx)
        bracket_end = prompt.find("]", bracket_start)
        if bracket_start < 0 or bracket_end < 0:
            continue
        raw_list = prompt[bracket_start + 1 : bracket_end]
        skill_ids = [
            s.strip().strip('"').strip("'")
            for s in raw_list.split(",")
            if s.strip().strip('"').strip("'")
        ]
        lines = []
        for sid in skill_ids:
            desc = descs.get(sid, "")
            if desc:
                lines.append(f'  - "{sid}": {desc[:150]}')
            else:
                lines.append(f'  - "{sid}"')
        new_block = "\n".join(lines)
        prompt = prompt[:idx] + label + "\n" + new_block + prompt[bracket_end + 1 :]
    return prompt


def load_segment_data(
    data_dir: str,
    games: List[str],
    skillbank_data_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load cold-start data for the ``segment`` adapter.

    Sources: ``teacher_io_coldstart.jsonl`` per game (segment_ranking,
    transition_ranking, pairwise_choice).

    When *skillbank_data_dir* is provided, enriches prompts with skill
    descriptions (name + strategy) from the skill bank so the model
    learns to rank skills by understanding what they do.
    """
    examples: List[Dict[str, str]] = []
    base = Path(data_dir)
    for game in games:
        descs: Dict[str, str] = {}
        if skillbank_data_dir:
            bank = _load_skill_bank(Path(skillbank_data_dir) / game / "skill_bank.jsonl")
            if bank:
                descs = _build_skill_descriptions(bank)
                logger.info("[segment] %s: loaded %d skill descriptions", game, len(descs))

        path = base / game / "teacher_io_coldstart.jsonl"
        rows = _read_jsonl(path)
        for row in rows:
            ex = _normalise_example(row)
            if descs and ex["prompt"]:
                ex["prompt"] = _enrich_segment_prompt(ex["prompt"], descs)
            if ex["prompt"] and ex["completion"]:
                examples.append(ex)
        if rows:
            logger.info("[segment] %s: %d examples", game, len(rows))
    logger.info("[segment] Total: %d examples", len(examples))
    return examples


def load_coldstart_io_data(
    data_dir: str,
    games: List[str],
    target_adapter: str,
) -> List[Dict[str, str]]:
    """Load cold-start data for ``contract`` or ``curator`` from
    ``coldstart_io_all.jsonl``, filtered by module.

    Module mapping (COLDSTART_IO_MODULE_MAP):
      boundary_proposal → contract
      pipeline          → contract
      skill_naming      → curator
    """
    target_modules = {
        mod for mod, adapter in COLDSTART_IO_MODULE_MAP.items()
        if adapter == target_adapter
    }
    examples: List[Dict[str, str]] = []
    base = Path(data_dir)
    for game in games:
        path = base / game / "coldstart_io_all.jsonl"
        rows = _read_jsonl(path)
        n_matched = 0
        for row in rows:
            module = row.get("module", "")
            if module not in target_modules:
                continue
            ex = _normalise_example(row)
            if ex["prompt"] and ex["completion"]:
                examples.append(ex)
                n_matched += 1
        if n_matched:
            logger.info("[%s] %s: %d examples (from %d total IO records)", target_adapter, game, n_matched, len(rows))
    logger.info("[%s] Total: %d examples", target_adapter, len(examples))
    return examples


def load_adapter_dataset(
    adapter_name: str,
    config: SFTConfig,
) -> List[Dict[str, str]]:
    """Load cold-start data for a single adapter.

    Returns list of ``{"prompt": ..., "completion": ...}`` dicts.
    """
    if adapter_name in DECISION_ADAPTERS:
        return load_decision_adapter_data(
            adapter_name, config.decision_data_dir, config.games,
            skillbank_data_dir=config.skillbank_data_dir,
        )
    elif adapter_name == "segment":
        return load_segment_data(
            config.skillbank_data_dir, config.games,
            skillbank_data_dir=config.skillbank_data_dir,
        )
    elif adapter_name in ("contract", "curator"):
        return load_coldstart_io_data(
            config.skillbank_data_dir, config.games, adapter_name,
        )
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")


def load_all_adapter_datasets(
    config: SFTConfig,
) -> Dict[str, List[Dict[str, str]]]:
    """Load cold-start data for all adapters specified in config.

    Returns ``{adapter_name: [{"prompt": ..., "completion": ...}, ...]}``
    """
    datasets: Dict[str, List[Dict[str, str]]] = {}
    for adapter_name in config.adapters_to_train:
        datasets[adapter_name] = load_adapter_dataset(adapter_name, config)
    return datasets
