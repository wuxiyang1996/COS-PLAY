"""
Action language formatters for skill contracts.

Converts ``SkillContract`` objects into formal planning representations:

  - **pddl**    — PDDL 2.1 ``:action`` blocks with ``:precondition`` /
                   ``:effect`` / ``:invariant`` (as comments).
  - **strips**  — Classic STRIPS operator notation
                   ``Pre: {...}, Add: {...}, Del: {...}``.
  - **sas**     — SAS+ style ``prevail`` / ``pre_post`` pairs.
  - **compact** — Readable one-liner per skill for LLM context injection.

All formatters operate on individual contracts or on a full ``SkillBank``.

Usage::

    from skill_agents_grpo.contract_verification.action_language import (
        contract_to_pddl,
        contract_to_strips,
        bank_to_pddl_domain,
        format_contract,
    )

    # Single contract
    pddl_str = contract_to_pddl(contract)

    # All active skills in a bank
    domain_str = bank_to_pddl_domain(bank, domain_name="game_skills")

    # Format-agnostic (driven by config string)
    text = format_contract(contract, fmt="strips")
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skill_agents_grpo.contract_verification.schemas import SkillContract

SUPPORTED_FORMATS = ("pddl", "strips", "sas", "compact")


# ── Helpers ──────────────────────────────────────────────────────────

def _pred_name(p: str) -> str:
    """Sanitise a predicate string for use in PDDL/STRIPS identifiers."""
    return p.replace(" ", "_").replace("-", "_").lower()


def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


# ── PDDL ─────────────────────────────────────────────────────────────

def contract_to_pddl(
    contract: "SkillContract",
    include_inv: bool = True,
    include_soft_pre: bool = True,
    include_version: bool = True,
) -> str:
    """Render a ``SkillContract`` as a PDDL ``:action`` block.

    Invariants are emitted as structured comments (``;; :invariant``)
    since standard PDDL has no invariant keyword.  Soft preconditions
    are emitted under a ``;; :soft-precondition`` comment block.
    """
    name = _pred_name(contract.skill_id)
    lines: List[str] = []

    header = f"(:action {name}"
    if include_version:
        header += f"  ; v{contract.version}"
    lines.append(header)

    lines.append("  :parameters ()")

    # Precondition
    pre_preds = sorted(contract.pre)
    if pre_preds:
        if len(pre_preds) == 1:
            lines.append(f"  :precondition ({_pred_name(pre_preds[0])})")
        else:
            lines.append("  :precondition (and")
            for p in pre_preds:
                lines.append(f"    ({_pred_name(p)})")
            lines.append("  )")
    else:
        lines.append("  :precondition ()")

    # Effect
    eff_parts: List[str] = []
    for p in sorted(contract.eff_add):
        eff_parts.append(f"({_pred_name(p)})")
    for p in sorted(contract.eff_del):
        eff_parts.append(f"(not ({_pred_name(p)}))")

    if eff_parts:
        if len(eff_parts) == 1:
            lines.append(f"  :effect {eff_parts[0]}")
        else:
            lines.append("  :effect (and")
            for ep in eff_parts:
                lines.append(f"    {ep}")
            lines.append("  )")
    else:
        lines.append("  :effect ()")

    # Invariants (as comments since PDDL has no native support)
    if include_inv and contract.inv:
        lines.append("  ;; :invariant (and")
        for p in sorted(contract.inv):
            lines.append(f"  ;;   ({_pred_name(p)})")
        lines.append("  ;; )")

    # Soft preconditions (as comments)
    if include_soft_pre and contract.soft_pre:
        lines.append("  ;; :soft-precondition (and")
        for p in sorted(contract.soft_pre):
            lines.append(f"  ;;   ({_pred_name(p)})")
        lines.append("  ;; )")

    # Termination cues (as comments)
    if contract.termination_cues:
        lines.append("  ;; :termination-cues (or")
        for p in sorted(contract.termination_cues):
            lines.append(f"  ;;   ({_pred_name(p)})")
        lines.append("  ;; )")

    lines.append(")")
    return "\n".join(lines)


# ── STRIPS ───────────────────────────────────────────────────────────

def contract_to_strips(
    contract: "SkillContract",
    include_inv: bool = True,
    include_version: bool = True,
) -> str:
    """Render a ``SkillContract`` in classic STRIPS operator notation."""
    name = contract.skill_id
    lines: List[str] = []

    header = f"Operator: {name}"
    if include_version:
        header += f" (v{contract.version})"
    lines.append(header)

    pre_str = ", ".join(sorted(contract.pre)) if contract.pre else "(none)"
    lines.append(f"  Pre:  {{{pre_str}}}")

    add_str = ", ".join(sorted(contract.eff_add)) if contract.eff_add else "(none)"
    lines.append(f"  Add:  {{{add_str}}}")

    del_str = ", ".join(sorted(contract.eff_del)) if contract.eff_del else "(none)"
    lines.append(f"  Del:  {{{del_str}}}")

    if include_inv and contract.inv:
        inv_str = ", ".join(sorted(contract.inv))
        lines.append(f"  Inv:  {{{inv_str}}}")

    if contract.soft_pre:
        soft_str = ", ".join(sorted(contract.soft_pre))
        lines.append(f"  Soft: {{{soft_str}}}")

    if contract.termination_cues:
        term_str = ", ".join(sorted(contract.termination_cues))
        lines.append(f"  Term: {{{term_str}}}")

    return "\n".join(lines)


# ── SAS+ ─────────────────────────────────────────────────────────────

def contract_to_sas(
    contract: "SkillContract",
    include_version: bool = True,
) -> str:
    """Render a ``SkillContract`` in SAS+ variable-value style.

    SAS+ represents state variables with finite domains.  Since our
    predicates are boolean, each predicate is a binary variable
    (0 = false, 1 = true).

    Format::

        operator <name>
        prevail: var=val, ...       (invariants + preconditions that don't change)
        pre_post: var pre->post, ... (effects)
    """
    name = contract.skill_id
    lines: List[str] = []

    header = f"operator {name}"
    if include_version:
        header += f"  ; v{contract.version}"
    lines.append(header)

    # Prevail: preconditions that are not deleted (remain true)
    prevail_vars = sorted(contract.pre - contract.eff_del)
    inv_only = sorted(contract.inv - contract.pre)
    all_prevail = prevail_vars + inv_only
    if all_prevail:
        pairs = [f"{v}=1" for v in all_prevail]
        lines.append(f"  prevail: {', '.join(pairs)}")
    else:
        lines.append("  prevail: (none)")

    # Pre-post: variables that change value
    pre_post_parts: List[str] = []
    for p in sorted(contract.eff_add):
        if p in contract.pre:
            continue  # already true, no change needed
        pre_post_parts.append(f"{p} 0->1")
    for p in sorted(contract.eff_del):
        pre_post_parts.append(f"{p} 1->0")
    # Preconditions that get deleted (consumed)
    for p in sorted(contract.pre & contract.eff_del):
        pre_post_parts.append(f"{p} 1->0 (consumed)")

    if pre_post_parts:
        lines.append(f"  pre_post: {', '.join(pre_post_parts)}")
    else:
        lines.append("  pre_post: (none)")

    return "\n".join(lines)


# ── Compact ──────────────────────────────────────────────────────────

def contract_to_compact(
    contract: "SkillContract",
    include_version: bool = True,
) -> str:
    """One-line compact representation for LLM context / logging."""
    parts: List[str] = [contract.skill_id]
    if include_version:
        parts[0] += f"/v{contract.version}"

    if contract.pre:
        parts.append(f"PRE({','.join(sorted(contract.pre))})")
    if contract.eff_add:
        parts.append(f"+({','.join(sorted(contract.eff_add))})")
    if contract.eff_del:
        parts.append(f"-({','.join(sorted(contract.eff_del))})")
    if contract.inv:
        parts.append(f"INV({','.join(sorted(contract.inv))})")
    if contract.soft_pre:
        parts.append(f"~PRE({','.join(sorted(contract.soft_pre))})")
    if contract.termination_cues:
        parts.append(f"TERM({','.join(sorted(contract.termination_cues))})")

    if not any(k in " ".join(parts) for k in ["PRE", "+", "-", "INV"]):
        parts.append("(empty contract)")

    return " | ".join(parts)


# ── Format dispatcher ───────────────────────────────────────────────

def format_contract(
    contract: "SkillContract",
    fmt: str = "pddl",
    **kwargs,
) -> str:
    """Format a single contract in the requested action language.

    Parameters
    ----------
    contract : SkillContract
    fmt : str
        One of ``"pddl"``, ``"strips"``, ``"sas"``, ``"compact"``.
    **kwargs
        Passed to the underlying formatter (e.g. ``include_inv``).
    """
    fmt = fmt.lower()
    if fmt == "pddl":
        return contract_to_pddl(contract, **kwargs)
    elif fmt == "strips":
        return contract_to_strips(contract, **kwargs)
    elif fmt == "sas":
        return contract_to_sas(contract, **kwargs)
    elif fmt == "compact":
        return contract_to_compact(contract, **kwargs)
    else:
        raise ValueError(
            f"Unknown action language format {fmt!r}. "
            f"Supported: {SUPPORTED_FORMATS}"
        )


# ── Bank-level formatting ───────────────────────────────────────────

def bank_to_pddl_domain(
    bank,
    domain_name: str = "skill_domain",
    include_inv: bool = True,
) -> str:
    """Export all active skills in a bank as a PDDL domain file.

    Automatically derives the ``:predicates`` section from all
    predicates referenced across contracts.
    """
    all_preds: set = set()
    action_blocks: List[str] = []

    for skill_id in sorted(bank.active_skill_ids):
        contract = bank.get_contract(skill_id)
        if contract is None:
            continue
        all_preds.update(contract.pre)
        all_preds.update(contract.eff_add)
        all_preds.update(contract.eff_del)
        all_preds.update(contract.inv)
        all_preds.update(contract.soft_pre)
        all_preds.update(contract.termination_cues)
        action_blocks.append(contract_to_pddl(contract, include_inv=include_inv))

    lines: List[str] = [
        f"(define (domain {_pred_name(domain_name)})",
        "  (:requirements :strips :typing)",
        "",
        "  (:predicates",
    ]
    for p in sorted(all_preds):
        lines.append(f"    ({_pred_name(p)})")
    lines.append("  )")
    lines.append("")

    for block in action_blocks:
        lines.append(_indent(block, 2))
        lines.append("")

    lines.append(")")
    return "\n".join(lines)


def bank_to_action_language(
    bank,
    fmt: str = "pddl",
    domain_name: str = "skill_domain",
    **kwargs,
) -> str:
    """Export all active skills in a bank in the requested format.

    Parameters
    ----------
    bank : SkillBank
    fmt : str
        One of ``"pddl"``, ``"strips"``, ``"sas"``, ``"compact"``.
    domain_name : str
        Used as the PDDL domain name (only relevant for ``"pddl"``).
    """
    fmt = fmt.lower()
    if fmt == "pddl":
        return bank_to_pddl_domain(bank, domain_name=domain_name, **kwargs)

    blocks: List[str] = []
    for skill_id in sorted(bank.active_skill_ids):
        contract = bank.get_contract(skill_id)
        if contract is None:
            continue
        blocks.append(format_contract(contract, fmt=fmt, **kwargs))

    separator = "\n" if fmt == "compact" else "\n\n"
    return separator.join(blocks)
