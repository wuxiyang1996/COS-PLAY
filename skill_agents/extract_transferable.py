"""
Transferable skill extraction pipeline.

Analyses skill banks across domains/games, discovers reusable patterns,
and produces ``TransferableSkill`` templates that can be instantiated
in any target domain.

Pipeline stages::

    Per-game skill banks
        ↓
    Stage A: Predicate normalisation
        Map game-specific predicates to abstract slot-based predicates.
        ↓
    Stage B: Structural clustering
        Group skills by effect-structure similarity, not surface tags.
        ↓
    Stage C: Template abstraction
        For each cluster, produce a TransferableSkill with reasoning
        protocol, slot bindings, and abstract effects.
        ↓
    Stage D: Transferability scoring
        Score each template on domain coverage, slot coverage,
        protocol quality, and evidence weight.
        ↓
    Stage E: Export
        Write transferable_skills.jsonl + transfer_index.json.

Usage::

    from skill_agents.extract_transferable import (
        extract_transferable_skills,
        TransferableSkillExtractor,
    )

    # From multiple per-game banks
    templates = extract_transferable_skills(
        banks={"2048": bank_2048, "tetris": bank_tetris, "avalon": bank_avalon},
        output_dir="output/transferable",
    )

    # Or step by step
    extractor = TransferableSkillExtractor()
    extractor.ingest_bank(bank_2048, domain="gymv", game="2048")
    extractor.ingest_bank(bank_tetris, domain="gymv", game="tetris")
    extractor.ingest_bank(bank_avalon, domain="gymv", game="avalon")
    extractor.run()
    extractor.export("output/transferable")
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.skill_template import (
    FAMILY_PROTOCOLS,
    SHARED_SLOTS,
    AbstractPredicate,
    HopStep,
    ReasoningProtocol,
    SlotBinding,
    TransferableSkill,
)
from skill_agents.stage3_mvp.schemas import Skill, SkillEffectsContract

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Predicate normalisation rules
# ═══════════════════════════════════════════════════════════════════════

# Regex patterns that map domain predicates to abstract slot names.
# Order matters: first match wins.
_SLOT_PATTERNS: List[Tuple[str, str]] = [
    # target slot
    (r"(?i)(target|goal|objective|focused|selected|anchor|highest)", "target"),
    # blocker slot
    (r"(?i)(block|obstacle|wall|prevent|disable|error|dead|stuck)", "blocker"),
    # constraint slot
    (r"(?i)(constraint|valid|legal|rule|limit|cannot|must|require)", "constraint"),
    # candidate_set slot
    (r"(?i)(candidate|option|choice|available|possible|adjacent|neighbor)", "candidate_set"),
    # history_anchor slot
    (r"(?i)(history|previous|prior|memory|recall|dialogue|conversation)", "history_anchor"),
]

# Regex patterns for classifying predicates into semantic roles.
_ROLE_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(position|place|locate|move|navigate|push)", "spatial"),
    (r"(?i)(merge|combine|clear|remove|delete|destroy)", "transform"),
    (r"(?i)(build|create|setup|construct|form)", "construct"),
    (r"(?i)(attack|defend|survive|protect|threat)", "conflict"),
    (r"(?i)(collect|gather|resource|score|reward|progress)", "accumulate"),
    (r"(?i)(explore|search|discover|find|identify)", "explore"),
    (r"(?i)(plan|decide|choose|select|compare)", "decide"),
]


def infer_slot(predicate: str) -> Optional[str]:
    """Infer which shared schema slot a predicate maps to."""
    for pattern, slot in _SLOT_PATTERNS:
        if re.search(pattern, predicate):
            return slot
    return None


def infer_role(predicate: str) -> str:
    """Classify a predicate into a semantic role."""
    for pattern, role in _ROLE_PATTERNS:
        if re.search(pattern, predicate):
            return role
    return "generic"


def normalise_predicate(predicate: str) -> str:
    """Strip domain-specific noise from a predicate key.

    Removes numeric suffixes, game-specific prefixes like ``world.``,
    ``event.``, and normalises separators.
    """
    p = predicate.strip()
    p = re.sub(r"^(world|event|game|env)\.", "", p)
    p = re.sub(r"[_\s]+", "_", p)
    p = re.sub(r"=\d+(\.\d+)?$", "", p)
    return p.lower()


# ═══════════════════════════════════════════════════════════════════════
# Structural similarity
# ═══════════════════════════════════════════════════════════════════════

def _effect_role_signature(skill: Skill) -> FrozenSet[str]:
    """Compute a role-based signature for a skill's effects.

    Instead of comparing literal predicate strings (game-specific),
    we compare the *semantic roles* of the predicates.
    """
    roles: Set[str] = set()
    if skill.contract:
        for p in skill.contract.eff_add:
            roles.add(f"+{infer_role(p)}")
        for p in skill.contract.eff_del:
            roles.add(f"-{infer_role(p)}")
        for p in skill.contract.eff_event:
            roles.add(f"!{infer_role(p)}")
    return frozenset(roles)


def _tag_signature(skill: Skill) -> str:
    """Extract the intention tag (MERGE, CLEAR, NAVIGATE, etc.)."""
    sid = skill.skill_id
    parts = sid.split(":")
    if len(parts) >= 2:
        return parts[-1].upper()
    for tag in skill.tags:
        t = tag.strip("[]").upper()
        if t and t != "UNKNOWN":
            return t
    return "GENERIC"


def structural_similarity(a: Skill, b: Skill) -> float:
    """Compute structural similarity between two skills (0..1).

    Combines:
    - Role signature Jaccard (semantic effect similarity)
    - Tag match bonus
    - Protocol step-count similarity
    """
    sig_a = _effect_role_signature(a)
    sig_b = _effect_role_signature(b)

    if not sig_a and not sig_b:
        role_sim = 0.5
    elif not sig_a or not sig_b:
        role_sim = 0.0
    else:
        role_sim = len(sig_a & sig_b) / len(sig_a | sig_b)

    tag_a = _tag_signature(a)
    tag_b = _tag_signature(b)
    tag_bonus = 0.2 if tag_a == tag_b else 0.0

    n_a = len(a.protocol.steps) if a.protocol else 0
    n_b = len(b.protocol.steps) if b.protocol else 0
    if n_a + n_b > 0:
        step_sim = 1.0 - abs(n_a - n_b) / max(n_a + n_b, 1)
    else:
        step_sim = 0.5

    return min(1.0, 0.5 * role_sim + 0.3 * tag_bonus + 0.2 * step_sim)


# ═══════════════════════════════════════════════════════════════════════
# Family classification
# ═══════════════════════════════════════════════════════════════════════

def classify_family(skill: Skill) -> str:
    """Assign a skill to one of the 4 transferable families.

    Heuristic based on effect roles and tag:
    - locate_filter_select: spatial/decide + accumulate effects
    - blocker_prerequisite_replan: conflict/blocker effects
    - history_hidden_state_act: decide effects with history dependency
    - compare_under_constraint: decide + constraint effects
    """
    roles = _effect_role_signature(skill)
    tag = _tag_signature(skill)
    role_strs = " ".join(roles)

    blocker_signals = any(
        k in role_strs for k in ("+conflict", "-conflict", "block", "defend")
    )
    if tag in ("DEFEND", "SURVIVE") or blocker_signals:
        return "blocker_prerequisite_replan"

    history_signals = any(
        k in role_strs for k in ("history", "recall")
    )
    if tag in ("EXPLORE",) or history_signals:
        return "history_hidden_state_act"

    constraint_signals = "+decide" in role_strs or "+explore" in role_strs
    if tag in ("OPTIMIZE", "SETUP") or constraint_signals:
        return "compare_under_constraint"

    return "locate_filter_select"


# ═══════════════════════════════════════════════════════════════════════
# Agglomerative clustering
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SkillEntry:
    """Enriched skill entry for clustering."""
    skill: Skill
    domain: str
    game: str
    role_sig: FrozenSet[str]
    tag: str
    family: str
    cluster_id: int = -1


def cluster_skills(
    entries: List[SkillEntry],
    sim_threshold: float = 0.45,
) -> List[List[SkillEntry]]:
    """Single-linkage agglomerative clustering by structural similarity.

    Skills from *different* games/domains that share role signatures
    and tags get grouped together.
    """
    n = len(entries)
    if n == 0:
        return []

    cluster_ids = list(range(n))

    def find(i: int) -> int:
        while cluster_ids[i] != i:
            cluster_ids[i] = cluster_ids[cluster_ids[i]]
            i = cluster_ids[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            cluster_ids[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if entries[i].game == entries[j].game:
                continue
            sim = structural_similarity(entries[i].skill, entries[j].skill)
            if sim >= sim_threshold:
                union(i, j)

    groups: Dict[int, List[SkillEntry]] = defaultdict(list)
    for i, e in enumerate(entries):
        e.cluster_id = find(i)
        groups[e.cluster_id].append(e)

    clusters = [g for g in groups.values() if len(g) >= 1]
    clusters.sort(key=lambda g: (-len(set(e.game for e in g)), -len(g)))
    return clusters


# ═══════════════════════════════════════════════════════════════════════
# Template abstraction
# ═══════════════════════════════════════════════════════════════════════

def abstract_cluster(
    cluster: List[SkillEntry],
    cluster_idx: int,
) -> TransferableSkill:
    """Produce a TransferableSkill template from a cluster of similar skills."""

    games = sorted(set(e.game for e in cluster))
    domains = sorted(set(e.domain for e in cluster))
    tags = Counter(e.tag for e in cluster)
    dominant_tag = tags.most_common(1)[0][0] if tags else "GENERIC"
    families = Counter(e.family for e in cluster)
    family = families.most_common(1)[0][0]

    template_id = f"xfer_{dominant_tag.lower()}_{cluster_idx:03d}"

    # Collect all abstract effects across the cluster
    abstract_add: Dict[str, Dict[str, str]] = defaultdict(dict)
    abstract_del: Dict[str, Dict[str, str]] = defaultdict(dict)
    slot_bindings: List[SlotBinding] = []

    for entry in cluster:
        sk = entry.skill
        if not sk.contract:
            continue

        for pred in sk.contract.eff_add:
            norm = normalise_predicate(pred)
            role = infer_role(pred)
            abstract_key = f"${role}.achieved"
            abstract_add[abstract_key][entry.game] = pred

            slot = infer_slot(pred)
            if slot:
                slot_bindings.append(SlotBinding(
                    slot=slot,
                    domain_predicate=pred,
                    domain=entry.domain,
                    direction="eff_add",
                ))

        for pred in sk.contract.eff_del:
            norm = normalise_predicate(pred)
            role = infer_role(pred)
            abstract_key = f"${role}.removed"
            abstract_del[abstract_key][entry.game] = pred

            slot = infer_slot(pred)
            if slot:
                slot_bindings.append(SlotBinding(
                    slot=slot,
                    domain_predicate=pred,
                    domain=entry.domain,
                    direction="eff_del",
                ))

    abstract_effects: List[AbstractPredicate] = []
    for template, instantiations in abstract_add.items():
        abstract_effects.append(AbstractPredicate(
            template=template,
            direction="eff_add",
            instantiations=dict(instantiations),
        ))
    for template, instantiations in abstract_del.items():
        abstract_effects.append(AbstractPredicate(
            template=template,
            direction="eff_del",
            instantiations=dict(instantiations),
        ))

    # Assign reasoning protocol from family
    reasoning_protocol = FAMILY_PROTOCOLS.get(
        family, FAMILY_PROTOCOLS["locate_filter_select"]
    )

    # Aggregate names and descriptions
    names = [e.skill.name for e in cluster if e.skill.name]
    descriptions = [
        e.skill.strategic_description
        for e in cluster if e.skill.strategic_description
    ]
    pass_rates = [e.skill.success_rate for e in cluster]

    ts = TransferableSkill(
        template_id=template_id,
        family=family,
        name=names[0] if names else f"{dominant_tag} pattern",
        description=(
            descriptions[0] if descriptions
            else f"Cross-domain {dominant_tag} pattern across {', '.join(games)}"
        ),
        tags=[dominant_tag] + [t for t, _ in tags.most_common(3) if t != dominant_tag],
        slot_bindings=_dedup_bindings(slot_bindings),
        abstract_effects=abstract_effects,
        reasoning_protocol=reasoning_protocol,
        source_skills=[
            {"skill_id": e.skill.skill_id, "domain": e.domain, "game": e.game}
            for e in cluster
        ],
        source_domains=domains,
        n_domain_instances=sum(e.skill.n_instances for e in cluster),
        n_domains=len(domains),
        avg_pass_rate=(
            sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        ),
    )
    ts.compute_transferability()
    return ts


def _dedup_bindings(bindings: List[SlotBinding]) -> List[SlotBinding]:
    """Remove duplicate slot bindings (same slot + domain_predicate)."""
    seen: Set[Tuple[str, str]] = set()
    result: List[SlotBinding] = []
    for b in bindings:
        key = (b.slot, b.domain_predicate)
        if key not in seen:
            seen.add(key)
            result.append(b)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main extractor
# ═══════════════════════════════════════════════════════════════════════

class TransferableSkillExtractor:
    """Orchestrates the full extraction pipeline.

    Usage::

        extractor = TransferableSkillExtractor()
        extractor.ingest_bank(bank_2048, domain="gymv", game="2048")
        extractor.ingest_bank(bank_tetris, domain="gymv", game="tetris")
        extractor.run()
        for ts in extractor.templates:
            print(ts.template_id, ts.transferability_score)
        extractor.export("output/transferable")
    """

    def __init__(
        self,
        sim_threshold: float = 0.45,
        min_cluster_games: int = 1,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.min_cluster_games = min_cluster_games
        self._entries: List[SkillEntry] = []
        self._clusters: List[List[SkillEntry]] = []
        self.templates: List[TransferableSkill] = []

    # ── Ingest ───────────────────────────────────────────────────

    def ingest_bank(
        self,
        bank: SkillBankMVP,
        domain: str = "gymv",
        game: str = "",
    ) -> int:
        """Add all skills from a bank into the extraction pool."""
        added = 0
        for sid in bank.skill_ids:
            skill = bank.get_skill(sid)
            if skill is None or skill.retired:
                continue
            entry = SkillEntry(
                skill=skill,
                domain=domain,
                game=game or sid.split(":")[0] if ":" in sid else game,
                role_sig=_effect_role_signature(skill),
                tag=_tag_signature(skill),
                family=classify_family(skill),
            )
            self._entries.append(entry)
            added += 1
        logger.info(
            "Ingested %d skills from %s/%s (total pool: %d)",
            added, domain, game, len(self._entries),
        )
        return added

    def ingest_skill(
        self,
        skill: Skill,
        domain: str = "gymv",
        game: str = "",
    ) -> None:
        """Add a single skill to the extraction pool."""
        self._entries.append(SkillEntry(
            skill=skill,
            domain=domain,
            game=game,
            role_sig=_effect_role_signature(skill),
            tag=_tag_signature(skill),
            family=classify_family(skill),
        ))

    # ── Run pipeline ─────────────────────────────────────────────

    def run(self) -> List[TransferableSkill]:
        """Execute the full extraction pipeline.

        Returns the list of TransferableSkill templates.
        """
        if not self._entries:
            logger.warning("No skills in pool — nothing to extract.")
            return []

        t0 = time.time()

        # Stage B: cluster
        self._clusters = cluster_skills(
            self._entries, sim_threshold=self.sim_threshold,
        )
        logger.info(
            "Clustering: %d skills → %d clusters",
            len(self._entries), len(self._clusters),
        )

        # Stage C: abstract each cluster into a template
        self.templates = []
        for idx, cluster in enumerate(self._clusters):
            n_games = len(set(e.game for e in cluster))
            if n_games < self.min_cluster_games:
                continue
            ts = abstract_cluster(cluster, idx)
            self.templates.append(ts)

        # Stage D: sort by transferability
        self.templates.sort(key=lambda t: -t.transferability_score)

        elapsed = time.time() - t0
        logger.info(
            "Extraction complete: %d templates (%.1fs). "
            "Top 3: %s",
            len(self.templates), elapsed,
            [(t.template_id, round(t.transferability_score, 3))
             for t in self.templates[:3]],
        )
        return self.templates

    # ── Export ────────────────────────────────────────────────────

    def export(self, output_dir: str) -> Dict[str, str]:
        """Write templates to disk.

        Produces:
        - ``transferable_skills.jsonl`` — one template per line
        - ``transfer_index.json`` — summary index for RAG retrieval
        - ``transfer_families.json`` — per-family breakdown

        Returns dict of output file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}

        # JSONL
        jsonl_path = out / "transferable_skills.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ts in self.templates:
                f.write(json.dumps(ts.to_dict(), default=str) + "\n")
        paths["templates"] = str(jsonl_path)

        # RAG index
        index_entries = []
        for ts in self.templates:
            index_entries.append({
                "id": ts.template_id,
                "type": "transferable_skill",
                "family": ts.family,
                "name": ts.name,
                "tags": ts.tags,
                "text": (
                    f"family={ts.family} | "
                    f"tags={','.join(ts.tags)} | "
                    f"domains={','.join(ts.source_domains)} | "
                    f"description={ts.description[:200]}"
                ),
                "transferability": round(ts.transferability_score, 3),
                "n_domains": ts.n_domains,
                "source_domains": ts.source_domains,
                "slot_bindings": [b.slot for b in ts.slot_bindings],
                "hop_actions": ts.reasoning_protocol.action_sequence,
            })

        index_path = out / "transfer_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.time(),
                "n_templates": len(self.templates),
                "entries": index_entries,
            }, f, indent=2, ensure_ascii=False)
        paths["index"] = str(index_path)

        # Per-family breakdown
        by_family: Dict[str, List[dict]] = defaultdict(list)
        for ts in self.templates:
            by_family[ts.family].append({
                "template_id": ts.template_id,
                "name": ts.name,
                "tags": ts.tags,
                "n_domains": ts.n_domains,
                "source_domains": ts.source_domains,
                "transferability": round(ts.transferability_score, 3),
                "n_source_skills": len(ts.source_skills),
            })

        families_path = out / "transfer_families.json"
        with open(families_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.time(),
                "families": {
                    fam: {
                        "n_templates": len(entries),
                        "protocol_hops": (
                            FAMILY_PROTOCOLS[fam].action_sequence
                            if fam in FAMILY_PROTOCOLS else []
                        ),
                        "templates": entries,
                    }
                    for fam, entries in sorted(by_family.items())
                },
            }, f, indent=2, ensure_ascii=False)
        paths["families"] = str(families_path)

        logger.info("Exported %d templates to %s", len(self.templates), out)
        return paths

    # ── Stats ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary of extraction results."""
        family_counts = Counter(t.family for t in self.templates)
        return {
            "n_input_skills": len(self._entries),
            "n_clusters": len(self._clusters),
            "n_templates": len(self.templates),
            "n_multi_domain": sum(
                1 for t in self.templates if t.n_domains >= 2
            ),
            "families": dict(family_counts),
            "avg_transferability": (
                sum(t.transferability_score for t in self.templates)
                / max(len(self.templates), 1)
            ),
            "top_5": [
                {
                    "id": t.template_id,
                    "family": t.family,
                    "score": round(t.transferability_score, 3),
                    "domains": t.source_domains,
                }
                for t in self.templates[:5]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════

def extract_transferable_skills(
    banks: Dict[str, SkillBankMVP],
    output_dir: str = "output/transferable",
    domain: str = "gymv",
    sim_threshold: float = 0.45,
    min_cluster_games: int = 1,
) -> List[TransferableSkill]:
    """One-call extraction from multiple per-game banks.

    Parameters
    ----------
    banks : dict
        Mapping of game_name → SkillBankMVP.
    output_dir : str
        Where to write output files.
    domain : str
        Domain for all input banks (default: gymv).
    sim_threshold : float
        Structural similarity threshold for clustering.
    min_cluster_games : int
        Minimum number of distinct games in a cluster.

    Returns
    -------
    list[TransferableSkill]
        Sorted by transferability score (descending).
    """
    extractor = TransferableSkillExtractor(
        sim_threshold=sim_threshold,
        min_cluster_games=min_cluster_games,
    )

    for game, bank in banks.items():
        extractor.ingest_bank(bank, domain=domain, game=game)

    extractor.run()
    extractor.export(output_dir)
    return extractor.templates
