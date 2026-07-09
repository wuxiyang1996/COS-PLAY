"""
Skill query engine — evolving from pure retriever to skill selection policy.

The query layer exposes two conceptual dimensions to the decision agent:

  1. **Retrieval relevance** — how well a skill's description / effects match
     the query (embedding similarity + keyword Jaccard).
  2. **Execution applicability** — how confident we are that the skill can
     actually execute successfully in the current state (contract match,
     verification pass rate, supporting evidence).

Decision agents should consume the rich ``SkillSelectionResult`` returned by
``select()`` rather than the flat dicts from ``query()``.  The old ``query()``
API is preserved for backward compatibility.

Usage::

    from skill_agents_grpo.query import SkillQueryEngine
    from skill_agents_grpo.skill_bank.bank import SkillBankMVP

    bank = SkillBankMVP("bank.jsonl"); bank.load()
    engine = SkillQueryEngine(bank)

    # Simple retrieval (backward compatible)
    results = engine.query("navigate to pot and place onion", top_k=3)

    # Rich skill selection for decision agents
    selection = engine.select(
        query="navigate to pot",
        current_state={"near_pot": 0.1, "holding_onion": 0.9},
        top_k=3,
    )
    for r in selection:
        print(r.skill_id, r.relevance, r.applicability, r.confidence)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class SelectionTracker:
    """Lightweight per-iteration counter for UCB exploration in skill selection."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = defaultdict(int)
        self._total: int = 0

    def increment(self, skill_id: str) -> None:
        self._counts[skill_id] += 1
        self._total += 1

    def get(self, skill_id: str) -> Tuple[int, int]:
        """Return (n_skill, n_total) for UCB computation."""
        return self._counts[skill_id], self._total

    def get_all_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def reset(self) -> None:
        self._counts.clear()
        self._total = 0

from skill_agents_grpo.skill_bank.bank import SkillBankMVP, _effects_compat_score
from skill_agents_grpo.stage3_mvp.schemas import SkillEffectsContract, VerificationReport


def _tokenize(text: str) -> Set[str]:
    """Lowercase tokenisation for keyword overlap scoring."""
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _effect_set_for_contract(c: SkillEffectsContract) -> Set[str]:
    return (c.eff_add or set()) | (c.eff_del or set()) | (c.eff_event or set())


def _contract_summary(c: SkillEffectsContract) -> Dict[str, Any]:
    return {
        "skill_id": c.skill_id,
        "version": c.version,
        "eff_add": sorted(c.eff_add) if c.eff_add else [],
        "eff_del": sorted(c.eff_del) if c.eff_del else [],
        "eff_event": sorted(c.eff_event) if c.eff_event else [],
        "n_literals": c.total_literals,
        "n_instances": c.n_instances,
    }


def _skill_description(
    sid: str,
    c: Optional[SkillEffectsContract],
    skill: Optional[Any] = None,
) -> str:
    """Build a textual description of a skill for embedding.

    When a ``Skill`` wrapper is available, includes strategic_description
    and protocol preconditions for richer semantic matching.
    """
    parts = [sid.replace("_", " ")]
    if skill is not None:
        if getattr(skill, "name", None):
            parts.append(skill.name)
        if getattr(skill, "strategic_description", None):
            parts.append(skill.strategic_description)
        protocol = getattr(skill, "protocol", None)
        if protocol is not None:
            for pc in getattr(protocol, "preconditions", [])[:5]:
                parts.append(pc)
            for step in getattr(protocol, "steps", [])[:3]:
                parts.append(step)
    if c is not None:
        if getattr(c, "name", None) and not (skill and getattr(skill, "name", None)):
            parts.append(c.name)
        if getattr(c, "description", None) and not (skill and getattr(skill, "strategic_description", None)):
            parts.append(c.description)
        for lit in sorted(c.eff_add or set()):
            parts.append(lit)
        for lit in sorted(c.eff_del or set()):
            parts.append(lit)
        for lit in sorted(c.eff_event or set()):
            parts.append(lit)
    return " ".join(parts)


# ── Rich result type for decision agents ─────────────────────────────

@dataclass
class SkillSelectionResult:
    """Structured guidance package returned by the skill selection policy.

    Decision agents use these fields to decide which skill to execute and
    how to carry it out.  The package goes well beyond retrieval relevance:
    it provides *why* the skill was selected, *what* it will achieve,
    *when* it is valid, *how* to execute it, *when* it is done, and *what*
    can go wrong.
    """

    skill_id: str
    skill_name: str = ""
    why_selected: str = ""
    applicability_score: float = 0.0
    relevance: float = 0.0
    confidence: float = 0.0

    expected_effects: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    termination_hint: str = ""
    failure_modes: List[str] = field(default_factory=list)
    execution_hint: str = ""

    contract_match_score: float = 0.0
    pass_rate: Optional[float] = None
    n_instances: int = 0
    matched_effects: List[str] = field(default_factory=list)
    missing_effects: List[str] = field(default_factory=list)
    contract: Dict[str, Any] = field(default_factory=dict)
    micro_plan: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "why_selected": self.why_selected,
            "applicability_score": round(self.applicability_score, 4),
            "relevance": round(self.relevance, 4),
            "confidence": round(self.confidence, 4),
            "expected_effects": self.expected_effects,
            "preconditions": self.preconditions,
            "termination_hint": self.termination_hint,
            "failure_modes": self.failure_modes,
            "execution_hint": self.execution_hint,
            "contract_match_score": round(self.contract_match_score, 4),
            "pass_rate": round(self.pass_rate, 3) if self.pass_rate is not None else None,
            "n_instances": self.n_instances,
            "matched_effects": self.matched_effects,
            "missing_effects": self.missing_effects,
            "contract": self.contract,
            "micro_plan": self.micro_plan,
        }


class SkillQueryEngine:
    """Query interface over a SkillBankMVP with RAG embedding support.

    Supports:
    - **embedding query**: cosine similarity over skill descriptions via RAG.
    - **keyword query**: match query tokens against skill ID and effect literals.
    - **effect query**: find skills whose effects overlap with desired state changes.
    - **select**: rich skill selection policy combining relevance + applicability.
    - **list / detail**: enumerate skills or inspect one in depth.

    The ``select()`` method is the preferred API for decision agents.  The old
    ``query()`` API is preserved for backward compatibility.
    """

    def __init__(
        self,
        bank: SkillBankMVP,
        embedder: Any = None,
        embedding_weight: float = 0.6,
    ) -> None:
        self._bank = bank
        self._embedding_weight = embedding_weight
        self._embedder = embedder
        self.selection_tracker = SelectionTracker()
        if self._embedder is None:
            try:
                from rag import get_text_embedder
                self._embedder = get_text_embedder(device="cpu", shared=True)
            except Exception:
                self._embedder = None
        self._skill_embeddings: Optional[np.ndarray] = None
        self._skill_id_order: List[str] = []
        self._build_index()

    def _build_index(self) -> None:
        """Pre-tokenise skill IDs/effects and embed skill descriptions."""
        self._id_tokens: Dict[str, Set[str]] = {}
        self._effect_tokens: Dict[str, Set[str]] = {}
        self._effect_sets: Dict[str, Set[str]] = {}
        self._strategic_tokens: Dict[str, Set[str]] = {}

        descs: List[str] = []
        self._skill_id_order = list(self._bank.skill_ids)

        has_get_skill = hasattr(self._bank, "get_skill")

        for sid in self._skill_id_order:
            self._id_tokens[sid] = _tokenize(sid)
            c = self._bank.get_contract(sid)
            skill = self._bank.get_skill(sid) if has_get_skill else None

            if c is not None:
                eff = _effect_set_for_contract(c)
                self._effect_sets[sid] = eff
                self._effect_tokens[sid] = set()
                for lit in eff:
                    self._effect_tokens[sid] |= _tokenize(lit)
            else:
                self._effect_sets[sid] = set()
                self._effect_tokens[sid] = set()

            strat_text = ""
            if skill is not None:
                strat_text = " ".join(filter(None, [
                    getattr(skill, "name", ""),
                    getattr(skill, "strategic_description", ""),
                ]))
            self._strategic_tokens[sid] = _tokenize(strat_text) if strat_text else set()

            descs.append(_skill_description(sid, c, skill))

        if self._embedder is not None and descs:
            try:
                self._skill_embeddings = self._embedder.encode(
                    descs, prompt_name="passage",
                )
            except Exception:
                self._skill_embeddings = None

    def rebuild_index(self) -> None:
        """Re-index after the skill bank has been mutated."""
        self._build_index()

    @property
    def has_embedder(self) -> bool:
        return self._skill_embeddings is not None

    def _embedding_scores(self, query: str) -> Optional[np.ndarray]:
        """Return per-skill cosine similarity scores for *query*, or None."""
        if self._embedder is None or self._skill_embeddings is None:
            return None
        try:
            q_emb = self._embedder.encode(query, prompt_name="query")
            q_emb = np.atleast_2d(q_emb).astype(np.float32)
            scores = (q_emb @ self._skill_embeddings.T).squeeze(0)
            return scores
        except Exception:
            return None

    # ── Retrieval relevance (internal) ───────────────────────────────

    def _compute_relevance(self, query: str) -> Dict[str, float]:
        """Compute retrieval relevance for all skills given a query.

        Incorporates strategic_description and protocol precondition tokens
        when available for richer semantic matching.
        """
        q_tokens = _tokenize(query)
        emb_scores = self._embedding_scores(query)
        w = self._embedding_weight if emb_scores is not None else 0.0

        scores: Dict[str, float] = {}
        for i, sid in enumerate(self._skill_id_order):
            id_score = _jaccard(q_tokens, self._id_tokens.get(sid, set()))
            eff_score = _jaccard(q_tokens, self._effect_tokens.get(sid, set()))
            strat_score = _jaccard(q_tokens, self._strategic_tokens.get(sid, set()))
            # Blend: strategic > id > effects
            kw_score = 0.35 * strat_score + 0.35 * id_score + 0.30 * eff_score
            emb = float(emb_scores[i]) if emb_scores is not None else 0.0
            scores[sid] = w * emb + (1.0 - w) * kw_score
        return scores

    # ── Execution applicability (internal) ───────────────────────────

    def _compute_applicability(
        self,
        sid: str,
        current_state: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Compute applicability score and effect match details for a skill.

        Returns (applicability_score, matched_effects, missing_effects).
        """
        c = self._bank.get_contract(sid)
        if c is None:
            return 0.0, [], []

        if current_state is None:
            # Without state info, rely on pass rate as a proxy
            r = self._bank.get_report(sid)
            pass_rate = r.overall_pass_rate if r else 0.5
            return pass_rate - 0.5, [], []  # center around 0

        # Use the effects compat scorer
        compat = _effects_compat_score(c, current_state, current_state)

        matched = []
        missing = []
        for lit in (c.eff_add or set()):
            val = current_state.get(lit)
            if val is not None:
                matched.append(lit)
            else:
                missing.append(lit)
        for lit in (c.eff_del or set()):
            val = current_state.get(lit)
            if val is not None:
                matched.append(lit)
            else:
                missing.append(lit)

        return compat, sorted(matched), sorted(missing)

    def _compute_confidence(
        self,
        sid: str,
        relevance: float,
        applicability: float,
        n_skill: int = 0,
        n_total: int = 0,
    ) -> float:
        """Combined confidence with UCB exploration bonus.

        exploit = 0.40 * relevance + 0.30 * norm_applicability + 0.30 * pass_rate
        explore = 0.15 * sqrt(ln(n_total + 1) / (n_skill + 1))
        """
        r = self._bank.get_report(sid)
        pass_rate = r.overall_pass_rate if r else 0.5

        norm_app = (applicability + 1.0) / 2.0  # map [-1,1] to [0,1]

        exploit = 0.40 * relevance + 0.30 * norm_app + 0.30 * pass_rate
        explore = 0.15 * math.sqrt(math.log(n_total + 1) / (n_skill + 1))
        return exploit + explore

    # ── Rich selection API (preferred for decision agents) ───────────

    def _build_guidance_fields(
        self,
        sid: str,
        query: str,
        relevance: float,
        applicability: float,
        matched: List[str],
        missing: List[str],
        current_state: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Derive structured guidance fields for a skill selection result.

        Returns a dict with: skill_name, why_selected, expected_effects,
        preconditions, termination_hint, failure_modes, execution_hint,
        micro_plan.
        """
        c = self._bank.get_contract(sid)
        r = self._bank.get_report(sid)
        has_get_skill = hasattr(self._bank, "get_skill")
        skill = self._bank.get_skill(sid) if has_get_skill else None

        skill_name = ""
        if skill is not None and getattr(skill, "name", ""):
            skill_name = skill.name
        elif c is not None and getattr(c, "name", ""):
            skill_name = c.name
        else:
            skill_name = sid.replace("_", " ")

        # --- why_selected ---
        reasons = []
        if relevance > 0.3:
            reasons.append(f"matches query '{query[:60]}' (rel={relevance:.2f})")
        if applicability > 0.2:
            reasons.append(f"applicable in current state (app={applicability:.2f})")
        if matched:
            reasons.append(f"effects align: {', '.join(matched[:3])}")
        pass_rate = r.overall_pass_rate if r else None
        if pass_rate is not None and pass_rate >= 0.7:
            reasons.append(f"reliable (pass_rate={pass_rate:.0%})")
        why_selected = "; ".join(reasons) if reasons else "best available match"

        # --- expected_effects ---
        expected_effects: List[str] = []
        if c is not None:
            for lit in sorted(c.eff_add or set()):
                expected_effects.append(f"+{lit}")
            for lit in sorted(c.eff_del or set()):
                expected_effects.append(f"-{lit}")
            for lit in sorted(c.eff_event or set()):
                expected_effects.append(f"event:{lit}")

        # --- preconditions ---
        preconditions: List[str] = []
        if skill is not None and skill.protocol.preconditions:
            preconditions = list(skill.protocol.preconditions)
        elif missing and current_state is not None:
            preconditions = [f"needs: {lit}" for lit in missing[:5]]

        # --- termination_hint ---
        termination_hint = ""
        if skill is not None and skill.protocol.success_criteria:
            termination_hint = "; ".join(skill.protocol.success_criteria[:3])
        elif c is not None and c.eff_add:
            termination_hint = "done when: " + ", ".join(
                sorted(c.eff_add)[:3]
            ) + " become true"

        # --- failure_modes ---
        failure_modes: List[str] = []
        if r is not None and r.failure_signatures:
            for sig, count in sorted(
                r.failure_signatures.items(), key=lambda x: -x[1]
            )[:3]:
                failure_modes.append(f"{sig} ({count}x)")
        if skill is not None and skill.protocol.abort_criteria:
            failure_modes.extend(skill.protocol.abort_criteria[:2])

        # --- execution_hint ---
        # Prefer strategic_description (clean natural-language strategy)
        # over ExecutionHint.execution_description (often contains raw game
        # state examples).  This keeps the "Strategy:" line in skill
        # selection prompts aligned with SFT training data.
        execution_hint = ""
        if skill is not None and skill.strategic_description:
            execution_hint = skill.strategic_description
        if skill is not None and getattr(skill, "execution_hint", None) is not None:
            eh = skill.execution_hint
            if not execution_hint:
                execution_hint = eh.execution_description or ""
                if not execution_hint and eh.state_transition_pattern:
                    execution_hint = eh.state_transition_pattern
            if eh.termination_cues and not termination_hint:
                termination_hint = "; ".join(eh.termination_cues[:3])
            if eh.common_failure_modes and not failure_modes:
                failure_modes = list(eh.common_failure_modes[:3])
            if eh.common_preconditions and not preconditions:
                preconditions = list(eh.common_preconditions[:5])
        if not execution_hint:
            if skill is not None and skill.protocol.steps:
                execution_hint = " → ".join(skill.protocol.steps[:4])
            elif c is not None and c.description:
                execution_hint = c.description

        # --- micro_plan ---
        if skill is not None and skill.protocol.steps:
            micro_plan = [{"action": step} for step in skill.protocol.steps[:7]]
        elif c:
            micro_plan = [
                {"action": None, "effect": lit}
                for lit in sorted(c.eff_add or set())[:7]
            ]
        else:
            micro_plan = []

        return {
            "skill_name": skill_name,
            "why_selected": why_selected,
            "expected_effects": expected_effects,
            "preconditions": preconditions,
            "termination_hint": termination_hint,
            "failure_modes": failure_modes,
            "execution_hint": execution_hint,
            "micro_plan": micro_plan,
        }

    def select(
        self,
        query: str,
        current_state: Optional[Dict[str, float]] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[SkillSelectionResult]:
        """Rich skill selection combining retrieval relevance with execution
        applicability and structured guidance.

        This is the preferred API for decision agents.  It separates "is this
        skill relevant to what I want?" from "can this skill execute now?"
        and provides actionable execution guidance.

        Parameters
        ----------
        query : str
            Natural-language description of the desired action/goal.
        current_state : dict, optional
            Current predicate state as ``{predicate: probability}``.
            When provided, enables contract-based applicability scoring.
        current_predicates : dict, optional
            Alias for *current_state* (either works; *current_state* takes
            precedence when both are provided).
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SkillSelectionResult]
            Sorted by confidence (highest first).  Each result includes
            structured guidance fields (why_selected, expected_effects,
            preconditions, termination_hint, failure_modes, execution_hint).
        """
        state = current_state or current_predicates
        relevance_scores = self._compute_relevance(query)

        results: List[SkillSelectionResult] = []
        for sid in self._skill_id_order:
            rel = relevance_scores.get(sid, 0.0)
            app, matched, missing = self._compute_applicability(sid, state)
            n_skill, n_total = self.selection_tracker.get(sid)
            conf = self._compute_confidence(sid, rel, app, n_skill=n_skill, n_total=n_total)

            c = self._bank.get_contract(sid)
            r = self._bank.get_report(sid)

            guidance = self._build_guidance_fields(
                sid, query, rel, app, matched, missing, state,
            )

            result = SkillSelectionResult(
                skill_id=sid,
                skill_name=guidance["skill_name"],
                why_selected=guidance["why_selected"],
                applicability_score=app,
                relevance=rel,
                confidence=conf,
                expected_effects=guidance["expected_effects"],
                preconditions=guidance["preconditions"],
                termination_hint=guidance["termination_hint"],
                failure_modes=guidance["failure_modes"],
                execution_hint=guidance["execution_hint"],
                contract_match_score=app,
                pass_rate=r.overall_pass_rate if r else None,
                n_instances=c.n_instances if c else 0,
                matched_effects=matched,
                missing_effects=missing,
                contract=_contract_summary(c) if c else {},
                micro_plan=guidance["micro_plan"],
            )
            results.append(result)

        results.sort(key=lambda r: -r.confidence)
        return results[:top_k]

    # ── Backward-compatible query API ────────────────────────────────

    def query(self, key: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Score every skill against a natural-language query key.

        Backward-compatible API.  For richer results, use ``select()``.
        """
        q_tokens = _tokenize(key)
        if not q_tokens:
            return self.list_all()[:top_k]

        emb_scores = self._embedding_scores(key)
        w = self._embedding_weight if emb_scores is not None else 0.0

        scored: List[tuple] = []
        for i, sid in enumerate(self._skill_id_order):
            id_score = _jaccard(q_tokens, self._id_tokens.get(sid, set()))
            eff_score = _jaccard(q_tokens, self._effect_tokens.get(sid, set()))
            kw_score = 0.6 * id_score + 0.4 * eff_score
            emb = float(emb_scores[i]) if emb_scores is not None else 0.0
            total = w * emb + (1.0 - w) * kw_score
            scored.append((total, sid))

        scored.sort(key=lambda x: -x[0])
        results: List[Dict[str, Any]] = []
        for score, sid in scored[:top_k]:
            c = self._bank.get_contract(sid)
            entry: Dict[str, Any] = {
                "skill_id": sid,
                "score": round(score, 4),
            }
            if c is not None:
                entry["contract"] = _contract_summary(c)
                entry["micro_plan"] = [
                    {"action": None, "effect": lit}
                    for lit in sorted(c.eff_add or set())[:7]
                ]
            else:
                entry["contract"] = {}
                entry["micro_plan"] = []
            results.append(entry)
        return results

    # ── Effect-based query ───────────────────────────────────────────

    def query_by_effects(
        self,
        desired_add: Optional[set] = None,
        desired_del: Optional[set] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find skills whose contract effects best match desired state changes."""
        query_set = (desired_add or set()) | (desired_del or set())
        if not query_set:
            return self.list_all()[:top_k]

        query_text = " ".join(sorted(query_set))
        emb_scores = self._embedding_scores(query_text)
        w = self._embedding_weight * 0.5 if emb_scores is not None else 0.0

        scored: List[tuple] = []
        for i, sid in enumerate(self._skill_id_order):
            eff = self._effect_sets.get(sid, set())
            sim = _jaccard(query_set, eff)
            if desired_add:
                c = self._bank.get_contract(sid)
                if c and c.eff_add:
                    add_overlap = len((desired_add or set()) & c.eff_add) / max(len(desired_add or set()), 1)
                    sim = 0.5 * sim + 0.5 * add_overlap
            emb = float(emb_scores[i]) if emb_scores is not None else 0.0
            total = w * emb + (1.0 - w) * sim
            scored.append((total, sid))

        scored.sort(key=lambda x: -x[0])
        results: List[Dict[str, Any]] = []
        for score, sid in scored[:top_k]:
            c = self._bank.get_contract(sid)
            entry: Dict[str, Any] = {
                "skill_id": sid,
                "effect_match_score": round(score, 4),
            }
            if c is not None:
                entry["contract"] = _contract_summary(c)
            results.append(entry)
        return results

    # ── List / detail ────────────────────────────────────────────────

    def list_all(self) -> List[Dict[str, Any]]:
        """Return a compact summary for every skill in the bank."""
        entries: List[Dict[str, Any]] = []
        for sid in self._bank.skill_ids:
            c = self._bank.get_contract(sid)
            r = self._bank.get_report(sid)
            entry: Dict[str, Any] = {"skill_id": sid}
            if c is not None:
                entry["version"] = c.version
                if getattr(c, "name", None):
                    entry["name"] = c.name
                if getattr(c, "description", None):
                    entry["description"] = c.description
                entry["n_add"] = len(c.eff_add) if c.eff_add else 0
                entry["n_del"] = len(c.eff_del) if c.eff_del else 0
                entry["n_event"] = len(c.eff_event) if c.eff_event else 0
                entry["n_instances"] = c.n_instances
            if r is not None:
                entry["pass_rate"] = round(r.overall_pass_rate, 3)
            entries.append(entry)
        return entries

    def get_detail(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Full detail for one skill: contract + report + quality info."""
        c = self._bank.get_contract(skill_id)
        if c is None:
            return None

        detail: Dict[str, Any] = {"skill_id": skill_id}
        if getattr(c, "name", None):
            detail["name"] = c.name
        if getattr(c, "description", None):
            detail["description"] = c.description
        detail["contract"] = _contract_summary(c)

        r = self._bank.get_report(skill_id)
        if r is not None:
            detail["report"] = {
                "pass_rate": round(r.overall_pass_rate, 3),
                "n_instances": r.n_instances,
                "worst_segments": r.worst_segments[:5],
                "failure_signatures": dict(list(r.failure_signatures.items())[:5]),
            }
            detail["eff_add_rates"] = {
                k: round(v, 3) for k, v in r.eff_add_success_rate.items()
            }
            detail["eff_del_rates"] = {
                k: round(v, 3) for k, v in r.eff_del_success_rate.items()
            }
        return detail

    # ── Decision-agent convenience ───────────────────────────────────

    def query_for_decision_agent(
        self,
        key: str,
        current_state: Optional[Dict[str, float]] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Convenience method for decision agents returning a single best-match.

        Defaults to the full selection policy (relevance + applicability +
        structured guidance) whenever *current_state* or *current_predicates*
        is available.  Falls back to retrieval only when no state is given.

        Returns a structured guidance dict including: skill_id, skill_name,
        why_selected, applicability_score, expected_effects, preconditions,
        termination_hint, failure_modes, execution_hint, micro_plan, etc.
        """
        state = current_state or current_predicates
        if state is not None:
            results = self.select(key, current_state=state, top_k=top_k)
            if not results:
                return {"skill_id": None, "micro_plan": [], "confidence": 0.0}
            return results[0].to_dict()

        # Backward-compatible path
        results = self.query(key, top_k=top_k)
        if not results:
            return {"skill_id": None, "micro_plan": []}

        best = results[0]
        return {
            "skill_id": best.get("skill_id"),
            "micro_plan": best.get("micro_plan", []) or [{"action": "proceed"}],
            "contract": best.get("contract", {}),
        }
