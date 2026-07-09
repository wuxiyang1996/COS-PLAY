"""OSWorld-only opt-in skill-bank retrieval (Improvement #7).

This module is *additive*: nothing in it runs unless ``--skill_bank_path``
is passed to ``cold_start/generate_cold_start_actor_osworld.py``. The
retriever loads a skill_bank.jsonl produced by either

  - ``labeling/extract_skillbank_gpt54.py`` (LLM-driven SkillBankAgent), or
  - ``skill_transfer_test/extract/`` (LLM-free per-episode lift)

and at the START of every OSWorld episode performs a BM25 search over

  ``strategic_description`` + ``protocol[*].notes`` + ``name``

against the task instruction. The top-K hits are formatted into an
in-context demonstration block and rendered in the actor's user prompt
under the SoM table. The actor is told to treat them as references for
action shape and ordering — not as literal copy-paste — because the
test-task UI may have moved/renamed elements.

This is the eval-side hook for the skill-bank research story (the
RQ3 "transfer matrix" discussed in the implementation_notes/ phase-5
plan): a student/teacher running with retrieval should beat the same
agent with retrieval disabled, by a margin proportional to how
relevant the bank is to the test task.

The retriever is intentionally simple (BM25, no embeddings, no
re-ranker, no caching tier) so the OSWorld eval path does not pull
in heavyweight ML deps. If ``rank_bm25`` is not installed it falls
back to a token-overlap scorer. Both work fine for ~10^3 skills.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation + scoring
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "at", "from",
    "i", "you", "we", "he", "she", "they", "my", "your", "our", "their",
    "do", "does", "did", "have", "has", "had", "will", "would", "should",
    "can", "could", "may", "might", "must", "shall", "if", "then", "else",
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "but", "not", "no", "yes", "so", "than", "too", "very", "also",
    "task", "osworld",
})


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if t not in _STOP_WORDS and len(t) > 1]


def _bm25_corpus(docs: List[List[str]]):
    """Return (idf_map, avg_doc_len) for a tiny pure-python BM25.

    We avoid pulling rank_bm25 / scikit-learn into the OSWorld eval
    path; for ≤10^3 skills the pure-python loop is well under 50ms.
    """
    n_docs = max(len(docs), 1)
    df: Dict[str, int] = {}
    for doc in docs:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1
    idf = {
        tok: math.log(1 + (n_docs - cnt + 0.5) / (cnt + 0.5))
        for tok, cnt in df.items()
    }
    avg_len = (sum(len(d) for d in docs) / n_docs) if docs else 1.0
    return idf, avg_len


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    idf: Dict[str, float],
    avg_doc_len: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    tf: Dict[str, int] = {}
    for tok in doc_tokens:
        tf[tok] = tf.get(tok, 0) + 1
    score = 0.0
    dl = len(doc_tokens)
    norm = 1.0 - b + b * (dl / avg_doc_len if avg_doc_len else 1.0)
    for q in query_tokens:
        f = tf.get(q, 0)
        if not f:
            continue
        score += idf.get(q, 0.0) * (f * (k1 + 1)) / (f + k1 * norm)
    return score


# ---------------------------------------------------------------------------
# Skill record adapter
# ---------------------------------------------------------------------------

@dataclass
class _IndexedSkill:
    """One row in the BM25 index — keeps both the doc tokens and the
    original skill payload so we can render the in-context demo from
    the JSON without re-parsing.
    """

    skill_id: str
    name: str
    description: str
    domains: List[str]
    protocol: List[Dict[str, Any]]
    doc_tokens: List[str]
    raw: Dict[str, Any]


def _extract_skill_text(skill: Dict[str, Any]) -> Tuple[str, str, List[str], List[Dict[str, Any]]]:
    """Pull ``(name, description, domains, protocol)`` out of either the
    LLM-driven SkillBankAgent shape or the LLM-free
    skill_transfer_test/extract/ shape. Both store the same fields
    under ``skill`` keys; this helper just normalises a few aliases.
    """
    name = (
        skill.get("name")
        or skill.get("skill_name")
        or skill.get("skill_id", "")
    )
    description = (
        skill.get("strategic_description")
        or skill.get("description")
        or skill.get("summary")
        or ""
    )
    domains = (
        skill.get("verified_domains")
        or skill.get("feasible_domains")
        or skill.get("applicable_domains")
        or []
    )
    if not isinstance(domains, list):
        domains = []
    protocol = skill.get("protocol") or []
    if not isinstance(protocol, list):
        protocol = []
    return str(name), str(description), [str(d) for d in domains], list(protocol)


def _skill_doc_tokens(name: str, description: str, protocol: List[Dict[str, Any]]) -> List[str]:
    parts = [name, description]
    for hop in protocol[:20]:  # cap so super-long protocols don't dominate
        if isinstance(hop, dict):
            parts.append(str(hop.get("notes", "")))
            payload = hop.get("payload") or {}
            if isinstance(payload, dict):
                for v in payload.values():
                    if isinstance(v, str):
                        parts.append(v)
    return _tokenize(" ".join(parts))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SkillBankRetriever:
    """BM25 retriever over a skill_bank.jsonl emitted by the lift pipeline.

    Loaded once at process start (the constructor reads the JSONL),
    then the actor calls :meth:`retrieve` once per episode. Stateless
    after construction; safe to share across threads.
    """

    bank_path: Path
    skills: List[_IndexedSkill] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    avg_doc_len: float = 1.0
    n_loaded: int = 0
    n_skipped: int = 0

    def __post_init__(self) -> None:
        self.bank_path = Path(self.bank_path).expanduser().resolve()
        if not self.bank_path.is_file():
            raise FileNotFoundError(
                f"skill_bank.jsonl not found at {self.bank_path}"
            )
        with open(self.bank_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    self.n_skipped += 1
                    continue
                # Both shapes wrap the actual skill under a ``skill`` key
                # (the LLM-free lift also packs a ``report``); fall back
                # to treating the row itself as the skill payload.
                skill = (
                    row.get("skill")
                    if isinstance(row, dict) and "skill" in row
                    else row
                )
                if not isinstance(skill, dict):
                    self.n_skipped += 1
                    continue
                name, desc, domains, protocol = _extract_skill_text(skill)
                if not name and not desc:
                    self.n_skipped += 1
                    continue
                doc_tokens = _skill_doc_tokens(name, desc, protocol)
                if not doc_tokens:
                    self.n_skipped += 1
                    continue
                self.skills.append(
                    _IndexedSkill(
                        skill_id=str(skill.get("skill_id", f"row{line_no}")),
                        name=name,
                        description=desc,
                        domains=domains,
                        protocol=protocol,
                        doc_tokens=doc_tokens,
                        raw=skill,
                    )
                )
        self.n_loaded = len(self.skills)
        logger.info(
            "[retriever] loaded %d skills from %s (%d skipped)",
            self.n_loaded, self.bank_path, self.n_skipped,
        )
        self.idf, self.avg_doc_len = _bm25_corpus(
            [s.doc_tokens for s in self.skills]
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        *,
        instruction: str,
        domain: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[float, _IndexedSkill]]:
        """Return ``[(score, skill), …]`` sorted by descending score.

        Domain filter: when ``domain`` is set, skills whose
        ``feasible_domains`` / ``verified_domains`` includes ``domain``
        get a small score boost (×1.25). We do NOT hard-filter on
        domain because cross-domain transfer is the whole point — a
        BrowserGym skill might still be the best answer for an
        OSWorld task that opens Chrome.
        """
        if not self.skills or top_k <= 0:
            return []
        q_tokens = _tokenize(instruction or "")
        if not q_tokens:
            return []
        scored: List[Tuple[float, _IndexedSkill]] = []
        for s in self.skills:
            base = _bm25_score(
                q_tokens, s.doc_tokens,
                self.idf, self.avg_doc_len,
            )
            if base <= 0.0:
                continue
            boost = 1.0
            if domain and (
                domain in s.domains
                or any(d.startswith(domain) for d in s.domains)
            ):
                boost = 1.25
            scored.append((base * boost, s))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Prompt-side rendering
    # ------------------------------------------------------------------

    def format_for_prompt(
        self,
        hits: List[Tuple[float, _IndexedSkill]],
        *,
        max_protocol_hops: int = 5,
    ) -> str:
        """Render the retrieved skills as an in-context demo block.

        Compact: skill name, description, and up to
        ``max_protocol_hops`` protocol steps with verb + notes. Pixel
        coordinates inside the original protocol notes are kept (they
        are demonstrations, not prescriptions) but the actor is told
        upstream to use them as references, not literal copy-paste.
        """
        if not hits:
            return ""
        lines: List[str] = []
        for rank, (score, skill) in enumerate(hits, start=1):
            lines.append(
                f"-- Skill {rank} (score={score:.2f}, "
                f"domains={','.join(skill.domains) or '?'}) --"
            )
            lines.append(f"Name: {skill.name[:120]}")
            if skill.description:
                lines.append(f"Goal: {skill.description[:240]}")
            if skill.protocol:
                lines.append("Protocol (truncated):")
                for hop in skill.protocol[:max_protocol_hops]:
                    if not isinstance(hop, dict):
                        continue
                    op = hop.get("op", "?")
                    notes = (hop.get("notes", "") or "").strip()
                    notes_short = (notes[:160] + "…") if len(notes) > 160 else notes
                    lines.append(f"  {op}: {notes_short}")
                if len(skill.protocol) > max_protocol_hops:
                    lines.append(
                        f"  … (+{len(skill.protocol) - max_protocol_hops} more hops)"
                    )
            lines.append("")
        return "\n".join(lines).rstrip()


__all__ = ["SkillBankRetriever"]
