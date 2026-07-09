"""Project promoted skills back into legacy per-game ``skill_bank.jsonl``.

Strict **one-way** projector: reads a `SkillRepository` ``bank_snapshots/<id>.json``
file (the on-disk format that
``orchestrator.PromotionOrchestrator`` writes inside ``promote()``) and
upserts the eligible skills into the legacy
``labeling/skill_bank_out/run_<ts>/<corpus>/<source>/skill_bank.jsonl`` that
``decision_agents.skill_interface.SkillBankProvider`` /
``skill_agents.pipeline.SkillBankAgent`` already consume.

Why this module exists
----------------------
Per ``implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md``
**D8 (Option A)** — we deliberately do *not* build a bidirectional bridge
between the legacy ``SkillBankMVP`` and the new ``SkillRepository``. Instead
the trainer's actor keeps reading the legacy per-game JSONL, and the new
Crafter / Promotion path writes *new* skills back through this projector
once per offline-promotion cycle. The legacy pipeline (Stage 1–4) is
unaffected; promoted skills land *additively* alongside what the curator
LoRA produces.

The projection is intentionally lossy in **only one direction**: typed
``protocol`` hops collapse to NL ``protocol.steps`` strings (matching the
cold-start bank shape the legacy reader was designed for). Going the
other direction (legacy → ``SkillRepository``) is out of scope here and
lives in the (still-pending) ``skill_bank.legacy_bridge`` module per the
package ``__init__`` docstring.

Architectural rules (mechanically followed by this module)
----------------------------------------------------------
* **No** ``skill_agents`` import — keeps the dependency direction one-way.
* **No** ``skill_bank.SkillRepository`` import — we read the on-disk JSON
  directly so this projector can run on snapshots produced by either an
  in-process or a subprocess-invoked ``decide_promotion_gpt54.py``.
* **Atomic write semantics** — the legacy JSONL is rewritten via
  ``tmpfile + os.replace`` so a concurrent ``SkillBankProvider`` reader
  never observes a torn file.
* **`report` blocks are preserved** for any skill_id that already exists
  in the legacy bank (otherwise we'd zero out usage stats every cycle).

Spec cross-refs
---------------
* `crafter-harness-orchestrator-roles.md` §3 (I/O ownership table) —
  Orchestrator is the only writer of ``status`` / ``verified_domains``;
  this module reads those fields, does not mutate them.
* ``harness/README.md`` §16.3 (bank-pointer mismatch) — D8 Option A
  resolution.
* ``labeling_supplement/decide_promotion_gpt54.py`` lines ~140–155
  (snapshot output schema) — the input shape this module consumes.
* ``labeling_supplement/reflect_per_episode_gpt54.py::_record_from_bank_entry``
  — the legacy envelope shape this module emits.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status policy
# ---------------------------------------------------------------------------

# F3: ``--gate-mode offline-synthetic`` caps every skill at ``provisional``.
# ``active`` only fires under real Harness gates. ``shadow`` is the
# intermediate state Stage 1 (replay) writes. All three are runnable per
# ``skill_bank.repository.SkillRepository.runnable(include_shadow=True)``
# and therefore eligible for the legacy bank too.
DEFAULT_ELIGIBLE_STATUSES: FrozenSet[str] = frozenset(
    {"active", "provisional", "shadow"}
)


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WritebackReport:
    """Summary of a single ``writeback_promotion`` invocation."""

    snapshot_path: Path
    legacy_bank_path: Path
    snapshot_id: str
    n_total_in_snapshot: int
    n_eligible: int
    n_inserted: int
    n_updated: int
    n_skipped_status: int
    n_skipped_invalid: int
    inserted_skill_ids: List[str] = field(default_factory=list)
    updated_skill_ids: List[str] = field(default_factory=list)
    eligible_statuses: FrozenSet[str] = DEFAULT_ELIGIBLE_STATUSES

    @property
    def n_written(self) -> int:
        return self.n_inserted + self.n_updated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_path": str(self.snapshot_path),
            "legacy_bank_path": str(self.legacy_bank_path),
            "snapshot_id": self.snapshot_id,
            "n_total_in_snapshot": self.n_total_in_snapshot,
            "n_eligible": self.n_eligible,
            "n_inserted": self.n_inserted,
            "n_updated": self.n_updated,
            "n_skipped_status": self.n_skipped_status,
            "n_skipped_invalid": self.n_skipped_invalid,
            "inserted_skill_ids": list(self.inserted_skill_ids),
            "updated_skill_ids": list(self.updated_skill_ids),
            "eligible_statuses": sorted(self.eligible_statuses),
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def writeback_promotion(
    *,
    snapshot_path: Path,
    legacy_bank_path: Path,
    eligible_statuses: Iterable[str] = DEFAULT_ELIGIBLE_STATUSES,
    dry_run: bool = False,
) -> WritebackReport:
    """Project the eligible skills from a ``SkillRepository`` snapshot
    into a legacy per-game ``skill_bank.jsonl``.

    Parameters
    ----------
    snapshot_path
        Path to a ``bank_snapshots/<id>.json`` file produced by
        ``decide_promotion_gpt54.py``.
    legacy_bank_path
        Path to the per-game ``skill_bank.jsonl`` that the trainer's
        actor reads (e.g.
        ``labeling/skill_bank_out/run_<ts>/<corpus>/<source>/skill_bank.jsonl``).
        Created if missing; parents are created as needed.
    eligible_statuses
        Statuses to project. Defaults to
        ``{"active", "provisional", "shadow"}`` per F3.
    dry_run
        If True, do everything except the final atomic write. Useful
        for tests and for the trainer's ``--dry-run`` debug mode.

    Returns
    -------
    :class:`WritebackReport`
        Strict counts; ``inserted_skill_ids`` / ``updated_skill_ids``
        for downstream logging.
    """
    snapshot_path = Path(snapshot_path)
    legacy_bank_path = Path(legacy_bank_path)
    statuses = frozenset(s.lower() for s in eligible_statuses)

    snap = _load_snapshot(snapshot_path)
    snapshot_id = snap.get("snapshot_id") or snapshot_path.stem
    skills_in: Sequence[Mapping[str, Any]] = (
        (snap.get("body") or {}).get("skills") or []
    )

    existing_envelopes = _load_legacy_jsonl(legacy_bank_path)
    by_id: Dict[str, Dict[str, Any]] = {
        e["skill"]["skill_id"]: e
        for e in existing_envelopes
        if isinstance(e.get("skill"), dict) and e["skill"].get("skill_id")
    }

    n_eligible = 0
    n_inserted = 0
    n_updated = 0
    n_skipped_status = 0
    n_skipped_invalid = 0
    inserted: List[str] = []
    updated: List[str] = []

    for skill in skills_in:
        skill_id = (skill or {}).get("skill_id")
        if not skill_id:
            n_skipped_invalid += 1
            continue
        status = str(skill.get("status") or "").lower()
        if status not in statuses:
            n_skipped_status += 1
            continue
        n_eligible += 1

        prior = by_id.get(skill_id)
        envelope = _project_to_legacy_envelope(skill, prior_envelope=prior)
        if envelope is None:
            n_skipped_invalid += 1
            continue

        if prior is None:
            by_id[skill_id] = envelope
            inserted.append(skill_id)
            n_inserted += 1
        else:
            by_id[skill_id] = envelope
            updated.append(skill_id)
            n_updated += 1

    report = WritebackReport(
        snapshot_path=snapshot_path,
        legacy_bank_path=legacy_bank_path,
        snapshot_id=snapshot_id,
        n_total_in_snapshot=len(skills_in),
        n_eligible=n_eligible,
        n_inserted=n_inserted,
        n_updated=n_updated,
        n_skipped_status=n_skipped_status,
        n_skipped_invalid=n_skipped_invalid,
        inserted_skill_ids=inserted,
        updated_skill_ids=updated,
        eligible_statuses=statuses,
    )

    if dry_run:
        logger.info(
            "writeback_promotion (dry-run) snapshot=%s -> %s | "
            "+%d inserted / ~%d updated / %d skipped_status / %d skipped_invalid",
            snapshot_id, legacy_bank_path,
            n_inserted, n_updated, n_skipped_status, n_skipped_invalid,
        )
        return report

    _atomic_write_jsonl(
        legacy_bank_path,
        # Preserve the original on-disk order for already-existing entries
        # (so SkillBankProvider's stable iteration order doesn't churn);
        # append new entries in snapshot order at the tail.
        _ordered_envelopes(existing_envelopes, by_id, inserted),
    )
    logger.info(
        "writeback_promotion snapshot=%s -> %s | "
        "+%d inserted / ~%d updated / %d skipped_status / %d skipped_invalid",
        snapshot_id, legacy_bank_path,
        n_inserted, n_updated, n_skipped_status, n_skipped_invalid,
    )
    return report


def find_latest_snapshot(promotion_pair_dir: Path) -> Optional[Path]:
    """Pick the most-recently-modified ``bank_snapshots/snap-*.json``
    under one ``promotion_decisions_out/run_<ts>/<corpus>/<source>/`` dir.

    Returns ``None`` if no snapshot exists (e.g. all proposals rejected).
    """
    promotion_pair_dir = Path(promotion_pair_dir)
    snap_dir = promotion_pair_dir / "bank_snapshots"
    if not snap_dir.is_dir():
        return None
    candidates = [
        p for p in snap_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name.startswith("snap-")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_VERSION_LEADING_INT_RE = re.compile(r"v?(\d+)")


def _load_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"snapshot not found: {snapshot_path}")
    with snapshot_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_legacy_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load existing legacy bank, returning ``[]`` if the file is missing
    or empty. Malformed lines are skipped with a debug log entry, never
    raised — the bank is a *running* artefact and we'd rather lose one
    bad line than refuse to write."""
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.debug(
                    "legacy_writeback: skipping malformed line %d in %s: %s",
                    line_no, path, exc,
                )
                continue
            if not isinstance(obj, dict):
                continue
            out.append(obj)
    return out


def _atomic_write_jsonl(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL via tmp + ``os.replace`` so concurrent readers never
    see a torn file. Creates parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile in the same directory ⇒ os.replace is atomic on
    # POSIX (rename within the same filesystem).
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            for entry in entries:
                tmp.write(json.dumps(entry, ensure_ascii=False, default=str))
                tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup; the next call's tempfile will succeed even
        # if this one leaks.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _ordered_envelopes(
    existing: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    inserted_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Stable output order: existing entries (in original order, with their
    upserted contents) first, then newly-inserted entries in insertion order.

    This keeps ``SkillBankProvider`` retrieval order stable across cycles
    for skills that haven't churned, which matters for any downstream code
    that uses position as a tiebreak.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for env in existing:
        if not isinstance(env.get("skill"), dict):
            continue
        sid = env["skill"].get("skill_id")
        if not sid or sid in seen:
            continue
        replacement = by_id.get(sid)
        if replacement is None:
            continue
        out.append(dict(replacement))
        seen.add(sid)
    for sid in inserted_ids:
        if sid in seen:
            continue
        env = by_id.get(sid)
        if env is None:
            continue
        out.append(dict(env))
        seen.add(sid)
    return out


# ---------------------------------------------------------------------------
# Projection — SkillRepository SkillRecord JSON  →  legacy envelope
# ---------------------------------------------------------------------------


def _project_to_legacy_envelope(
    skill: Mapping[str, Any],
    *,
    prior_envelope: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Project one ``SkillRepository`` SkillRecord JSON into the legacy
    ``{"skill": ..., "report": ...}`` envelope.

    If the legacy entry already exists (``prior_envelope is not None``),
    the ``report`` block from the prior envelope is preserved so usage
    statistics survive a writeback.

    Returns ``None`` if the input is malformed (no ``skill_id``).
    """
    skill_id = skill.get("skill_id")
    if not skill_id:
        return None

    contract = skill.get("contract") or {}
    typed_protocol: Sequence[Mapping[str, Any]] = skill.get("protocol") or []
    nl_steps = _typed_protocol_to_nl_steps(typed_protocol)

    legacy_protocol: Dict[str, Any] = {
        "preconditions": list(contract.get("preconditions") or []),
        "steps": nl_steps,
        "success_criteria": list(contract.get("success_criteria") or []),
        "abort_criteria": list(contract.get("abort_criteria") or []),
        "expected_duration": int(skill.get("expected_duration") or len(nl_steps) or 1),
        "source": "promotion-writeback",
    }
    # Carry over any extra protocol metadata the legacy reader respects
    # (predicate_success / predicate_abort / step_checks) when present
    # in the prior envelope so we don't drop human annotations.
    if prior_envelope is not None:
        prior_proto = (prior_envelope.get("skill") or {}).get("protocol") or {}
        for k in ("step_checks", "predicate_success", "predicate_abort"):
            if k in prior_proto and k not in legacy_protocol:
                legacy_protocol[k] = list(prior_proto[k])

    evidence_role = ""
    expected_roles = contract.get("expected_evidence_roles") or []
    if expected_roles:
        evidence_role = str(expected_roles[0]).upper()

    feasible: Sequence[str] = skill.get("feasible_domains") or []
    legacy_skill: Dict[str, Any] = {
        "skill_id": skill_id,
        "version": _semver_to_int(skill.get("version")),
        "name": skill.get("name") or skill_id,
        "strategic_description": (
            skill.get("strategic_description")
            or contract.get("description", "")
            or ""
        ),
        "tags": list(skill.get("tags") or []),
        "protocol": legacy_protocol,
        "contract": {
            "skill_id": skill_id,
            "version": _semver_to_int(skill.get("version")),
            "name": contract.get("name") or skill.get("name") or skill_id,
            "description": contract.get("description", ""),
            "eff_add": list(contract.get("effects_add") or []),
            "eff_del": list(contract.get("effects_del") or []),
        },
        "evidence_role": evidence_role,
        "applicable_domains": list(feasible),
        # Task-axis fields (harness/README §22). When the upstream
        # ``SkillRecord`` carries non-empty ``feasible_tasks`` /
        # ``verified_tasks`` (e.g. set by the cross-game translator at
        # ``skill_agents.skill_bank.translate_for_target`` or by the
        # foundry's lifecycle gates), we round-trip them onto the legacy
        # JSONL so ``trainer.coevolution._crafter_hook._record_from_bank_entry``
        # rehydrates the same eligibility-relevant metadata. Empty lists
        # remain task-agnostic — back-compat default.
        "feasible_tasks": list(skill.get("feasible_tasks") or []),
        "verified_tasks": list(skill.get("verified_tasks") or []),
        # Cross-game translation provenance (shared-bank mode). When
        # ``derived_from`` is set, the curator + retirement passes can
        # find the lineage; when ``confidence_tag != "stable"``, the
        # eligibility filter / skill_selection prompt may down-weight
        # the candidate.
        "derived_from": skill.get("derived_from"),
        "confidence_tag": skill.get("confidence_tag") or "stable",
        # Round-trip metadata so a future bidirectional bridge can detect
        # writeback-projected entries and refuse to clobber the upstream
        # SkillRecord.
        "_writeback_status": str(skill.get("status") or "").lower(),
        "_writeback_source_type": skill.get("source_type") or "",
        "_writeback_verified_domains": list(skill.get("verified_domains") or []),
        "_writeback_version_str": str(skill.get("version") or ""),
    }

    if prior_envelope is not None:
        prior_report = prior_envelope.get("report") or {}
        # Filter to known VerificationReport fields — the legacy
        # ``SkillBankMVP.load()`` round-trips ``report`` through
        # ``VerificationReport.from_dict(d) → cls(**d)``, which raises
        # TypeError on any unexpected key. Carrying through *only* the
        # known schema keeps usage stats while keeping load() safe.
        report = _project_to_verification_report(prior_report, skill_id=skill_id)
    else:
        report = _empty_report(skill_id=skill_id)

    return {"skill": legacy_skill, "report": report}


def _typed_protocol_to_nl_steps(
    typed_protocol: Sequence[Mapping[str, Any]],
) -> List[str]:
    """Collapse typed hops ``[{action, payload, notes}]`` into NL strings
    that the legacy ``protocol.steps: List[str]`` consumer expects.

    Mirrors the inverse of
    ``labeling_supplement/reflect_per_episode_gpt54.py::_wrap_protocol_steps``
    so a protocol that round-trips through writeback ↔ readback returns
    to its original NL shape.
    """
    out: List[str] = []
    for hop in typed_protocol:
        if not isinstance(hop, Mapping):
            out.append(str(hop))
            continue
        notes = hop.get("notes")
        if isinstance(notes, str) and notes.strip():
            out.append(notes.strip())
            continue
        action = hop.get("action")
        if isinstance(action, str) and action.strip():
            payload = hop.get("payload")
            if isinstance(payload, Mapping) and payload:
                out.append(f"{action.strip()} {json.dumps(payload, sort_keys=True)}")
            else:
                out.append(action.strip())
            continue
        # Last resort: serialise the whole hop so we don't lose information.
        out.append(json.dumps(hop, sort_keys=True, default=str))
    return out


def _semver_to_int(version: Any) -> int:
    """Best-effort extraction of a leading integer from a SkillRecord
    version string (``"v1.offline.recond_0"`` → 1).  Legacy bank's
    ``version`` field is ``int``; we keep that contract.

    Falls back to 1 (not 0) so brand-new entries don't conflict with the
    legacy convention that 0 means "uninitialised".
    """
    if isinstance(version, int):
        return version
    if version is None:
        return 1
    m = _VERSION_LEADING_INT_RE.search(str(version))
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            pass
    return 1


# Canonical VerificationReport keys (skill_agents/stage3_mvp/schemas.py:158-187).
# ``SkillBankMVP.load()`` deserializes ``entry["report"]`` via
# ``VerificationReport.from_dict(d) → cls(**d)``, so the report dict MUST
# carry only these keys — any extra key raises TypeError at load time.
_VERIFICATION_REPORT_KEYS: FrozenSet[str] = frozenset({
    "skill_id",
    "n_instances",
    "eff_add_success_rate",
    "eff_del_success_rate",
    "eff_event_rate",
    "overall_pass_rate",
    "worst_segments",
    "failure_signatures",
})


def _empty_report(*, skill_id: str) -> Dict[str, Any]:
    """Synthesize a minimal ``VerificationReport``-compatible report
    block for newly-inserted skills.

    Field set matches ``skill_agents.stage3_mvp.schemas.VerificationReport``
    exactly — adding any extra key would crash
    ``SkillBankMVP.load() → VerificationReport.from_dict(d) → cls(**d)``.
    Empirically validated against the real loader after the audit caught
    the first iteration's ``selection_count`` / ``pass_rate`` keys
    breaking ``SkillBankMVP.load()``; see
    ``tests/test_legacy_writeback_real_loader.py`` for the gate.
    """
    return {
        "skill_id": skill_id,
        "n_instances": 0,
        "eff_add_success_rate": {},
        "eff_del_success_rate": {},
        "eff_event_rate": {},
        "overall_pass_rate": 0.0,
        "worst_segments": [],
        "failure_signatures": {},
    }


def _project_to_verification_report(
    raw_report: Mapping[str, Any], *, skill_id: str,
) -> Dict[str, Any]:
    """Return ``raw_report`` filtered to canonical VerificationReport keys
    and back-filled with the empty defaults for any missing field.

    Defensive against:
      * older writeback runs that emitted non-canonical keys
        (``selection_count``, ``pass_rate``, ``source``, …); these are
        dropped silently.
      * cold-start banks that have a real ``VerificationReport`` block;
        every key is preserved verbatim.
      * ``raw_report`` accidentally not being a ``Mapping`` (e.g. a
        legacy ``None``); we synthesise an empty report from scratch.
    """
    if not isinstance(raw_report, Mapping):
        return _empty_report(skill_id=skill_id)
    out = _empty_report(skill_id=skill_id)
    for k, v in raw_report.items():
        if k in _VERIFICATION_REPORT_KEYS:
            out[k] = v
    # Force skill_id consistency — some legacy reports may have a stale id
    # if the underlying skill was renamed/upserted.
    out["skill_id"] = skill_id
    return out


__all__ = [
    "DEFAULT_ELIGIBLE_STATUSES",
    "WritebackReport",
    "find_latest_snapshot",
    "writeback_promotion",
]
