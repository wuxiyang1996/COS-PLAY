"""
Bank Maintenance — Local re-decode interface.

Instead of re-running Stage 2 globally, this module provides:
  - ``redecode_windows``: run DP/beam only on specified trajectory windows.
  - ``relabel_via_alias``: cheap relabelling when only skill IDs change.
  - ``collect_affected_trajectories``: gather traj_ids from segment records.

All three avoid full-trajectory DP when possible.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from skill_agents.bank_maintenance.config import BankMaintenanceConfig
from skill_agents.bank_maintenance.schemas import RedecodeRequest

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# 1) Cheap alias relabelling
# ═════════════════════════════════════════════════════════════════════


def relabel_via_alias(
    segments: List[dict],
    alias_map: Dict[str, str],
) -> int:
    """Relabel segments in-place using *alias_map* (retired → canonical).

    Each segment dict should have an ``assigned_skill`` or ``skill_label`` key.
    Returns the number of relabelled segments.
    """
    count = 0
    for seg in segments:
        key = "assigned_skill" if "assigned_skill" in seg else "skill_label"
        old = seg.get(key, "")
        if old in alias_map:
            seg[key] = alias_map[old]
            count += 1
    return count


# ═════════════════════════════════════════════════════════════════════
# 2) Collect affected trajectories
# ═════════════════════════════════════════════════════════════════════


def collect_affected_trajectories(
    skill_ids: List[str],
    all_segments: List[dict],
) -> Dict[str, List[Tuple[int, int]]]:
    """Return {traj_id: [(t_start, t_end), ...]} for segments matching *skill_ids*.

    Works with either SegmentRecord.to_dict() or SegmentDiagnostic-style dicts.
    """
    skill_set = set(skill_ids)
    result: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for seg in all_segments:
        label = seg.get("assigned_skill") or seg.get("skill_label", "")
        if label not in skill_set:
            continue
        traj = seg.get("traj_id", "")
        start = seg.get("t_start") or seg.get("start", 0)
        end = seg.get("t_end") or seg.get("end", 0)
        if traj:
            result[traj].append((start, end))

    return dict(result)


# ═════════════════════════════════════════════════════════════════════
# 3) Build RedecodeRequests from affected trajectory windows
# ═════════════════════════════════════════════════════════════════════


def build_redecode_requests(
    affected: Dict[str, List[Tuple[int, int]]],
    reason: str,
    affected_skills: List[str],
    config: BankMaintenanceConfig,
    traj_lengths: Optional[Dict[str, int]] = None,
) -> List[RedecodeRequest]:
    """Convert raw (traj_id, windows) into merged, padded RedecodeRequests."""
    pad = config.redecode_window_pad
    requests: List[RedecodeRequest] = []

    for traj_id, windows in affected.items():
        max_t = (traj_lengths or {}).get(traj_id, 10**9)
        merged = _merge_windows(windows, pad)
        for ws, we in merged:
            requests.append(RedecodeRequest(
                traj_id=traj_id,
                window_start=max(0, ws - pad),
                window_end=min(we + pad, max_t),
                reason=reason,
                affected_skills=affected_skills,
            ))

    return requests


# ═════════════════════════════════════════════════════════════════════
# 4) Execute local re-decode (calls into Stage 2 decoder)
# ═════════════════════════════════════════════════════════════════════


def redecode_windows(
    requests: List[RedecodeRequest],
    decode_fn: Callable[
        [str, int, int, List[str]],
        List[dict],
    ],
) -> Dict[str, List[dict]]:
    """Execute local re-decode for each request using *decode_fn*.

    Parameters
    ----------
    requests : list[RedecodeRequest]
        Windows to re-decode.
    decode_fn : callable
        ``(traj_id, window_start, window_end, skill_ids) -> list[segment_dict]``
        A thin wrapper around Stage 2's ``viterbi_decode`` or ``beam_decode``
        restricted to the given window and skill set.

    Returns
    -------
    dict[str, list[dict]]
        ``traj_id -> [segment_dicts]`` for all re-decoded windows.
    """
    results: Dict[str, List[dict]] = defaultdict(list)

    for req in requests:
        try:
            new_segments = decode_fn(
                req.traj_id,
                req.window_start,
                req.window_end,
                req.affected_skills,
            )
            results[req.traj_id].extend(new_segments)
            logger.info(
                "Re-decoded %s [%d..%d]: %d segments (%s)",
                req.traj_id, req.window_start, req.window_end,
                len(new_segments), req.reason,
            )
        except Exception:
            logger.exception(
                "Re-decode failed for %s [%d..%d]",
                req.traj_id, req.window_start, req.window_end,
            )

    return dict(results)


# ── Utility ──────────────────────────────────────────────────────────


def _merge_windows(
    windows: List[Tuple[int, int]], gap: int,
) -> List[Tuple[int, int]]:
    if not windows:
        return []
    sorted_w = sorted(windows)
    merged = [sorted_w[0]]
    for s, e in sorted_w[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + gap:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged
