"""
Step 2 — Compute per-instance effects from booleanized predicates.

After segment summarization (Step 1), derive eff_add / eff_del / eff_event
for each ``SegmentRecord``.  Only predicates that meet the reliability
threshold are included.
"""

from __future__ import annotations

from skill_agents.stage3_mvp.config import Stage3MVPConfig
from skill_agents.stage3_mvp.schemas import SegmentRecord
from skill_agents.stage3_mvp.predicate_vocab import PredicateVocab, normalize_event


def compute_effects(
    rec: SegmentRecord,
    config: Stage3MVPConfig,
    vocab: PredicateVocab,
) -> SegmentRecord:
    """Populate ``eff_add``, ``eff_del``, ``eff_event`` on *rec* in-place.

    Reliability filter: only predicates with
    ``vocab.reliability(p) >= config.reliability_min_for_effects``
    are considered.

    Parameters
    ----------
    rec : SegmentRecord
        Must already have ``B_start`` and ``B_end`` populated (Step 1).
    config : Stage3MVPConfig
    vocab : PredicateVocab

    Returns
    -------
    SegmentRecord
        The same object, mutated with effects filled in.
    """
    threshold = config.reliability_min_for_effects

    reliable_start = vocab.filter_reliable(rec.B_start, threshold)
    reliable_end = vocab.filter_reliable(rec.B_end, threshold)

    rec.eff_add = reliable_end - reliable_start
    rec.eff_del = reliable_start - reliable_end

    # Event-like effects from UI event log
    rec.eff_event = set()
    for raw in rec.events:
        normalized = raw if raw.startswith("event.") else normalize_event(raw)
        if vocab.is_reliable(normalized, threshold):
            rec.eff_event.add(normalized)

    return rec
