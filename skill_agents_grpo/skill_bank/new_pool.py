"""
NEW-pool manager: accumulate, cluster, and promote ``__NEW__`` segments.

Instead of treating ``__NEW__`` as a catch-all bucket, the ``NewPoolManager``
tracks rich metadata per candidate, clusters by effect similarity (not exact
signature match), and promotes clusters that meet support, consistency, and
separability criteria.

Usage::

    pool = NewPoolManager(config=NewPoolConfig())
    pool.add(record)                    # accumulate
    candidates = pool.get_candidates()  # inspect mature clusters
    promoted = pool.promote(bank, observations_by_traj)  # materialize

Design principles:
  - Modular: all NEW logic in one place (not scattered across pipeline.py).
  - Reuses existing clustering from ``contract_verification.clustering`` when
    available; falls back to effect-signature Jaccard otherwise.
  - Configurable thresholds for ablation experiments.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from skill_agents_grpo.stage3_mvp.schemas import (
    ProtoSkill,
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class NewPoolConfig:
    """Thresholds for NEW-pool management and promotion."""

    # Minimum cluster size to consider for promotion
    min_cluster_size: int = 5

    # Minimum pass rate from Stage 3 verification to promote
    min_pass_rate: float = 0.7

    # Minimum Jaccard distance from any existing skill to promote
    min_distinctiveness: float = 0.25

    # Minimum consistency: fraction of cluster members sharing the majority
    # effect signature pattern
    min_consistency: float = 0.5

    # Jaccard similarity threshold for merging NEW candidates into one cluster
    cluster_similarity_thresh: float = 0.4

    # Maximum number of candidates to hold before forcing a clustering pass
    max_pool_size: int = 500

    # How many top clusters to attempt promotion per call
    max_promotions_per_call: int = 10


# ── Candidate metadata ──────────────────────────────────────────────

@dataclass
class NewCandidate:
    """Rich metadata for one ``__NEW__`` segment candidate."""

    record: SegmentRecord
    effect_sig: str = ""
    duration: int = 0
    predecessor_skill: Optional[str] = None
    successor_skill: Optional[str] = None
    added_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.effect_sig = self.record.effect_signature()
        self.duration = max(1, self.record.t_end - self.record.t_start + 1)


@dataclass
class ClusterSummary:
    """Summary statistics for one NEW candidate cluster."""

    cluster_id: int
    size: int
    effect_centroid_add: Set[str]
    effect_centroid_del: Set[str]
    effect_centroid_event: Set[str]
    mean_duration: float
    std_duration: float
    consistency: float  # fraction sharing majority effect pattern
    representative_sig: str
    predecessor_distribution: Dict[str, int] = field(default_factory=dict)
    successor_distribution: Dict[str, int] = field(default_factory=dict)


# ── Pool manager ─────────────────────────────────────────────────────

class NewPoolManager:
    """Accumulates, clusters, and promotes ``__NEW__`` segment candidates.

    Replaces the scattered NEW handling in ``pipeline.py`` with a single
    modular component.
    """

    def __init__(self, config: Optional[NewPoolConfig] = None) -> None:
        self.config = config or NewPoolConfig()
        self._candidates: List[NewCandidate] = []
        self._promoted_ids: Set[str] = set()  # seg_ids already promoted
        self._cluster_labels: Optional[np.ndarray] = None
        self._cluster_summaries: List[ClusterSummary] = []

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def records(self) -> List[SegmentRecord]:
        return [c.record for c in self._candidates]

    # ── Accumulate ───────────────────────────────────────────────────

    def add(
        self,
        record: SegmentRecord,
        predecessor_skill: Optional[str] = None,
        successor_skill: Optional[str] = None,
    ) -> None:
        """Add a ``__NEW__`` segment to the pool with context metadata."""
        if record.seg_id in self._promoted_ids:
            return
        cand = NewCandidate(
            record=record,
            predecessor_skill=predecessor_skill,
            successor_skill=successor_skill,
        )
        self._candidates.append(cand)
        self._cluster_labels = None  # invalidate

    def add_batch(
        self,
        records: List[SegmentRecord],
        context: Optional[List[Tuple[Optional[str], Optional[str]]]] = None,
    ) -> None:
        """Add multiple records.  ``context`` is list of (pred_skill, succ_skill)."""
        for i, rec in enumerate(records):
            pred, succ = (None, None) if context is None else context[i]
            self.add(rec, predecessor_skill=pred, successor_skill=succ)

    def remove_promoted(self, seg_ids: Set[str]) -> None:
        """Remove promoted records from the pool."""
        self._promoted_ids |= seg_ids
        self._candidates = [c for c in self._candidates if c.record.seg_id not in seg_ids]
        self._cluster_labels = None

    # ── Clustering ───────────────────────────────────────────────────

    def _build_effect_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """Build binary effect vectors for all candidates."""
        vocab: Set[str] = set()
        for c in self._candidates:
            vocab.update(c.record.eff_add)
            vocab.update(c.record.eff_del)
            vocab.update(c.record.eff_event)
        vocab_list = sorted(vocab)
        if not vocab_list:
            return np.zeros((len(self._candidates), 0), dtype=np.float32), []

        pred_to_idx = {p: i for i, p in enumerate(vocab_list)}
        N = len(self._candidates)
        V = len(vocab_list)
        mat = np.zeros((N, V), dtype=np.float32)
        for row, cand in enumerate(self._candidates):
            for p in cand.record.eff_add:
                if p in pred_to_idx:
                    mat[row, pred_to_idx[p]] = 1.0
            for p in cand.record.eff_del:
                if p in pred_to_idx:
                    mat[row, pred_to_idx[p]] = 1.0
            for p in cand.record.eff_event:
                if p in pred_to_idx:
                    mat[row, pred_to_idx[p]] = 1.0
        return mat, vocab_list

    def cluster(self) -> List[ClusterSummary]:
        """Cluster candidates by effect similarity (Jaccard-based).

        Uses agglomerative clustering from sklearn when available,
        falls back to signature-based grouping otherwise.
        """
        if len(self._candidates) < 2:
            if self._candidates:
                self._cluster_labels = np.zeros(1, dtype=int)
                self._cluster_summaries = [self._build_summary(0, list(range(1)))]
            return self._cluster_summaries

        mat, vocab = self._build_effect_vectors()
        if mat.shape[1] == 0:
            self._cluster_labels = np.zeros(len(self._candidates), dtype=int)
            self._cluster_summaries = [self._build_summary(0, list(range(len(self._candidates))))]
            return self._cluster_summaries

        try:
            labels, n_clusters = self._agglomerative_cluster(mat)
        except ImportError:
            labels, n_clusters = self._signature_cluster()

        self._cluster_labels = labels

        clusters_idx: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            clusters_idx[int(lbl)].append(i)

        self._cluster_summaries = []
        for cid in sorted(clusters_idx.keys()):
            self._cluster_summaries.append(
                self._build_summary(cid, clusters_idx[cid])
            )

        return self._cluster_summaries

    def _agglomerative_cluster(self, mat: np.ndarray) -> Tuple[np.ndarray, int]:
        from sklearn.cluster import AgglomerativeClustering

        N = mat.shape[0]
        dist = self._jaccard_distance_matrix(mat)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.config.cluster_similarity_thresh,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)
        return labels, len(set(labels))

    def _signature_cluster(self) -> Tuple[np.ndarray, int]:
        """Fallback: group by exact effect signature."""
        sig_to_id: Dict[str, int] = {}
        labels = []
        for cand in self._candidates:
            sig = cand.effect_sig
            if sig not in sig_to_id:
                sig_to_id[sig] = len(sig_to_id)
            labels.append(sig_to_id[sig])
        return np.array(labels, dtype=int), len(sig_to_id)

    @staticmethod
    def _jaccard_distance_matrix(mat: np.ndarray) -> np.ndarray:
        N = mat.shape[0]
        dist = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                union = np.sum((mat[i] + mat[j]) > 0)
                inter = np.sum((mat[i] * mat[j]) > 0)
                d = 1.0 - (inter / union) if union > 0 else 0.0
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _build_summary(self, cid: int, indices: List[int]) -> ClusterSummary:
        """Build summary for one cluster."""
        cands = [self._candidates[i] for i in indices]
        n = len(cands)

        # Effect centroid: literals appearing in >= 50% of instances
        add_counter: Counter = Counter()
        del_counter: Counter = Counter()
        evt_counter: Counter = Counter()
        sig_counter: Counter = Counter()
        pred_dist: Counter = Counter()
        succ_dist: Counter = Counter()
        durations = []

        for c in cands:
            for p in c.record.eff_add:
                add_counter[p] += 1
            for p in c.record.eff_del:
                del_counter[p] += 1
            for p in c.record.eff_event:
                evt_counter[p] += 1
            sig_counter[c.effect_sig] += 1
            if c.predecessor_skill:
                pred_dist[c.predecessor_skill] += 1
            if c.successor_skill:
                succ_dist[c.successor_skill] += 1
            durations.append(c.duration)

        thresh = n * 0.5
        centroid_add = {p for p, c in add_counter.items() if c >= thresh}
        centroid_del = {p for p, c in del_counter.items() if c >= thresh}
        centroid_evt = {p for p, c in evt_counter.items() if c >= thresh}

        most_common_sig, most_common_count = sig_counter.most_common(1)[0]
        consistency = most_common_count / n

        dur_arr = np.array(durations, dtype=np.float64)

        return ClusterSummary(
            cluster_id=cid,
            size=n,
            effect_centroid_add=centroid_add,
            effect_centroid_del=centroid_del,
            effect_centroid_event=centroid_evt,
            mean_duration=float(dur_arr.mean()),
            std_duration=float(dur_arr.std()),
            consistency=consistency,
            representative_sig=most_common_sig,
            predecessor_distribution=dict(pred_dist),
            successor_distribution=dict(succ_dist),
        )

    # ── Candidate inspection ─────────────────────────────────────────

    def get_candidates(self, min_size: Optional[int] = None) -> List[ClusterSummary]:
        """Return cluster summaries for mature candidates.

        Runs clustering if needed.  Filters to clusters with
        ``size >= min_size`` (defaults to ``config.min_cluster_size``).
        """
        if self._cluster_labels is None or len(self._cluster_summaries) == 0:
            self.cluster()

        cutoff = min_size or self.config.min_cluster_size
        return [s for s in self._cluster_summaries if s.size >= cutoff]

    def get_cluster_records(self, cluster_id: int) -> List[SegmentRecord]:
        """Return all records belonging to a given cluster."""
        if self._cluster_labels is None:
            self.cluster()
        return [
            self._candidates[i].record
            for i, lbl in enumerate(self._cluster_labels)
            if int(lbl) == cluster_id
        ]

    # ── Promotion ────────────────────────────────────────────────────

    def promote(
        self,
        bank,
        observations_by_traj: Dict[str, list],
        llm_config=None,
    ) -> List[str]:
        """Attempt to promote qualifying clusters into real skills.

        This replaces the old ``materialize_new_skills()`` logic in pipeline.py
        with richer criteria: support + consistency + separability.

        Parameters
        ----------
        bank : SkillBankMVP
            The skill bank to add new skills to.
        observations_by_traj : dict
            Trajectory observations for Stage 3 processing.
        llm_config : LLMTeacherConfig, optional
            If provided, use LLM to suggest human-readable names.

        Returns
        -------
        list[str]
            IDs of newly created skills.
        """
        from skill_agents_grpo.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
        )
        from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig

        candidates = self.get_candidates()
        if not candidates:
            return []

        # Sort by (consistency * size) descending — prefer well-supported clusters
        candidates.sort(key=lambda c: -(c.consistency * c.size))

        created_ids: List[str] = []
        ts = int(time.time())

        for rank, summary in enumerate(candidates[:self.config.max_promotions_per_call]):
            if summary.size < self.config.min_cluster_size:
                continue
            if summary.consistency < self.config.min_consistency:
                continue

            cluster_recs = self.get_cluster_records(summary.cluster_id)
            if not cluster_recs:
                continue

            # Check distinctiveness against existing bank
            centroid_effects = summary.effect_centroid_add | summary.effect_centroid_del
            if not self._is_distinctive(centroid_effects, bank):
                continue

            new_id = f"S_new_{ts}_{rank}"
            for rec in cluster_recs:
                rec.skill_label = new_id

            specs = [
                SegmentSpec(
                    seg_id=rec.seg_id,
                    traj_id=rec.traj_id,
                    t_start=rec.t_start,
                    t_end=rec.t_end,
                    skill_label=new_id,
                    ui_events=list(rec.events) if getattr(rec, "events", None) else [],
                )
                for rec in cluster_recs
            ]

            s3_config = Stage3MVPConfig(
                min_instances_per_skill=max(1, self.config.min_cluster_size),
            )

            s3_summary = run_stage3_mvp(
                segments=specs,
                observations_by_traj=observations_by_traj,
                config=s3_config,
                bank=bank,
            )

            if new_id in s3_summary.skill_results:
                sr = s3_summary.skill_results[new_id]
                if sr.get("pass_rate", 0) >= self.config.min_pass_rate:
                    created_ids.append(new_id)
                    self.remove_promoted({rec.seg_id for rec in cluster_recs})

                    # Try to name the skill via LLM
                    contract = bank.get_contract(new_id)
                    if contract is not None and llm_config is not None:
                        self._try_name_skill(
                            contract, cluster_recs, observations_by_traj, llm_config,
                        )
                        bank.add_or_update(contract)
                else:
                    bank.remove(new_id)
                    for rec in cluster_recs:
                        rec.skill_label = "__NEW__"

        return created_ids

    def _is_distinctive(self, centroid_effects: Set[str], bank) -> bool:
        """Check candidate effects are sufficiently different from existing skills."""
        if not centroid_effects:
            return False

        for sid in bank.skill_ids:
            contract = bank.get_contract(sid)
            if contract is None:
                continue
            existing = (contract.eff_add or set()) | (contract.eff_del or set())
            if not existing:
                continue
            union = len(centroid_effects | existing)
            inter = len(centroid_effects & existing)
            jacc_dist = 1.0 - (inter / union) if union > 0 else 1.0
            if jacc_dist < self.config.min_distinctiveness:
                return False
        return True

    @staticmethod
    def _try_name_skill(
        contract: SkillEffectsContract,
        cluster_recs: List[SegmentRecord],
        observations_by_traj: Dict[str, list],
        llm_config,
    ) -> None:
        """Best-effort LLM naming for a newly promoted skill."""
        try:
            from skill_agents_grpo.infer_segmentation.llm_teacher import suggest_skill_name

            observation_slices = []
            for rec in cluster_recs[:3]:
                obs = observations_by_traj.get(rec.traj_id, [])
                if rec.t_start is not None and rec.t_end is not None:
                    sl = obs[rec.t_start: rec.t_end + 1]
                    if len(sl) > 0:
                        observation_slices.append(sl)

            if observation_slices:
                naming = suggest_skill_name(
                    observation_slices,
                    eff_add=list(contract.eff_add) or None,
                    eff_del=list(contract.eff_del) or None,
                    eff_event=list(contract.eff_event) or None,
                    config=llm_config,
                )
                if naming and naming.get("name"):
                    contract.name = naming["name"]
                    contract.description = naming.get("description")
        except Exception:
            pass  # naming is best-effort

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save pool state to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [
                {
                    "record": c.record.to_dict(),
                    "predecessor_skill": c.predecessor_skill,
                    "successor_skill": c.successor_skill,
                    "added_at": c.added_at,
                }
                for c in self._candidates
            ],
            "promoted_ids": sorted(self._promoted_ids),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load pool state from JSON."""
        path = Path(filepath)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._candidates = []
        for entry in data.get("candidates", []):
            rec = SegmentRecord.from_dict(entry["record"])
            cand = NewCandidate(
                record=rec,
                predecessor_skill=entry.get("predecessor_skill"),
                successor_skill=entry.get("successor_skill"),
                added_at=entry.get("added_at", 0.0),
            )
            self._candidates.append(cand)
        self._promoted_ids = set(data.get("promoted_ids", []))
        self._cluster_labels = None

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging."""
        candidates = self.get_candidates()
        return {
            "pool_size": len(self._candidates),
            "n_clusters": len(self._cluster_summaries) if self._cluster_summaries else 0,
            "n_mature_clusters": len(candidates),
            "mature_clusters": [
                {
                    "id": c.cluster_id,
                    "size": c.size,
                    "consistency": round(c.consistency, 3),
                    "mean_duration": round(c.mean_duration, 1),
                    "sig": c.representative_sig,
                }
                for c in candidates[:5]
            ],
        }


# ═════════════════════════════════════════════════════════════════════
# Proto-Skill Manager
# ═════════════════════════════════════════════════════════════════════


@dataclass
class ProtoSkillConfig:
    """Thresholds for proto-skill formation and promotion."""

    min_cluster_size: int = 3
    min_consistency: float = 0.4
    min_separability: float = 0.2
    promotion_min_support: int = 5
    promotion_min_consistency: float = 0.5
    promotion_min_pass_rate: float = 0.6
    max_proto_skills: int = 50


class ProtoSkillManager:
    """Manages the proto-skill layer between __NEW__ and real skills.

    Proto-skills are lightweight intermediate representations that allow
    NEW clusters to participate in Stage 2 decoding as candidate labels
    *before* full promotion.  This accelerates new-skill growth by giving
    the system a meaningful intermediate target earlier.

    Lifecycle::

        __NEW__ → cluster (via NewPoolManager) → ProtoSkill
        → light verification → if is_promotable → real Skill

    Stage 2 can include proto-skill candidate labels in decoding.
    """

    def __init__(self, config: Optional[ProtoSkillConfig] = None) -> None:
        self.config = config or ProtoSkillConfig()
        self._protos: Dict[str, ProtoSkill] = {}

    @property
    def proto_ids(self) -> List[str]:
        return list(self._protos.keys())

    @property
    def size(self) -> int:
        return len(self._protos)

    def get(self, proto_id: str) -> Optional[ProtoSkill]:
        return self._protos.get(proto_id)

    def candidate_labels(self) -> List[str]:
        """Return Stage-2-compatible labels for all proto-skills."""
        return [p.candidate_label for p in self._protos.values()]

    # ── Formation ────────────────────────────────────────────────────

    def form_from_cluster(
        self,
        cluster_summary: ClusterSummary,
        member_records: List[SegmentRecord],
        existing_bank_skills: Optional[Set[str]] = None,
    ) -> Optional[ProtoSkill]:
        """Attempt to form a proto-skill from a NEW cluster.

        Criteria: cluster must meet min_cluster_size, min_consistency,
        and have enough distinctiveness from existing skills and protos.
        """
        cfg = self.config
        if cluster_summary.size < cfg.min_cluster_size:
            return None
        if cluster_summary.consistency < cfg.min_consistency:
            return None
        if len(self._protos) >= cfg.max_proto_skills:
            return None

        centroid = cluster_summary.effect_centroid_add | cluster_summary.effect_centroid_del
        if existing_bank_skills and centroid:
            for existing_id in existing_bank_skills:
                pass  # separability checked at promotion time

        proto_id = f"proto_{int(time.time())}_{cluster_summary.cluster_id}"
        if proto_id in self._protos:
            proto_id += f"_{len(self._protos)}"

        durations = [max(1, r.t_end - r.t_start + 1) for r in member_records]
        dur_arr = np.array(durations, dtype=np.float64)

        tag_dist: Counter = Counter()
        for r in member_records:
            for e in (r.eff_event or set()):
                if e.startswith("tag_"):
                    tag_dist[e] += 1

        context_before: Counter = Counter()
        context_after: Counter = Counter()

        proto = ProtoSkill(
            proto_id=proto_id,
            member_seg_ids=[r.seg_id for r in member_records],
            candidate_effects_add=set(cluster_summary.effect_centroid_add),
            candidate_effects_del=set(cluster_summary.effect_centroid_del),
            candidate_effects_event=set(cluster_summary.effect_centroid_event),
            support=cluster_summary.size,
            consistency=cluster_summary.consistency,
            tag_distribution=dict(tag_dist),
            typical_length_mean=float(dur_arr.mean()),
            typical_length_std=float(dur_arr.std()),
            context_before=[
                sk for sk, _ in cluster_summary.predecessor_distribution.items()
            ][:5],
            context_after=[
                sk for sk, _ in cluster_summary.successor_distribution.items()
            ][:5],
        )

        self._protos[proto_id] = proto
        return proto

    def form_from_pool(
        self,
        pool_manager: NewPoolManager,
        existing_bank_skills: Optional[Set[str]] = None,
    ) -> List[ProtoSkill]:
        """Scan all mature clusters in a NewPoolManager and form proto-skills.

        Returns list of newly created ProtoSkills.
        """
        candidates = pool_manager.get_candidates()
        created: List[ProtoSkill] = []

        for summary in candidates:
            records = pool_manager.get_cluster_records(summary.cluster_id)
            proto = self.form_from_cluster(
                summary, records, existing_bank_skills,
            )
            if proto is not None:
                created.append(proto)

        return created

    # ── Verification ────────────────────────────────────────────────

    def verify(
        self,
        proto_id: str,
        bank,
        observations_by_traj: Dict[str, list],
    ) -> Optional[float]:
        """Run light verification on a proto-skill.

        Uses Stage 3 to compute pass rate for the proto-skill's members.
        Returns the pass rate or None if verification failed.
        """
        proto = self._protos.get(proto_id)
        if proto is None:
            return None

        from skill_agents_grpo.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
        )
        from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig

        specs = []
        for seg_id in proto.member_seg_ids[:20]:
            parts = seg_id.rsplit("_seg", 1)
            if len(parts) == 2:
                traj_id = parts[0]
            else:
                traj_id = seg_id
            specs.append(SegmentSpec(
                seg_id=seg_id,
                traj_id=traj_id,
                t_start=0,
                t_end=0,
                skill_label=proto.proto_id,
            ))

        if not specs:
            return None

        s3_config = Stage3MVPConfig(min_instances_per_skill=1)
        try:
            summary = run_stage3_mvp(
                segments=specs,
                observations_by_traj=observations_by_traj,
                config=s3_config,
                bank=bank,
            )
            if proto.proto_id in summary.skill_results:
                pass_rate = summary.skill_results[proto.proto_id].get("pass_rate", 0.0)
            else:
                pass_rate = 0.0
        except Exception:
            pass_rate = 0.0

        proto.verified = True
        proto.verification_pass_rate = pass_rate
        proto.n_verifications += 1
        proto.updated_at = time.time()

        # Clean up trial contract
        if bank.has_skill(proto.proto_id):
            bank.remove(proto.proto_id)

        return pass_rate

    # ── Promotion ────────────────────────────────────────────────────

    def promote_ready(self, bank) -> List[str]:
        """Promote all proto-skills that pass the promotion criteria.

        Returns list of newly created skill IDs.
        """
        promoted: List[str] = []
        to_remove: List[str] = []

        for pid, proto in self._protos.items():
            if not proto.is_promotable:
                continue

            skill = proto.to_skill()
            bank.add_or_update(skill.contract)
            bank.add_or_update_skill(skill)
            promoted.append(pid)
            to_remove.append(pid)

        for pid in to_remove:
            del self._protos[pid]

        return promoted

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "proto_skills": {
                pid: p.to_dict() for pid, p in self._protos.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._protos = {}
        for pid, pd in data.get("proto_skills", {}).items():
            self._protos[pid] = ProtoSkill.from_dict(pd)

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging."""
        promotable = [p for p in self._protos.values() if p.is_promotable]
        return {
            "n_protos": len(self._protos),
            "n_promotable": len(promotable),
            "protos": [
                {
                    "id": p.proto_id,
                    "support": p.support,
                    "consistency": round(p.consistency, 3),
                    "verified": p.verified,
                    "pass_rate": round(p.verification_pass_rate, 3),
                }
                for p in list(self._protos.values())[:10]
            ],
        }
