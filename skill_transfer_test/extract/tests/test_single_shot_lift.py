"""Single-shot lift coverage tests across the four QA corpora.

Closes TODO-4. Drives :func:`skill_transfer_test.extract.single_shot_lift.lift_one_sample`
against ONE real sample drawn from each of the four single-shot corpora
the extractor pipeline supports (``visual_toolbench``, ``tir_bench``,
``video_holmes``, ``siv_bench``) and asserts the lifted
``{report, skill}`` envelope satisfies every contract the downstream
``runner.py`` / ``archetype_aggregator.py`` / ``_unify.py`` consumers
rely on.

The fixtures are *not* checked in. Each parametrized case finds the
first ``correct=True`` sample on disk under
``CorpusSpec.default_input_root`` and skips the case if no usable
sample is available -- this keeps the test suite green on CI runners
that don't ship the cold-start data tree, while still exercising the
real lift logic against real samples whenever the data IS available.

A second test verifies the negative path: ``correct=False`` samples
return ``None`` when ``include_incorrect`` is left at its default.

The third test verifies entity-reference binding: when the prose
reasoning cites ``e1`` / ``e2`` / etc., the resulting typed protocol
must contain hops whose payloads reference the same entity IDs (so the
downstream harness can ground them against the schema).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from skill_transfer_test.extract import _corpus_specs as cs
from skill_transfer_test.extract.single_shot_lift import lift_one_sample


SINGLE_SHOT_CORPORA: tuple[str, ...] = (
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)


# ---------------------------------------------------------------------------
# Helpers -- find a real sample for each corpus on demand
# ---------------------------------------------------------------------------


def _find_first_sample(
    corpus: str, *, want_correct: Optional[bool] = True,
) -> Optional[Path]:
    """Return the first sample under the corpus' ``default_input_root``.

    Filters by ``correct == want_correct`` when ``want_correct is not None``.
    Returns ``None`` (so the caller can ``pytest.skip``) if either the
    root is missing or no sample matches the filter.
    """
    spec = cs.get_spec(corpus)
    root = spec.default_input_root
    if not root.exists():
        return None
    for sample_path in sorted(root.glob(spec.sample_glob)):
        try:
            payload = json.loads(sample_path.read_text())
        except Exception:
            continue
        if want_correct is None:
            return sample_path
        if bool(payload.get("correct", False)) == want_correct:
            return sample_path
    return None


def _load_sample(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Happy path -- lift one correct sample per corpus and validate the envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", SINGLE_SHOT_CORPORA)
def test_lift_one_sample_returns_well_formed_envelope(corpus: str):
    spec = cs.get_spec(corpus)
    sample_path = _find_first_sample(corpus, want_correct=True)
    if sample_path is None:
        pytest.skip(
            f"{corpus}: no correct samples on disk under "
            f"{spec.default_input_root}"
        )
    sample = _load_sample(sample_path)
    record = lift_one_sample(sample, spec=spec)
    assert record is not None, (
        f"{corpus}: lift_one_sample returned None on a correct sample"
    )

    # Top-level envelope shape
    assert set(record.keys()) >= {"report", "skill"}, record.keys()
    report = record["report"]
    skill = record["skill"]

    # Skill identity + lineage
    assert skill["skill_id"], f"{corpus}: skill_id is empty"
    assert skill["skill_id"] == report["skill_id"], (
        f"{corpus}: report.skill_id and skill.skill_id disagree"
    )
    assert skill["name"], f"{corpus}: skill.name is empty"
    assert skill["name"].startswith("answer/"), (
        f"{corpus}: skill.name does not start with 'answer/' ({skill['name']!r})"
    )

    # Domain/feasibility carry the corpus' canonical domain
    assert spec.domain in skill["feasible_domains"]
    assert spec.domain in skill["applicable_domains"]
    assert spec.domain in skill["verified_domains"], (
        f"{corpus}: correct=True sample should populate verified_domains"
    )

    # Protocol shape
    protocol = skill["protocol"]
    assert isinstance(protocol, list) and len(protocol) >= 1, (
        f"{corpus}: protocol must be a non-empty list of typed hops"
    )
    for i, hop in enumerate(protocol):
        assert isinstance(hop, dict), f"{corpus}: hop[{i}] is not a dict"
        # Typed hops use `op` (the canonical InnerAction verb) plus
        # `payload`, `slot_types`, `effects_add/del`, `evidence_role`.
        assert "op" in hop, (
            f"{corpus}: hop[{i}] missing 'op' (keys: {sorted(hop.keys())})"
        )
        assert "payload" in hop, f"{corpus}: hop[{i}] missing 'payload'"
        assert "evidence_role" in hop, (
            f"{corpus}: hop[{i}] missing 'evidence_role'"
        )

    # Single-shot v4 contract: at least one effect should be one of the
    # canonical single-shot QA predicates so the reverse-bind / few-shot
    # adapter has something to gate on.
    eff_add = skill["contract"]["effects_add"]
    assert isinstance(eff_add, list) and len(eff_add) >= 1, (
        f"{corpus}: contract.effects_add is empty -- single-shot lift "
        f"must mine answer-level predicates"
    )
    eff_types = {e.get("type") for e in eff_add if isinstance(e, dict)}
    assert eff_types & {"answer_emitted", "answer_matches_gold", "entity_grounded"}, (
        f"{corpus}: no single-shot v4 predicates in effects_add "
        f"(saw types: {sorted(t for t in eff_types if t)})"
    )

    # Provenance is the cross-corpus glue _unify and archetype_aggregator
    # rely on.
    prov = skill["provenance"]
    assert prov["corpus"] == spec.name
    assert prov["benchmark"] == spec.extra.get("benchmark", spec.name)
    assert prov["modality"] == spec.modality
    assert prov["bank_kind"] == "per_sample"
    assert prov["source_sample"] == sample.get("sample_id"), (
        f"{corpus}: provenance.source_sample drifted from sample_id"
    )
    if spec.archetype_cluster_field:
        assert prov.get("cluster_key"), (
            f"{corpus}: cluster_field={spec.archetype_cluster_field!r} "
            f"set but provenance.cluster_key not populated"
        )

    # Report book-keeping
    assert report["expected_answer"] == sample.get("gold_answer", "")
    assert report["judge_correct"] is True
    assert report["lift_stats"]["n_hops"] >= 1, (
        f"{corpus}: lift_stats.n_hops is 0"
    )


# ---------------------------------------------------------------------------
# Negative path -- correct=False is filtered when include_incorrect is False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", SINGLE_SHOT_CORPORA)
def test_incorrect_sample_filtered_by_default(corpus: str):
    spec = cs.get_spec(corpus)
    sample_path = _find_first_sample(corpus, want_correct=False)
    if sample_path is None:
        pytest.skip(
            f"{corpus}: no correct=False samples to exercise the filter"
        )
    sample = _load_sample(sample_path)
    out = lift_one_sample(sample, spec=spec)
    assert out is None, (
        f"{corpus}: incorrect sample should be filtered when "
        f"include_incorrect=False (got {type(out).__name__})"
    )
    # ...but with include_incorrect=True, the lift must still succeed
    # (provided schema + reasoning are non-empty -- these are the only
    # other gates lift_one_sample applies up-front).
    if sample.get("schema") and sample.get("answer_reasoning"):
        out2 = lift_one_sample(sample, spec=spec, include_incorrect=True)
        assert out2 is not None, (
            f"{corpus}: include_incorrect=True must lift an incorrect "
            f"sample with non-empty schema + reasoning"
        )
        # verified_domains must be empty because correct=False
        assert out2["skill"]["verified_domains"] == [], (
            f"{corpus}: include_incorrect=True kept verified_domains "
            f"populated despite correct=False"
        )


# ---------------------------------------------------------------------------
# Entity-ref binding -- e\d+ tokens in reasoning must land in hop payloads
# ---------------------------------------------------------------------------


_ENTITY_REF = re.compile(r"\be(\d+)\b")


def _payload_to_string(hop: Dict[str, Any]) -> str:
    """Flatten a hop's payload to a string for substring search."""
    return json.dumps(hop.get("payload", {}), default=str)


@pytest.mark.parametrize("corpus", SINGLE_SHOT_CORPORA)
def test_entity_refs_bound_into_hop_payloads(corpus: str):
    spec = cs.get_spec(corpus)
    # Try up to 5 correct samples until we find one whose reasoning
    # actually cites e\d+ entities -- video corpora sometimes have prose
    # reasoning that doesn't ground to schema entities.
    candidate: Optional[Path] = None
    cited_eids: list[str] = []
    root = spec.default_input_root
    if not root.exists():
        pytest.skip(f"{corpus}: default_input_root missing")
    for sp in sorted(root.glob(spec.sample_glob))[:50]:
        try:
            payload = json.loads(sp.read_text())
        except Exception:
            continue
        if not payload.get("correct"):
            continue
        eids = list({m.group(0) for m in _ENTITY_REF.finditer(
            payload.get("answer_reasoning") or ""
        )})
        if eids:
            candidate = sp
            cited_eids = sorted(eids)
            break
    if candidate is None:
        pytest.skip(
            f"{corpus}: no correct sample with e\\d+ entity refs in "
            f"reasoning across first 50 samples"
        )
    sample = _load_sample(candidate)
    record = lift_one_sample(sample, spec=spec)
    assert record is not None
    # Walk every hop payload + notes and look for AT LEAST ONE of the
    # cited entity IDs. The lift is allowed to drop refs that the
    # protocol-lift classifier treats as redundant; we just require it
    # doesn't drop ALL of them.
    seen = False
    for hop in record["skill"]["protocol"]:
        haystack = _payload_to_string(hop) + " " + (hop.get("notes") or "")
        for eid in cited_eids:
            if re.search(rf"\b{eid}\b", haystack):
                seen = True
                break
        if seen:
            break
    assert seen, (
        f"{corpus}: reasoning cites {cited_eids[:5]} but none of those "
        f"IDs landed in any hop payload/notes (sample={candidate.name})"
    )
