"""Field-validation tests for ``skill_transfer_test.extract._corpus_specs``.

Closes TODO-3. Verifies the six cross-domain ``CorpusSpec``s the
extractor pipeline registers are well-formed and that each
``archetype_cluster_field`` (when set) is a navigation path that
actually resolves on a sample drawn from that corpus' canonical
on-disk layout.

Tests are pure / hermetic where they can be -- the
``archetype_cluster_field`` resolution test is *opt-in* and skipped if
the cold-start sample referenced by ``default_input_root`` is not on
disk in the test environment. CI runners without the cold-start data
still exercise the pure-data shape checks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skill_transfer_test.extract import _corpus_specs as cs


# Authoritative inventory -- if these don't all show up the registry
# regressed.
EXPECTED_CORPORA: tuple[str, ...] = (
    "browsergym",
    "osworld",
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)

VALID_LIFT_KINDS: tuple[str, ...] = ("sequence", "single_shot")
VALID_MODALITIES: tuple[str, ...] = (
    "games", "desktop", "browser", "image", "video",
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_all_six_corpora_registered():
    """The registry must export exactly the canonical six corpora."""
    names = set(cs.all_names())
    assert set(EXPECTED_CORPORA) == names, (
        f"registry drift: expected {sorted(EXPECTED_CORPORA)}, "
        f"got {sorted(names)}"
    )


def test_all_specs_returns_one_per_name():
    specs = cs.all_specs()
    assert len(specs) == len(set(s.name for s in specs)), (
        "duplicate spec.name in all_specs()"
    )
    assert {s.name for s in specs} == set(cs.all_names())


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError, match="unknown corpus"):
        cs.get_spec("__no_such_corpus__")


@pytest.mark.parametrize("name", EXPECTED_CORPORA)
def test_get_spec_roundtrip(name: str):
    spec = cs.get_spec(name)
    assert spec.name == name


# ---------------------------------------------------------------------------
# Per-spec field shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_CORPORA)
def test_required_fields_present_and_typed(name: str):
    spec = cs.get_spec(name)
    assert isinstance(spec.name, str) and spec.name
    assert spec.lift_kind in VALID_LIFT_KINDS
    assert spec.modality in VALID_MODALITIES
    assert isinstance(spec.domain, str) and spec.domain
    assert isinstance(spec.default_input_root, Path)
    assert isinstance(spec.sample_glob, str) and spec.sample_glob
    # archetype_cluster_field is optional but if set must be a string
    # (empty string is treated the same as None -- catch both).
    if spec.archetype_cluster_field is not None:
        assert isinstance(spec.archetype_cluster_field, str)
        assert spec.archetype_cluster_field, (
            f"{name}: archetype_cluster_field set to empty string"
        )


def test_resolve_input_root_override_wins():
    spec = cs.get_spec("visual_toolbench")
    override = Path("/tmp/__override__")
    assert spec.resolve_input_root(override) == override
    assert spec.resolve_input_root(None) == spec.default_input_root


# ---------------------------------------------------------------------------
# archetype_cluster_field navigation -- resolved against a real sample
# when available, otherwise skipped.
# ---------------------------------------------------------------------------


def _navigate(d: Any, dotted_path: str) -> Any:
    """Walk ``a.b.c`` against a nested dict; raise KeyError on miss."""
    cur = d
    for key in dotted_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(dotted_path)
        cur = cur[key]
    return cur


@pytest.mark.parametrize("name", [
    n for n in EXPECTED_CORPORA
    if cs.get_spec(n).archetype_cluster_field
])
def test_archetype_cluster_field_navigates_real_sample(name: str):
    """The cluster_field must be reachable on a real on-disk sample.

    Skipped when the corpus' default_input_root has no matching samples
    -- some CI environments don't ship the cold-start data tree.
    """
    spec = cs.get_spec(name)
    root = spec.default_input_root
    if not root.exists():
        pytest.skip(f"{name}: default_input_root missing ({root})")
    samples = list(root.glob(spec.sample_glob))
    if not samples:
        pytest.skip(
            f"{name}: no samples match {root}/{spec.sample_glob}"
        )
    # Walk only the FIRST sample -- the schema is supposed to be
    # corpus-uniform; if even one sample's path can't navigate, the
    # cluster_field is wrong for the whole corpus.
    sample_payload = json.loads(samples[0].read_text())
    try:
        _navigate(sample_payload, spec.archetype_cluster_field)
    except KeyError:
        pytest.fail(
            f"{name}: archetype_cluster_field={spec.archetype_cluster_field!r}"
            f" did not resolve in {samples[0]}"
        )
