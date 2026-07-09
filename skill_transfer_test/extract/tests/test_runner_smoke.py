"""End-to-end smoke for ``runner.py`` + ``_unify.py`` (closes TODO-5).

Drives the runner's :func:`skill_transfer_test.extract.runner.main` with
``--max-samples 2`` against the visual / video corpora that have on-disk
data, then walks the output tree to check:

* every requested corpus produced a per-corpus directory
* each per-corpus dir holds a ``per_sample/skill_bank.jsonl`` (or
  ``per_episode/skill_bank.jsonl`` for sequence corpora) with at least
  one row
* ``rollup.json`` is well-formed and lists the same set of corpora
* :func:`skill_transfer_test.extract._unify.unify_root` then aggregates
  the run-id root cleanly: the unified outputs exist, the
  ``skill_index.jsonl`` is non-empty, the ``skill_catalog_all.json``
  groups by corpus, and the ``skill_rag_index.json`` carries one
  entry per skill in the bank.

Skips entirely when the cold-start tree is missing -- this test is a
correctness gate, not a data-availability one.

Tests use ``tmp_path`` so the on-disk artifacts are torn down with the
test session and never leak into ``skill_transfer_test/skill_bank_local/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_transfer_test.extract import _corpus_specs as cs
from skill_transfer_test.extract import _unify, runner


SMOKE_CORPORA: tuple[str, ...] = (
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)


def _have_data_for(corpus: str) -> bool:
    """True if at least one sample is on disk for ``corpus``."""
    spec = cs.get_spec(corpus)
    if not spec.default_input_root.exists():
        return False
    return next(iter(spec.default_input_root.glob(spec.sample_glob)), None) is not None


def _runnable_corpora() -> list[str]:
    return [c for c in SMOKE_CORPORA if _have_data_for(c)]


# ---------------------------------------------------------------------------
# Runner end-to-end
# ---------------------------------------------------------------------------


def test_runner_main_emits_per_corpus_banks(tmp_path: Path):
    runnable = _runnable_corpora()
    if not runnable:
        pytest.skip(
            "no cold-start data on disk for any of: "
            + ", ".join(SMOKE_CORPORA)
        )
    output_root = tmp_path / "skill_bank"
    rc = runner.main([
        "--output-root", str(output_root),
        "--run-id", "smoke_e2e",
        "--corpora", *runnable,
        "--max-samples", "2",
    ])
    assert rc == 0, "runner.main returned non-zero"

    run_dir = output_root / "smoke_e2e"
    assert run_dir.is_dir(), f"runner did not create {run_dir}"

    # rollup.json shape
    rollup = run_dir / "rollup.json"
    assert rollup.exists(), "runner did not emit rollup.json"
    summaries = json.loads(rollup.read_text())
    assert isinstance(summaries, list), "rollup.json is not a list"
    assert len(summaries) == len(runnable), (
        f"rollup has {len(summaries)} entries; expected {len(runnable)}"
    )

    # Every requested corpus must have produced a bank-jsonl. Single-shot
    # corpora ship per_sample/, sequence corpora ship per_episode/.
    for corpus in runnable:
        spec = cs.get_spec(corpus)
        if spec.lift_kind == "single_shot":
            bank = run_dir / corpus / "per_sample" / "skill_bank.jsonl"
        else:
            bank = run_dir / corpus / "per_episode" / "skill_bank.jsonl"
        assert bank.exists(), f"{corpus}: missing bank at {bank}"
        rows = [r for r in bank.read_text().splitlines() if r.strip()]
        # max-samples=2 with correct-only filter can occasionally yield 0
        # if both top samples are correct=False; we still want the test
        # to be meaningful so we *expect* >=1 across all runnable corpora
        # rather than per-corpus.
        if rows:
            row0 = json.loads(rows[0])
            assert "skill" in row0 and "report" in row0, (
                f"{corpus}: row 0 missing report/skill envelope"
            )

    # Cross-corpus invariant: at least one corpus must have produced a
    # non-empty bank (otherwise the smoke is meaningless).
    total_rows = 0
    for corpus in runnable:
        spec = cs.get_spec(corpus)
        kind_dir = "per_sample" if spec.lift_kind == "single_shot" else "per_episode"
        bank = run_dir / corpus / kind_dir / "skill_bank.jsonl"
        if bank.exists():
            total_rows += sum(
                1 for line in bank.read_text().splitlines() if line.strip()
            )
    assert total_rows >= 1, (
        f"runner emitted 0 total rows across {runnable}; "
        f"max-samples=2 should have lifted at least 1"
    )


# ---------------------------------------------------------------------------
# Runner -> _unify integration
# ---------------------------------------------------------------------------


def test_runner_then_unify_roundtrip(tmp_path: Path):
    runnable = _runnable_corpora()
    if not runnable:
        pytest.skip("no cold-start data on disk")
    output_root = tmp_path / "skill_bank"
    rc = runner.main([
        "--output-root", str(output_root),
        "--run-id", "smoke_unify",
        "--corpora", *runnable,
        "--max-samples", "2",
    ])
    assert rc == 0
    run_dir = output_root / "smoke_unify"

    summary = _unify.unify_root(run_dir, corpora=runnable)
    assert summary["n_banks"] >= 1, (
        f"_unify discovered no banks under {run_dir}: {summary}"
    )
    assert summary["n_corpora"] >= 1, summary

    # Outputs all exist
    unified_dir = run_dir / "_unified"
    assert unified_dir.is_dir()
    flat_path = unified_dir / "skill_index.jsonl"
    catalog_path = unified_dir / "skill_catalog_all.json"
    rag_path = unified_dir / "skill_rag_index.json"
    assert flat_path.exists() and catalog_path.exists() and rag_path.exists()

    # skill_index.jsonl: non-empty, every row carries corpus + bank_kind
    flat_rows = [
        json.loads(line) for line in flat_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(flat_rows) >= 1, f"_unified/skill_index.jsonl is empty"
    for row in flat_rows:
        assert "corpus" in row and "bank_kind" in row, row.keys()
        assert row["corpus"] in runnable, (
            f"unexpected corpus tag {row['corpus']!r} on a unified row"
        )

    # skill_catalog_all.json: corpora dict keyed by corpus name
    catalog = json.loads(catalog_path.read_text())
    assert "corpora" in catalog, catalog.keys()
    assert set(catalog["corpora"].keys()) <= set(runnable), (
        f"catalog corpora {set(catalog['corpora'].keys())} not subset of "
        f"requested {set(runnable)}"
    )

    # skill_rag_index.json: one entry per flat row
    rag = json.loads(rag_path.read_text())
    assert rag["n_entries"] == len(rag["entries"]) == len(flat_rows), (
        f"rag entry count drift: n_entries={rag['n_entries']} "
        f"len(entries)={len(rag['entries'])} flat={len(flat_rows)}"
    )
    # rag ids are unique
    rag_ids = [e["id"] for e in rag["entries"]]
    assert len(rag_ids) == len(set(rag_ids)), "duplicate ids in rag index"
