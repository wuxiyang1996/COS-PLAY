"""Tiny loader for the frozen evaluation-dataset manifests.

Other modules (gate, few-shot adapter, baselines, eval scripts) should
read ID lists through this helper rather than re-parsing the text files
inline — that way every consumer goes through one provenance-checked
path and a hash-mismatch is caught immediately.

Typical use:

    from cold_start.evaluation_dataset.load_manifests import (
        load_ids, load_osworld_catalog, verify_integrity,
    )

    pool_ids = load_ids("video_holmes", split="pool")     # list[str]
    held_ids = load_ids("osworld",       split="holdout")
    catalog  = load_osworld_catalog("pool")               # dict[str, list[str]]

    drift = verify_integrity()                            # raises on hash mismatch

The manifests live in ``cold_start/evaluation_dataset/{pool,holdout}/``;
``manifest.json`` next to this file records seed, build timestamp, and
SHA-256 per file (regenerate with
``python cold_start/evaluation_dataset/build_pool_and_holdout.py``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Literal

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
POOL_DIR = ROOT / "pool"
HOLDOUT_DIR = ROOT / "holdout"

Split = Literal["pool", "holdout"]
KNOWN_BENCHMARKS = (
    "osworld",
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)


def _split_dir(split: Split) -> Path:
    if split == "pool":
        return POOL_DIR
    if split == "holdout":
        return HOLDOUT_DIR
    raise ValueError(f"unknown split {split!r}; want 'pool' or 'holdout'")


def load_ids(benchmark: str, split: Split = "pool") -> List[str]:
    """Return the list of sample ids for ``benchmark`` in ``split``.

    Matches the IDs the actor launchers emit, see manifest header for
    the per-benchmark format.
    """
    if benchmark not in KNOWN_BENCHMARKS:
        raise KeyError(
            f"unknown benchmark {benchmark!r}; known: {KNOWN_BENCHMARKS}"
        )
    path = _split_dir(split) / f"{benchmark}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"manifest {path} missing — run build_pool_and_holdout.py"
        )
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_osworld_catalog(split: Split = "pool") -> Dict[str, List[str]]:
    """Return the OSWorld catalog (``--task_catalog`` format)."""
    path = _split_dir(split) / "osworld_catalog.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"OSWorld catalog {path} missing — run build_pool_and_holdout.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest() -> dict:
    """Return the parsed top-level ``manifest.json`` (sizes + hashes)."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} missing — run build_pool_and_holdout.py"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_integrity(strict: bool = True) -> Dict[str, str]:
    """Re-hash every recorded file and compare against the manifest.

    Returns a dict mapping changed-file rel-paths → reason (``mismatch``
    or ``missing``). When ``strict=True`` (default), raises
    ``RuntimeError`` if any drift is detected.
    """
    manifest = load_manifest()
    drift: Dict[str, str] = {}
    for rel, meta in manifest["files"].items():
        path = ROOT / rel
        if not path.is_file():
            drift[rel] = "missing"
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            drift[rel] = "mismatch"
    if drift and strict:
        raise RuntimeError(
            "evaluation_dataset drift detected — re-run "
            "build_pool_and_holdout.py or restore from VCS:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in drift.items())
        )
    return drift


if __name__ == "__main__":
    # Smoke-test: print a one-line summary per benchmark + verify hashes.
    m = load_manifest()
    print(f"manifest seed={m['seed']}  built_at={m['built_at_utc']}  "
          f"benchmarks={list(m['sizes'])}")
    for bench, sizes in m["sizes"].items():
        print(f"  {bench:<18} pool={sizes['pool']:>4}  holdout={sizes['holdout']:>4}")
    drift = verify_integrity(strict=False)
    if drift:
        print(f"\nDRIFT detected for {len(drift)} files:")
        for k, v in drift.items():
            print(f"  {k}: {v}")
    else:
        print("\nintegrity ok — all files match recorded hashes")
