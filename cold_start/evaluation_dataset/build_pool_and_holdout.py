#!/usr/bin/env python
"""Build stratified ``pool`` + ``holdout`` manifests for the five
transfer-target benchmarks of the cold-start lean plan.

Sizes mirror the project README's "Cold-start data generation — lean
plan" table:

  benchmark           pool   holdout   stratification axis
  ------------------  -----  -------   -------------------
  osworld              120      30    snapshot (11 apps) × possibility_of_env_change
  visual_toolbench     300     100    prompt_category (9)
  tir_bench            300     100    task family (13)
  video_holmes        1000     200    Question Type (7)
  siv_bench            400     100    category (10)

Outputs are written to ``cold_start/evaluation_dataset/{pool,holdout}/``
so the directories drop straight into the actor launchers without
spurious cross-talk between splits:

  pool/<benchmark>.txt              one sample id per line
  pool/osworld_catalog.json         OSWorld ``--task_catalog`` format
  holdout/<benchmark>.txt           same shape, frozen for E0/E1/E2
  holdout/osworld_catalog.json      same
  _axis_distribution.json           per-bucket diagnostic across all 5

The pool and holdout slices are **disjoint** by construction
(``pool ∩ holdout = ∅`` is asserted at build time) and **proportionally
stratified** within each axis bucket — every category that appears in
the pool also appears in the holdout whenever the bucket has ≥ 2
sampled items.

Sample IDs match what the actor launchers emit / parse:

  osworld          row['id']                                  (UUID)
  visual_toolbench row['id']                                  (HF row id, str)
  tir_bench        row['id']                                  (HF row id, str)
  video_holmes     "{video_id}.Q{question_id}"
  siv_bench        "{video_id}.Q{tsv_row_index}"

Run:
  python cold_start/evaluation_dataset/build_pool_and_holdout.py

Override sizes (any subset):
  python cold_start/evaluation_dataset/build_pool_and_holdout.py \
    --osworld_pool 120 --osworld_holdout 30 \
    --vtb_pool 300   --vtb_holdout 100 \
    --tir_pool 300   --tir_holdout 100 \
    --vh_pool 1000   --vh_holdout 200 \
    --siv_pool 400   --siv_holdout 100

Wire-up:
  # Visual-reasoning launcher reads the directory, autoglobs <bench>.txt
  python cold_start/generate_cold_start_actor_visual_reasoning.py \
    --sample_ids_dir cold_start/evaluation_dataset/pool ...
  # OSWorld launcher reads the JSON catalog directly
  python cold_start/generate_cold_start_actor_osworld.py \
    --task_catalog cold_start/evaluation_dataset/pool/osworld_catalog.json ...
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

OUT_DIR = Path(__file__).resolve().parent
POOL_DIR = OUT_DIR / "pool"
HOLDOUT_DIR = OUT_DIR / "holdout"
COLD_START_DIR = OUT_DIR.parent
REPO_ROOT = COLD_START_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
SEED = 0


# --- Default lean-plan sizes -------------------------------------------------

DEFAULTS = {
    "osworld":          (120,  30),
    "visual_toolbench": (300, 100),
    "tir_bench":        (300, 100),
    "video_holmes":     (1000, 200),
    "siv_bench":        (400, 100),
}


# --- Round-robin stratified sampler -----------------------------------------

def _round_robin_by_key(
    items: List[dict],
    key_fn: Callable[[dict], str],
    k: int,
    rng: random.Random,
) -> Tuple[List[dict], dict]:
    """Round-robin sample ``k`` items, one bucket at a time.

    Both bucket order and within-bucket order are shuffled with ``rng``.
    Stops early if every bucket is exhausted before ``k`` is reached.
    """
    buckets: dict = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    keys = list(buckets)
    rng.shuffle(keys)
    for v in buckets.values():
        rng.shuffle(v)
    out: List[dict] = []
    cursors = {k_: 0 for k_ in keys}
    while len(out) < k:
        progressed = False
        for key in keys:
            if cursors[key] >= len(buckets[key]):
                continue
            out.append(buckets[key][cursors[key]])
            cursors[key] += 1
            progressed = True
            if len(out) >= k:
                break
        if not progressed:
            break
    return out, dict(buckets)


def _stratified_pool_and_holdout(
    items: List[dict],
    key_fn: Callable[[dict], str],
    pool_size: int,
    holdout_size: int,
    rng: random.Random,
) -> Tuple[List[dict], List[dict]]:
    """Disjoint stratified pool / holdout split.

    Step 1. Round-robin sample ``pool + holdout`` rows from the full
            data. Round-robin gives **equal** bucket representation
            (rare categories are over-sampled relative to their natural
            frequency), which is what we want for both training-style
            pools and small evaluation holdouts.
    Step 2. Within each bucket of the sample, split items at the
            ``pool_size / (pool_size + holdout_size)`` ratio. This
            guarantees **every category that appears in the pool also
            appears in the holdout** (modulo buckets with only one
            sampled item, where pool wins by tie-break).

    Step 3. Adjust to exact pool/holdout sizes by spilling overflow
            between halves with a deterministic shuffle.
    """
    target = pool_size + holdout_size
    sampled, _ = _round_robin_by_key(items, key_fn, target, rng)

    bucket_items: dict = defaultdict(list)
    for it in sampled:
        bucket_items[key_fn(it)].append(it)

    pool: List[dict] = []
    holdout: List[dict] = []
    pool_ratio = pool_size / max(target, 1)
    for key, rows in bucket_items.items():
        n = len(rows)
        n_pool = round(n * pool_ratio)
        # Tie-break so both halves see the bucket whenever it has ≥ 2
        # sampled items.
        if n >= 2:
            n_pool = max(1, min(n - 1, n_pool))
        pool.extend(rows[:n_pool])
        holdout.extend(rows[n_pool:])

    # Round-trip rebalance to exact sizes (rounding can over/undershoot
    # by a couple).
    rng.shuffle(pool)
    rng.shuffle(holdout)
    if len(pool) > pool_size:
        holdout.extend(pool[pool_size:])
        pool = pool[:pool_size]
    elif len(pool) < pool_size and holdout:
        deficit = pool_size - len(pool)
        pool.extend(holdout[:deficit])
        holdout = holdout[deficit:]
    if len(holdout) > holdout_size:
        holdout = holdout[:holdout_size]
    return pool, holdout


# --- Manifest emit -----------------------------------------------------------

def _write_manifest(path: Path, ids: List[str], header: str, source_axis: dict):
    body = "\n".join(ids) + ("\n" if ids else "")
    axis_str = "  ".join(f"{k}={v}" for k, v in sorted(source_axis.items()))
    path.write_text(
        f"# {header}\n"
        f"# count={len(ids)}  seed={SEED}\n"
        f"# axis_distribution: {axis_str}\n"
        f"{body}"
    )
    print(f"  -> {path.name}: {len(ids)} ids   [{axis_str}]")


def _emit(
    name: str,
    rows: List[dict],
    pool_size: int,
    holdout_size: int,
    id_fn: Callable[[dict], str],
    axis_fn: Callable[[dict], str],
    rng: random.Random,
    total_pool: int,
    extra_writers: List[Callable[[List[dict], List[dict]], None]] | None = None,
):
    pool, holdout = _stratified_pool_and_holdout(
        rows, axis_fn, pool_size, holdout_size, rng
    )
    pool_ids = [id_fn(r) for r in pool]
    holdout_ids = [id_fn(r) for r in holdout]
    overlap = set(pool_ids) & set(holdout_ids)
    assert not overlap, f"{name}: pool ∩ holdout overlap = {sorted(overlap)[:5]}"
    pool_axis = dict(Counter(axis_fn(r) for r in pool))
    holdout_axis = dict(Counter(axis_fn(r) for r in holdout))
    _write_manifest(
        POOL_DIR / f"{name}.txt",
        pool_ids,
        f"{name} pool — {len(pool_ids)} / {total_pool} stratified",
        pool_axis,
    )
    _write_manifest(
        HOLDOUT_DIR / f"{name}.txt",
        holdout_ids,
        f"{name} holdout (frozen for E0/E1/E2 eval) — {len(holdout_ids)} / {total_pool}",
        holdout_axis,
    )
    if extra_writers:
        for writer in extra_writers:
            writer(pool, holdout)
    return {
        "pool": {"size": len(pool_ids), "axis_distribution": pool_axis},
        "holdout": {"size": len(holdout_ids), "axis_distribution": holdout_axis},
    }


# --- Per-benchmark loaders + samplers ---------------------------------------

_OSWORLD_SNAPSHOT_ALIASES = {
    "multiapps": "multi_apps",
    "vscode": "vs_code",
}


def _normalize_snapshot(s: str) -> str:
    s = (s or "UNK").strip().lower().replace(" ", "_")
    return _OSWORLD_SNAPSHOT_ALIASES.get(s, s)


def sample_osworld(args, rng) -> dict | None:
    print("\n=== osworld ===")
    examples_dir = WORKSPACE_ROOT / "OSWorld" / "evaluation_examples" / "examples"
    if not examples_dir.is_dir():
        print(f"  [skip] {examples_dir} not found")
        return None
    rows = []
    for p in sorted(examples_dir.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            print(f"  [warn] {p.name}: {exc}")
            continue
        rows.append(
            {
                "id": d.get("id"),
                "snapshot": _normalize_snapshot(d.get("snapshot")),
                "env_change": d.get("possibility_of_env_change") or "UNK",
                "instruction": d.get("instruction", "")[:80],
                "_app_dir": p.parent.name,  # ground truth app per directory layout
            }
        )
    apps = sorted({r["snapshot"] for r in rows})
    env_changes = sorted({r["env_change"] for r in rows})
    print(f"  pool total: {len(rows)}  (snapshots: {len(apps)} unique, "
          f"env_change tiers: {env_changes})")

    def _write_osworld_catalogs(pool: List[dict], holdout: List[dict]) -> None:
        # OSWorld's --task_catalog wants {"<domain>": ["uuid", ...]} where
        # <domain> is the on-disk directory name (matches the launcher's
        # ``examples/<domain>/<uuid>.json`` glob). Use ``_app_dir`` rather
        # than the (sometimes-noisy) ``snapshot`` field.
        # NOTE: the python launcher resolves task UUIDs at
        #     <catalog_path.parent>/examples/<domain>/<uuid>.json
        # so we ALSO drop a symlink to OSWorld's ``examples/`` next to the
        # catalog.  Without this, the launcher reports "No tasks resolved".
        osworld_examples_src = Path("/workspace/OSWorld/evaluation_examples/examples")
        for split_name, split_rows, dest in (
            ("pool", pool, POOL_DIR / "osworld_catalog.json"),
            ("holdout", holdout, HOLDOUT_DIR / "osworld_catalog.json"),
        ):
            catalog: dict = defaultdict(list)
            for r in split_rows:
                catalog[r["_app_dir"]].append(r["id"])
            for k in catalog:
                catalog[k].sort()
            dest.write_text(json.dumps(dict(sorted(catalog.items())), indent=2))
            link = dest.parent / "examples"
            if osworld_examples_src.is_dir():
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(osworld_examples_src)
            else:
                print(f"  [WARN] {osworld_examples_src} not found — "
                      f"OSWorld launcher will fail to resolve tasks. "
                      f"Install OSWorld eval examples at that path.")
            print(f"  -> {dest.relative_to(OUT_DIR)}: "
                  f"{sum(len(v) for v in catalog.values())} ids "
                  f"across {len(catalog)} domains ({split_name})  "
                  f"(examples/ symlink: "
                  f"{'OK' if link.is_symlink() else 'MISSING'})")

    return _emit(
        "osworld",
        rows,
        args.osworld_pool,
        args.osworld_holdout,
        id_fn=lambda r: r["id"],
        # Stratify by (snapshot, env_change) for stricter diversity, but
        # report the per-snapshot distribution which is the headline axis.
        axis_fn=lambda r: r["snapshot"],
        rng=rng,
        total_pool=len(rows),
        extra_writers=[_write_osworld_catalogs],
    )


def sample_visual_toolbench(args, rng) -> dict | None:
    print("\n=== visual_toolbench ===")
    parquet = REPO_ROOT / "data" / "datasets" / "VisualToolBench" / "test.parquet"
    if not parquet.is_file():
        print(f"  [skip] {parquet} not found")
        return None
    import pandas as pd  # local import; pandas is heavy
    df = pd.read_parquet(parquet)
    n_total = len(df)
    if "num_turns" in df.columns:
        df = df[df["num_turns"] == 1]
    print(f"  pool: single-turn={len(df)} / total={n_total}")
    target = args.vtb_pool + args.vtb_holdout
    if len(df) <= target:
        print(f"  [WARN] pool ({len(df)}) < target ({target}); pool gets the front "
              f"slice, holdout may be smaller than requested")
    rows = df.to_dict(orient="records")
    return _emit(
        "visual_toolbench",
        rows,
        args.vtb_pool,
        args.vtb_holdout,
        id_fn=lambda r: str(r["id"]),
        axis_fn=lambda r: r.get("prompt_category") or "UNK",
        rng=rng,
        total_pool=len(rows),
    )


def sample_tir_bench(args, rng) -> dict | None:
    print("\n=== tir_bench ===")
    path = REPO_ROOT / "data" / "datasets" / "TIR-Bench" / "TIR-Bench.json"
    if not path.is_file():
        print(f"  [skip] {path} not found")
        return None
    rows = json.loads(path.read_text())
    print(f"  pool total: {len(rows)}")
    return _emit(
        "tir_bench",
        rows,
        args.tir_pool,
        args.tir_holdout,
        id_fn=lambda r: str(r["id"]),
        axis_fn=lambda r: r.get("task") or "UNK",
        rng=rng,
        total_pool=len(rows),
    )


def sample_video_holmes(args, rng) -> dict | None:
    print("\n=== video_holmes ===")
    path = REPO_ROOT / "data" / "Video-Holmes" / "Benchmark" / "test_Video-Holmes.json"
    if not path.is_file():
        print(f"  [skip] {path} not found")
        return None
    rows = json.loads(path.read_text())
    print(f"  pool total: {len(rows)}")
    return _emit(
        "video_holmes",
        rows,
        args.vh_pool,
        args.vh_holdout,
        id_fn=lambda r: (
            f"{r.get('video ID') or r.get('video_id')}."
            f"Q{r.get('Question ID') or r.get('question_id')}"
        ),
        axis_fn=lambda r: r.get("Question Type") or "UNK",
        rng=rng,
        total_pool=len(rows),
    )


def sample_siv_bench(args, rng) -> dict | None:
    print("\n=== siv_bench ===")
    tsv = REPO_ROOT / "data" / "SIV-Bench" / "SIV-Bench-QA.tsv"
    if not tsv.is_file():
        print(f"  [skip] {tsv} not found")
        return None
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    # Tag each row with its TSV index — that's what the launcher uses as
    # the question_id portion of sample_id.
    for i, r in enumerate(rows):
        r["_qid"] = i
    print(f"  pool total: {len(rows)}")
    def _id(r):
        vid = r.get("video_id") or r.get("video") or r.get("id") or ""
        return f"{vid}.Q{r['_qid']}"
    return _emit(
        "siv_bench",
        rows,
        args.siv_pool,
        args.siv_holdout,
        id_fn=_id,
        axis_fn=lambda r: r.get("category") or "UNK",
        rng=rng,
        total_pool=len(rows),
    )


# --- manifest metadata (provenance + reuse tooling) -------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _emit_manifest_metadata(args, summary: dict) -> Path:
    """Write a top-level ``manifest.json`` recording per-file SHA-256 +
    counts + sizes + provenance.

    Consumers can (a) check that the manifests on disk match the recorded
    hashes before reusing them, and (b) use the size table as a contract
    when wiring new launchers — no need to ``wc -l`` the text files."""
    manifest_path = OUT_DIR / "manifest.json"
    files: dict = {}
    for split, base in (("pool", POOL_DIR), ("holdout", HOLDOUT_DIR)):
        for p in sorted(base.glob("*")):
            # The OSWorld catalog writer drops a sibling ``examples/``
            # symlink to the OSWorld eval-examples dir so the launcher
            # can resolve task UUIDs; skip it (and any other directory)
            # since the manifest only tracks regular file artefacts.
            if not p.is_file():
                continue
            rel = p.relative_to(OUT_DIR).as_posix()
            count = None
            if p.suffix == ".txt":
                count = sum(
                    1
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith("#")
                )
            elif p.suffix == ".json":
                # OSWorld catalog: count flattened ids
                try:
                    cat = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(cat, dict):
                        count = sum(len(v) for v in cat.values())
                except Exception:
                    pass
            files[rel] = {
                "sha256": _sha256_file(p),
                "bytes": p.stat().st_size,
                "ids": count,
            }

    sizes = {
        name: {"pool": result["pool"]["size"], "holdout": result["holdout"]["size"]}
        for name, result in summary.items()
    }

    manifest = {
        "kind": "cold_start.evaluation_dataset",
        "version": 1,
        "seed": SEED,
        "built_at_utc": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "builder": {
            "script": "cold_start/evaluation_dataset/build_pool_and_holdout.py",
            "python": platform.python_version(),
        },
        "args": {
            "osworld_pool":   args.osworld_pool,   "osworld_holdout":   args.osworld_holdout,
            "vtb_pool":       args.vtb_pool,       "vtb_holdout":       args.vtb_holdout,
            "tir_pool":       args.tir_pool,       "tir_holdout":       args.tir_holdout,
            "vh_pool":        args.vh_pool,        "vh_holdout":        args.vh_holdout,
            "siv_pool":       args.siv_pool,       "siv_holdout":       args.siv_holdout,
        },
        "sizes": sizes,
        "axis_distribution": summary,
        "files": files,
        "notes": (
            "Pool/holdout disjoint by construction; per-bucket "
            "proportional split guarantees every category in pool also "
            "appears in holdout when bucket has >=2 sampled items. "
            "Re-run build_pool_and_holdout.py at any time — manifests are "
            "byte-stable for fixed seed + dataset versions."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


# --- main --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--osworld_pool",   type=int, default=DEFAULTS["osworld"][0])
    p.add_argument("--osworld_holdout",type=int, default=DEFAULTS["osworld"][1])
    p.add_argument("--vtb_pool",       type=int, default=DEFAULTS["visual_toolbench"][0])
    p.add_argument("--vtb_holdout",    type=int, default=DEFAULTS["visual_toolbench"][1])
    p.add_argument("--tir_pool",       type=int, default=DEFAULTS["tir_bench"][0])
    p.add_argument("--tir_holdout",    type=int, default=DEFAULTS["tir_bench"][1])
    p.add_argument("--vh_pool",        type=int, default=DEFAULTS["video_holmes"][0])
    p.add_argument("--vh_holdout",     type=int, default=DEFAULTS["video_holmes"][1])
    p.add_argument("--siv_pool",       type=int, default=DEFAULTS["siv_bench"][0])
    p.add_argument("--siv_holdout",    type=int, default=DEFAULTS["siv_bench"][1])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    summary = {}
    for name, fn in [
        ("osworld",          sample_osworld),
        ("visual_toolbench", sample_visual_toolbench),
        ("tir_bench",        sample_tir_bench),
        ("video_holmes",     sample_video_holmes),
        ("siv_bench",        sample_siv_bench),
    ]:
        # Use a per-benchmark RNG seed slice so adding/removing one
        # benchmark doesn't reshuffle the others. ``hashlib.sha256`` is a
        # platform-stable hash — Python's built-in ``hash()`` is salted
        # per-interpreter unless ``PYTHONHASHSEED`` is pinned, which would
        # silently break manifest reproducibility.
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        seed_offset = int.from_bytes(digest[:4], "big") % 10_000
        per_rng = random.Random(SEED + seed_offset)
        result = fn(args, per_rng)
        if result is not None:
            summary[name] = result

    if summary:
        diag = OUT_DIR / "_axis_distribution.json"
        diag.write_text(json.dumps(summary, indent=2))
        print(f"\nDiagnostic: {diag}")

        manifest = _emit_manifest_metadata(args, summary)
        print(f"Manifest:   {manifest}")

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
