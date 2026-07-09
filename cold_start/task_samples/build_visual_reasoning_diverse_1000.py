#!/usr/bin/env python
"""Build diverse ~1,000-sample subsamples for each visual-reasoning benchmark.

Sampling strategy (deterministic, seed=0):

  visual_toolbench  all 603 single-turn rows (the wrapper's default
                    iterator only yields single-turn anyway). Pool is
                    smaller than the 1,000 target.
  tir_bench         1,000 / 1,215. Round-robin over the 13 ``task``
                    families so every family is exhausted at the same
                    rate. Hits all 13 families with ~77 each.
  video_holmes      1,000 / 1,837. Round-robin over the 7
                    ``Question Type`` buckets so MHR / SR / IMC / TCI /
                    CTI / TA / PAR are sampled at ~143 each.
  siv_bench         1,000 / 8,728. Round-robin over the 10 SIV-Bench
                    ``category`` buckets so the long-tail buckets
                    (Human Attribute Identification = 121, Facial
                    Expression = 218) are not crowded out.

Outputs (one sample id per line, headers comment-prefixed):

  visual_toolbench_1000.txt
  tir_bench_1000.txt
  video_holmes_1000.txt
  siv_bench_1000.txt
  visual_reasoning_all_diverse.txt   (concat; ~3,603 ids)

Each file contains ``sample_id`` strings that match the format the
visual-reasoning launcher emits per sample:

  visual_toolbench   row['id'] (HF row id, str)
  tir_bench          row['id'] (HF row id, str)
  video_holmes       "{video_id}.Q{question_id}"
  siv_bench          "{video_id}.Q{tsv_row_index}"

Use the new ``--sample_ids_dir`` flag on
``generate_cold_start_actor_visual_reasoning.py`` to filter the
iterators by these manifests at run time.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parent.parent
SEED = 0
TARGET_PER_BENCHMARK = 1000


# ---------------------------------------------------------------------------
# Round-robin stratified sampler (mirrors build_browsergym_diverse_200.py)
# ---------------------------------------------------------------------------

def _round_robin_by_key(items, key_fn, k, rng):
    buckets: dict = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    keys = list(buckets)
    rng.shuffle(keys)
    for v in buckets.values():
        rng.shuffle(v)
    out = []
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


def _write_manifest(path: Path, ids, header):
    body = "\n".join(ids) + "\n"
    path.write_text(f"# {header}\n# count={len(ids)}  seed={SEED}\n{body}")
    print(f"  wrote {path.name}: {len(ids)} sample ids")


# ---------------------------------------------------------------------------
# Per-benchmark loaders + samplers
# ---------------------------------------------------------------------------

def sample_visual_toolbench():
    print("=== visual_toolbench ===")
    parquet = REPO_ROOT / "data" / "datasets" / "VisualToolBench" / "test.parquet"
    if not parquet.exists():
        print(f"  [skip] {parquet} not found — run hf download first")
        return None
    import pandas as pd  # local import; pandas is heavy
    df = pd.read_parquet(parquet)
    n_total = len(df)
    if "num_turns" in df.columns:
        df = df[df["num_turns"] == 1]
    print(f"  pool: single-turn={len(df)} / total={n_total}")
    if len(df) <= TARGET_PER_BENCHMARK:
        ids = [str(x) for x in df["id"].tolist()]
        cat = Counter(df["prompt_category"].fillna("UNK"))
        print(f"  taking all (pool < target). prompt_category dist: {dict(cat)}")
        return ids, "visual_toolbench: all 603 single-turn rows (pool < 1000 target)"

    rng = random.Random(SEED)
    rows = df.to_dict(orient="records")
    sampled, buckets = _round_robin_by_key(
        rows,
        key_fn=lambda r: r.get("prompt_category") or "UNK",
        k=TARGET_PER_BENCHMARK,
        rng=rng,
    )
    cat = Counter((r.get("prompt_category") or "UNK") for r in sampled)
    ef = Counter((r.get("eval_focus") or "UNK") for r in sampled)
    print(f"  prompt_category covered: {dict(cat)}")
    print(f"  eval_focus covered: {dict(ef)}")
    ids = [str(r["id"]) for r in sampled]
    return ids, f"visual_toolbench: {len(ids)}/{n_total} stratified by prompt_category"


def sample_tir_bench():
    print("\n=== tir_bench ===")
    path = REPO_ROOT / "data" / "datasets" / "TIR-Bench" / "TIR-Bench.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    rows = json.loads(path.read_text())
    print(f"  pool total: {len(rows)}")
    rng = random.Random(SEED)
    sampled, buckets = _round_robin_by_key(
        rows,
        key_fn=lambda r: r.get("task") or "UNK",
        k=TARGET_PER_BENCHMARK,
        rng=rng,
    )
    fam = Counter(r.get("task") or "UNK" for r in sampled)
    print(f"  task families covered: {dict(fam)}")
    ids = [str(r["id"]) for r in sampled]
    return ids, f"tir_bench: {len(ids)}/{len(rows)} stratified by 13 task families"


def sample_video_holmes():
    print("\n=== video_holmes ===")
    path = REPO_ROOT / "data" / "Video-Holmes" / "Benchmark" / "test_Video-Holmes.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    rows = json.loads(path.read_text())
    print(f"  pool total: {len(rows)}")
    rng = random.Random(SEED)
    sampled, buckets = _round_robin_by_key(
        rows,
        key_fn=lambda r: r.get("Question Type") or "UNK",
        k=TARGET_PER_BENCHMARK,
        rng=rng,
    )
    qtype = Counter(r.get("Question Type") or "UNK" for r in sampled)
    print(f"  question types covered: {dict(qtype)}")
    ids = [
        f"{r.get('video ID') or r.get('video_id')}.Q{r.get('Question ID') or r.get('question_id')}"
        for r in sampled
    ]
    return ids, f"video_holmes: {len(ids)}/{len(rows)} stratified by 7 Question Types"


def sample_siv_bench():
    print("\n=== siv_bench ===")
    tsv = REPO_ROOT / "data" / "SIV-Bench" / "SIV-Bench-QA.tsv"
    if not tsv.exists():
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

    rng = random.Random(SEED)
    sampled, buckets = _round_robin_by_key(
        rows,
        key_fn=lambda r: r.get("category") or "UNK",
        k=TARGET_PER_BENCHMARK,
        rng=rng,
    )
    cat = Counter(r.get("category") or "UNK" for r in sampled)
    pool_cat = Counter(r.get("category") or "UNK" for r in rows)
    print(f"  pool   category dist: {dict(pool_cat)}")
    print(f"  sample category dist: {dict(cat)}")
    ids = []
    # The video_id alias chain prefers ``video_id`` then ``video`` then
    # ``id``. Match that order so the manifest matches the launcher.
    for r in sampled:
        vid = (
            r.get("video_id")
            or r.get("video")
            or r.get("id")
            or ""
        )
        ids.append(f"{vid}.Q{r['_qid']}")
    return ids, f"siv_bench: {len(ids)}/{len(rows)} stratified by 10 categories"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {}
    for name, fn in [
        ("visual_toolbench", sample_visual_toolbench),
        ("tir_bench", sample_tir_bench),
        ("video_holmes", sample_video_holmes),
        ("siv_bench", sample_siv_bench),
    ]:
        result = fn()
        if result is None:
            continue
        ids, header = result
        path = OUT_DIR / f"{name}_1000.txt"
        _write_manifest(path, ids, header)
        out[name] = ids

    if out:
        all_ids = []
        for name, ids in out.items():
            all_ids.extend(f"{name}\t{x}" for x in ids)
        path = OUT_DIR / "visual_reasoning_all_diverse.txt"
        path.write_text(
            "# all four visual-reasoning benchmarks — diverse subsample\n"
            f"# total={len(all_ids)}  seed={SEED}\n"
            "# format: <benchmark>\\t<sample_id>\n"
            + "\n".join(all_ids) + "\n"
        )
        print(f"\n  wrote {path.name}: {len(all_ids)} ids across {len(out)} benchmarks")

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
