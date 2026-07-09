#!/usr/bin/env python
"""Build diverse ~200-task subsamples for each BrowserGym suite.

Sampling strategy (deterministic, seed=0):

  miniwob          all 125 tasks (already < 200; every task is its own family).
  webarena         200 / 812. Stratified by primary site (5 buckets), then
                   round-robin over intent_template_id within each bucket so
                   every distinct intent is covered before doubling up.
  assistantbench   all 181 test rows (test_general + test_expert);
                   already < 200, full coverage.

Outputs (one file per suite, one task id per line):

  browsergym_miniwob_200.txt
  browsergym_webarena_200.txt
  browsergym_assistantbench_200.txt
  browsergym_all_diverse.txt    (concatenation of the above; ~506 tasks)

Each file is consumable by ``run_coldstart_actor_browsergym.sh --tasks``
via xargs:

    xargs -a browsergym_webarena_200.txt \\
        bash cold_start/run_coldstart_actor_browsergym.sh --tasks

NOTE: The visualwebarena branch of this builder was retired on
2026-05-03 — see ``legacy/visualwebarena/README.md``. The historical
``sample_visualwebarena`` function and its 200/910 manifest are
preserved unchanged in the legacy archive for reproducibility.
"""

from __future__ import annotations

import importlib.resources
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEED = 0
TARGET_PER_SUITE = 200


# ---------------------------------------------------------------------------
# Sources of truth
# ---------------------------------------------------------------------------

def _load_webarena_configs():
    import webarena
    raw = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
    return json.loads(raw)


def _load_miniwob_ids():
    import gymnasium as gym
    import browsergym.miniwob  # noqa: F401  registers tasks
    return sorted(
        k for k in gym.envs.registry.keys()
        if k.startswith("browsergym/miniwob.")
    )


def _load_assistantbench_test_ids():
    import gymnasium as gym
    import browsergym.assistantbench  # noqa: F401
    return sorted(
        k for k in gym.envs.registry.keys()
        if k.startswith("browsergym/assistantbench.test.")
    )


# ---------------------------------------------------------------------------
# Stratified samplers
# ---------------------------------------------------------------------------

def _round_robin_by_key(items, key_fn, k, rng):
    """Sample k items so every distinct key is hit before any key doubles up."""
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
    return out


def sample_webarena(configs, k, seed):
    rng = random.Random(seed)
    by_site = defaultdict(list)
    for c in configs:
        primary = c["sites"][0] if c.get("sites") else "_other"
        by_site[primary].append(c)

    # Quota per primary site, proportional to count, +1 floor.
    total = sum(len(v) for v in by_site.values())
    quotas = {s: max(1, round(k * len(v) / total)) for s, v in by_site.items()}
    while sum(quotas.values()) > k:
        s = max(quotas, key=lambda x: quotas[x])
        quotas[s] -= 1
    while sum(quotas.values()) < k:
        s = max(by_site, key=lambda x: len(by_site[x]))
        quotas[s] += 1

    sampled = []
    for site, q in quotas.items():
        chunk = _round_robin_by_key(
            by_site[site],
            key_fn=lambda c: c["intent_template_id"],
            k=q,
            rng=rng,
        )
        sampled.extend(chunk)
    rng.shuffle(sampled)
    return sampled, quotas


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write(path: Path, lines, header):
    body = "\n".join(lines) + "\n"
    path.write_text(f"# {header}\n# count={len(lines)}  seed={SEED}\n{body}")
    print(f"  wrote {path.name}: {len(lines)} tasks")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== miniwob ===")
    miniwob_ids = _load_miniwob_ids()
    print(f"  total registered: {len(miniwob_ids)} (taking ALL — every slug is a unique family)")
    _write(
        OUT_DIR / "browsergym_miniwob_200.txt",
        miniwob_ids,
        "miniwob: full coverage (125 distinct task families, < target of 200)",
    )

    print("\n=== webarena ===")
    wa_cfg = _load_webarena_configs()
    wa_sampled, wa_quotas = sample_webarena(wa_cfg, TARGET_PER_SUITE, SEED)
    site_dist = Counter(c["sites"][0] for c in wa_sampled)
    tmpl_dist = Counter(c["intent_template_id"] for c in wa_sampled)
    print(f"  sampled: {len(wa_sampled)}  sites covered: {dict(site_dist)}")
    print(f"  distinct intent_templates covered: {len(tmpl_dist)} / 190")
    wa_ids = [f"browsergym/webarena.{c['task_id']}" for c in wa_sampled]
    _write(
        OUT_DIR / "browsergym_webarena_200.txt",
        wa_ids,
        f"webarena: 200/812 stratified by site × intent_template (covers {len(tmpl_dist)}/190 templates, all 5 sites)",
    )

    print("\n=== assistantbench ===")
    ab_ids = _load_assistantbench_test_ids()
    print(f"  total test rows: {len(ab_ids)} (taking ALL — already below target of 200)")
    _write(
        OUT_DIR / "browsergym_assistantbench_200.txt",
        ab_ids,
        "assistantbench: full test set coverage (test_general + test_expert)",
    )

    print("\n=== combined manifest ===")
    all_ids = miniwob_ids + wa_ids + ab_ids
    _write(
        OUT_DIR / "browsergym_all_diverse.txt",
        all_ids,
        f"three suites (miniwob + webarena + assistantbench) — diverse subsample (~200/suite); total={len(all_ids)}",
    )

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
