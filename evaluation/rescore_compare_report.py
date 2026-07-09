#!/usr/bin/env python3
"""Re-score an existing compare_base_vs_lora.json report with a corrected
section-aware extractor.

Background:
  evaluation/smoke_schema_gen_5tasks.py uses ``<(\\w+)>(.*?)</\\1>`` to extract
  per-section bodies, which assumes paired open/close tags. The canonical
  schema_gen format actually uses *section markers* — ``<entities>``,
  ``<state_flags>``, etc. open a section that runs until the next ``<tag>`` or
  the closing ``</state>``. There are no ``</entities>`` / ``</state_flags>``
  closes anywhere in gold or in predictions. The old scorer therefore reported
  field_overlap = 0/7 across the board, hiding the real per-field signal.

This script reads an existing report, recomputes per-section content, and
prints the corrected per-mode summary plus a side-by-side per-field overlap.

Usage:
    python evaluation/rescore_compare_report.py runs/t1_1prime/compare_base_vs_lora.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

EXPECTED_SCHEMA_FIELDS = (
    "entities",
    "relations",
    "state_flags",
    "targets",
    "uncertainty",
    "actions",
    "evidence",
)

_STATE_RE = re.compile(r"<state>(.*?)(?:</state>|\Z)", re.DOTALL)
# Section marker = an opening tag on its own line (possibly indented)
_SECTION_RE = re.compile(r"(?m)^[ \t]*<(\w+)>[ \t]*$")


def _split_sections(schema: str) -> Dict[str, str]:
    """Return ``{section_name: body_text}`` for the canonical schema format.

    Sections are delimited by ``<name>`` markers on their own line; the body
    runs until the next marker (or the closing ``</state>`` / end of string).
    The pre-amble between ``<state>`` and the first ``<section>`` marker
    (typically the ``domain=...`` / ``task=...`` / ``goal=...`` / ``step=...``
    header) is returned under the synthetic key ``"_header"``.
    """
    if not schema:
        return {}
    m = _STATE_RE.search(schema)
    if not m:
        return {}
    body = m.group(1)
    markers = list(_SECTION_RE.finditer(body))
    sections: Dict[str, str] = {}
    if not markers:
        sections["_header"] = body.strip()
        return sections
    sections["_header"] = body[: markers[0].start()].strip()
    for i, mk in enumerate(markers):
        name = mk.group(1)
        body_start = mk.end()
        body_end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        sections[name] = body[body_start:body_end].strip()
    return sections


def _norm_body(s: str) -> str:
    """Light normalization for body comparison: strip + collapse whitespace."""
    return re.sub(r"[ \t]+", " ", s.strip())


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    args = ap.parse_args(argv)

    rep = json.loads(args.report.read_text())
    rows = rep["results"]

    by_mode: Dict[str, List[dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    print(f"\n=== rescored from {args.report} ===")
    print(f"modes: {list(by_mode.keys())}, n_per_mode: "
          f"{ {m: len(rs) for m, rs in by_mode.items()} }\n")

    per_mode_agg: Dict[str, dict] = {}
    per_task_per_mode: Dict[str, Dict[str, dict]] = {}

    for mode, rs in by_mode.items():
        n_tasks = len(rs)
        section_present: Dict[str, int] = {f: 0 for f in EXPECTED_SCHEMA_FIELDS}
        section_body_exact: Dict[str, int] = {f: 0 for f in EXPECTED_SCHEMA_FIELDS}
        section_body_prefix: Dict[str, float] = {f: 0.0 for f in EXPECTED_SCHEMA_FIELDS}
        n_state_closed = 0
        n_pa = 0  # path-a: <state>...</state> closed AND >=3 expected sections present

        for r in rs:
            pred = r["predicted_full"] or ""
            gold = r["gold_full"] or ""
            ps = _split_sections(pred)
            gs = _split_sections(gold)
            if "</state>" in pred:
                n_state_closed += 1
            n_present = sum(1 for f in EXPECTED_SCHEMA_FIELDS if f in ps)
            if "</state>" in pred and n_present >= 3:
                n_pa += 1
            for f in EXPECTED_SCHEMA_FIELDS:
                if f in ps:
                    section_present[f] += 1
                if f in ps and f in gs:
                    p_body = _norm_body(ps[f])
                    g_body = _norm_body(gs[f])
                    if p_body == g_body:
                        section_body_exact[f] += 1
                    cpl = _common_prefix_len(p_body, g_body)
                    section_body_prefix[f] += cpl / max(1, len(g_body))

            per_task_per_mode.setdefault(r["sample_id"], {})[mode] = {
                "task_id": r["task_id"],
                "n_present": n_present,
                "state_closed": "</state>" in pred,
                "ps": ps,
                "gs": gs,
            }

        present_pct = {f: section_present[f] / n_tasks for f in EXPECTED_SCHEMA_FIELDS}
        body_em_pct = {f: section_body_exact[f] / n_tasks for f in EXPECTED_SCHEMA_FIELDS}
        body_prefix_avg = {
            f: section_body_prefix[f] / n_tasks for f in EXPECTED_SCHEMA_FIELDS
        }

        per_mode_agg[mode] = {
            "n_tasks": n_tasks,
            "n_state_closed": n_state_closed,
            "n_path_a_accept": n_pa,
            "section_present_pct": present_pct,
            "section_body_em_pct": body_em_pct,
            "section_body_prefix_avg": body_prefix_avg,
        }

        print(f"[mode={mode}]"
              f"  n={n_tasks}"
              f"  </state>={n_state_closed}/{n_tasks}"
              f"  pathA={n_pa}/{n_tasks}")
        print(f"  per-section presence (% of {n_tasks} tasks the section is present):")
        for f in EXPECTED_SCHEMA_FIELDS:
            print(f"    - {f:<14s} present={present_pct[f]*100:5.1f}%   "
                  f"body_EM={body_em_pct[f]*100:5.1f}%   "
                  f"mean body-prefix={body_prefix_avg[f]*100:5.1f}%")
        print()

    if "base" in per_mode_agg and "lora" in per_mode_agg:
        b = per_mode_agg["base"]
        l = per_mode_agg["lora"]
        print("=== per-section LoRA uplift over base ===")
        print(f"  {'section':<14s}  "
              f"{'base_present':>13s}  {'lora_present':>13s}  "
              f"{'base_bodyEM':>12s}  {'lora_bodyEM':>12s}  "
              f"{'base_bodyPfx':>13s}  {'lora_bodyPfx':>13s}")
        for f in EXPECTED_SCHEMA_FIELDS:
            print(f"  {f:<14s}  "
                  f"{b['section_present_pct'][f]*100:>12.1f}%  "
                  f"{l['section_present_pct'][f]*100:>12.1f}%  "
                  f"{b['section_body_em_pct'][f]*100:>11.1f}%  "
                  f"{l['section_body_em_pct'][f]*100:>11.1f}%  "
                  f"{b['section_body_prefix_avg'][f]*100:>12.1f}%  "
                  f"{l['section_body_prefix_avg'][f]*100:>12.1f}%")

        print("\n=== per-task header-section comparison ===")
        print(f"  {'task':<40s}  "
              f"{'header_base_pfx%':>16s}  {'header_lora_pfx%':>16s}")
        for sid, by_m in per_task_per_mode.items():
            if "base" not in by_m or "lora" not in by_m:
                continue
            gb = by_m["base"]["gs"].get("_header", "")
            pb_b = by_m["base"]["ps"].get("_header", "")
            pb_l = by_m["lora"]["ps"].get("_header", "")
            cpl_b = _common_prefix_len(_norm_body(pb_b), _norm_body(gb))
            cpl_l = _common_prefix_len(_norm_body(pb_l), _norm_body(gb))
            denom = max(1, len(_norm_body(gb)))
            print(f"  {by_m['base']['task_id']:<40s}  "
                  f"{cpl_b/denom*100:>15.1f}%  "
                  f"{cpl_l/denom*100:>15.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
