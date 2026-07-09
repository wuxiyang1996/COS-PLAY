#!/usr/bin/env python3
"""Build the run-wide SFT adapter manifest (T2.9).

Walks ``runs/sft_*`` and emits ``runs/sft_coldstart/sft_summary_all.json``
listing every trained LoRA adapter the trainer can resume / hot-load.
This subsumes the per-adapter ``sft_summary.json`` (which only carries
the *most recently trained* adapter) and gives the
``pre-training-readiness-audit.md`` a single lookup point for "is the
checkpoint ready, and where exactly does it live?".

Spec:
  - ``implementation_notes/pre-training-readiness-audit.md`` §0.1  (T2.9)
  - ``implementation_notes/pre-training-readiness-audit.md`` §0.2  (the
    six trained adapters this manifest reconciles against ``runs/``)

Manifest schema (stable, versioned via ``manifest_version``):

    {
      "manifest_version": "v1.0.0",
      "generated_at": "2026-05-01T07:23:11.000Z",
      "n_adapters": 6,
      "adapters": [
        {
          "name": "schema_gen",
          "phase": "schema_gen",         # or "decision" / "skillbank"
          "base_model": "Qwen/Qwen3.5-35B-A3B",
          "path": "runs/sft_schema_gen/schema_gen_20260430_091831",
          "adapter_config": "...adapter_config.json",
          "adapter_weights": "...adapter_model.safetensors",
          "lora_r": 16,
          "lora_alpha": 32,
          "target_modules": [...],
          "training_artifacts": {
            "checkpoints": ["checkpoint-200", "checkpoint-400", "checkpoint-477"],
            "training_args": "training_args.bin",
            "train_config": "train_config.json"
          },
          "size_bytes": 12345678
        },
        ...
      ]
    }

Usage:
    python scripts/build_sft_manifest.py \\
        --runs-root runs \\
        --output runs/sft_coldstart/sft_summary_all.json

Output is read-only metadata; this script never touches the adapter
files themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


MANIFEST_VERSION = "v1.0.0"


def _phase_for_adapter(name: str, parent_dir: str) -> str:
    """Classify an adapter into the trainer's phase taxonomy.

    `decision`   — Phase A (actor)        — skill_selection, action_taking
    `skillbank`  — Phase B  (skill bank)  — segment, contract, curator
    `schema_gen` — Phase 1 (visual ground.) — schema_gen
    """
    if name == "schema_gen":
        return "schema_gen"
    if "decision" in parent_dir:
        return "decision"
    if "skillbank" in parent_dir:
        return "skillbank"
    return "unknown"


def _resolve_adapter_dir(candidate: Path, name: str) -> Optional[Path]:
    """Find the dir that actually carries ``adapter_config.json``.

    PEFT's ``save_pretrained`` sometimes nests the artifact one level
    deep — the top-level dir contains a sub-dir named after the
    adapter that holds the actual files. Probe both layouts.
    """
    if (candidate / "adapter_config.json").is_file():
        return candidate
    nested = candidate / name
    if (nested / "adapter_config.json").is_file():
        return nested
    return None


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _checkpoints_under(root: Path) -> List[str]:
    """Return any ``checkpoint-*`` subdirs (sorted by step)."""
    if not root.is_dir():
        return []
    out: List[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("checkpoint-"):
            out.append(child.name)
    return out


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _adapter_record(
    name: str,
    home: Path,
    runs_root: Path,
    *,
    parent_dir: str = "",
) -> Optional[Dict[str, Any]]:
    """Build one adapter row. Returns ``None`` when not loadable."""
    real = _resolve_adapter_dir(home, name)
    if real is None:
        return None

    cfg_path = real / "adapter_config.json"
    cfg = _read_json(cfg_path) or {}
    weights = real / "adapter_model.safetensors"
    if not weights.is_file():
        # Defensive: a half-trained adapter without weights is not loadable.
        return None

    # Surrounding directory for taxonomy hints (e.g. .../skillbank/contract → "skillbank").
    if not parent_dir:
        parent_dir = str(home.relative_to(runs_root)).replace("\\", "/")

    train_artifacts: Dict[str, Any] = {
        "checkpoints": _checkpoints_under(real) or _checkpoints_under(home),
    }
    for fname in ("training_args.bin", "train_config.json", "trainer_state.json"):
        candidate = real / fname
        if candidate.is_file():
            train_artifacts[fname.split(".")[0]] = str(
                candidate.relative_to(runs_root.parent)
                if runs_root.parent in candidate.parents
                else candidate
            )

    return {
        "name": name,
        "phase": _phase_for_adapter(name, parent_dir),
        "base_model": cfg.get("base_model_name_or_path", ""),
        "path": str(real.relative_to(runs_root.parent) if runs_root.parent in real.parents else real),
        "adapter_config": str(cfg_path.relative_to(runs_root.parent) if runs_root.parent in cfg_path.parents else cfg_path),
        "adapter_weights": str(weights.relative_to(runs_root.parent) if runs_root.parent in weights.parents else weights),
        "lora_r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_dropout": cfg.get("lora_dropout"),
        "target_modules": sorted(cfg.get("target_modules") or []),
        "training_artifacts": train_artifacts,
        "size_bytes": _safe_size(weights),
        "peft_version": cfg.get("peft_version"),
        "task_type": cfg.get("task_type"),
        "inference_mode": bool(cfg.get("inference_mode", True)),
    }


def discover_adapters(runs_root: Path) -> List[Dict[str, Any]]:
    """Walk runs_root, return one record per loadable adapter."""
    rows: List[Dict[str, Any]] = []

    # ── schema_gen — pick the most recent timestamped subdir ─────────
    sg_root = runs_root / "sft_schema_gen"
    if sg_root.is_dir():
        ts_dirs = sorted(
            (d for d in sg_root.iterdir() if d.is_dir() and d.name.startswith("schema_gen_")),
            key=lambda d: d.name,
            reverse=True,
        )
        for ts in ts_dirs:
            rec = _adapter_record(
                "schema_gen", ts, runs_root, parent_dir="sft_schema_gen",
            )
            if rec is not None:
                rec["latest_for_name"] = (ts == ts_dirs[0])
                rows.append(rec)

    # ── cold-start — decision (skill_selection / action_taking) ──────
    cs_root = runs_root / "sft_coldstart"
    for sub in ("decision", "skillbank"):
        sub_root = cs_root / sub
        if not sub_root.is_dir():
            continue
        for adapter_dir in sorted(sub_root.iterdir()):
            if not adapter_dir.is_dir():
                continue
            name = adapter_dir.name
            rec = _adapter_record(
                name, adapter_dir, runs_root,
                parent_dir=f"sft_coldstart/{sub}",
            )
            if rec is not None:
                rec["latest_for_name"] = True
                rows.append(rec)

    return rows


def build_manifest(runs_root: Path) -> Dict[str, Any]:
    rows = discover_adapters(runs_root)
    rows.sort(key=lambda r: (r["phase"], r["name"]))
    by_phase: Dict[str, List[str]] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r["name"])
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "runs_root": str(runs_root),
        "n_adapters": len(rows),
        "by_phase": by_phase,
        "adapters": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the run-wide SFT adapter manifest (T2.9).",
    )
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
        help="Root of the runs/ tree (default: <repo>/runs).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Output path (default: <runs-root>/sft_coldstart/sft_summary_all.json).",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Print manifest to stdout, do not write to disk.",
    )
    args = parser.parse_args(argv)

    runs_root = args.runs_root.resolve()
    if not runs_root.is_dir():
        parser.error(f"--runs-root does not exist: {runs_root}")

    manifest = build_manifest(runs_root)

    print(
        f"Discovered {manifest['n_adapters']} adapter(s) under "
        f"{runs_root}:",
        file=sys.stderr,
    )
    for r in manifest["adapters"]:
        ck = r["training_artifacts"]["checkpoints"]
        print(
            f"  • {r['phase']:>11s} / {r['name']:<16s} | "
            f"{r['base_model']:<28s} | "
            f"r={r['lora_r']:>3} α={r['lora_alpha']:>3} | "
            f"checkpoints={len(ck):>2}",
            file=sys.stderr,
        )

    if args.print_only:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
        return 0

    out_path = args.output or (runs_root / "sft_coldstart" / "sft_summary_all.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nWrote manifest → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
