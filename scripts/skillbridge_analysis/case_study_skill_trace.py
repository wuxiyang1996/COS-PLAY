"""Provenance trail for a single ``skill_id`` (block E8).

Given a run directory and a ``skill_id``, walks the available logs and
prints (and optionally writes) a chronological case study:

* lifecycle transitions (DRAFT → PROVISIONAL → ACTIVE → DEPRECATED → …)
* every rejection (eligibility veto code + reason) involving the skill
* every validate_invocation diagnostic involving the skill
* every step in ``reward_log.jsonl`` where the skill was chosen
* every audit.jsonl event referencing the skill (crafter mutations,
  promotion judge decisions, dump-driver rebinds, etc.)

The output is a single Markdown document suitable for embedding into a
NeurIPS appendix case-study panel.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--skill-id", required=True)
    p.add_argument("--out-path", type=Path, default=None)
    p.add_argument(
        "--max-events-per-stream", type=int, default=200,
        help="Truncate each stream to keep the trace readable.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _filter(
    rows: Iterable[Dict[str, Any]],
    skill_id: str,
    *,
    extra_keys: tuple[str, ...] = (),
    max_events: int = 200,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("skill_id") == skill_id:
            out.append(r)
            continue
        for k in extra_keys:
            if r.get(k) == skill_id or skill_id in str(r.get(k, "")):
                out.append(r)
                break
        if len(out) >= max_events:
            break
    return out


def _section(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"### {title}\n\n_(no events)_\n\n"
    lines = [f"### {title} ({len(rows)} events)\n"]
    lines.append("```jsonl")
    for r in rows[:200]:
        lines.append(json.dumps(r, default=str, ensure_ascii=False))
    lines.append("```\n")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sid = args.skill_id

    lifecycle = _filter(
        _iter_jsonl(args.run_dir / "lifecycle_log" / "transitions.jsonl"),
        sid, max_events=args.max_events_per_stream,
    )
    rejections = _filter(
        _iter_jsonl(args.run_dir / "harness_log" / "rejections.jsonl"),
        sid, max_events=args.max_events_per_stream,
    )
    validates = _filter(
        _iter_jsonl(args.run_dir / "harness_log" / "validate.jsonl"),
        sid, max_events=args.max_events_per_stream,
    )
    reward = _filter(
        _iter_jsonl(args.run_dir / "reward_log.jsonl"),
        sid, extra_keys=("chosen_skill_id",),
        max_events=args.max_events_per_stream,
    )

    audit_paths = [
        args.run_dir / "audit.jsonl",
        args.run_dir / "_artifacts" / "audit.jsonl",
    ]
    audit: List[Dict[str, Any]] = []
    for p in audit_paths:
        if p.exists():
            audit.extend(
                _filter(
                    _iter_jsonl(p), sid,
                    extra_keys=(
                        "skill",
                        "proposal_id",
                        "target_skill_id",
                    ),
                    max_events=args.max_events_per_stream,
                )
            )

    md_parts = [
        f"# Skill case study: `{sid}`\n",
        f"_run_dir_: `{args.run_dir}`\n",
        _section("Lifecycle transitions", lifecycle),
        _section("Eligibility rejections (filter_candidates)", rejections),
        _section("Validate-invocation diagnostics", validates),
        _section("Reward log (chosen_skill_id matches)", reward),
        _section("Audit (crafter / promotion / orchestrator)", audit),
    ]
    md = "\n".join(md_parts)

    out_path = args.out_path or (
        args.run_dir / "analysis" / f"case_study_{sid}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    logger.info("wrote %s", out_path)

    print(md[:4000])
    if len(md) > 4000:
        print("\n... (truncated, see file)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
