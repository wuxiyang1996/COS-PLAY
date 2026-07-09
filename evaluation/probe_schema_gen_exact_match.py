#!/usr/bin/env python3
"""Schema-gen exact-match probe (T1.1′).

Loads the trained ``schema_gen`` LoRA on top of its base
(``Qwen/Qwen3.5-35B-A3B`` per ``runs/sft_schema_gen/<run>/adapter_config.json``)
and runs *N* held-out grounding triples through generation. Reports:

  - **Exact-match rate** — full-string equality between predicted and gold
    ``<state>...</state>`` blocks (after whitespace normalisation).
  - **Field-level accuracy** — per-tag agreement on the 5 standard
    schema sections (``intention``, ``observations``, ``entities``,
    ``relations``, ``ambiguities``). Coarse counter-style approximation
    of the §13 "field accuracy ≥85%" target since deep semantic equiv
    needs a teacher judge.
  - **Path-A acceptance** — fraction of predictions that the validator
    accepts without escalating to tools (PLAN-VISUAL-GROUNDING-MILESTONES §4).

Spec:
  - ``implementation_notes/pre-training-readiness-audit.md`` §0.1 (T1.1′)
  - ``plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md`` §13
    (Phase 1 exit: 8B field accuracy ≥85% on Gym-V, ≥75% on browser;
    Path A ≥70%)

Cost / runtime:
  ~3 min on H100 / 35B-A3B for ``--n 50`` samples (default). Set
  ``--n 200`` for the full §13-aligned probe (~12 min).

Usage:
    # Smoke probe — 30 samples, default holdout from the manifest.
    python evaluation/probe_schema_gen_exact_match.py --n 30

    # Full §13 probe.
    python evaluation/probe_schema_gen_exact_match.py \\
        --n 200 \\
        --report runs/probe_schema_gen_<ts>.json

    # Override the adapter / data location.
    python evaluation/probe_schema_gen_exact_match.py \\
        --adapter runs/sft_schema_gen/schema_gen_20260430_091831 \\
        --gymv-root labeling/output/grounding/gymv \\
        --domains gymv,env_wrappers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

# Whitespace canonicalisation — applied to both gold and predicted
# schemas before string equality. Whitespace collapse only; we do NOT
# touch tag content.
_WS_RE = re.compile(r"[ \t]+\n")
_BLANKS_RE = re.compile(r"\n{3,}")


def _canon(text: str) -> str:
    text = _WS_RE.sub("\n", text.strip())
    text = _BLANKS_RE.sub("\n\n", text)
    return text


_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def _extract_tags(schema: str) -> Dict[str, str]:
    """Return a {tag → inner-text} mapping over all `<tag>...</tag>` blocks."""
    out: Dict[str, str] = {}
    for m in _TAG_RE.finditer(schema or ""):
        tag = m.group(1).strip()
        body = m.group(2).strip()
        if tag:
            out[tag] = body
    return out


# Five schema sections per PLAN-VISUAL-GROUNDING-MILESTONES §3.
EXPECTED_SCHEMA_FIELDS = (
    "intention",
    "observations",
    "entities",
    "relations",
    "ambiguities",
)


def _path_a_accepts(predicted: str) -> bool:
    """Coarse Path-A validator: schema must contain ``<state>`` and
    at least 3 of the 5 expected sub-tags. Mirrors the validator's
    minimum bar from §3 / §11."""
    if "<state>" not in predicted or "</state>" not in predicted:
        return False
    tags = _extract_tags(predicted)
    n_expected = sum(1 for f in EXPECTED_SCHEMA_FIELDS if f in tags)
    return n_expected >= 3


@dataclass
class ProbeRow:
    sample_id: str
    domain: str
    image_path: str
    n_prompt_tokens: int = 0
    n_predicted_tokens: int = 0
    elapsed_s: float = 0.0
    exact_match: bool = False
    path_a_accept: bool = False
    field_overlap: Dict[str, bool] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeReport:
    adapter_path: str
    base_model: str
    n_samples: int
    n_exact_match: int
    n_path_a_accept: int
    field_accuracy: Dict[str, float]
    overall_field_accuracy: float
    elapsed_s: float
    rows: List[ProbeRow]
    config: Dict[str, Any]

    def summary(self) -> str:
        em = self.n_exact_match / max(1, self.n_samples)
        pa = self.n_path_a_accept / max(1, self.n_samples)
        return (
            f"\n=== schema_gen exact-match probe ===\n"
            f"adapter:               {self.adapter_path}\n"
            f"base:                  {self.base_model}\n"
            f"n_samples:             {self.n_samples}\n"
            f"exact_match_rate:      {em:.3f}  ({self.n_exact_match}/{self.n_samples})\n"
            f"path_a_accept_rate:    {pa:.3f}  ({self.n_path_a_accept}/{self.n_samples})\n"
            f"overall_field_acc:     {self.overall_field_accuracy:.3f}\n"
            + "".join(
                f"  - {k:<14s} {v:.3f}\n" for k, v in self.field_accuracy.items()
            )
            + f"elapsed:               {self.elapsed_s:.1f}s\n"
            f"\nTarget (§13 PLAN-VISUAL-GROUNDING-MILESTONES):\n"
            f"  • field accuracy ≥0.85 on gymv\n"
            f"  • Path-A acceptance ≥0.70 (Phase-1)\n"
            f"  • exact match: no spec target — informational only\n"
        )


def _load_holdout_samples(
    *,
    gymv_root: Path,
    env_wrappers_root: Path,
    browser_root: Path,
    image_qa_jsonl: Path,
    video_qa_jsonl: Path,
    domains: List[str],
    n: int,
    seed: int,
) -> List["SchemaGenSample"]:                          # noqa: F821
    """Sample ``n`` held-out triples deterministically across domains."""
    sys.path.insert(0, str(REPO_ROOT))
    from trainer.SFT.schema_gen.config import SchemaGenConfig
    from trainer.SFT.schema_gen.data_loader import load_schema_gen_dataset

    cfg = SchemaGenConfig(
        domains=domains,
        gymv_triple_root=str(gymv_root),
        env_wrappers_triple_root=str(env_wrappers_root),
        browser_triple_root=str(browser_root),
        image_qa_jsonl=str(image_qa_jsonl),
        video_qa_jsonl=str(video_qa_jsonl),
        # Probe deliberately includes hard-cases — that's the populace
        # the §13 threshold is measured against in deployment.
        drop_hard_cases=False,
        max_samples_per_domain=None,
    )
    pool = load_schema_gen_dataset(cfg)
    if not pool:
        raise SystemExit(
            f"No samples found. Looked in:\n  gymv={gymv_root}\n"
            f"  env_wrappers={env_wrappers_root}\n  browser={browser_root}\n"
            "Hint: re-run labeling/build_schema_gen_triples.py first."
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _generate(
    *,
    model: Any,
    processor: Any,
    sample: "SchemaGenSample",                            # noqa: F821
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    """One generation pass — returns the raw assistant string."""
    import torch

    messages = [
        {"role": "user", "content": [
            *({"type": "image", "image": p} for p in sample.images),
            {"type": "text", "text": sample.prompt},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id,
        )
    in_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[0, in_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def _score_one(
    sample: "SchemaGenSample",                            # noqa: F821
    predicted: str,
    elapsed_s: float,
    *,
    n_prompt_tokens: int,
    n_predicted_tokens: int,
) -> ProbeRow:
    gold = _canon(sample.target_schema or "")
    pred = _canon(predicted or "")
    em = bool(gold) and gold == pred

    gold_tags = _extract_tags(gold)
    pred_tags = _extract_tags(pred)
    field_overlap = {
        f: (f in pred_tags and pred_tags[f] == gold_tags.get(f, ""))
        for f in EXPECTED_SCHEMA_FIELDS
    }

    return ProbeRow(
        sample_id=sample.sample_id,
        domain=sample.domain,
        image_path=str(sample.images[0]) if sample.images else "",
        n_prompt_tokens=n_prompt_tokens,
        n_predicted_tokens=n_predicted_tokens,
        elapsed_s=round(elapsed_s, 3),
        exact_match=em,
        path_a_accept=_path_a_accepts(pred),
        field_overlap=field_overlap,
    )


def _resolve_adapter(adapter_arg: Optional[str]) -> Path:
    """Find the most recent ``runs/sft_schema_gen/<ts>/`` if not specified."""
    if adapter_arg:
        return Path(adapter_arg).resolve()
    sg = REPO_ROOT / "runs" / "sft_schema_gen"
    if not sg.is_dir():
        raise SystemExit(f"No --adapter specified and {sg} does not exist.")
    candidates = sorted(
        (d for d in sg.iterdir() if d.is_dir() and d.name.startswith("schema_gen_")),
        key=lambda d: d.name,
        reverse=True,
    )
    for c in candidates:
        if (c / "adapter_config.json").is_file() and (
            c / "adapter_model.safetensors"
        ).is_file():
            return c
    raise SystemExit(
        f"No loadable schema_gen adapter under {sg}. "
        "Train one first or pass --adapter."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Schema-gen exact-match probe (T1.1′).",
    )
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of held-out samples to probe (default 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Sampling seed; pin for reproducibility.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
    )
    parser.add_argument(
        "--domains", type=str, default="gymv,env_wrappers",
        help="Comma-separated subset of {gymv,env_wrappers,browser,image_qa,video_qa}.",
    )
    parser.add_argument(
        "--gymv-root", type=Path,
        default=REPO_ROOT / "labeling" / "output" / "grounding" / "gymv",
    )
    parser.add_argument(
        "--env-wrappers-root", type=Path,
        default=REPO_ROOT / "labeling" / "output" / "grounding" / "env_wrappers",
    )
    parser.add_argument(
        "--browser-root", type=Path,
        default=REPO_ROOT / "labeling" / "output" / "grounding" / "browser",
    )
    parser.add_argument(
        "--image-qa-jsonl", type=Path,
        default=REPO_ROOT / "labeling" / "output" / "grounding" / "image_qa" / "labels.jsonl",
    )
    parser.add_argument(
        "--video-qa-jsonl", type=Path,
        default=REPO_ROOT / "labeling" / "output" / "grounding" / "video_qa" / "labels.jsonl",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Where to write the JSON report (default: print to stdout).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu"], default="auto",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HOME", "/workspace/huggingface")
    os.environ.setdefault(
        "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"),
    )

    adapter_path = _resolve_adapter(args.adapter)
    adapter_cfg = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8"),
    )
    base_model = adapter_cfg.get("base_model_name_or_path", "")
    print(
        f"Probing schema_gen adapter:\n"
        f"  adapter = {adapter_path}\n"
        f"  base    = {base_model}\n"
        f"  n       = {args.n}\n"
        f"  seed    = {args.seed}\n"
        f"  domains = {args.domains}",
        file=sys.stderr,
    )

    # ── Load model + adapter ────────────────────────────────────────
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
    except ImportError:                                         # pragma: no cover
        AutoModelForImageTextToText = None                      # type: ignore

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    is_multimodal = hasattr(cfg, "text_config") or hasattr(cfg, "vision_config")
    loader_cls = (
        AutoModelForImageTextToText
        if (is_multimodal and AutoModelForImageTextToText is not None)
        else AutoModelForCausalLM
    )

    device_map = "cpu" if args.device == "cpu" else "auto"
    base = loader_cls.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Sample held-out triples ─────────────────────────────────────
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    pool = _load_holdout_samples(
        gymv_root=args.gymv_root,
        env_wrappers_root=args.env_wrappers_root,
        browser_root=args.browser_root,
        image_qa_jsonl=args.image_qa_jsonl,
        video_qa_jsonl=args.video_qa_jsonl,
        domains=domains,
        n=args.n,
        seed=args.seed,
    )
    print(f"Loaded {len(pool)} held-out samples", file=sys.stderr)

    # ── Run inference ───────────────────────────────────────────────
    rows: List[ProbeRow] = []
    t0 = time.monotonic()
    for i, sample in enumerate(pool):
        s0 = time.monotonic()
        try:
            predicted = _generate(
                model=model, processor=processor, sample=sample,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
            elapsed_s = time.monotonic() - s0
            row = _score_one(
                sample, predicted, elapsed_s,
                n_prompt_tokens=0, n_predicted_tokens=0,
            )
        except Exception as exc:                            # noqa: BLE001
            row = ProbeRow(
                sample_id=sample.sample_id,
                domain=sample.domain,
                image_path=str(sample.images[0]) if sample.images else "",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.monotonic() - s0,
            )
        rows.append(row)
        marker = "✓" if row.exact_match else ("a" if row.path_a_accept else "✗")
        print(
            f"  [{i+1:>3d}/{len(pool):<3d}] {marker} {row.domain:>14s} "
            f"{row.sample_id[:40]:<40s} {row.elapsed_s:>5.1f}s",
            file=sys.stderr,
        )

    elapsed_total = time.monotonic() - t0

    # ── Aggregate ───────────────────────────────────────────────────
    n = len(rows)
    n_em = sum(1 for r in rows if r.exact_match)
    n_pa = sum(1 for r in rows if r.path_a_accept)
    field_acc = {
        f: (sum(1 for r in rows if r.field_overlap.get(f)) / max(1, n))
        for f in EXPECTED_SCHEMA_FIELDS
    }
    overall_field_acc = sum(field_acc.values()) / max(1, len(field_acc))

    report = ProbeReport(
        adapter_path=str(adapter_path),
        base_model=base_model,
        n_samples=n,
        n_exact_match=n_em,
        n_path_a_accept=n_pa,
        field_accuracy=field_acc,
        overall_field_accuracy=overall_field_acc,
        elapsed_s=round(elapsed_total, 1),
        rows=rows,
        config={
            "n": args.n,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "domains": domains,
            "device": args.device,
            "do_sample": False,
        },
    )

    print(report.summary())

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "adapter_path": report.adapter_path,
                    "base_model": report.base_model,
                    "n_samples": report.n_samples,
                    "n_exact_match": report.n_exact_match,
                    "n_path_a_accept": report.n_path_a_accept,
                    "field_accuracy": report.field_accuracy,
                    "overall_field_accuracy": report.overall_field_accuracy,
                    "elapsed_s": report.elapsed_s,
                    "config": report.config,
                    "rows": [r.to_dict() for r in report.rows],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Report → {args.report}", file=sys.stderr)

    # Exit non-zero if we missed both Phase-1 thresholds — the whole
    # point of the audit gate is to refuse to launch training under
    # those conditions.
    field_thresh = 0.85
    path_a_thresh = 0.70
    pa_rate = n_pa / max(1, n)
    if overall_field_acc < field_thresh and pa_rate < path_a_thresh:
        print(
            f"\n!! Both §13 thresholds missed: "
            f"field_acc={overall_field_acc:.3f} < {field_thresh} AND "
            f"path_a={pa_rate:.3f} < {path_a_thresh}. "
            f"Block training launch.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
