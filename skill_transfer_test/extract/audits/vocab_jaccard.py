"""Vocabulary-Jaccard audit (closes ``TODO-6``).

Computes per-layer Jaccard overlap between the **game** and **cross-domain**
skill-bank clusters, plus per-corpus pairwise Jaccards within the
cross-domain cluster. Reproduces the §11.5.1 (rollout memo) /
§9.3.1 (`skill_transfer_test/README.md`) vocabulary-alignment numbers from
live banks rather than the hardcoded analytical estimates in the memo.

Layers compared:

- ``protocol_ops``           — ``hop["op"]`` strings
- ``slot_types``             — values of ``hop["slot_types"]`` dicts
- ``hop_predicates``         — predicate ``type``s from per-hop ``effects_add/del``
- ``contract_predicates``    — predicate ``type``s from skill-level ``contract.effects_*``
                              (always empty for game banks — they use ``eff_add``/``eff_del``
                              of bare strings; that signal lives in ``hop_predicates`` instead)
- ``predicates_combined``    — ``hop_predicates ∪ contract_predicates``; this is the layer
                              that matters for Stage 1-6 transferability because it is
                              shape-invariant across game vs cross-domain banks.

Outputs ``vocab_jaccard.json`` and ``vocab_jaccard.md`` under
``<output-root>/<run-id>/``.

Invocation::

    python -m skill_transfer_test.extract.audits.vocab_jaccard --verbose

See ``audits/__init__.py`` and the ``_loaders`` module for the shared
discovery / extraction surface.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _utcnow() -> datetime:
    """Timezone-aware UTC ``datetime.now`` (avoids ``utcnow`` deprecation)."""
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    """ISO-8601 UTC timestamp ending in ``Z`` for stable JSON / MD output."""
    return _utcnow().replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

from ._loaders import (
    BankInfo,
    CorpusVocab,
    collect_corpus_vocab,
    discover_banks,
)


_DEFAULT_BANKS_ROOT: tuple[Path, ...] = (
    Path("labeling/skill_bank_out"),
    Path("skill_transfer_test/skill_bank_local/full_v5"),
)
_DEFAULT_OUTPUT_ROOT = Path("cross_domain_results/_phase0")

_LAYERS: tuple[str, ...] = (
    "protocol_ops",
    "slot_types",
    "hop_predicates",
    "contract_predicates",
    "predicates_combined",
)

_REFERENCE_JACCARDS: dict[str, float] = {
    # Rollout memo §11.5.1 analytical estimates; deviations >0.05 get flagged.
    "protocol_ops": 0.82,
    "slot_types": 1.00,
}
_REFERENCE_TOLERANCE = 0.05


def _layer_sets(cv: CorpusVocab) -> dict[str, frozenset[str]]:
    """Return all five vocabulary layers for one corpus."""
    return {
        "protocol_ops": cv.protocol_ops,
        "slot_types": cv.slot_types,
        "hop_predicates": cv.hop_predicates,
        "contract_predicates": cv.contract_predicates,
        "predicates_combined": cv.hop_predicates | cv.contract_predicates,
    }


def _union_layers(corpus_vocabs: Iterable[CorpusVocab]) -> dict[str, frozenset[str]]:
    """Union each layer across a sequence of CorpusVocabs (empty -> empty set)."""
    accum: dict[str, set[str]] = {layer: set() for layer in _LAYERS}
    for cv in corpus_vocabs:
        layers = _layer_sets(cv)
        for layer in _LAYERS:
            accum[layer].update(layers[layer])
    return {layer: frozenset(s) for layer, s in accum.items()}


def _jaccard_record(a: frozenset[str], b: frozenset[str]) -> dict:
    """Jaccard of two sets plus shoulder-side diff lists.

    Returns ``jaccard=None`` when the union is empty (undefined Jaccard);
    callers serialize that as JSON ``null``. Diff lists are sorted for
    diff-stable output.
    """
    inter = a & b
    union = a | b
    if not union:
        return {
            "jaccard": None,
            "intersect": 0,
            "union": 0,
            "a_only": [],
            "b_only": [],
        }
    return {
        "jaccard": round(len(inter) / len(union), 4),
        "intersect": len(inter),
        "union": len(union),
        "a_only": sorted(a - b),
        "b_only": sorted(b - a),
    }


def _format_jaccard(j: float | None) -> str:
    return "—" if j is None else f"{j:.2f}"


def _write_json(
    out_path: Path,
    *,
    args: argparse.Namespace,
    banks: list[BankInfo],
    corpus_vocabs: list[CorpusVocab],
    cluster_vocabs: dict[str, dict[str, frozenset[str]]],
    cluster_sizes: dict[str, dict[str, int]],
    game_vs_cross: dict[str, dict],
    pairwise: dict[str, dict[str, dict[str, float | None]]],
    n_warnings_suppressed: int,
) -> None:
    payload = {
        "run_id": args.run_id,
        "generated_at": _utc_iso(),
        "banks_root": [str(p) for p in args.banks_root],
        "n_banks_total": len(banks),
        "n_banks_game": sum(1 for b in banks if b.cluster == "game"),
        "n_banks_cross_domain": sum(1 for b in banks if b.cluster == "cross_domain"),
        "warnings_suppressed": n_warnings_suppressed,
        "cluster_vocabulary_sizes": cluster_sizes,
        "game_vs_cross_jaccard": game_vs_cross,
        "per_bank_vocab_sizes": [
            {
                "label": cv.bank_info.label,
                "corpus_subdir": cv.bank_info.corpus_subdir,
                "cluster": cv.bank_info.cluster,
                "n_skills": cv.n_skills,
                "protocol_ops": len(cv.protocol_ops),
                "slot_types": len(cv.slot_types),
                "hop_predicates": len(cv.hop_predicates),
                "contract_predicates": len(cv.contract_predicates),
            }
            for cv in corpus_vocabs
        ],
        "pairwise_within_cross_domain": pairwise,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_md(
    out_path: Path,
    *,
    args: argparse.Namespace,
    banks: list[BankInfo],
    cluster_sizes: dict[str, dict[str, int]],
    game_vs_cross: dict[str, dict],
    n_warnings_suppressed: int,
) -> None:
    n_game = sum(1 for b in banks if b.cluster == "game")
    n_cross = sum(1 for b in banks if b.cluster == "cross_domain")
    lines: list[str] = []
    lines.append(f"# Vocab Jaccard audit — `{args.run_id}`")
    lines.append("")
    lines.append(f"- Generated: {_utc_iso()}")
    lines.append(f"- Banks root: {', '.join(str(p) for p in args.banks_root)}")
    lines.append(
        f"- Banks discovered: **{len(banks)}** total "
        f"({n_game} game, {n_cross} cross-domain); "
        f"{n_warnings_suppressed} discovery warnings suppressed."
    )
    lines.append("")
    lines.append("## Cluster vocabulary sizes")
    lines.append("")
    lines.append("| layer | game | cross-domain |")
    lines.append("|---|---:|---:|")
    for layer in _LAYERS:
        lines.append(
            f"| `{layer}` | {cluster_sizes['game'].get(layer, 0)} "
            f"| {cluster_sizes['cross_domain'].get(layer, 0)} |"
        )
    lines.append("")
    lines.append("## Game vs cross-domain Jaccard")
    lines.append("")
    if n_game == 0 or n_cross == 0:
        absent = "game" if n_game == 0 else "cross-domain"
        lines.append(f"_({absent} banks absent — comparison skipped)_")
        lines.append("")
    else:
        lines.append("| layer | jaccard | ∩ | ∪ | reference | flag |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for layer in _LAYERS:
            rec = game_vs_cross[layer]
            ref = _REFERENCE_JACCARDS.get(layer)
            if ref is None or rec["jaccard"] is None:
                ref_str, flag = "—", ""
            else:
                ref_str = f"{ref:.2f}"
                delta = abs(rec["jaccard"] - ref)
                flag = f"⚠ Δ={delta:.2f}" if delta > _REFERENCE_TOLERANCE else "ok"
            lines.append(
                f"| `{layer}` | {_format_jaccard(rec['jaccard'])} "
                f"| {rec['intersect']} | {rec['union']} | {ref_str} | {flag} |"
            )
        lines.append("")
    lines.append("## Headline interpretation")
    lines.append("")
    proto = game_vs_cross.get("protocol_ops", {}).get("jaccard")
    slot = game_vs_cross.get("slot_types", {}).get("jaccard")
    pred = game_vs_cross.get("predicates_combined", {}).get("jaccard")
    lines.append(
        "Per rollout-memo §11.5.1, the protocol-op and slot-type vocabularies "
        "are expected to be near-universal across the game and cross-domain "
        "clusters, while the predicate vocabularies diverge at the surface and "
        "must be bridged operationally via per-domain schema producers. "
        f"This run measured protocol_ops={_format_jaccard(proto)}, "
        f"slot_types={_format_jaccard(slot)}, "
        f"predicates_combined={_format_jaccard(pred)} — "
        "high protocol/slot agreement confirms the shared protocol layer is "
        "transferable; low predicate agreement is the expected surface-level "
        "disjointness that Stages 1–6 close via target-side schema producers."
    )
    lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Vocabulary-Jaccard audit (game vs cross-domain).",
    )
    p.add_argument("--banks-root", nargs="+", type=Path, default=list(_DEFAULT_BANKS_ROOT))
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--run-id",
        type=str,
        default=f"phase0_{_utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        banks = discover_banks(args.banks_root)
    n_warnings_suppressed = len(caught)

    if not banks:
        print(
            f"vocab_jaccard: ERROR no banks discovered under {args.banks_root}; "
            "nothing to audit",
        )
        return 1

    if args.verbose:
        print(f"vocab_jaccard: discovered {len(banks)} banks "
              f"({n_warnings_suppressed} warnings suppressed)")
        for bi in banks:
            print(f"  [{bi.cluster:12s}] {bi.label} ({bi.corpus_subdir})")

    corpus_vocabs: list[CorpusVocab] = []
    for bi in banks:
        cv = collect_corpus_vocab(bi)
        corpus_vocabs.append(cv)
        if args.verbose:
            print(f"  {bi.label:30s} n_skills={cv.n_skills:4d} "
                  f"|ops|={len(cv.protocol_ops)} |slots|={len(cv.slot_types)} "
                  f"|hop_pred|={len(cv.hop_predicates)} "
                  f"|contract_pred|={len(cv.contract_predicates)}")

    cluster_vocabs: dict[str, dict[str, frozenset[str]]] = {}
    cluster_sizes: dict[str, dict[str, int]] = {}
    for cluster in ("game", "cross_domain"):
        members = [cv for cv in corpus_vocabs if cv.bank_info.cluster == cluster]
        cluster_vocabs[cluster] = _union_layers(members)
        cluster_sizes[cluster] = {
            layer: len(cluster_vocabs[cluster][layer]) for layer in _LAYERS
        }

    game_vs_cross: dict[str, dict] = {}
    have_both = bool(cluster_vocabs["game"]["protocol_ops"] or cluster_sizes["game"]["protocol_ops"]) and \
                any(cv.bank_info.cluster == "game" for cv in corpus_vocabs) and \
                any(cv.bank_info.cluster == "cross_domain" for cv in corpus_vocabs)
    for layer in _LAYERS:
        if not have_both:
            game_vs_cross[layer] = {
                "jaccard": None, "intersect": 0, "union": 0,
                "game_only": [], "cross_only": [],
            }
            continue
        rec = _jaccard_record(cluster_vocabs["game"][layer],
                              cluster_vocabs["cross_domain"][layer])
        game_vs_cross[layer] = {
            "jaccard": rec["jaccard"],
            "intersect": rec["intersect"],
            "union": rec["union"],
            "game_only": rec["a_only"],
            "cross_only": rec["b_only"],
        }

    cross = [cv for cv in corpus_vocabs if cv.bank_info.cluster == "cross_domain"]
    pairwise: dict[str, dict[str, dict[str, float | None]]] = {layer: {} for layer in _LAYERS}
    for layer in _LAYERS:
        for a in cross:
            row: dict[str, float | None] = {}
            a_layers = _layer_sets(a)
            for b in cross:
                rec = _jaccard_record(a_layers[layer], _layer_sets(b)[layer])
                row[b.bank_info.label] = rec["jaccard"]
            pairwise[layer][a.bank_info.label] = row

    json_path = out_dir / "vocab_jaccard.json"
    md_path = out_dir / "vocab_jaccard.md"
    _write_json(
        json_path,
        args=args,
        banks=banks,
        corpus_vocabs=corpus_vocabs,
        cluster_vocabs=cluster_vocabs,
        cluster_sizes=cluster_sizes,
        game_vs_cross=game_vs_cross,
        pairwise=pairwise,
        n_warnings_suppressed=n_warnings_suppressed,
    )
    _write_md(
        md_path,
        args=args,
        banks=banks,
        cluster_sizes=cluster_sizes,
        game_vs_cross=game_vs_cross,
        n_warnings_suppressed=n_warnings_suppressed,
    )

    proto = game_vs_cross["protocol_ops"]["jaccard"]
    slot = game_vs_cross["slot_types"]["jaccard"]
    pred = game_vs_cross["predicates_combined"]["jaccard"]
    print(
        f"vocab_jaccard: {json_path},{md_path} "
        f"| game-vs-cross protocol_ops={_format_jaccard(proto)} "
        f"slot_types={_format_jaccard(slot)} "
        f"predicates_combined={_format_jaccard(pred)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
