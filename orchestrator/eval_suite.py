"""orchestrator/eval_suite.py — frozen evaluation-suite loader for Stage-4.

Spec: ``PLAN-UNIFIED-SKILL-GATE.md`` §4 + ``§5 — Stage-4 Non-regression``.

The Stage-4 (non-regression) gate must compare a *proposed* bank
snapshot against the *last release* on a **named, frozen suite** so
re-evaluations are reproducible. The runtime payload is already
defined in :mod:`harness.gate_runner` (:class:`EvalSuite` —
``suite_id``, ``pre_score``, ``post_score``, ``metrics``).

What was missing — and what this module ships — is the on-disk
*loader* that:

1. Discovers suite *specs* (their dataset slice, metric keys,
   description, version) from a registry directory.
2. Loads per-snapshot *scoreboards* (one file per
   ``bank_snapshot_id``).
3. Builds a frozen :class:`EvalSuite` from a (``pre_snapshot_id``,
   ``post_snapshot_id``) pair so the gate can run G5 without the
   caller having to wire ``(baseline_score, post_score)`` by hand.

On-disk layout (default registry root: ``<repo>/evaluation/suites``):

    evaluation/suites/<suite_id>/
        suite.yaml           # frozen spec (datasets, metric keys, version)
        scoreboards/
            <bank_snapshot_id>.json   # produced by T2.3 evaluation/driver.py

Each scoreboard JSON has the shape::

    {
        "bank_snapshot_id": "snap-…",
        "suite_id": "gymv-smoke-v1",
        "score": 0.78,
        "metrics": {"gymv_holdout.pass_rate": 0.78, …},
        "evaluated_at_utc": "2026-05-02T…"
    }

The loader is pure I/O — no harness, no torch, no network — so it
imports cheaply from :class:`orchestrator.GateService` and the
upcoming :mod:`evaluation.driver`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EvalSuite",
    "EvalSuiteSpec",
    "Scoreboard",
    "EvalSuiteLoader",
    "default_suites_root",
    "load_eval_suite_spec",
    "load_scoreboard",
    "load_eval_suite",
]


# ---------------------------------------------------------------------------
# Runtime payload (canonical home — re-exported by harness.gate_runner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSuite:
    """Frozen evaluation-suite reference for Stage-4 non-regression.

    PLAN-UNIFIED-SKILL-GATE §7 Stage-4 expects the gate to compare the
    proposed bank against the last release on a *named, frozen* suite
    so re-evaluations are reproducible. ``suite_id`` is the canonical
    name (e.g. ``"gymv-smoke-v1"``); ``pre_score`` / ``post_score`` are
    the per-suite aggregate metrics; ``metrics`` may carry per-task
    breakdowns.

    This dataclass is the canonical runtime payload; it lives in
    :mod:`orchestrator.eval_suite` so the loader can produce it
    without a circular import. :mod:`harness.gate_runner` re-exports
    the same symbol for legacy callers.
    """

    suite_id: str
    pre_score: float
    post_score: float
    metrics: Mapping[str, float] = field(default_factory=dict)

    def delta(self) -> float:
        return float(self.post_score) - float(self.pre_score)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_suites_root() -> str:
    """Return the default suites registry root.

    Resolution order:
      1. ``$EVAL_SUITES_ROOT`` (environment override).
      2. ``<repo>/evaluation/suites`` (repo-relative).
    """
    env_root = os.environ.get("EVAL_SUITES_ROOT")
    if env_root:
        return env_root
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "evaluation", "suites"))


# ---------------------------------------------------------------------------
# Frozen on-disk dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSuiteSpec:
    """Frozen description of an evaluation suite (the ``suite.yaml``).

    Two suites with the same ``suite_id`` but different ``version``
    strings are considered *different* suites — Stage-4 reproducibility
    relies on the spec being immutable once published.
    """

    suite_id: str
    version: str
    datasets: Tuple[Mapping[str, Any], ...] = ()
    metric_keys: Tuple[str, ...] = ()
    description: str = ""

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "EvalSuiteSpec":
        try:
            suite_id = str(d["suite_id"])
            version = str(d["version"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"EvalSuiteSpec missing required field {exc.args[0]!r}"
            ) from exc
        raw_datasets = d.get("datasets", []) or []
        if not isinstance(raw_datasets, (list, tuple)):
            raise ValueError("EvalSuiteSpec.datasets must be a list of mappings")
        datasets = tuple(
            dict(item) for item in raw_datasets if isinstance(item, Mapping)
        )
        metric_keys = tuple(str(k) for k in d.get("metric_keys", ()) or ())
        description = str(d.get("description", ""))
        return EvalSuiteSpec(
            suite_id=suite_id,
            version=version,
            datasets=datasets,
            metric_keys=metric_keys,
            description=description,
        )


@dataclass(frozen=True)
class Scoreboard:
    """Single per-snapshot scoreboard for a frozen suite.

    Produced by :mod:`evaluation.driver` (T2.3); consumed here only.
    """

    bank_snapshot_id: str
    suite_id: str
    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    evaluated_at_utc: str = ""

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Scoreboard":
        try:
            bank_snapshot_id = str(d["bank_snapshot_id"])
            suite_id = str(d["suite_id"])
            score = float(d["score"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Scoreboard missing required field {exc.args[0]!r}"
            ) from exc
        raw_metrics = d.get("metrics", {}) or {}
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("Scoreboard.metrics must be a mapping")
        metrics = {str(k): float(v) for k, v in raw_metrics.items()}
        evaluated_at_utc = str(d.get("evaluated_at_utc", ""))
        return Scoreboard(
            bank_snapshot_id=bank_snapshot_id,
            suite_id=suite_id,
            score=score,
            metrics=metrics,
            evaluated_at_utc=evaluated_at_utc,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class EvalSuiteLoader:
    """Disk-backed registry of frozen evaluation suites + scoreboards."""

    def __init__(self, suites_root: Optional[str] = None) -> None:
        self._root = suites_root or default_suites_root()

    # -- introspection -----------------------------------------------------

    @property
    def suites_root(self) -> str:
        return self._root

    def list_suites(self) -> List[str]:
        if not os.path.isdir(self._root):
            return []
        out: List[str] = []
        for name in sorted(os.listdir(self._root)):
            spec_dir = os.path.join(self._root, name)
            if os.path.isdir(spec_dir) and self._spec_path(name) is not None:
                out.append(name)
        return out

    def list_scoreboards(self, suite_id: str) -> List[str]:
        sb_dir = os.path.join(self._root, suite_id, "scoreboards")
        if not os.path.isdir(sb_dir):
            return []
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(sb_dir)
            if name.endswith(".json")
        )

    # -- spec --------------------------------------------------------------

    def load_spec(self, suite_id: str) -> EvalSuiteSpec:
        path = self._spec_path(suite_id)
        if path is None:
            raise FileNotFoundError(
                f"EvalSuite spec for {suite_id!r} not found under {self._root!r}"
            )
        data = self._read_spec_file(path)
        spec = EvalSuiteSpec.from_dict(data)
        if spec.suite_id != suite_id:
            raise ValueError(
                f"suite.yaml suite_id={spec.suite_id!r} does not match "
                f"directory name {suite_id!r}"
            )
        return spec

    # -- scoreboards -------------------------------------------------------

    def load_scoreboard(self, suite_id: str, snapshot_id: str) -> Scoreboard:
        sb_path = os.path.join(
            self._root, suite_id, "scoreboards", f"{snapshot_id}.json"
        )
        if not os.path.isfile(sb_path):
            raise FileNotFoundError(
                f"Scoreboard for suite={suite_id!r} snapshot={snapshot_id!r} "
                f"not found at {sb_path!r}"
            )
        with open(sb_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        sb = Scoreboard.from_dict(data)
        if sb.suite_id != suite_id:
            raise ValueError(
                f"scoreboard.suite_id={sb.suite_id!r} does not match "
                f"requested suite_id={suite_id!r} (file {sb_path!r})"
            )
        if sb.bank_snapshot_id != snapshot_id:
            raise ValueError(
                f"scoreboard.bank_snapshot_id={sb.bank_snapshot_id!r} does "
                f"not match requested snapshot={snapshot_id!r} "
                f"(file {sb_path!r})"
            )
        return sb

    # -- compose runtime payload ------------------------------------------

    def build_eval_suite(
        self,
        suite_id: str,
        *,
        pre_snapshot_id: str,
        post_snapshot_id: str,
    ) -> EvalSuite:
        """Assemble a frozen :class:`EvalSuite` from disk scoreboards.

        ``metrics`` on the returned :class:`EvalSuite` is the *delta*
        from pre to post for every metric key the spec declared,
        plus the pre/post scalars under the ``"pre."`` / ``"post."``
        prefixes for downstream visibility.
        """

        spec = self.load_spec(suite_id)
        pre_sb = self.load_scoreboard(suite_id, pre_snapshot_id)
        post_sb = self.load_scoreboard(suite_id, post_snapshot_id)

        merged: dict[str, float] = {}
        keys: Sequence[str]
        if spec.metric_keys:
            keys = spec.metric_keys
        else:
            keys = sorted(set(pre_sb.metrics) | set(post_sb.metrics))
        for k in keys:
            pre_v = float(pre_sb.metrics.get(k, 0.0))
            post_v = float(post_sb.metrics.get(k, 0.0))
            merged[f"pre.{k}"] = pre_v
            merged[f"post.{k}"] = post_v
            merged[f"delta.{k}"] = post_v - pre_v

        return EvalSuite(
            suite_id=suite_id,
            pre_score=float(pre_sb.score),
            post_score=float(post_sb.score),
            metrics=merged,
        )

    # -- internals ---------------------------------------------------------

    def _spec_path(self, suite_id: str) -> Optional[str]:
        for ext in ("yaml", "yml", "json"):
            candidate = os.path.join(self._root, suite_id, f"suite.{ext}")
            if os.path.isfile(candidate):
                return candidate
        return None

    def _read_spec_file(self, path: str) -> Mapping[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith(".json"):
            return json.loads(text)
        # YAML — soft dependency. Fall back to a tiny parser for the
        # subset we actually emit (flat keys + nested lists/maps) so
        # tests don't require PyYAML.
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            if not isinstance(data, Mapping):
                raise ValueError(f"suite.yaml at {path!r} did not parse as a mapping")
            return data
        except ImportError:
            return _parse_minimal_yaml(text, source=path)


# ---------------------------------------------------------------------------
# Module-level conveniences
# ---------------------------------------------------------------------------


def load_eval_suite_spec(
    suite_id: str, *, suites_root: Optional[str] = None
) -> EvalSuiteSpec:
    return EvalSuiteLoader(suites_root).load_spec(suite_id)


def load_scoreboard(
    suite_id: str, snapshot_id: str, *, suites_root: Optional[str] = None
) -> Scoreboard:
    return EvalSuiteLoader(suites_root).load_scoreboard(suite_id, snapshot_id)


def load_eval_suite(
    suite_id: str,
    *,
    pre_snapshot_id: str,
    post_snapshot_id: str,
    suites_root: Optional[str] = None,
) -> EvalSuite:
    return EvalSuiteLoader(suites_root).build_eval_suite(
        suite_id,
        pre_snapshot_id=pre_snapshot_id,
        post_snapshot_id=post_snapshot_id,
    )


# ---------------------------------------------------------------------------
# Minimal YAML fallback
# ---------------------------------------------------------------------------


def _parse_minimal_yaml(text: str, *, source: str) -> Mapping[str, Any]:
    """Parse a tiny YAML subset sufficient for the suite.yaml schema.

    Supported:
      - top-level ``key: value`` scalars (str / int / float / bool / null)
      - one level of ``key:`` followed by ``- value`` list items
      - one level of ``- key: value`` mapping list items
      - blank lines + ``#`` comments

    Anything more complex must use real PyYAML.
    """

    out: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            i += 1
            continue
        if stripped.startswith(" "):
            raise ValueError(
                f"_parse_minimal_yaml({source!r}): unexpected indent at line {i + 1}: "
                f"{raw!r}"
            )
        if ":" not in stripped:
            raise ValueError(
                f"_parse_minimal_yaml({source!r}): expected 'key:' at line {i + 1}: "
                f"{raw!r}"
            )
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()
        if val:
            out[key] = _parse_scalar(val)
            i += 1
            continue
        # Block value follows.
        items: List[Any] = []
        i += 1
        while i < len(lines):
            sub = lines[i]
            sub_stripped = sub.split("#", 1)[0].rstrip()
            if not sub_stripped.strip():
                i += 1
                continue
            if not sub_stripped.startswith((" ", "\t")):
                break
            content = sub_stripped.lstrip()
            if not content.startswith("- "):
                raise ValueError(
                    f"_parse_minimal_yaml({source!r}): expected list item at "
                    f"line {i + 1}: {sub!r}"
                )
            item_text = content[2:].strip()
            if ":" in item_text and not item_text.startswith('"'):
                # Inline mapping item: collect this line + further indented siblings.
                mapping: dict[str, Any] = {}
                first_key, _, first_val = item_text.partition(":")
                mapping[first_key.strip()] = _parse_scalar(first_val.strip())
                i += 1
                base_indent = len(sub_stripped) - len(content)
                while i < len(lines):
                    nxt = lines[i]
                    nxt_stripped = nxt.split("#", 1)[0].rstrip()
                    if not nxt_stripped.strip():
                        i += 1
                        continue
                    if not nxt_stripped.startswith((" ", "\t")):
                        break
                    indent = len(nxt_stripped) - len(nxt_stripped.lstrip())
                    if indent <= base_indent:
                        break
                    nxt_content = nxt_stripped.lstrip()
                    if nxt_content.startswith("- "):
                        break
                    sub_key, _, sub_val = nxt_content.partition(":")
                    if not sub_val:
                        raise ValueError(
                            f"_parse_minimal_yaml({source!r}): nested block "
                            f"values not supported at line {i + 1}"
                        )
                    mapping[sub_key.strip()] = _parse_scalar(sub_val.strip())
                    i += 1
                items.append(mapping)
            else:
                items.append(_parse_scalar(item_text))
                i += 1
        out[key] = items
    return out


def _parse_scalar(val: str) -> Any:
    if val == "" or val.lower() in {"null", "~"}:
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val
