"""JSONL writer for SFT cold-start data, matching trainer/SFT loader.

The output rows follow the contract :func:`trainer.SFT.data_loader.\
load_decision_adapter_data` and :func:`_normalise_example` consume:

* required: ``prompt`` (full text the actor sees), ``completion`` (the
  raw LLM reply — ``SUBGOAL: …\\nREASONING: …\\nACTION: <number>``);
* used by ``_align_action_taking_to_coevolution``: ``intention``
  (the ``[TAG] phrase`` from ``infer_intention``), ``active_skill``
  (the active skill id, when one is selected).

We write into the directory layout ``trainer.SFT.config.SFTConfig`` already
points at::

    <out_dir>/<game>/skill_selection.jsonl
    <out_dir>/<game>/action_taking.jsonl

so the cold-start trainer ingests the artefacts without any conversion.

Multimodal extras
-----------------
Each row also carries an optional ``image`` block::

    "image": {
        "path":       "rollouts/.../step_0007.png",
        "mime_type":  "image/png",
        "caption":    "browser viewport @ 1280x720",
        "width":      1280,
        "height":     720,
    }

The existing text-only ``data_loader.py`` ignores unknown keys, so the
records remain forward-compatible with the Stage B (tool-trajectory) /
Qwen3-VL distillation pipeline without changing the cold-start trainer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from decision_agents.core.multimodal import VisualInput

_LOGGER = logging.getLogger(__name__)

DEFAULT_SFT_OUTPUT_DIR = Path("labeling/output/gpt54_skill_labeled/grpo_coldstart")
"""Mirrors :data:`trainer.SFT.config.DECISION_DATA_DIR` so the trainer
picks the artefacts up out of the box."""

ADAPTER_SKILL_SELECTION = "skill_selection"
ADAPTER_ACTION_TAKING = "action_taking"
DECISION_ADAPTERS = (ADAPTER_SKILL_SELECTION, ADAPTER_ACTION_TAKING)


# ──────────────────────────────────────────────────────────────────────
# Record dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SFTRecord:
    """One JSONL row.

    Mirrors the fields :func:`trainer.SFT.data_loader._normalise_example`,
    :func:`_align_action_taking_to_coevolution`, and
    :func:`_inject_context_lines` look at, plus a multimodal ``image``
    block that the loader silently passes through.
    """

    prompt: str
    completion: str
    adapter: str = ADAPTER_ACTION_TAKING
    game: str = "unknown"
    intention: str = ""
    active_skill: str = ""
    image: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Drop empty / None fields so the JSONL stays compact and the
        loader's ``row.get(...)`` lookups behave as if the field were
        missing rather than empty."""
        d: Dict[str, Any] = {"prompt": self.prompt, "completion": self.completion}
        if self.intention:
            d["intention"] = self.intention
        if self.active_skill:
            d["active_skill"] = self.active_skill
        if self.image:
            d["image"] = self.image
        for k, v in self.extras.items():
            if v is not None and k not in d:
                d[k] = v
        return d


# ──────────────────────────────────────────────────────────────────────
# Recorder
# ──────────────────────────────────────────────────────────────────────


class SFTRecorder:
    """Append-only JSONL writer with per-(game, adapter) file fanout.

    Thread-safe; the GPT-4o actor calls :meth:`record` once per outer
    step (action-taking) and once per skill reselect (skill-selection).
    Files are opened lazily on first write so empty games don't
    pollute the output tree.

    Parameters
    ----------
    output_dir
        Root directory.  Defaults to :data:`DEFAULT_SFT_OUTPUT_DIR` so
        :class:`trainer.SFT.config.SFTConfig` finds the artefacts
        without any path overrides.
    flush_each
        If True (default), flush after every write.  Cheap for the
        rates we expect (≤ a few writes/sec) and protects against
        partial-file corruption when a long-running rollout crashes.
    """

    def __init__(
        self,
        output_dir: Optional[os.PathLike] = None,
        *,
        flush_each: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_SFT_OUTPUT_DIR
        self.flush_each = flush_each
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────

    def record(self, record: SFTRecord) -> Path:
        """Append *record* to ``<output_dir>/<game>/<adapter>.jsonl``.

        Returns the path written to so callers can log it on the
        :class:`Experience` extras for traceability.
        """
        if record.adapter not in DECISION_ADAPTERS:
            raise ValueError(
                f"Unknown SFT adapter {record.adapter!r}; expected one "
                f"of {DECISION_ADAPTERS}"
            )
        if not record.prompt or not record.completion:
            # ``data_loader._normalise_example`` skips rows where either
            # is empty — we drop them here to keep stats accurate.
            return Path(os.devnull)

        target = self.output_dir / record.game / f"{record.adapter}.jsonl"
        line = json.dumps(record.to_jsonl_dict(), ensure_ascii=False, default=str)
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                if self.flush_each:
                    f.flush()
            key = f"{record.game}/{record.adapter}"
            self._counts[key] = self._counts.get(key, 0) + 1
        return target

    def record_action_taking(
        self,
        *,
        prompt: str,
        completion: str,
        game: str,
        intention: str = "",
        active_skill: str = "",
        image: Optional[VisualInput] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Convenience wrapper for the per-step action-taking record."""
        return self.record(
            SFTRecord(
                prompt=prompt,
                completion=completion,
                adapter=ADAPTER_ACTION_TAKING,
                game=game,
                intention=intention,
                active_skill=active_skill,
                image=_image_block(image),
                extras=extras or {},
            )
        )

    def record_skill_selection(
        self,
        *,
        prompt: str,
        completion: str,
        game: str,
        active_skill: str = "",
        image: Optional[VisualInput] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Convenience wrapper for the per-reselect skill-selection record.

        The skill-selection adapter doesn't get an ``intention`` field
        in :func:`trainer.SFT.data_loader._inject_context_lines`, so we
        omit it here to keep the schema clean.
        """
        return self.record(
            SFTRecord(
                prompt=prompt,
                completion=completion,
                adapter=ADAPTER_SKILL_SELECTION,
                game=game,
                active_skill=active_skill,
                image=_image_block(image),
                extras=extras or {},
            )
        )

    def stats(self) -> Dict[str, int]:
        """Return ``{"<game>/<adapter>": count}`` for the current run."""
        with self._lock:
            return dict(self._counts)

    def write_manifest(self) -> Path:
        """Dump :meth:`stats` to ``<output_dir>/_manifest.json``.

        Useful for the smoke-check that asserts the GPT-4o run actually
        produced labels for every requested game before we kick off
        SFT training.
        """
        manifest_path = self.output_dir / "_manifest.json"
        with self._lock:
            payload = {
                "output_dir": str(self.output_dir),
                "counts": dict(self._counts),
                "total_records": sum(self._counts.values()),
            }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _image_block(image: Optional[VisualInput]) -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    block = image.to_dict()
    return {k: v for k, v in block.items() if v is not None}


def iter_records(jsonl_path: os.PathLike) -> Iterable[Dict[str, Any]]:
    """Helper for tests / inspection scripts.

    Mirrors :func:`trainer.SFT.data_loader._read_jsonl` so callers can
    cross-check the recorder's output against what the trainer will
    eventually consume.
    """
    p = Path(jsonl_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                _LOGGER.warning("Skipping malformed JSONL row in %s: %s", p, exc)
