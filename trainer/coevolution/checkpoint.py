"""Checkpointing and resume for the co-evolution loop.

Saves a full snapshot every ``checkpoint_interval`` steps:
  - Skill bank (``skill_bank.jsonl``)
  - All 5 LoRA adapter weights
  - Step metadata (step number, bank version, metrics)

Supports resuming from any checkpoint.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DECISION_ADAPTERS = ["skill_selection", "action_taking"]
SKILLBANK_ADAPTERS = ["segment", "contract", "curator"]
ADAPTER_NAMES = DECISION_ADAPTERS + SKILLBANK_ADAPTERS

ADAPTER_SUBDIR = {name: "decision" for name in DECISION_ADAPTERS}
ADAPTER_SUBDIR.update({name: "skillbank" for name in SKILLBANK_ADAPTERS})


def _save_one_bank(agent: Any, path: Path, label: str = "") -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        bank = getattr(agent, "bank", agent)
        if hasattr(bank, "save"):
            bank.save(str(path))
            logger.info("Saved bank%s to %s", f" [{label}]" if label else "", path)
    except Exception as exc:
        logger.warning("Bank save failed%s: %s", f" [{label}]" if label else "", exc)


def _load_one_bank(agent: Any, path: Path, label: str = "") -> None:
    try:
        bank = getattr(agent, "bank", agent)
        if hasattr(bank, "load"):
            bank.load(str(path))
            logger.info("Restored bank%s from %s", f" [{label}]" if label else "", path)
    except Exception as exc:
        logger.warning("Bank restore failed%s: %s", f" [{label}]" if label else "", exc)


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    *,
    bank_agent: Any = None,
    bank_agents: Optional[Dict[str, Any]] = None,
    adapter_dir: str = "runs/lora_adapters",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a co-evolution checkpoint.

    Creates ``{checkpoint_dir}/step_{step:04d}/`` with:
      - Per-game skill banks in ``banks/{game}/skill_bank.jsonl``
        (or legacy ``skill_bank.jsonl`` for a single agent)
      - ``adapters/{name}/`` — LoRA adapter weights for each adapter
      - ``metadata.json`` — step info, bank version, metrics
    """
    ckpt_path = Path(checkpoint_dir) / f"step_{step:04d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    if bank_agents:
        banks_dir = ckpt_path / "banks"
        banks_dir.mkdir(exist_ok=True)
        for game, agent in bank_agents.items():
            if agent is None:
                continue
            game_dir = banks_dir / game
            game_dir.mkdir(parents=True, exist_ok=True)
            _save_one_bank(agent, game_dir / "skill_bank.jsonl", label=game)
    elif bank_agent is not None:
        _save_one_bank(bank_agent, ckpt_path / "skill_bank.jsonl")

    # Copy LoRA adapters (preserving decision/ and skillbank/ subdirs)
    adapters_dir = ckpt_path / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    src_dir = Path(adapter_dir)
    for name in ADAPTER_NAMES:
        sub = ADAPTER_SUBDIR[name]
        src = src_dir / sub / name
        if src.exists() and src.is_dir():
            dst = adapters_dir / sub / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except Exception as exc:
                logger.warning("Adapter copy failed for '%s': %s", name, exc)

    # Save metadata
    meta = {
        "step": step,
        "timestamp": time.time(),
        "adapter_names": ADAPTER_NAMES,
    }
    if metadata:
        meta.update(metadata)
    meta_path = ckpt_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Checkpoint saved: step %d → %s", step, ckpt_path)
    return ckpt_path


def load_checkpoint(
    checkpoint_dir: str,
    step: int,
    *,
    adapter_dir: str = "runs/lora_adapters",
    bank_agent: Any = None,
    bank_agents: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a co-evolution checkpoint.

    Restores adapter weights to ``adapter_dir`` and optionally reloads
    the skill bank(s).

    Returns the metadata dict.
    """
    ckpt_path = Path(checkpoint_dir) / f"step_{step:04d}"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    meta_path = ckpt_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    adapters_dir = ckpt_path / "adapters"
    dst_dir = Path(adapter_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_NAMES:
        sub = ADAPTER_SUBDIR[name]
        src = adapters_dir / sub / name
        if not src.exists():
            src = adapters_dir / name  # backward compat: flat layout
        if src.exists() and src.is_dir():
            dst = dst_dir / sub / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                logger.info("Restored adapter '%s' from checkpoint", name)
            except Exception as exc:
                logger.warning("Adapter restore failed for '%s': %s", name, exc)

    banks_dir = ckpt_path / "banks"
    if banks_dir.exists() and bank_agents:
        for game, agent in bank_agents.items():
            if agent is None:
                continue
            bp = banks_dir / game / "skill_bank.jsonl"
            if bp.exists():
                _load_one_bank(agent, bp, label=game)
    else:
        bank_path = ckpt_path / "skill_bank.jsonl"
        if bank_path.exists() and bank_agent is not None:
            _load_one_bank(bank_agent, bank_path)

    logger.info("Checkpoint loaded: step %d from %s", step, ckpt_path)
    return metadata


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """Find the latest checkpoint step number, or None if no checkpoints exist."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                meta = d / "metadata.json"
                if meta.exists():
                    steps.append(step)
            except (ValueError, IndexError):
                pass

    return max(steps) if steps else None


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
) -> List[int]:
    """Remove old checkpoints, keeping the most recent ``keep_last``."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                steps.append(step)
            except (ValueError, IndexError):
                pass

    if len(steps) <= keep_last:
        return []

    steps.sort()
    to_remove = steps[:-keep_last]
    removed = []
    for step in to_remove:
        p = ckpt_dir / f"step_{step:04d}"
        try:
            shutil.rmtree(p)
            removed.append(step)
        except Exception as exc:
            logger.warning("Failed to remove checkpoint step_%04d: %s", step, exc)

    return removed
