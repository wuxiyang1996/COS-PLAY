"""
Structured logging for both trainers: Decision Agent (GRPO) and SkillBank (Hard-EM).

Logs metrics to console, JSON files, and optionally W&B.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.common.metrics import DecisionMetrics, SkillBankMetrics

logger = logging.getLogger("trainer")


class TrainLogger:
    """Unified logger for training metrics, checkpoints, and diff reports.

    Writes JSON-lines log files and optionally forwards to W&B.
    """

    def __init__(
        self,
        log_dir: str = "runs/trainer",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._decision_log = self.log_dir / "decision_metrics.jsonl"
        self._skillbank_log = self.log_dir / "skillbank_metrics.jsonl"
        self._event_log = self.log_dir / "events.jsonl"

        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project or "game-ai-trainer",
                           name=wandb_run_name, reinit=True)
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; falling back to file logging")

        self._step = 0

    def log_decision_metrics(
        self,
        metrics: DecisionMetrics,
        episode: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log decision agent training metrics."""
        d = metrics.to_dict()
        d["episode"] = episode
        d["timestamp"] = time.time()
        if extra:
            d.update(extra)

        self._append_jsonl(self._decision_log, d)
        logger.info(
            "Decision ep=%d  win=%.2f  r_env=%.3f  r_total=%.3f  qskill=%.3f  qmem=%.3f",
            episode, metrics.win_rate, metrics.mean_r_env,
            metrics.mean_reward, metrics.query_skill_rate, metrics.query_mem_rate,
        )

        if self._wandb:
            self._wandb.log({f"decision/{k}": v for k, v in d.items()}, step=episode)

    def log_skillbank_metrics(
        self,
        metrics: SkillBankMetrics,
        version: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log skillbank EM iteration metrics."""
        d = metrics.to_dict()
        d["version"] = version
        d["timestamp"] = time.time()
        if extra:
            d.update(extra)

        self._append_jsonl(self._skillbank_log, d)
        logger.info(
            "SkillBank v=%d  skills=%d  new=%d  pass=%.2f  margin=%.3f",
            version, metrics.n_skills, metrics.new_pool_size,
            metrics.mean_pass_rate, metrics.mean_margin,
        )

        if self._wandb:
            self._wandb.log({f"skillbank/{k}": v for k, v in d.items()}, step=version)

    def log_bank_diff(self, version: int, diff_report: Dict[str, Any]) -> None:
        """Log a bank version diff report."""
        diff_path = self.log_dir / f"bank_diff_v{version}.json"
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(diff_report, f, indent=2, default=str)
        self._append_jsonl(self._event_log, {
            "event": "bank_diff",
            "version": version,
            "timestamp": time.time(),
            "path": str(diff_path),
        })

    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a generic training event."""
        entry = {"event": event_type, "timestamp": time.time()}
        if data:
            entry.update(data)
        self._append_jsonl(self._event_log, entry)

    def log_eval(
        self,
        metrics: DecisionMetrics,
        episode: int,
        bank_version: int,
        seeds_used: Optional[List[int]] = None,
    ) -> None:
        """Log fixed-seed evaluation results."""
        d = metrics.to_dict()
        d["episode"] = episode
        d["bank_version"] = bank_version
        d["eval_seeds"] = seeds_used
        d["timestamp"] = time.time()

        eval_log = self.log_dir / "eval_results.jsonl"
        self._append_jsonl(eval_log, d)
        logger.info(
            "EVAL ep=%d bank_v=%d  win=%.2f  r_env=%.3f",
            episode, bank_version, metrics.win_rate, metrics.mean_r_env,
        )

        if self._wandb:
            self._wandb.log({f"eval/{k}": v for k, v in d.items()}, step=episode)

    @staticmethod
    def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def close(self) -> None:
        if self._wandb:
            self._wandb.finish()
