"""vLLM server lifecycle manager.

Manages N independent vLLM instances (one per GPU, TP=1) that run
persistently throughout training.  After each GRPO step, LoRA adapters
are hot-reloaded via the vLLM REST API — no restart required.

Typical lifecycle::

    manager.start()              # Launch 4 × TP=1 instances (once)
    await manager.wait_healthy() # Wait for all to respond
    for step in range(total_steps):
        # ... Phase A+B: rollouts + skill bank ...
        # ... Phase C: FSDP GRPO training on other GPUs ...
        await manager.reload_adapters()  # hot-reload updated weights
    manager.stop()               # Cleanup at end

Memory budget per instance (Qwen/Qwen3.5-9B, H200-141GB, TP=1, text-only):
  - Model weights (bf16):     ~18 GB   (vision tower skipped via --language-model-only)
  - MTP head (bf16):          ~0.5 GB  (built-in multi-token-prediction)
  - KV cache (gpu_util=0.95): ~115 GB  →  ample margin even at long contexts
  - Total:                    ~134 GB / 141 GB

If you flip to draft-model speculative decoding instead of MTP:
  - Draft model (Qwen3.5-9B has no smaller sibling — pair manually if used)
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_HEALTH_MONITOR_INTERVAL_S = float(os.environ.get("VLLM_HEALTH_MONITOR_S", "30"))
_HEALTH_RESTART_COOLDOWN_S = float(os.environ.get("VLLM_RESTART_COOLDOWN_S", "120"))


class VLLMServerManager:
    """Manages multiple TP=1 vLLM inference servers across GPUs."""

    def __init__(
        self,
        model_name: str,
        adapter_dir: str,
        gpu_ids: List[int],
        base_port: int = 8000,
        gpu_util: float = 0.95,
        max_num_seqs: int = 128,
        enforce_eager: bool = False,
        log_dir: Optional[str] = None,
        speculative_model: Optional[str] = None,
        num_speculative_tokens: int = 1,
        speculative_method: str = "mtp",
        language_model_only: bool = True,
        reasoning_parser: Optional[str] = "qwen3",
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.gpu_ids = gpu_ids
        self.base_port = base_port
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.log_dir = log_dir
        self.speculative_model = speculative_model
        self.num_speculative_tokens = num_speculative_tokens
        self.speculative_method = speculative_method
        self.language_model_only = language_model_only
        self.reasoning_parser = reasoning_parser
        self._processes: List[subprocess.Popen] = []
        self._log_files: list = []
        self._speculative_config_json = self._build_speculative_config()

        atexit.register(self.stop)

    def _build_speculative_config(self) -> Optional[str]:
        """Build the JSON string for ``--speculative_config``.

        Two methods are supported:

        ``mtp``
            Use the model's built-in Multi-Token-Prediction head (Qwen3.5,
            Qwen3-Next, DeepSeek-V3, …).  No external draft model required.

        ``draft_model``
            Use a small external model as the drafter.  Required for plain
            Qwen3 (no MTP head); pass ``speculative_model="Qwen/Qwen3-0.6B"``.
        """
        import json
        method = (self.speculative_method or "").lower()
        if method == "mtp":
            if self.num_speculative_tokens <= 0:
                return None
            return json.dumps({
                "method": "mtp",
                "num_speculative_tokens": self.num_speculative_tokens,
            })
        if method == "draft_model":
            if not self.speculative_model:
                return None
            return json.dumps({
                "method": "draft_model",
                "model": self.speculative_model,
                "num_speculative_tokens": self.num_speculative_tokens,
            })
        return None

    @property
    def n_instances(self) -> int:
        return len(self.gpu_ids)

    @property
    def base_urls(self) -> List[str]:
        return [
            f"http://localhost:{self.base_port + i}/v1"
            for i in range(self.n_instances)
        ]

    @property
    def ports(self) -> List[int]:
        return [self.base_port + i for i in range(self.n_instances)]

    def _build_lora_args(self) -> List[str]:
        """Build --lora-modules flag values from existing adapter dirs."""
        modules = []
        for sub, names in [
            ("decision", ["skill_selection", "action_taking"]),
            ("skillbank", ["segment", "contract", "curator"]),
        ]:
            for name in names:
                path = Path(self.adapter_dir) / sub / name
                if (path / "adapter_config.json").exists():
                    modules.append(f"{name}={path}")
        return modules

    def _has_shared_gpus(self) -> bool:
        """True if any GPU hosts more than one instance."""
        from collections import Counter
        return max(Counter(self.gpu_ids).values()) > 1

    def start(self) -> None:
        """Start one vLLM TP=1 instance per entry in gpu_ids.

        When multiple instances share a GPU, they are launched in waves
        (unique GPUs first, duplicates after the first wave is healthy)
        to avoid OOM during the memory-heavy init phase.
        """
        if self._processes:
            logger.warning("Instances already running — stopping first")
            self.stop()

        lora_modules = self._build_lora_args()
        shared_gpus = self._has_shared_gpus()

        log_dir_path = Path(self.log_dir) if self.log_dir else None
        if log_dir_path:
            log_dir_path.mkdir(parents=True, exist_ok=True)

        seen_gpus: set = set()
        first_wave: list = []
        second_wave: list = []
        for i, gpu_id in enumerate(self.gpu_ids):
            entry = (i, gpu_id)
            if gpu_id not in seen_gpus:
                first_wave.append(entry)
                seen_gpus.add(gpu_id)
            else:
                second_wave.append(entry)

        self._launch_wave(first_wave, lora_modules, log_dir_path,
                          shared_gpus)
        self._first_wave_count = len(first_wave)
        self._second_wave = second_wave
        self._lora_modules = lora_modules
        self._log_dir_path = log_dir_path
        self._shared_gpus = shared_gpus

        spec_msg = self._spec_log_suffix()
        if lora_modules:
            logger.info(
                "Wave 1: started %d vLLM instances (ports %d–%d) "
                "with %d LoRA adapters%s",
                len(first_wave),
                self.base_port,
                self.base_port + len(first_wave) - 1,
                len(lora_modules),
                spec_msg,
            )
        else:
            logger.info(
                "Wave 1: started %d vLLM instances (ports %d–%d), "
                "no LoRA adapters yet%s",
                len(first_wave),
                self.base_port,
                self.base_port + len(first_wave) - 1,
                spec_msg,
            )
        if second_wave:
            logger.info(
                "Wave 2 (%d instances) will start after wave 1 is healthy",
                len(second_wave),
            )

    def _spec_log_suffix(self) -> str:
        """Human-readable spec-decode summary for log messages."""
        if not self._speculative_config_json:
            return ""
        method = (self.speculative_method or "").lower()
        if method == "mtp":
            return f", spec_decode=MTP ({self.num_speculative_tokens} tok)"
        if method == "draft_model" and self.speculative_model:
            return (f", spec_decode={self.speculative_model} "
                    f"({self.num_speculative_tokens} tok)")
        return ""

    def _build_serve_cmd(
        self,
        port: int,
        lora_modules: list,
        shared_gpus: bool,
    ) -> List[str]:
        """Build the ``vllm serve`` argv for one TP=1 instance.

        Centralises the Qwen3.5-aware flag selection so initial launch and
        crash-restart paths stay in lockstep.
        """
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(self.gpu_util),
            "--enable-lora", "--max-loras", "5", "--max-lora-rank", "64",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-seqs", str(self.max_num_seqs),
            # batch-level token budget (sum across in-flight requests in
            # one decode step).  Bumped 16384 → 32768 (v10, 2026-05-04)
            # to let the engine pack more concurrent rollouts per step
            # — H200 KV cache headroom is ~10 GB free per instance at
            # gpu_util=0.9, well above the 32K bump.  This is per-batch,
            # NOT per-request, so it cannot truncate any single
            # completion (which is gated by ``max_tokens``).
            "--max-num-batched-tokens", "32768",
            "--port", str(port),
            "--trust-remote-code",
        ]
        if self.enforce_eager or shared_gpus:
            cmd.append("--enforce-eager")
        if self.language_model_only:
            # Qwen3.5 / Qwen3-VL: drop the vision encoder for text-only
            # rollouts.  Frees several GB of weights + activation memory.
            cmd.append("--language-model-only")
        if self.reasoning_parser:
            # Required by Qwen3.5 / Qwen3 to expose the model's <think> /
            # </think> reasoning blocks via the OpenAI-compatible API.
            cmd.extend(["--reasoning-parser", self.reasoning_parser])
        if self._speculative_config_json:
            cmd.extend(["--speculative_config", self._speculative_config_json])
        if lora_modules:
            cmd.extend(["--lora-modules"] + lora_modules)
        return cmd

    def _launch_wave(
        self,
        entries: list,
        lora_modules: list,
        log_dir_path: Optional[Path],
        shared_gpus: bool,
    ) -> None:
        for i, gpu_id in entries:
            port = self.base_port + i

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
            env.setdefault("HF_HOME", "/workspace/huggingface")
            env.setdefault("HF_HUB_CACHE",
                           os.path.join(env["HF_HOME"], "hub"))
            # T2.14: vLLM 0.20+ unconditionally enumerates FP8-eligible layers
            # during kernel warmup (deep_gemm_warmup) and crashes if the
            # `deep_gemm` package is missing — even on bf16 / fp16 weights
            # like Qwen3.5-9B that have no FP8 path.  Disable by default;
            # callers can re-enable by exporting VLLM_USE_DEEP_GEMM=1.
            env.setdefault("VLLM_USE_DEEP_GEMM", "0")
            # Some clusters expose only the CUDA runtime, not the full toolkit
            # with nvcc.  vLLM's flashinfer sampler may JIT-compile on first
            # profile_run and fail with "Could not find nvcc"; use vLLM's
            # built-in sampler unless callers explicitly opt back in.
            env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

            cmd = self._build_serve_cmd(port, lora_modules, shared_gpus)

            log_fh = None
            if log_dir_path:
                log_path = log_dir_path / f"vllm_{i}_gpu{gpu_id}.log"
                log_fh = open(log_path, "w")

            logger.info(
                "Starting vLLM [%d/%d]: GPU %d → port %d",
                i + 1, self.n_instances, gpu_id, port,
            )

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=log_fh or subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
            )
            self._processes.append(proc)
            if log_fh:
                self._log_files.append(log_fh)

        spec_msg = self._spec_log_suffix()
        if lora_modules:
            logger.info(
                "Started %d vLLM instances (ports %d–%d) with %d LoRA adapters%s",
                self.n_instances,
                self.base_port,
                self.base_port + self.n_instances - 1,
                len(lora_modules),
                spec_msg,
            )
        else:
            logger.info(
                "Started %d vLLM instances (ports %d–%d), no LoRA adapters yet%s",
                self.n_instances,
                self.base_port,
                self.base_port + self.n_instances - 1,
                spec_msg,
            )

    def stop(self) -> None:
        """Terminate all vLLM instances and free GPU memory."""
        self.stop_health_monitor()
        if not self._processes:
            return

        n = len(self._processes)
        logger.info("Stopping %d vLLM instances...", n)

        # Send SIGTERM to all
        for proc in self._processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        # Wait for graceful shutdown, SIGKILL stragglers
        deadline = time.monotonic() + 15
        for proc in self._processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

        self._processes.clear()

        for fh in self._log_files:
            try:
                fh.close()
            except Exception:
                pass
        self._log_files.clear()

        logger.info("All %d vLLM instances stopped", n)

    async def _wait_for_indices(
        self,
        indices: list,
        timeout: float,
        poll_interval: float,
        label: str = "",
    ) -> bool:
        """Wait until the given instance indices all respond to /health."""
        import httpx as _httpx

        start = time.monotonic()
        healthy: set = set()
        n_target = len(indices)
        last_progress_log = start

        async with _httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() - start < timeout:
                for i in indices:
                    if i in healthy:
                        continue
                    if i >= len(self._processes):
                        continue
                    proc = self._processes[i]
                    if proc.poll() is not None:
                        logger.error(
                            "vLLM instance %d (GPU %d, port %d) exited "
                            "with code %d",
                            i, self.gpu_ids[i],
                            self.base_port + i, proc.returncode,
                        )
                        return False

                    port = self.base_port + i
                    try:
                        resp = await client.get(
                            f"http://localhost:{port}/health",
                        )
                        if resp.status_code == 200:
                            healthy.add(i)
                            elapsed = time.monotonic() - start
                            logger.info(
                                "vLLM GPU %d (port %d) healthy "
                                "[%d/%d %s, %.1fs]",
                                self.gpu_ids[i], port,
                                len(healthy), n_target, label, elapsed,
                            )
                    except Exception:
                        pass

                if len(healthy) >= n_target:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "All %d %s instances healthy (%.1fs)",
                        n_target, label, elapsed,
                    )
                    return True

                now = time.monotonic()
                if now - last_progress_log >= 30:
                    elapsed = now - start
                    logger.info(
                        "Waiting for vLLM %s: %d/%d healthy (%.0fs elapsed)",
                        label, len(healthy), n_target, elapsed,
                    )
                    last_progress_log = now

                await asyncio.sleep(poll_interval)

        missing = [self.base_port + i for i in indices if i not in healthy]
        logger.error(
            "Timeout (%.0fs): %d %s instances not ready (ports %s)",
            timeout, len(missing), label, missing,
        )
        return False

    async def wait_healthy(
        self,
        timeout: float = 600,
        poll_interval: float = 5,
    ) -> bool:
        """Wait for all instances to pass health checks.

        When GPUs are shared (2+ instances per GPU), instances are
        started in two waves to avoid OOM during init.  Wave 1
        (one instance per unique GPU) must be healthy before wave 2
        is launched.
        """
        first_count = getattr(self, "_first_wave_count", self.n_instances)
        second_wave = getattr(self, "_second_wave", [])
        wave1_indices = list(range(first_count))

        ok = await self._wait_for_indices(
            wave1_indices, timeout, poll_interval, label="wave-1",
        )
        if not ok:
            return False

        if second_wave:
            logger.info(
                "Wave 1 healthy — launching wave 2 (%d instances)...",
                len(second_wave),
            )
            self._launch_wave(
                second_wave,
                getattr(self, "_lora_modules", []),
                getattr(self, "_log_dir_path", None),
                getattr(self, "_shared_gpus", False),
            )
            wave2_indices = [i for i, _ in second_wave]
            ok = await self._wait_for_indices(
                wave2_indices, timeout, poll_interval, label="wave-2",
            )
            if not ok:
                return False

        logger.info(
            "All %d vLLM instances healthy", self.n_instances,
        )
        return True

    # ------------------------------------------------------------------
    # Background health monitor + auto-restart
    # ------------------------------------------------------------------

    def _restart_instance(self, idx: int) -> bool:
        """Restart a single dead vLLM instance in-place.

        Returns True if the new process was launched (it still needs
        to become healthy before it can serve requests).
        """
        if idx >= len(self._processes):
            return False
        proc = self._processes[idx]
        if proc.poll() is None:
            return False  # still running

        gpu_id = self.gpu_ids[idx]
        port = self.base_port + idx
        logger.warning(
            "vLLM instance %d (GPU %d, port %d) is dead (rc=%s) — restarting",
            idx, gpu_id, port, proc.returncode,
        )

        lora_modules = getattr(self, "_lora_modules", self._build_lora_args())
        shared_gpus = getattr(self, "_shared_gpus", self._has_shared_gpus())
        log_dir_path = getattr(self, "_log_dir_path", None)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        env.setdefault("HF_HOME", "/workspace/huggingface")
        env.setdefault("HF_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        cmd = self._build_serve_cmd(port, lora_modules, shared_gpus)

        log_fh = None
        if log_dir_path:
            log_path = log_dir_path / f"vllm_{idx}_gpu{gpu_id}.log"
            log_fh = open(log_path, "a")  # append to keep crash context

        new_proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
        )
        self._processes[idx] = new_proc
        if log_fh and idx < len(self._log_files):
            try:
                self._log_files[idx].close()
            except Exception:
                pass
            self._log_files[idx] = log_fh

        logger.info(
            "Restarted vLLM instance %d (GPU %d, port %d, pid %d)",
            idx, gpu_id, port, new_proc.pid,
        )
        return True

    def start_health_monitor(self) -> None:
        """Launch a background daemon thread that monitors vLLM instances.

        If a process is found dead, it is automatically restarted.
        The monitor respects a cooldown to avoid restart storms.
        """
        if getattr(self, "_monitor_stop", None) is not None:
            return  # already running
        self._monitor_stop = threading.Event()
        self._last_restart: dict[int, float] = {}

        def _monitor() -> None:
            while not self._monitor_stop.wait(_HEALTH_MONITOR_INTERVAL_S):
                now = time.monotonic()
                for idx, proc in enumerate(self._processes):
                    if proc.poll() is not None:
                        last = self._last_restart.get(idx, 0.0)
                        if (now - last) < _HEALTH_RESTART_COOLDOWN_S:
                            continue
                        if self._restart_instance(idx):
                            self._last_restart[idx] = now

        t = threading.Thread(target=_monitor, daemon=True, name="vllm-health-monitor")
        t.start()
        self._monitor_thread = t
        logger.info(
            "vLLM health monitor started (interval=%.0fs, cooldown=%.0fs)",
            _HEALTH_MONITOR_INTERVAL_S, _HEALTH_RESTART_COOLDOWN_S,
        )

    def stop_health_monitor(self) -> None:
        """Stop the background health monitor if running."""
        stop_evt = getattr(self, "_monitor_stop", None)
        if stop_evt is not None:
            stop_evt.set()
            self._monitor_stop = None

    async def reload_adapters(self) -> None:
        """Hot-reload all LoRA adapters on every running vLLM instance.

        For each adapter found on disk, issues an unload + load cycle
        via the vLLM REST API so the instance picks up freshly-trained
        weights without a full restart.
        """
        import httpx as _httpx

        adapter_groups = [
            ("decision", ["skill_selection", "action_taking"]),
            ("skillbank", ["segment", "contract", "curator"]),
        ]

        adapters_to_reload: list[tuple[str, str]] = []
        for sub, names in adapter_groups:
            for name in names:
                path = Path(self.adapter_dir) / sub / name
                if (path / "adapter_config.json").exists():
                    adapters_to_reload.append((name, str(path)))

        if not adapters_to_reload:
            logger.warning("No adapters found on disk to reload")
            return

        n_ok, n_fail = 0, 0
        async with _httpx.AsyncClient(timeout=30.0) as client:
            for port in self.ports:
                base = f"http://localhost:{port}"
                for adapter_name, adapter_path in adapters_to_reload:
                    try:
                        await client.post(
                            f"{base}/v1/unload_lora_adapter",
                            json={"lora_name": adapter_name},
                        )
                    except Exception:
                        pass

                    try:
                        resp = await client.post(
                            f"{base}/v1/load_lora_adapter",
                            json={
                                "lora_name": adapter_name,
                                "lora_path": adapter_path,
                            },
                        )
                        if resp.status_code == 200:
                            n_ok += 1
                        else:
                            logger.warning(
                                "Reload %s on port %d: HTTP %d — %s",
                                adapter_name, port,
                                resp.status_code, resp.text[:200],
                            )
                            n_fail += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to reload %s on port %d: %s",
                            adapter_name, port, exc,
                        )
                        n_fail += 1

        logger.info(
            "Adapter hot-reload: %d/%d successful across %d instances "
            "(%d adapters)",
            n_ok, n_ok + n_fail, len(self.ports),
            len(adapters_to_reload),
        )

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
