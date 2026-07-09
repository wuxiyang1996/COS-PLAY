"""Stable Actor wrapper for cross-domain SkillBridge eval.

Loads a SkillBridge checkpoint (LoRA adapters + skill bank) and exposes
a uniform ``Actor`` interface every per-domain eval driver can call.

Design goals:
  * Domain-agnostic — same Actor class drives gymv, browsergym, osworld,
    visual_reasoning, video.  The per-domain runner is responsible for
    constructing the env-specific ``state`` dict and consuming the
    returned ``action`` string.
  * Deterministic LoRA loading — the actor expects a vLLM endpoint
    already running with the LoRA modules registered (the eval
    launchers spin one up).
  * Optional harness gate — when ``--harness-mode full`` is passed, the
    actor runs ``select_eligible_skills`` + ``validate_invocation``
    around the LLM's pick.
  * No GRPO state — eval is rollout-only; reward attribution and
    advantage estimation are handled by the per-domain runner.

Public API:

    actor = SkillBridgeActor.from_checkpoint(
        checkpoint_dir=Path("runs/.../checkpoints/step_0150"),
        bank_dir=Path("runs/.../skillbank"),
        vllm_base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3.5-9B",
        harness_mode="full",  # or "plain-text-skills" / "off"
        actor_bank_cap_k=0,
    )
    while not done:
        action = actor.act(
            game="gymv_thunder_force_iii",
            obs_nl=observation_text,
            structured_state={"hp": 80, "score": 1200},
            action_names=["UP", "DOWN", "LEFT", "RIGHT", "A", "B"],
        )
        obs_nl, reward, done, info = env.step(action)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ActStats:
    """Per-act diagnostic returned alongside the chosen action."""
    action: str
    chosen_skill_id: Optional[str] = None
    intention: str = ""
    reasoning: str = ""
    n_candidates: int = 0
    n_admitted: int = 0
    harness_validate_ok: Optional[bool] = None
    harness_validate_diag: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "chosen_skill_id": self.chosen_skill_id,
            "intention": self.intention,
            "reasoning": self.reasoning,
            "n_candidates": self.n_candidates,
            "n_admitted": self.n_admitted,
            "harness_validate_ok": self.harness_validate_ok,
            "harness_validate_diag": self.harness_validate_diag,
            "latency_ms": self.latency_ms,
            "extra": dict(self.extra),
        }


class SkillBridgeActor:
    """Stateless-per-episode actor: loads (LoRA + bank), exposes .act().

    Construct via :meth:`from_checkpoint`.  Each ``act()`` call performs:

      1. RAG retrieval over the skill bank (top-K candidates).
      2. (optional) Harness eligibility filter.
      3. ``skill_selection`` LoRA call → pick one candidate.
      4. (optional) Harness ``validate_invocation`` veto.
      5. ``action_taking`` LoRA call → pick env action.

    Failures at every layer are non-fatal — a missing bank, dead vLLM
    endpoint, or unparseable LLM response degrades to a uniform-random
    valid action so the eval harness keeps producing rows.
    """

    def __init__(
        self,
        *,
        vllm_base_url: str,
        model: str,
        skill_bank: Optional[Any] = None,
        harness_hooks: Optional[Mapping[str, Any]] = None,
        harness_mode: str = "full",
        actor_bank_cap_k: int = 0,
        temperature: float = 0.3,
        max_tokens: int = 512,
        top_k_candidates: int = 3,
    ) -> None:
        self._base_url = vllm_base_url
        self._model = model
        self._skill_bank = skill_bank
        self._harness_hooks = dict(harness_hooks or {})
        self._harness_mode = harness_mode
        self._actor_bank_cap_k = int(actor_bank_cap_k)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._top_k = int(top_k_candidates)
        self._client = self._make_client()

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_dir: Optional[Path],
        bank_dir: Optional[Path],
        vllm_base_url: str,
        model: str,
        harness_mode: str = "full",
        actor_bank_cap_k: int = 0,
        games_for_harness: Optional[List[str]] = None,
        temperature: float = 0.3,
    ) -> "SkillBridgeActor":
        """Build an actor from a trainer run directory.

        Parameters
        ----------
        checkpoint_dir : Path | None
            Pointer to ``runs/<run>/checkpoints/step_NNNN``.  When
            ``None`` the actor uses the base model only (no LoRA) —
            useful for reproducing the "base model" baseline row.
        bank_dir : Path | None
            Pointer to ``runs/<run>/skillbank`` (per-game bank.jsonl
            files live in subdirs).  ``None`` means cold-start
            (no skill candidates surfaced).
        vllm_base_url : str
            Already-running vLLM endpoint.  The launcher must have
            registered the LoRA adapters at the standard slugs
            (``skill_selection``, ``action_taking``).
        model : str
            Base model identifier (e.g. ``"Qwen/Qwen3.5-9B"``).
        harness_mode : str
            ``"full"`` / ``"plain-text-skills"`` / ``"off"`` — same
            semantics as the trainer's block-B1 flag.
        actor_bank_cap_k : int
            Top-K bank cap (block-B5).  ``0`` = no cap.
        games_for_harness : list[str] | None
            Optional list of game names for which to construct
            ``SkillHarnessHook`` instances.  When omitted the harness
            is skipped (no veto layer; suitable for non-gymv domains
            that don't have a bank-backed harness yet).
        """
        # Skill bank — load via PerGameSkillBankManager when present,
        # else fall back to a single SkillQueryEngine over the
        # consolidated jsonl.
        skill_bank = _load_skill_bank(bank_dir)

        # Harness hooks — one per game for which a bank.jsonl exists.
        # Eval is read-only so we don't need the LLM validator path.
        harness_hooks: Dict[str, Any] = {}
        if (
            harness_mode != "off"
            and games_for_harness
            and bank_dir is not None
        ):
            try:
                from trainer.coevolution._harness_hook import SkillHarnessHook
                for g in games_for_harness:
                    bank_path = bank_dir / g / "skill_bank.jsonl"
                    if not bank_path.exists():
                        # Fall back to flat bank layout.
                        bank_path = bank_dir / "skill_bank.jsonl"
                    if not bank_path.exists():
                        continue
                    hook = SkillHarnessHook.for_game(
                        game=g,
                        bank_path=bank_path,
                        domain="gymv",
                        allow_shadow=True,
                        mode=harness_mode,
                    )
                    harness_hooks[g] = hook
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "skillbridge_eval: harness_hook construction failed (%s) "
                    "— continuing without harness", exc,
                )
                harness_hooks = {}

        return cls(
            vllm_base_url=vllm_base_url,
            model=model,
            skill_bank=skill_bank,
            harness_hooks=harness_hooks,
            harness_mode=harness_mode,
            actor_bank_cap_k=actor_bank_cap_k,
            temperature=temperature,
        )

    def _make_client(self) -> Any:
        """Return a thin async vLLM client.  Reuses the trainer's
        :class:`AsyncVLLMClient` so adapter routing matches."""
        from trainer.coevolution.vllm_client import AsyncVLLMClient
        return AsyncVLLMClient(
            base_url=self._base_url,
            model=self._model,
            default_temperature=self._temperature,
            default_max_tokens=self._max_tokens,
        )

    # ── Per-step API ─────────────────────────────────────────────────

    def act(
        self,
        *,
        game: str,
        obs_nl: str,
        structured_state: Optional[Dict[str, Any]] = None,
        action_names: List[str],
        intention: str = "",
        episode_id: str = "",
        inner_step: int = 0,
    ) -> ActStats:
        """Synchronous wrapper around :meth:`_act_async`.  Constructs
        an event loop on demand so callers don't need asyncio plumbing.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Inside an outer event loop — caller should use _act_async
                # directly.  Fall through to a fresh loop in a thread.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        asyncio.run,
                        self._act_async(
                            game=game, obs_nl=obs_nl,
                            structured_state=structured_state,
                            action_names=action_names,
                            intention=intention,
                            episode_id=episode_id,
                            inner_step=inner_step,
                        ),
                    )
                    return fut.result()
        except RuntimeError:
            pass  # No current loop; fall through.
        return asyncio.run(self._act_async(
            game=game, obs_nl=obs_nl,
            structured_state=structured_state,
            action_names=action_names,
            intention=intention,
            episode_id=episode_id,
            inner_step=inner_step,
        ))

    async def _act_async(
        self,
        *,
        game: str,
        obs_nl: str,
        structured_state: Optional[Dict[str, Any]],
        action_names: List[str],
        intention: str,
        episode_id: str,
        inner_step: int,
    ) -> ActStats:
        t0 = time.monotonic()
        stats = ActStats(action="")

        # ── 1. Retrieve top-K candidates (skipped when no bank). ──
        candidates: List[Dict[str, Any]] = []
        if self._skill_bank is not None and self._harness_mode != "off":
            try:
                from scripts.qwen3_decision_agent import (
                    get_top_k_skill_candidates,
                )
                candidates = get_top_k_skill_candidates(
                    self._skill_bank,
                    obs_nl,
                    game_name=game,
                    intention=intention,
                    structured_state=structured_state or {},
                    top_k=self._top_k,
                    bank_cap_k=self._actor_bank_cap_k,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("eval_actor: candidate retrieval failed: %s", exc)
                candidates = []
        stats.n_candidates = len(candidates)

        # ── 2. Optional harness eligibility filter. ───────────────
        admitted: List[Dict[str, Any]] = list(candidates)
        hook = self._harness_hooks.get(game)
        if hook is not None and self._harness_mode == "full" and candidates:
            try:
                state = hook.state_for_step(
                    game=game, summary_state=obs_nl,
                    intention=intention, inner_step=inner_step,
                    outer_step=inner_step,
                )
                admitted, _diag = hook.filter_candidates(
                    list(candidates), state, episode_id=episode_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("eval_actor: harness filter failed: %s", exc)
                admitted = list(candidates)
        stats.n_admitted = len(admitted)

        # ── 3. skill_selection LLM call (only when ≥2 candidates). ─
        chosen_skill_id: Optional[str] = None
        chosen_skill_dict: Optional[Dict[str, Any]] = None
        if len(admitted) >= 2:
            chosen_idx, _reasoning = await self._pick_skill(
                obs_nl=obs_nl, game=game,
                candidates=admitted, intention=intention,
            )
            if 0 <= chosen_idx < len(admitted):
                chosen_skill_dict = admitted[chosen_idx]
                chosen_skill_id = chosen_skill_dict.get("skill_id")
        elif len(admitted) == 1:
            chosen_skill_dict = admitted[0]
            chosen_skill_id = chosen_skill_dict.get("skill_id")
        stats.chosen_skill_id = chosen_skill_id

        # ── 4. Optional validate_invocation veto. ─────────────────
        if (
            chosen_skill_id is not None
            and hook is not None
            and self._harness_mode == "full"
        ):
            try:
                state = hook.state_for_step(
                    game=game, summary_state=obs_nl,
                    intention=intention, inner_step=inner_step,
                    outer_step=inner_step,
                )
                ok, vd = hook.validate_choice(
                    chosen_skill_id, state,
                    episode_id=episode_id, inner_step=inner_step,
                )
                stats.harness_validate_ok = bool(ok)
                stats.harness_validate_diag = vd
                if not ok:
                    # Walk to next admitted candidate; degrade to no-skill
                    # if no admit survives.
                    for cand in admitted:
                        sid = cand.get("skill_id")
                        if not sid or sid == chosen_skill_id:
                            continue
                        ok2, _ = hook.validate_choice(
                            sid, state, episode_id=episode_id,
                            inner_step=inner_step,
                        )
                        if ok2:
                            chosen_skill_id = sid
                            chosen_skill_dict = cand
                            stats.chosen_skill_id = sid
                            stats.harness_validate_ok = True
                            break
                    else:
                        chosen_skill_id = None
                        chosen_skill_dict = None
                        stats.chosen_skill_id = None
            except Exception as exc:  # noqa: BLE001
                logger.debug("eval_actor: validate_choice failed: %s", exc)

        # ── 5. action_taking LLM call. ────────────────────────────
        action, reasoning = await self._pick_action(
            obs_nl=obs_nl, game=game,
            action_names=action_names,
            intention=intention,
            skill=chosen_skill_dict,
        )
        stats.action = action
        stats.reasoning = reasoning
        stats.intention = intention
        stats.latency_ms = (time.monotonic() - t0) * 1000.0
        return stats

    # ── LLM call helpers ─────────────────────────────────────────────

    async def _pick_skill(
        self,
        *,
        obs_nl: str,
        game: str,
        candidates: List[Dict[str, Any]],
        intention: str,
    ) -> Tuple[int, str]:
        """Run the ``skill_selection`` LoRA on a candidate menu, return
        ``(chosen_idx, reasoning)``.  Defaults to index 0 on parse fail.
        """
        from scripts.qwen3_decision_agent import _format_candidates_for_selection
        # Reuse the trainer's compact menu prompt for parity with training.
        try:
            menu = _format_candidates_for_selection(candidates)
        except Exception:
            menu = "\n".join(
                f"{i + 1}. {c.get('skill_id', '?')}"
                for i, c in enumerate(candidates)
            )
        prompt = (
            f"Game state:\n{obs_nl[:2500]}\n\n"
            f"Current intention: {intention[:500]}\n\n"
            f"Available strategies (pick ONE by number):\n{menu}\n\n"
            f"Choose the best strategy. Output REASONING then SKILL number."
        )
        result = await self._client.generate_chat(
            [{"role": "user", "content": prompt}],
            adapter="skill_selection",
            temperature=self._temperature,
            max_tokens=128,
        )
        text = (result.text or "").strip()
        idx = _parse_skill_number(text, n=len(candidates))
        return idx, text

    async def _pick_action(
        self,
        *,
        obs_nl: str,
        game: str,
        action_names: List[str],
        intention: str,
        skill: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Run the ``action_taking`` LoRA, return ``(action, reasoning)``.

        Falls back to the first valid action on parse failure.
        """
        skill_block = ""
        if skill:
            skill_block = (
                f"\n\nSelected skill: {skill.get('skill_id', '?')}\n"
                f"Why: {skill.get('why_selected', '')[:300]}\n"
                f"Hint: {skill.get('execution_hint', '')[:300]}\n"
            )
        actions_block = ", ".join(action_names) or "no_op"
        prompt = (
            f"Game: {game.replace('_', ' ')}\n"
            f"State:\n{obs_nl[:2500]}\n\n"
            f"Intention: {intention[:500]}{skill_block}\n\n"
            f"Available actions: {actions_block}\n"
            "Choose ONE action.  Output REASONING then ACTION."
        )
        result = await self._client.generate_chat(
            [{"role": "user", "content": prompt}],
            adapter="action_taking",
            temperature=self._temperature,
            max_tokens=128,
        )
        text = (result.text or "").strip()
        action = _parse_action(text, action_names)
        return action, text


# ── Module-level helpers ──────────────────────────────────────────────


def _load_skill_bank(bank_dir: Optional[Path]) -> Optional[Any]:
    """Build a ``SkillQueryEngine`` over ``bank_dir/skill_bank.jsonl``.

    Returns ``None`` when ``bank_dir`` is missing or empty.  Per-game
    sub-banks (``bank_dir/<game>/skill_bank.jsonl``) are concatenated
    into a single merged JSONL that backs the SkillBankMVP load path.
    """
    if bank_dir is None:
        return None
    bank_path = Path(bank_dir)
    if not bank_path.exists():
        return None

    candidates = list(bank_path.glob("**/skill_bank.jsonl"))
    if not candidates:
        return None

    try:
        from skill_agents.query import SkillQueryEngine
        from skill_agents.skill_bank.bank import SkillBankMVP
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval_actor: skill_bank imports failed: %s", exc)
        return None

    merged_path = bank_path / "_merged_for_eval.jsonl"
    seen: set = set()
    n_lines = 0
    try:
        with open(merged_path, "w", encoding="utf-8") as out:
            for p in sorted(candidates):
                if p == merged_path:
                    continue
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # Skill records carry their id under .skill.skill_id
                        # (new format) or .skill_id / .contract.skill_id
                        # (legacy).  Use whichever is present.
                        sid = (
                            (obj.get("skill") or {}).get("skill_id")
                            or obj.get("skill_id")
                            or (obj.get("contract") or {}).get("skill_id")
                        )
                        if not sid or sid in seen:
                            continue
                        seen.add(sid)
                        out.write(line + "\n")
                        n_lines += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval_actor: merge bank failed: %s", exc)
        return None

    if n_lines == 0:
        return None
    try:
        bank = SkillBankMVP(path=str(merged_path))
        bank.load(str(merged_path))
        return SkillQueryEngine(bank)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval_actor: SkillQueryEngine load failed: %s", exc)
        return None


def _parse_skill_number(text: str, *, n: int) -> int:
    """Extract the trailing skill number (1-based) from the LLM output.

    Returns the 0-based index, clamped to ``[0, n-1]``.  Defaults to 0
    on parse failure.
    """
    import re
    if n <= 0:
        return 0
    matches = re.findall(r"(?:SKILL|skill)\s*[:#]?\s*(\d+)", text)
    if not matches:
        # Look for the last bare integer in the text.
        matches = re.findall(r"\b(\d+)\b", text)
    if not matches:
        return 0
    try:
        idx = int(matches[-1]) - 1
    except ValueError:
        return 0
    return max(0, min(n - 1, idx))


def _parse_action(text: str, action_names: List[str]) -> str:
    """Find the first action_name occurrence in the text.

    Falls back to ``action_names[0]`` (or the empty string) on no match.
    """
    if not action_names:
        return ""
    text_lower = text.lower()
    # Prefer ``ACTION:`` line if present.
    for line in text.splitlines()[::-1]:
        line_low = line.strip().lower()
        if not line_low:
            continue
        for a in action_names:
            if a.lower() in line_low:
                return a
    # Fallback: any occurrence in the whole text.
    for a in action_names:
        if a.lower() in text_lower:
            return a
    return action_names[0]
