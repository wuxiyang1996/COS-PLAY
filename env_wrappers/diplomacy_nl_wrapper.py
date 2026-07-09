"""
Diplomacy game env wrapper: state <-> natural language, gym-style multi-agent interface.

Wraps the diplomacy.Game engine into a synchronous reset()/step() loop where every
power receives a natural-language observation and returns natural-language orders.

Supports multi-agent (all 7 powers controlled externally) and single-agent (one power
controlled, the rest use a partner_policy callback).

Usage (multi-agent, from codebase root):

    from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper

    env = DiplomacyNLWrapper()
    obs, info = env.reset()
    # obs: {power_name: nl_str}  e.g. {"FRANCE": "...", "ENGLAND": "...", ...}

    # Each step: pass orders for each power
    actions = {
        "FRANCE": ["A PAR - BUR", "F BRE - ENG", "A MAR H"],
        "ENGLAND": ["F LON - ENG", "F EDI - NTH", "A LVP - YOR"],
        # ... other powers ...
    }
    obs, rewards, terminated, truncated, info = env.step(actions)

Usage (single-agent):

    env = DiplomacyNLWrapper(controlled_power="FRANCE")
    obs, info = env.reset()           # obs: str (NL for FRANCE)
    obs, reward, term, trunc, info = env.step(["A PAR - BUR", "F BRE H", "A MAR - SPA"])

Negotiation mode:
    If negotiation_rounds > 0, each movement phase has a negotiation step before orders.
    Use step_negotiate() to exchange messages, then step() to submit orders.
    Or set auto_negotiate=True to skip negotiation (default).
"""

import itertools
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_log = logging.getLogger(__name__)

# Prefer local AI_Diplomacy/diplomacy package so "from diplomacy import Game" finds it
import sys
from pathlib import Path as _Path
_diplomacy_wrapper_dir = _Path(__file__).resolve().parent
_codebase_root = _diplomacy_wrapper_dir.parent
_ai_diplomacy_dir = _codebase_root / "AI_Diplomacy"
if _ai_diplomacy_dir.exists() and str(_ai_diplomacy_dir) not in sys.path:
    sys.path.insert(0, str(_ai_diplomacy_dir))

try:
    from diplomacy import Game
except ImportError as _e:
    # If still failing, try pip package or re-raise with a helpful message
    Game = None  # type: ignore
    _dip_import_error = _e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

_ORDER_TYPE_HELP = {
    "M": (
        "Movement phase. Issue orders for each unit:\n"
        "  Hold:         A PAR H\n"
        "  Move:         A PAR - BUR\n"
        "  Support hold: A MAR S A PAR\n"
        "  Support move: A MAR S A PAR - BUR\n"
        "  Convoy:       F ENG C A LON - BRE"
    ),
    "R": (
        "Retreat phase. Dislodged units must retreat or disband:\n"
        "  Retreat: A PAR R MAR\n"
        "  Disband: A PAR D"
    ),
    "A": (
        "Adjustment phase. Build or disband units:\n"
        "  Build:   A PAR B  or  F BRE B\n"
        "  Disband: A PAR D"
    ),
}


# ---------------------------------------------------------------------------
# Compact structured state summary (for agent context / retrieval)
# ---------------------------------------------------------------------------

def build_structured_state_summary(
    game: "Game",
    power_name: str,
    prev_sc_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """Build a compact structured dict for one Diplomacy power.

    Designed to be fed into ``compact_structured_state()`` from
    ``decision_agents.agent_helper``.

    Args:
        prev_sc_counts: SC counts from the previous phase — if provided,
            a ``delta`` field is included showing which powers gained/lost.

    Returns:
        Dict with short key=value-friendly fields.
    """
    current_phase = game.get_current_phase()
    phase_type = current_phase[-1] if current_phase and current_phase not in ("FORMING", "COMPLETED") else ""
    power = game.powers.get(power_name)

    if power is None or power.is_eliminated():
        return {
            "game": "diplomacy",
            "phase": current_phase,
            "self": f"power:{power_name} ELIMINATED",
        }

    units = list(power.units) if power.units else []
    centers = list(power.centers) if power.centers else []
    units_short = ",".join(units[:6])
    if len(units) > 6:
        units_short += f"..+{len(units) - 6}"

    orderable_locs = game.get_orderable_locations(power_name) or []
    fronts_short = ",".join(orderable_locs[:5])
    if len(orderable_locs) > 5:
        fronts_short += f"..+{len(orderable_locs) - 5}"

    objective = {
        "M": "issue_move_orders",
        "R": "issue_retreat_orders",
        "A": "build_or_disband",
    }.get(phase_type, "wait")

    # All-powers SC snapshot (compact)
    rivals_parts = []
    for pname, pobj in game.powers.items():
        if pname == power_name or pobj.is_eliminated():
            continue
        rivals_parts.append(f"{pname[:3]}:{len(pobj.centers)}")
    rivals_str = ",".join(rivals_parts)

    summary: dict = {
        "game": "diplomacy",
        "phase": current_phase,
        "objective": objective,
        "self": f"power:{power_name} centers:{len(centers)}",
        "resources": f"units:{units_short}" if units_short else "units:none",
    }

    if rivals_str:
        summary["rivals"] = rivals_str

    if fronts_short:
        summary["critical"] = f"locs:{fronts_short}"

    if power.retreats:
        ret_units = ",".join(str(u) for u in list(power.retreats.keys())[:3])
        summary["critical"] = summary.get("critical", "") + f" retreat:{ret_units}"

    # SC delta since last phase
    if prev_sc_counts:
        deltas = []
        for pname, pobj in game.powers.items():
            cur = len(pobj.centers) if not pobj.is_eliminated() else 0
            prev = prev_sc_counts.get(pname, cur)
            diff = cur - prev
            if diff != 0:
                deltas.append(f"{pname[:3]}{'+' if diff > 0 else ''}{diff}")
        if deltas:
            summary["delta"] = ",".join(deltas)

    return summary


# ---------------------------------------------------------------------------
# State -> NL
# ---------------------------------------------------------------------------

def state_to_natural_language(
    game: "Game",
    power_name: str,
    negotiation_messages: Optional[List[str]] = None,
    phase_history: Optional[List[str]] = None,
) -> str:
    """
    Convert Diplomacy game state to natural-language description for one power.

    Args:
        game: diplomacy.Game instance.
        power_name: The power receiving this observation (e.g. "FRANCE").
        negotiation_messages: Recent negotiation messages (optional).
        phase_history: Summary of recent phase results (optional).

    Returns:
        Natural-language state description string.
    """
    current_phase = game.get_current_phase()
    phase_type = current_phase[-1] if current_phase and current_phase not in ("FORMING", "COMPLETED") else ""
    power = game.powers.get(power_name)

    lines = [f"=== Diplomacy — Phase: {current_phase} ==="]
    lines.append(f"You are: {power_name}.")
    lines.append("")

    # Your forces
    if power:
        lines.append(f"Your units: {power.units if power.units else 'None (eliminated)'}")
        lines.append(f"Your supply centers: {list(power.centers)} ({len(power.centers)} total)")
        lines.append(f"Your home centers: {list(power.homes) if power.homes else 'N/A'}")
        if power.retreats:
            lines.append(f"Your dislodged units needing retreat: {dict(power.retreats)}")
        lines.append("")

    # All powers summary
    lines.append("--- All Powers Status ---")
    for pname, pobj in game.powers.items():
        marker = " (you)" if pname == power_name else ""
        if pobj.is_eliminated():
            lines.append(f"  {pname}{marker}: ELIMINATED")
        else:
            lines.append(
                f"  {pname}{marker}: {len(pobj.centers)} centers, "
                f"units={pobj.units}"
            )
    lines.append("")

    # Possible orders for this power
    if power and not power.is_eliminated():
        orderable_locs = game.get_orderable_locations(power_name)
        possible_orders = game.get_all_possible_orders()

        if orderable_locs:
            lines.append("--- Your Possible Orders ---")
            for loc in orderable_locs:
                if loc in possible_orders and possible_orders[loc]:
                    orders_str = ", ".join(possible_orders[loc][:15])
                    more = f" ... (+{len(possible_orders[loc]) - 15} more)" if len(possible_orders[loc]) > 15 else ""
                    lines.append(f"  {loc}: {orders_str}{more}")
            lines.append("")
        else:
            lines.append("No units requiring orders this phase.")
            lines.append("")

    # Order format help
    if phase_type in _ORDER_TYPE_HELP:
        lines.append("--- Order Format ---")
        lines.append(_ORDER_TYPE_HELP[phase_type])
        lines.append("")

    # Phase action prompt
    if game.is_game_done:
        lines.append("Game is over.")
    elif power and power.is_eliminated():
        lines.append("You have been eliminated. No action needed.")
    else:
        lines.append("Submit your orders as a list of order strings.")
        lines.append("Example: [\"A PAR - BUR\", \"F BRE H\", \"A MAR S A PAR - BUR\"]")

    # Recent negotiation messages
    if negotiation_messages:
        lines.append("")
        lines.append("--- Recent Negotiation ---")
        for msg in negotiation_messages[-15:]:
            lines.append(msg)

    # Phase history
    if phase_history:
        lines.append("")
        lines.append("--- Recent History ---")
        for entry in phase_history[-10:]:
            lines.append(entry)

    return "\n".join(lines)


def state_to_natural_language_for_all(
    game: "Game",
    **kwargs,
) -> Dict[str, str]:
    """Return NL observation for every non-eliminated power."""
    return {
        power_name: state_to_natural_language(game, power_name, **kwargs)
        for power_name in game.powers
    }


# ---------------------------------------------------------------------------
# Order parsing
# ---------------------------------------------------------------------------

def parse_orders(text: Any, game: "Game", power_name: str) -> List[str]:
    """
    Parse order strings from various input formats.

    Accepts:
      - List[str]: ["A PAR - BUR", "F BRE H"]
      - str: "A PAR - BUR, F BRE H" or "A PAR - BUR\\nF BRE H" or JSON list
      - None/empty: returns empty list

    Validates against possible orders. Invalid orders are dropped.
    """
    if text is None:
        return []

    # Already a list
    if isinstance(text, (list, tuple)):
        raw_orders = [str(o).strip() for o in text if o]
    elif isinstance(text, str):
        text = text.strip()
        # Try JSON parse
        if text.startswith("["):
            import json
            try:
                raw_orders = json.loads(text)
                raw_orders = [str(o).strip() for o in raw_orders]
            except (json.JSONDecodeError, TypeError):
                raw_orders = []
        else:
            # Split by newlines or commas (but not commas inside order like "S A PAR - BUR")
            # Orders are generally separated by newlines or semicolons
            import re
            raw_orders = re.split(r"[;\n]+", text)
            raw_orders = [o.strip().strip(",").strip("'\"") for o in raw_orders if o.strip()]
    else:
        return []

    if not raw_orders:
        return []

    # Validate against possible orders
    possible_orders = game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power_name) or []
    valid_set = set()
    for loc in orderable_locs:
        if loc in possible_orders:
            valid_set.update(possible_orders[loc])

    validated = [o for o in raw_orders if o in valid_set]
    return validated


# ---------------------------------------------------------------------------
# Default partner policy (random orders)
# ---------------------------------------------------------------------------

def _random_partner_orders(game: "Game", power_name: str, **_kwargs) -> List[str]:
    """Generate random valid orders for a power."""
    possible_orders = _kwargs.get("all_possible_orders") or game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power_name) or []
    orders = []
    for loc in orderable_locs:
        if loc in possible_orders and possible_orders[loc]:
            orders.append(random.choice(possible_orders[loc]))
    return orders


def _retrieve_skill_hint(skill_bank: Any, obs_text: str, game: str = "diplomacy") -> str:
    """Fast CPU-only top-1 skill retrieval.  Returns a short hint or ''."""
    if skill_bank is None:
        return ""
    try:
        has_items = (
            (hasattr(skill_bank, "__len__") and len(skill_bank) > 0)
            or (hasattr(skill_bank, "skill_ids")
                and len(list(skill_bank.skill_ids)) > 0)
        )
        if not has_items:
            return ""
        from scripts.qwen3_decision_agent import get_top_k_skill_candidates
        candidates = get_top_k_skill_candidates(
            skill_bank, obs_text, game_name=game, top_k=1,
        )
        if candidates:
            name = candidates[0].get("skill_name", "")
            hint = candidates[0].get("execution_hint", "")
            if name or hint:
                return f"Strategy hint: {name}. {hint}".strip()[:200]
    except Exception:
        pass
    return ""


def _build_shared_board_context(game: "Game") -> str:
    """Build a power-agnostic board summary reusable by all partner calls."""
    current_phase = game.get_current_phase()
    lines = [f"=== Diplomacy — Phase: {current_phase} ===", ""]
    lines.append("--- All Powers Status ---")
    for pname, pobj in game.powers.items():
        if pobj.is_eliminated():
            lines.append(f"  {pname}: ELIMINATED")
        else:
            lines.append(
                f"  {pname}: {len(pobj.centers)} centers, "
                f"units={pobj.units}"
            )
    lines.append("")
    return "\n".join(lines)


def _make_llm_partner_policy(
    base_urls: List[str],
    model_name: str,
    adapter_name: str = "action_taking",
    skill_bank: Any = None,
    game_name: str = "diplomacy",
) -> Callable:
    """Create an LLM-based partner policy that queries vLLM for orders.

    Returns a callable with the same signature as ``_random_partner_orders``.
    Uses round-robin across *base_urls* for load balancing.
    Opponents use the ``action_taking`` LoRA so they improve alongside the
    controlled agent (self-play dynamics) without the skill-selection overhead.
    When a *skill_bank* is provided the top-1 skill hint is injected into the
    prompt (CPU-only retrieval, zero extra LLM calls).
    Falls back to random for any unparseable orders.

    The callable accepts optional *shared_ctx* and *all_possible_orders* kwargs
    so that ``_submit_orders_single`` can compute these once per phase and reuse
    them for all 6 opponent calls.
    """
    import requests as _requests

    _url_cycle = itertools.cycle(base_urls)

    def _llm_partner_orders(
        game: "Game",
        power_name: str,
        *,
        shared_ctx: Optional[str] = None,
        all_possible_orders: Optional[dict] = None,
    ) -> List[str]:
        try:
            possible_orders = all_possible_orders or game.get_all_possible_orders()
            orderable_locs = game.get_orderable_locations(power_name) or []
            if not orderable_locs:
                return []

            power = game.powers.get(power_name)
            if shared_ctx is None:
                shared_ctx = _build_shared_board_context(game)

            power_lines = [f"You are: {power_name}."]
            if power:
                power_lines.append(f"Your units: {power.units if power.units else 'None'}")
                power_lines.append(f"Your supply centers: {list(power.centers)} ({len(power.centers)} total)")
            obs_text = shared_ctx + "\n".join(power_lines)

            loc_options = []
            for loc in orderable_locs:
                if loc in possible_orders and possible_orders[loc]:
                    opts = possible_orders[loc][:10]
                    loc_options.append(f"  {loc}: {opts}")
            options_str = "\n".join(loc_options)

            skill_line = _retrieve_skill_hint(skill_bank, obs_text, game_name)
            skill_prefix = f"{skill_line}\n" if skill_line else ""

            prompt = (
                f"{obs_text}\n\n{skill_prefix}"
                f"Pick one order per unit from these options:\n"
                f"{options_str}\n\n"
                f"Reply with one order per line, nothing else.\nOrders:\n"
            )

            url = next(_url_cycle)
            resp = _requests.post(
                f"{url}/completions",
                json={
                    "model": adapter_name,
                    "prompt": prompt,
                    "max_tokens": 128,
                    "temperature": 0.4,
                    "stop": ["\n\n"],
                },
                timeout=15,
            )
            text = resp.json()["choices"][0]["text"].strip()

            valid_set: set = set()
            for loc in orderable_locs:
                if loc in possible_orders:
                    valid_set.update(possible_orders[loc])

            parsed = [line.strip() for line in text.split("\n") if line.strip()]
            valid_orders = [o for o in parsed if o in valid_set]

            covered_locs = set()
            for o in valid_orders:
                for loc in orderable_locs:
                    if loc in possible_orders and o in possible_orders[loc]:
                        covered_locs.add(loc)
                        break

            for loc in orderable_locs:
                if loc not in covered_locs and loc in possible_orders and possible_orders[loc]:
                    valid_orders.append(random.choice(possible_orders[loc]))

            return valid_orders

        except Exception as exc:
            _log.debug("LLM partner orders failed for %s: %s", power_name, exc)
            return _random_partner_orders(game, power_name)

    return _llm_partner_orders


def _make_api_partner_policy(
    api_model: str,
    api_base: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
) -> Callable:
    """Create an API-based partner policy for external opponent models.

    Calls an external chat API (e.g. OpenRouter) instead of local vLLM,
    enabling training against a fixed strong opponent to break self-play
    reward inflation.  Falls back to random on any failure.
    """
    import openai as _openai

    if api_key is None:
        try:
            from api_keys import open_router_api_key
            api_key = open_router_api_key
        except ImportError:
            import os
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

    _client = _openai.OpenAI(base_url=api_base, api_key=api_key, max_retries=2)

    def _api_partner_orders(
        game: "Game",
        power_name: str,
        *,
        shared_ctx: Optional[str] = None,
        all_possible_orders: Optional[dict] = None,
    ) -> List[str]:
        try:
            possible_orders = all_possible_orders or game.get_all_possible_orders()
            orderable_locs = game.get_orderable_locations(power_name) or []
            if not orderable_locs:
                return []

            power = game.powers.get(power_name)
            if shared_ctx is None:
                shared_ctx = _build_shared_board_context(game)

            power_lines = [f"You are: {power_name}."]
            if power:
                power_lines.append(f"Your units: {power.units if power.units else 'None'}")
                power_lines.append(f"Your supply centers: {list(power.centers)} ({len(power.centers)} total)")
            obs_text = shared_ctx + "\n".join(power_lines)

            loc_options = []
            for loc in orderable_locs:
                if loc in possible_orders and possible_orders[loc]:
                    opts = possible_orders[loc][:10]
                    loc_options.append(f"  {loc}: {opts}")
            options_str = "\n".join(loc_options)

            prompt = (
                f"{obs_text}\n\n"
                f"Pick one order per unit from these options:\n"
                f"{options_str}\n\n"
                f"Reply with one order per line, nothing else.\nOrders:\n"
            )

            resp = _client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()

            valid_set: set = set()
            for loc in orderable_locs:
                if loc in possible_orders:
                    valid_set.update(possible_orders[loc])

            parsed = [line.strip() for line in text.split("\n") if line.strip()]
            valid_orders = [o for o in parsed if o in valid_set]

            covered_locs = set()
            for o in valid_orders:
                for loc in orderable_locs:
                    if loc in possible_orders and o in possible_orders[loc]:
                        covered_locs.add(loc)
                        break

            for loc in orderable_locs:
                if loc not in covered_locs and loc in possible_orders and possible_orders[loc]:
                    valid_orders.append(random.choice(possible_orders[loc]))

            return valid_orders

        except Exception as exc:
            _log.debug("API partner orders failed for %s: %s", power_name, exc)
            return _random_partner_orders(game, power_name)

    return _api_partner_orders


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class DiplomacyNLWrapper:
    """
    Gym-style wrapper for the Diplomacy game engine with natural-language observations
    and actions.

    Multi-agent mode (default):
        - reset() -> (obs_dict, info)   where obs_dict = {power_name: nl_string}
        - step(actions) -> (obs_dict, rewards_dict, terminated, truncated, info)
          actions: {power_name: orders} where orders is List[str] or a single str

    Single-agent mode (controlled_power is set):
        - reset() -> (obs_str, info)
        - step(orders) -> (obs_str, reward, terminated, truncated, info)
          Other powers use partner_policy.

    Each call to step() processes one full game phase (submit orders for all powers,
    then process). The game advances through Spring Movement -> Spring Retreat ->
    Fall Movement -> Fall Retreat -> Fall Adjustment -> next year.
    """

    def __init__(
        self,
        controlled_power: Optional[str] = None,
        partner_policy: Optional[Callable] = None,
        map_name: str = "standard",
        max_phases: int = 100,
        seed: int = 42,
        vllm_base_urls: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        skill_bank: Any = None,
        opponent_model: Optional[str] = None,
        opponent_api_base: Optional[str] = None,
    ):
        """
        Args:
            controlled_power: If set (e.g. "FRANCE"), single-agent mode.
                If None, multi-agent mode (all 7 powers controlled externally).
            partner_policy: Callable(game, power_name) -> List[str] for non-controlled
                powers in single-agent mode.
                If None and opponent_model is set, uses API policy.
                If None and vllm_base_urls is set, uses LLM policy.
                If None and no URLs, uses random orders.
            map_name: Diplomacy map name (default "standard").
            max_phases: Maximum number of phases before truncation.
            seed: Random seed.
            vllm_base_urls: vLLM server URLs for LLM-based partner policy.
            model_name: Model name for vLLM requests.
            skill_bank: Per-game skill bank for opponent skill hints.
            opponent_model: External API model name for opponents (e.g.
                "gpt-5-mini").  Takes priority over vLLM self-play.
            opponent_api_base: API base URL (default: OpenRouter).
        """
        if Game is None:
            _msg = (
                "Cannot import diplomacy.Game. "
                "Either install the diplomacy package (pip install diplomacy>=1.1.2) or use the "
                "local AI_Diplomacy copy: run from codebase root so AI_Diplomacy is on PYTHONPATH, "
                "and install AI_Diplomacy dependencies (e.g. pip install coloredlogs; see AI_Diplomacy/)."
            )
            _err = globals().get("_dip_import_error")
            if _err is not None:
                _msg += f" Original error: {_err}"
            raise ImportError(_msg)
        self._controlled_power = controlled_power.upper() if controlled_power else None
        self._multi_agent = controlled_power is None
        if partner_policy is not None:
            self._partner_policy = partner_policy
        elif opponent_model:
            self._partner_policy = _make_api_partner_policy(
                opponent_model,
                api_base=opponent_api_base or "https://openrouter.ai/api/v1",
            )
        elif vllm_base_urls and model_name:
            self._partner_policy = _make_llm_partner_policy(
                vllm_base_urls, model_name,
                skill_bank=skill_bank, game_name="diplomacy",
            )
        else:
            self._partner_policy = _random_partner_orders
        self._map_name = map_name
        self._max_phases = max_phases
        self._seed = seed

        self.game: Optional[Game] = None
        self._phases_processed: int = 0
        self._phase_history: List[str] = []
        self._negotiation_log: List[str] = []

    # ---- Gym-like interface ----

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        """
        Reset the Diplomacy game.

        Returns:
            Multi-agent:  ({power_name: nl_str, ...}, info)
            Single-agent: (nl_str, info)
        """
        if seed is not None:
            self._seed = seed

        self.game = Game(map_name=self._map_name, seed=self._seed)
        self._phases_processed = 0
        self._phase_history = []
        self._negotiation_log = []
        self._prev_centers = None

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self,
        action: Union[Dict[str, Any], List[str], str, None] = None,
    ) -> Tuple[Any, Any, bool, bool, dict]:
        """
        Submit orders for the current phase and process it.

        Args:
            Multi-agent:  action = {power_name: orders} where orders is List[str] or str.
            Single-agent: action = List[str] or str (orders for controlled power).

        Returns:
            Multi-agent:  (obs_dict, rewards_dict, terminated, truncated, info)
            Single-agent: (obs_str, reward_float, terminated, truncated, info)
        """
        if self.game is None or self.game.is_game_done:
            raise RuntimeError("Game is not running. Call reset() first.")

        current_phase = self.game.get_current_phase()

        if self._multi_agent:
            actions = action if isinstance(action, dict) else {}
            self._submit_orders_multi(actions)
        else:
            self._submit_orders_single(action)

        # Record phase info before processing
        pre_phase = current_phase

        # Process the phase
        self.game.process()
        self._phases_processed += 1

        # Record what happened
        self._record_phase_result(pre_phase)

        terminated = self.game.is_game_done
        truncated = self._phases_processed >= self._max_phases

        obs = self._build_obs()
        info = self._build_info()
        rewards = self._build_rewards()

        if self._multi_agent:
            return obs, rewards, terminated, truncated, info
        else:
            cp = self._controlled_power
            raw = rewards.get(cp, 0.0)
            shaped = raw + self._shaping_delta(cp)
            return obs, shaped, terminated, truncated, info

    def step_negotiate(
        self,
        messages: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> Tuple[Any, dict]:
        """
        Exchange negotiation messages (optional, before submitting orders).

        This is a separate step that does NOT advance the game phase. It only
        records messages that will appear in subsequent observations.

        Args:
            messages: {sender_power: [{"recipient": power_or_"GLOBAL", "content": str}, ...]}

        Returns:
            (obs, info) with updated observations including the new messages.
        """
        if messages:
            for sender, msg_list in messages.items():
                for msg in msg_list:
                    recipient = msg.get("recipient", "GLOBAL")
                    content = msg.get("content", "")
                    self._negotiation_log.append(f"{sender} -> {recipient}: {content}")

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    # ---- Order submission ----

    def _submit_orders_multi(self, actions: Dict[str, Any]) -> None:
        """Submit orders for all powers from the actions dict."""
        for power_name, power in self.game.powers.items():
            if power.is_eliminated():
                continue
            raw_orders = actions.get(power_name, None)
            if raw_orders is not None:
                orders = parse_orders(raw_orders, self.game, power_name)
            else:
                # No orders provided -> random fallback
                orders = _random_partner_orders(self.game, power_name)
            if orders:
                self.game.set_orders(power_name, orders)

    def _submit_orders_single(self, action: Any) -> None:
        """Submit orders for controlled power + partner policy for others."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        controlled_orders = None
        partner_powers = []
        for power_name, power in self.game.powers.items():
            if power.is_eliminated():
                continue
            if power_name == self._controlled_power:
                controlled_orders = (power_name, parse_orders(action, self.game, power_name))
            else:
                partner_powers.append(power_name)

        if controlled_orders:
            pn, orders = controlled_orders
            if orders:
                self.game.set_orders(pn, orders)

        if partner_powers:
            shared_ctx = _build_shared_board_context(self.game)
            all_orders = self.game.get_all_possible_orders()

            def _call_partner(pn):
                try:
                    return self._partner_policy(
                        self.game, pn,
                        shared_ctx=shared_ctx,
                        all_possible_orders=all_orders,
                    )
                except TypeError:
                    return self._partner_policy(self.game, pn)

            with ThreadPoolExecutor(max_workers=len(partner_powers)) as pool:
                futures = {
                    pool.submit(_call_partner, pn): pn
                    for pn in partner_powers
                }
                for fut in as_completed(futures):
                    pn = futures[fut]
                    try:
                        orders = fut.result()
                    except Exception:
                        orders = _random_partner_orders(self.game, pn)
                    if orders:
                        self.game.set_orders(pn, orders)

    # ---- Phase history ----

    def _record_phase_result(self, phase_name: str) -> None:
        """Record a summary of what happened after processing a phase."""
        sc_summary = []
        for pname, pobj in self.game.powers.items():
            if not pobj.is_eliminated():
                sc_summary.append(f"{pname}:{len(pobj.centers)}")
        self._phase_history.append(
            f"[{phase_name}] Processed. Supply centers: {', '.join(sc_summary)}"
        )
        # Clear negotiation log for next phase
        self._negotiation_log = []

    # ---- Observations ----

    def _build_obs(self) -> Any:
        """Build observation(s) for the current state."""
        if self.game is None:
            return {} if self._multi_agent else ""
        if self._multi_agent:
            return state_to_natural_language_for_all(
                self.game,
                negotiation_messages=self._negotiation_log,
                phase_history=self._phase_history,
            )
        else:
            return state_to_natural_language(
                self.game,
                self._controlled_power,
                negotiation_messages=self._negotiation_log,
                phase_history=self._phase_history,
            )

    # ---- Rewards ----

    def _build_rewards(self) -> Dict[str, float]:
        """
        Rewards for all powers. Based on supply center count normalized by 18
        (solo victory threshold). Eliminated powers get -1.0.
        """
        rewards = {}
        for power_name, power in self.game.powers.items():
            if power.is_eliminated():
                rewards[power_name] = -1.0
            else:
                rewards[power_name] = len(power.centers) / 18.0
        return rewards

    def _shaping_delta(self, power_name: str) -> float:
        """Potential-based shaping: reward delta for gaining/losing centers.

        ``delta_centers * 0.5`` per phase so the agent gets dense signal
        from territorial changes.  Policy-invariant (potential-based).
        """
        power = self.game.powers.get(power_name)
        if power is None:
            return 0.0
        current = len(power.centers)
        prev = getattr(self, "_prev_centers", None)
        if prev is None:
            self._prev_centers = current
            return 0.0
        delta = (current - prev) * 0.5
        self._prev_centers = current
        return delta

    # ---- Info ----

    def _build_info(self) -> dict:
        """Build info dict with game metadata."""
        if self.game is None:
            return {}

        current_phase = self.game.get_current_phase()
        phase_type = current_phase[-1] if current_phase and current_phase not in ("FORMING", "COMPLETED") else ""

        powers_info = {}
        for pname, pobj in self.game.powers.items():
            powers_info[pname] = {
                "units": list(pobj.units),
                "centers": list(pobj.centers),
                "homes": list(pobj.homes) if pobj.homes else [],
                "num_centers": len(pobj.centers),
                "eliminated": pobj.is_eliminated(),
                "retreats": dict(pobj.retreats) if pobj.retreats else {},
            }

        # Orderable locations per power
        active_powers = {}
        for pname, pobj in self.game.powers.items():
            if not pobj.is_eliminated():
                locs = self.game.get_orderable_locations(pname) or []
                if locs:
                    active_powers[pname] = locs

        # Structured summary for the primary power
        primary_power = self._controlled_power or next(iter(self.game.powers), None)
        structured = (
            build_structured_state_summary(self.game, primary_power)
            if primary_power else {}
        )

        info = {
            "env_name": "diplomacy",
            "game_name": "diplomacy",
            "phase": current_phase,
            "phase_type": phase_type,
            "phases_processed": self._phases_processed,
            "max_phases": self._max_phases,
            "is_game_done": self.game.is_game_done,
            "powers": powers_info,
            "active_powers": active_powers,
            "phase_history": list(self._phase_history),
            "negotiation_log": list(self._negotiation_log),
            "order_history": dict(self.game.order_history) if self.game.order_history else {},
            "structured_state": structured,
        }

        # Add possible orders (can be large; include for active powers only)
        possible_orders = self.game.get_all_possible_orders()
        info["possible_orders"] = {}
        for pname, locs in active_powers.items():
            info["possible_orders"][pname] = {
                loc: possible_orders.get(loc, []) for loc in locs
            }

        return info

    def get_map_svg(self) -> Optional[str]:
        """
        Render the current game map to SVG for the web visualizer.
        Returns None if rendering fails (e.g. game not initialized or renderer unavailable).
        """
        if self.game is None:
            return None
        try:
            from diplomacy.engine.renderer import Renderer
            renderer = Renderer(self.game)
            svg_content = renderer.render(output_path=None, incl_abbrev=True)
            if svg_content:
                try:
                    from games.games.diplomacy.utils import add_legend_to_svg
                    svg_content = add_legend_to_svg(svg_content, renderer.metadata.get("color", {}))
                except Exception:
                    pass
            return svg_content
        except Exception:
            return None

    # ---- Properties ----

    @property
    def powers(self) -> List[str]:
        """List of power names in the game."""
        if self.game:
            return list(self.game.powers.keys())
        return DEFAULT_POWERS

    @property
    def num_agents(self) -> int:
        return len(self.powers)

    @property
    def multi_agent(self) -> bool:
        return self._multi_agent

    @property
    def done(self) -> bool:
        if self.game is None:
            return True
        return self.game.is_game_done or self._phases_processed >= self._max_phases

    @property
    def action_space(self):
        """Informal; actions are lists of order strings. See info['possible_orders']."""
        return None

    @property
    def observation_space(self):
        """Observations are NL strings; no formal gym Space."""
        return None
