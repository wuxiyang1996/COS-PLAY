#!/usr/bin/env python
"""
Cold-start actor-agent rollouts for **BrowserGym** (gpt-5.5 vision pipeline).

Pipeline (one outer step):

  1. Pull a multimodal observation from a real BrowserGym env (Playwright +
     headless Chromium): PIL screenshot, AXTree,
     ``extra_element_properties``, focused element, URL, last action / error.
     Two target modes are supported:

       - ``--tasks <env_id>``  pre-registered BrowserGym task ids
                              (e.g. ``browsergym/miniwob.click-button``,
                              ``browsergym/webarena.42``,
                              ``browsergym/visualwebarena.0``,
                              ``browsergym/assistantbench.test.0``).
                              The task ships with its own goal / success
                              criterion / reward.
       - ``--urls <url>``      open-ended browsing on top of
                              ``browsergym/openended`` (no built-in
                              task — reward is always 0, useful for
                              data collection on arbitrary live pages).

  2. Visual schema (deterministic fallback) — :func:`browsergym_wrapper
     .heuristic.obs_to_schema` walks the AXTree into the canonical
     ``<state>...</state>`` block. Free, fast, always available.
  3. Visual schema (VLM, primary) — gpt-5.5 (vision) reads the **screenshot**
     plus the AXTree-as-text grounding context and emits the canonical schema
     via :mod:`vlm_wrapper.schema`. The screenshot is the primary input.
  4. Action selection — gpt-5.5 reads the schema + the BrowserGym candidate
     actions list (``click(bid)`` / ``fill(bid, "...")`` / ``scroll`` /
     ``go_back`` / ``noop``) and picks ONE action via OpenAI function
     calling. The function-call schema separates ``action_type`` / ``bid`` /
     ``text`` so the actor can also fill text into an input box.
  5. ``env.step(action_string)`` and an :class:`Experience` is appended to
     the :class:`Episode`. The schema, raw VLM output, action reasoning,
     reward, terminate / truncate flags, and image path are all preserved
     on the Experience for SFT/GRPO consumers.

Companion to ``cold_start/generate_cold_start_actor_gymv.py`` (gym-v) and
``cold_start/generate_cold_start_actor.py`` (env_wrappers) — same Episode/
Experience output format, but driven through BrowserGym's live screenshot +
AXTree obs API. Real Chromium is required: there is no offline / synthetic
mode. Install via ``install/install_browsergym.sh``.

Output layout (``<codebase_root>/Cold-start-out-browsergym/<safe_id>/``):

  - ``episode_NNN.json``       individual Episode (Episode.to_dict())
  - ``episode_buffer.json``    Episode_Buffer (loadable for trainer)
  - ``rollouts.jsonl``         append-only JSONL, one Episode per line
  - ``rollout_summary.json``   per-target stats
  - ``frames/<ep>/step_NNN.png``  rendered frames sent to the VLM (debug)

Usage::

    export OPENAI_API_KEY="sk-..."          # or OPENROUTER_API_KEY

    # Default: 1 episode against Google + Wikipedia (openended), 8 steps each
    python cold_start/generate_cold_start_actor_browsergym.py

    # Real benchmark tasks (one episode each)
    python cold_start/generate_cold_start_actor_browsergym.py \\
        --tasks browsergym/miniwob.click-button \\
                browsergym/miniwob.enter-text \\
                browsergym/assistantbench.test.0 \\
        --episodes 1 --max_steps 12 --save_frames -v

    # Discover the registered task IDs (no rollout, just a listing)
    python cold_start/generate_cold_start_actor_browsergym.py --list_tasks

    # Custom open-ended URLs
    python cold_start/generate_cold_start_actor_browsergym.py \\
        --urls https://en.wikipedia.org/wiki/Reinforcement_learning \\
        --episodes 2 --max_steps 12 --save_frames -v

    # Visible (non-headless) Chromium for debugging
    python cold_start/generate_cold_start_actor_browsergym.py --no_headless

    # Cheap baseline: skip the vision call (AXTree-walked heuristic schema only)
    python cold_start/generate_cold_start_actor_browsergym.py --no_vision

Suite infra requirements (set ONLY if the corresponding suite is in --tasks):

    miniwob          MINIWOB_URL=file:///path/to/miniwob-plusplus/html/miniwob/
    webarena         WA_HOMEPAGE / WA_SHOPPING / WA_REDDIT / WA_GITLAB /
                     WA_WIKIPEDIA / WA_MAP — see github.com/web-arena-x/webarena
    visualwebarena   VWA_HOMEPAGE / VWA_CLASSIFIEDS / VWA_SHOPPING /
                     VWA_REDDIT — see github.com/web-arena-x/visualwebarena
    assistantbench   no extra infra (loads from HuggingFace dataset cache)
    openended        no extra infra (any live URL works)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make the codebase + workspace root importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in [str(CODEBASE_ROOT), str(WORKSPACE_ROOT)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_api_keys_from_file() -> Optional[Path]:
    """Seed ``os.environ`` from a sibling ``api_keys.py`` if present.

    Looked-up locations (first hit wins):
      1. ``$COSPLAY_API_KEYS_FILE``
      2. ``cold_start/api_keys.py``
      3. ``<codebase_root>/api_keys.py``
      4. ``<codebase_root>/../api_keys.py``  (workspace root)

    The file should define any of:
      ``openrouter_api_key``, ``openai_api_key``, ``claude_api_key``
    Existing env vars are NOT overwritten.
    """
    import importlib.util

    candidates: List[Path] = []
    env_override = os.environ.get("COSPLAY_API_KEYS_FILE")
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend([
        SCRIPT_DIR / "api_keys.py",
        CODEBASE_ROOT / "api_keys.py",
        CODEBASE_ROOT.parent / "api_keys.py",
    ])

    for path in candidates:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        try:
            spec = importlib.util.spec_from_file_location("_cosplay_api_keys", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to load api_keys file {path}: {exc}", file=sys.stderr)
            continue

        mapping = {
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
            "claude_api_key": "ANTHROPIC_API_KEY",
        }
        for attr, env_name in mapping.items():
            val = getattr(mod, attr, None)
            if isinstance(val, str) and val.strip() and not os.environ.get(env_name):
                os.environ[env_name] = val.strip()
        return path
    return None


_API_KEYS_FILE_USED = _bootstrap_api_keys_from_file()


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

import openai

try:
    from common.models import BACKBONE_SFT_TEACHER_MODEL as _SFT_TEACHER_MODEL
except Exception:  # pragma: no cover — keep script runnable in isolation
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_MODEL = _SFT_TEACHER_MODEL  # gpt-5.5

try:
    from API_func import OPENROUTER_BASE, make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    make_openai_client = None
    effective_openai_model = None

logger = logging.getLogger("cold_start.actor_browsergym")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_URLS: List[str] = [
    "https://www.google.com",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
]

# Optional task suites to import. Each pulls in its own registered env ids
# (browsergym/miniwob.<task>, browsergym/webarena.<id>, …). Failures are
# logged once and the suite is dropped — most installs only have a subset.
_OPTIONAL_TASK_SUITE_MODULES: List[str] = [
    "browsergym.miniwob",
    "browsergym.webarena",
    "browsergym.visualwebarena",
    "browsergym.assistantbench",
    "browsergym.workarena",
    # In-tree bridge: WebShop's Flask server fronted as
    # ``browsergym/webshop.<goal_idx>`` envs (see webshop_wrapper/README.md).
    # Importing the package side-effect-registers the gym ids; if the
    # WebShop server isn't running the import still succeeds, the
    # WebShopTask just fails at reset() time with a clear connection
    # error pointing at install/install_webshop.sh.
    "webshop_wrapper",
]


# ---------------------------------------------------------------------------
# Target = (kind, payload, safe_id)
#
# kind == "task"   : payload is a registered BrowserGym env id
#                    (e.g. "browsergym/miniwob.click-button"). The task
#                    carries its own goal + reward function — we just call
#                    gym.make(env_id).
# kind == "url"    : payload is a live URL. We boot browsergym/openended
#                    with task_kwargs={"start_url": payload}.
# safe_id          : filesystem-safe slug used for the per-target output dir.
# ---------------------------------------------------------------------------

# How many outer steps per episode.
#
# Bumped from 8 to 30 on 2026-05-03 to match the VWA / WebArena
# literature default. ``max_steps=8`` was a mini-WoB-era number that
# starves multi-constraint search-and-filter tasks (e.g. classifieds
# "find the most expensive TV from Maryland that displays an ongoing
# NFL game") of action budget — the canonical solve path is 5 actions
# of real work, plus 5 of recovery, plus 2 of verification. Empirical
# diagnostic on 2026-05-03 (visualwebarena.92, gpt-5.5 low) showed the
# agent thrashing through ``scroll/go_back/go_forward`` for 11 of 12
# steps under the old budget. See ``legacy/visualwebarena/
# vwa-improvement-plan.md`` §3 (Tier-1 change A).
DEFAULT_MAX_STEPS = 30
# Default episode count per URL/target when ``--episodes`` is not given.
DEFAULT_EPISODES = 1
# Anti-noop: force a different action after this many consecutive steps
# whose URL+focused-bid is identical AND no error was raised.
_MAX_CONSECUTIVE_NOOPS = 2
# Anti-error: force a different action_type/bid after this many consecutive
# steps that hit ``last_action_error`` with the same action.
_MAX_CONSECUTIVE_ERRORS = 2
# Anti-thrash: when the agent has done ``_MAX_CONSECUTIVE_NAV`` actions in a
# row from the navigation-only set ({scroll, go_back, go_forward, noop}) AND
# the current page surfaces a ``fill(...)`` candidate, force the next
# action to be that fill with a goal-derived query. Catches the post-
# search blocked-page recovery pattern surfaced in the May-3 VWA
# visualwebarena.92 diagnostic, where gpt-5.5 low looped through 28 nav-
# only steps without realising the bid for the search box had been re-
# numbered after the Magento interstitial. Set to 3 — small enough to fire
# fast on real thrashing, large enough to not interrupt legitimate scroll-
# to-find loops on pages without input candidates.
_MAX_CONSECUTIVE_NAV = 3
_NAV_ONLY_PREFIXES = ("scroll(", "go_back(", "go_forward(", "noop(")
# Anti-repetition (#6e): track action signatures in a sliding window and
# discourage any signature that has appeared ``_MAX_REPEATS_BEFORE_DISCOURAGE``
# times in the last ``_REPEAT_WINDOW`` steps. Discouraged actions are dropped
# from the candidate-action list shown to the action LLM; if the LLM still
# picks one (off-list), the rollout-loop swap-fallback (#6e) substitutes a
# non-discouraged candidate. Catches the May-3 ``visualwebarena.96`` pattern
# (``click("211")`` repeated 7× → ``go_back()`` repeated 10× → 73 % of steps
# wasted on repeats) and the ``visualwebarena.433`` pattern
# (``fill("54", "f/music")`` + ``press("54", "Enter")`` repeated 4× each).
#
# **Protected** action types (NOT discouraged even if they exceed the
# threshold): ``go_back``, ``go_forward``, ``noop``. These are the agent's
# recovery escape hatches — discouraging them removes its only path back
# from a dead-end click. ``scroll`` IS discouraged (the existing
# consecutive-NOOP override at #6 handles back-to-back identical scrolls;
# this window-based mechanism additionally catches ``scroll(+) → click →
# scroll(+) → click → scroll(+)`` interleaved loops).
_REPEAT_WINDOW = 8
_MAX_REPEATS_BEFORE_DISCOURAGE = 2
_REPEAT_PROTECTED_PREFIXES = ("go_back(", "go_forward(", "noop(")
# Minimum candidates to keep after filtering — backs off discouragement if
# every candidate would be filtered out.
_MIN_CANDIDATES_AFTER_FILTER = 3
# Number of recent action results to surface in the action-selection prompt.
_HISTORY_WINDOW = 5
# Substrings (case-insensitive) on a node's text/role that mark it as a
# cookie / consent / GDPR dismissal button. The actor pre-empts the LLM and
# auto-clicks the first such bid to unblock benchmarks (esp. assistantbench
# starting on google.com behind a consent wall).
#
# NOTE: We accept BOTH "accept all" and "reject all" / "deny" variants —
# either one closes Google's consent dialog and lets the agent proceed.
# Reject is actually preferred (privacy-preserving + still skips the wall),
# so its keywords are listed first so they win the rank-sort tiebreak.
_CONSENT_ACCEPT_KEYWORDS = (
    "reject all", "reject cookies", "decline all", "decline",
    "tout refuser",                  # fr  reject
    "alle ablehnen", "ablehnen",     # de  reject
    "rifiuta tutto",                 # it  reject
    "rechazar todo",                 # es  reject
    "rejeitar tudo",                 # pt  reject
    "全部拒否", "拒否",                # ja  reject
    "全部拒绝", "拒绝",                # zh  reject
    "모두 거부", "거부",               # ko  reject
    "accept all", "accept cookies", "i agree", "i accept", "agree to all",
    "agree all", "agree", "got it", "allow all", "allow cookies",
    "tout accepter",                 # fr
    "alle akzeptieren", "akzeptieren", "akzeptiere",  # de
    "accetta tutto", "accetto",      # it
    "aceptar todo", "aceptar",       # es
    "aceitar tudo", "aceitar",       # pt
    "hyväksy kaikki", "hyväksy",     # fi
    "godta alle", "godta",           # no
    "godkänn alla", "godkänn",       # sv
    "acepteer alle", "acepteer",     # nl approx
    "全部承諾", "承諾", "全て承認",    # ja
    "全部同意", "同意",               # zh
    "동의", "모두 동의",               # ko
)
# Default token budgets (output cap).
_ACTION_MAX_TOKENS = 400
_SCHEMA_MAX_TOKENS = 4000
# Reasoning models burn output tokens on hidden thinking — give them more.
_SCHEMA_MAX_TOKENS_REASONING = 12000
# Cap on entities per schema (keeps the action prompt focused).
_DEFAULT_MAX_ENTITIES = 25
# Cap on candidate-action list emitted to the actor LLM.
_MAX_CANDIDATE_ACTIONS = 18

# Models that require ``max_completion_tokens`` (no ``temperature``).
_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:gpt-5(?:[\.\-]\w+)?|o[134](?:[\.\-]\w+)?)(?:$|[^\w])",
    re.IGNORECASE,
)


def _is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI-style reasoning models (gpt-5.x, o1/o3/o4)."""
    if not model:
        return False
    return bool(_REASONING_MODEL_RE.search(model))


def _url_safe(url: str) -> str:
    """Best-effort filesystem-safe slug for a URL."""
    s = (url or "").lower()
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    out: List[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in "-._/":
            out.append("_")
    slug = "".join(out).strip("_")[:120] or "page"
    return slug


def _task_safe(env_id: str) -> str:
    """Filesystem-safe slug for a BrowserGym task id (``browsergym/foo.42``)."""
    s = env_id.replace("browsergym/", "", 1).replace("/", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-._":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")[:140] or "task"


def _import_optional_task_suites() -> Tuple[List[str], List[str]]:
    """Import all known BrowserGym task suite packages, returning (ok, fail).

    Each successful import registers ``browsergym/<suite>.*`` env ids in
    Gymnasium's global registry.  Suites that fail to import are simply
    skipped — the user only needs the subset they're targeting via
    ``--tasks``.  ``browsergym.core`` (which registers ``openended``) is
    imported separately, eagerly, in ``main()``.
    """
    ok: List[str] = []
    fail: List[str] = []
    for mod_name in _OPTIONAL_TASK_SUITE_MODULES:
        try:
            __import__(mod_name)
            ok.append(mod_name)
        except Exception as exc:
            fail.append(f"{mod_name}: {type(exc).__name__}: {str(exc)[:120]}")
    return ok, fail


def _list_registered_task_ids() -> Dict[str, List[str]]:
    """Return registered BrowserGym task ids bucketed by suite prefix."""
    import gymnasium as gym
    buckets: Dict[str, List[str]] = {}
    for k in gym.envs.registry.keys():
        if not k.startswith("browsergym/"):
            continue
        prefix = k.split("/", 1)[1].split(".")[0] or "openended"
        buckets.setdefault(prefix, []).append(k)
    for v in buckets.values():
        v.sort()
    return buckets


def _suite_of(env_id: str) -> str:
    """``browsergym/miniwob.click-button`` -> ``"miniwob"``."""
    if not env_id.startswith("browsergym/"):
        return ""
    return env_id.split("/", 1)[1].split(".")[0]


# (suite, env_var, hint) — preflight check on ``--tasks``: if any task in
# the suite is requested and the env var is missing, fail fast with a clear
# install message instead of crashing inside env.reset() per-episode.
_SUITE_INFRA_REQUIREMENTS: List[Tuple[str, List[str], str]] = [
    ("miniwob",
     ["MINIWOB_URL"],
     "MiniWoB++ HTML pages are not bundled with the pip package. Clone "
     "https://github.com/Farama-Foundation/miniwob-plusplus.git (frozen "
     "commit 7fd85d71a4b60325c6585396ec4f48377d049838) and export "
     "MINIWOB_URL=file:///path/to/miniwob-plusplus/miniwob/html/miniwob/ . "
     "The launcher script auto-resolves common locations."),
    ("webarena",
     # Exact list from browsergym.webarena's __init__ assertion (7 vars,
     # all required even if a particular task only touches one site).
     ["WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT", "WA_GITLAB",
      "WA_WIKIPEDIA", "WA_MAP", "WA_HOMEPAGE"],
     "WebArena requires the self-hosted Shopping / Shopping-Admin / "
     "Reddit / GitLab / Wikipedia / Map / Homepage Docker images. "
     "See github.com/web-arena-x/webarena#setup-instructions."),
    ("visualwebarena",
     # Exact list from browsergym.visualwebarena's __init__ assertion.
     ["VWA_SHOPPING", "VWA_REDDIT", "VWA_WIKIPEDIA", "VWA_HOMEPAGE",
      "VWA_CLASSIFIEDS", "VWA_CLASSIFIEDS_RESET_TOKEN"],
     "VisualWebArena requires the self-hosted Shopping / Reddit / Wikipedia "
     "/ Homepage / Classifieds Docker images plus a reset token. "
     "See github.com/web-arena-x/visualwebarena."),
    # assistantbench loads from the HF dataset cache and points the browser at
    # public URLs (e.g. google.com); no extra env vars are needed.
    # openended just needs a --urls argument; nothing to check here.
]


def _preflight_task_infra(task_ids: List[str]) -> List[str]:
    """Return a list of error strings; empty list means infra is OK."""
    if not task_ids:
        return []
    needed = {_suite_of(t) for t in task_ids}
    errors: List[str] = []
    for suite, env_vars, hint in _SUITE_INFRA_REQUIREMENTS:
        if suite not in needed:
            continue
        missing = [v for v in env_vars if not os.environ.get(v)]
        if missing:
            errors.append(
                f"  [{suite}] missing env var(s): {', '.join(missing)}\n"
                f"    {hint}"
            )
    return errors


# ---------------------------------------------------------------------------
# Image / observation helpers
# ---------------------------------------------------------------------------

def _to_pil(image: Any):
    """Coerce ``obs['screenshot']`` (PIL/np/bytes) into a single PIL RGB."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def _save_frame(image: Any, path: Path) -> Optional[str]:
    pil = _to_pil(image)
    if pil is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pil.save(str(path), format="PNG")
        return str(path)
    except Exception:
        return None


def _flatten_axtree(obs: Dict[str, Any], max_chars: int = 3000) -> str:
    """Flatten ``obs['axtree_object']`` into a compact text dump."""
    axtree = obs.get("axtree_object")
    if axtree is None:
        return ""

    try:
        from browsergym.utils.obs import flatten_axtree_to_str  # type: ignore
        text = flatten_axtree_to_str(
            axtree, extra_properties=obs.get("extra_element_properties", {}),
        )
    except Exception:
        lines: List[str] = []
        for node in axtree.get("nodes", [])[:80]:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            bid = node.get("browsergym_id", "")
            if not (role and (name or bid)):
                continue
            lines.append(f"[{bid}] {role}: {name[:60]}".rstrip())
        text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _extract_goal(obs: Dict[str, Any]) -> str:
    goal = obs.get("goal", "") or ""
    if not goal:
        goal_obj = obs.get("goal_object", ()) or ()
        goal = " ".join(
            m.get("text", "") for m in goal_obj if m.get("type") == "text"
        )
    return goal.strip() or "Explore the page"


def _extract_goal_images(obs: Dict[str, Any]) -> List[Any]:
    """Pull goal-side images out of ``goal_object``.

    BrowserGym's VisualWebArena task fills ``goal_object`` with OpenAI-
    style multimodal content: a sequence of dicts including
    ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
    entries that describe the visual reference (e.g. "buy a chair like
    *this image*"). The base actor flow drops these on the floor when it
    flattens the goal to text — meaning the VLM never sees the reference
    image and the task is unsolvable. This helper reconstructs the PIL
    images from the data URIs so we can forward them as additional image
    parts in the user message.

    Returns an (possibly empty) list of PIL.Image objects in goal order.
    """
    goal_obj = obs.get("goal_object") or ()
    if not goal_obj:
        return []
    try:
        from PIL import Image
    except ImportError:
        return []
    import base64

    images: List[Any] = []
    for entry in goal_obj:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "image_url":
            continue
        url = (entry.get("image_url") or {}).get("url") or ""
        if not url.startswith("data:"):
            # Skip remote URLs — fetching them at agent step time would
            # add a lot of latency and is not needed for VWA which
            # always inlines as data URIs.
            continue
        try:
            _header, _, b64 = url.partition(",")
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            logger.debug("goal image decode failed: %s", exc)
            continue
        images.append(img)
    return images


def _count_som_telemetry(obs: Dict[str, Any]) -> Dict[str, int]:
    """Cheap, allocation-free probe of how rich the SoM annotation is.

    The Set-of-Marks overlay only matters for the actor if the
    underlying ``extra_element_properties`` actually flags interactable
    elements with ``set_of_marks=True``. Empirically (VWA diagnostic
    2026-05-03) some pages — especially fallback ``about:blank`` and
    a few VWA classifieds list views — return populated extras with
    **zero** ``set_of_marks`` flags, which silently degrades SoM-on
    behaviour to the same as SoM-off plus an unnecessary overlay
    render. We surface this as per-episode telemetry so a parent run
    can ``grep '\\[SOM WARN\\]'`` and immediately tell whether to
    investigate the agent's poor pass-rate on a given task.
    """
    extras = obs.get("extra_element_properties") or {}
    n_total = len(extras)
    n_som = 0
    n_clickable = 0
    for v in extras.values():
        if not isinstance(v, dict):
            continue
        if v.get("set_of_marks"):
            n_som += 1
        if v.get("clickable"):
            n_clickable += 1

    # Role lives on the AXTree node, NOT in ``extra_element_properties``,
    # so cross-reference the two by browsergym_id. Skip silently if either
    # side is missing/malformed.
    n_input = 0
    axtree = obs.get("axtree_object") or {}
    nodes = axtree.get("nodes", []) if isinstance(axtree, dict) else []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        bid = node.get("browsergym_id")
        if not bid or bid not in extras:
            continue
        role_field = node.get("role") or {}
        role = role_field.get("value") if isinstance(role_field, dict) else ""
        if role in ("textbox", "searchbox", "combobox", "spinbutton"):
            n_input += 1

    return {
        "n_extras": n_total,
        "n_set_of_marks": n_som,
        "n_clickable": n_clickable,
        "n_input_role": n_input,
    }


def _render_som_screenshot(obs: Dict[str, Any]) -> Optional[Any]:
    """Return a Set-of-Marks-annotated PIL image of the current viewport.

    Uses BrowserGym's built-in ``overlay_som`` helper, which draws a
    dashed bounding box + black tag containing the bid for every
    interactable element flagged with ``set_of_marks=True`` in
    ``extra_element_properties``. With this overlay the VLM can read
    bids straight off the pixels rather than having to cross-reference
    the AXTree text — empirically the single biggest leverage point for
    web-VLM action accuracy (WebVoyager / SeeAct / Aria-UI all rely on
    SoM).

    Returns ``None`` if the screenshot or extra_element_properties are
    unavailable, so the caller can fall back to the plain screenshot.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return None
    extras = obs.get("extra_element_properties") or {}
    try:
        from browsergym.utils.obs import overlay_som  # type: ignore
    except Exception as exc:
        logger.debug("overlay_som unavailable: %s", exc)
        return None
    try:
        arr = np.asarray(screenshot)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            return None
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        som_img = overlay_som(arr, extras)
        return som_img.convert("RGB") if hasattr(som_img, "convert") else som_img
    except Exception as exc:
        logger.debug("overlay_som failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# OpenAI client / model routing
# ---------------------------------------------------------------------------

def _build_client_and_route(
    *, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """Return ``(client, routed_model)`` or ``(None, model)`` on failure."""
    client = None
    if make_openai_client is not None:
        try:
            client = make_openai_client(api_key=api_key, base_url=base_url)
        except Exception:
            client = None

    if client is None:
        kw: Dict[str, Any] = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        try:
            client = openai.OpenAI(**kw) if kw else openai.OpenAI()
        except Exception:
            return None, model

    if api_key or base_url:
        return client, model
    if effective_openai_model is not None:
        try:
            return client, effective_openai_model(model)
        except Exception:
            return client, model
    return client, model


_VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high")


def _maybe_disable_thinking_kwargs(model: Any, tool_choice: Any) -> Dict[str, Any]:
    """Return ``extra_body`` kwargs to disable Qwen ``thinking`` mode.

    Qwen3/3.5/3.6 multimodal flagships (e.g. ``qwen/qwen3.5-plus-20260420``,
    ``qwen/qwen3.6-plus``) ship with thinking-mode ON.  When a strict
    ``tool_choice={"type":"function",...}`` payload is sent, the upstream
    DashScope endpoint rejects it with HTTP 400
    (``InvalidParameter ... in thinking mode``).  Our actor pipeline
    relies on strict tool-choice to force structured action JSON, so the
    only safe option is to turn thinking OFF.

    We forward both supported parameter names so the same payload works
    whether the endpoint is local-vLLM or DashScope/OpenRouter:

      - DashScope / OpenRouter:    ``extra_body.enable_thinking = False``
      - vLLM-OpenAI-compat:        ``extra_body.chat_template_kwargs
                                     .enable_thinking = False``

    Each server silently ignores the parameter it does not recognise, so
    the payload is portable.  No-ops for non-Qwen models and for calls
    that don't set ``tool_choice``.
    """
    if not isinstance(model, str):
        return {}
    if "qwen" not in model.lower():
        return {}
    if tool_choice is None:
        return {}
    return {
        "extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[list] = None,
    tool_choice: Any = None,
    reasoning_effort: Optional[str] = None,
):
    """Cross-model chat-completion wrapper.

    Reasoning models (gpt-5.x, o1/o3/o4) reject ``max_tokens`` and
    ``temperature``; they require ``max_completion_tokens`` and burn part of
    that budget on hidden thinking tokens.  Detect reasoning models up front
    and route them through with a generous output cap; classic models keep
    the legacy path with a single fallback retry.

    ``reasoning_effort`` (one of ``minimal`` / ``low`` / ``medium`` / ``high``)
    is forwarded only for reasoning models; ignored otherwise.  Setting
    ``minimal`` suppresses hidden thinking tokens — the right default for
    cold-start data generation, where the SFT student never consumes the
    teacher's hidden chain anyway.
    """
    if _is_reasoning_model(model):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max(6000, max_tokens * 4),
        }
        # OpenAI hard-rejects ``reasoning_effort`` together with ``tools`` on
        # /v1/chat/completions for the gpt-5.x family (HTTP 400: "Function
        # tools with reasoning_effort are not supported for gpt-5.x in
        # /v1/chat/completions. Please use /v1/responses instead."). The
        # /v1/responses migration is a much bigger refactor; for now we
        # silently drop ``reasoning_effort`` whenever the call ships
        # tools so the action-LLM step does not 400-fail and degrade to
        # the candidate-list fallback. Schema-VLM calls (which are tool-
        # less) keep ``reasoning_effort`` and benefit from it.
        # Detection: gpt-5.x model + tools/tool_choice present + the
        # routed model id contains no provider prefix (i.e. direct
        # OpenAI Chat Completions). OpenRouter tunnels the same model
        # under ``openai/gpt-5.x`` and historically accepts the
        # parameter — only strip on direct OpenAI.
        is_direct_openai_gpt5 = (
            isinstance(model, str)
            and model.lower().startswith("gpt-5")
            and "/" not in model  # OpenRouter ids contain a "/"
        )
        tools_present = (tools is not None) or (tool_choice is not None)
        suppress_reasoning = (
            reasoning_effort is not None
            and is_direct_openai_gpt5
            and tools_present
        )
        if reasoning_effort and not suppress_reasoning:
            if reasoning_effort not in _VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort must be one of {_VALID_REASONING_EFFORTS}, "
                    f"got {reasoning_effort!r}"
                )
            kwargs["reasoning_effort"] = reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        kwargs.update(_maybe_disable_thinking_kwargs(model, tool_choice))
        return client.chat.completions.create(**kwargs)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    kwargs.update(_maybe_disable_thinking_kwargs(model, tool_choice))

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc)
        if not (
            "max_completion_tokens" in msg
            or ("max_tokens" in msg and "Unsupported" in msg)
            or ("temperature" in msg and "Unsupported" in msg)
        ):
            raise
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        kwargs["max_completion_tokens"] = max(6000, max_tokens * 5)
        return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Lazy imports for optional deps
# ---------------------------------------------------------------------------

def _import_browsergym_heuristic():
    from browsergym_wrapper.heuristic import obs_to_schema as bg_obs_to_schema
    return bg_obs_to_schema


def _import_browser_tools():
    """List-valid-actions handler from the browser tool registry."""
    try:
        from browsergym_wrapper.tools import build_browser_registry
        return build_browser_registry
    except Exception as exc:
        logger.debug("browsergym_wrapper.tools unavailable: %s", exc)
        return None


def _import_schema_helpers():
    try:
        from vlm_wrapper.schema import (
            build_system_prompt,
            build_user_message,
            parse_schema_output,
        )
        return {
            "build_system_prompt": build_system_prompt,
            "build_user_message": build_user_message,
            "parse_schema_output": parse_schema_output,
        }
    except Exception as exc:
        logger.debug("vlm_wrapper.schema unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Stage 1 — visual schema generation (gpt-5.5 vision)
# ---------------------------------------------------------------------------

# Pre-compiled patterns for the lenient parser.
_LENIENT_STATE_OPEN_RE = re.compile(r"<state\b[^>]*>", re.IGNORECASE)
_LENIENT_STATE_CLOSE_RE = re.compile(r"</state\s*>", re.IGNORECASE)
_LENIENT_FENCE_RE = re.compile(
    r"^\s*```(?:xml|html|text|state)?\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LENIENT_SECTION_RE = re.compile(
    r"<(entities|attributes|affordances|relations|state_flags|targets|actions)\b",
    re.IGNORECASE,
)


def _lenient_parse_schema(raw: str, strict_parser) -> Tuple[Optional[str], str]:
    """Salvage a ``<state>...</state>`` schema from messy VLM output.

    Returns ``(parsed_schema_or_None, recovery_kind)``.
    """
    if not raw:
        return None, ""

    parsed = strict_parser(raw)
    if parsed:
        return parsed, "strict"

    cleaned = _LENIENT_FENCE_RE.sub("", raw).strip()
    if cleaned and cleaned != raw:
        parsed = strict_parser(cleaned)
        if parsed:
            return parsed, "fenced"

    candidate = cleaned or raw

    open_m = _LENIENT_STATE_OPEN_RE.search(candidate)
    close_m = _LENIENT_STATE_CLOSE_RE.search(candidate)
    if open_m and not close_m:
        salvaged = candidate[open_m.start():].rstrip() + "\n</state>"
        parsed = strict_parser(salvaged)
        if parsed:
            return parsed, "truncated"

    if not open_m and _LENIENT_SECTION_RE.search(candidate):
        salvaged = "<state>\n" + candidate.strip() + "\n</state>"
        parsed = strict_parser(salvaged)
        if parsed:
            return parsed, "untagged"

    return None, ""


def generate_schema_from_image(
    *,
    pil_image,
    obs: Dict[str, Any],
    task_id: str,
    goal: str,
    step: int,
    candidate_actions: List[str],
    client: Any,
    routed_model: str,
    schema_helpers: Dict[str, Any],
    canonical_fallback: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = _SCHEMA_MAX_TOKENS,
    max_entities: int = _DEFAULT_MAX_ENTITIES,
    use_som: bool = True,
    goal_images: Optional[List[Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call gpt-5.5 (vision) on the screenshot to produce a ``<state>`` schema.

    The screenshot is the **primary** input; the AXTree-as-text rides along
    as grounding context (so the VLM can emit correct ``bid``s).  Returns a
    dict with the parsed ``schema`` (or ``None``), raw output, finish reason,
    and any exception captured.

    Visual enhancements:
      - When ``use_som`` is True (default) and BrowserGym's
        ``extra_element_properties`` provides Set-of-Marks bbox metadata,
        we send the SoM-overlayed screenshot (bids drawn on the pixels)
        rather than the raw screenshot. This is the canonical input
        format for VLM web agents.
      - When ``goal_images`` is non-empty (typically VisualWebArena's
        reference product images extracted from ``goal_object``), each
        image is appended as an additional ``image_url`` part so the VLM
        can ground the goal visually.

    On failure or no schema, falls back to ``canonical_fallback`` (the
    deterministic AXTree-walked schema from
    :mod:`browsergym_wrapper.heuristic`).
    """
    if pil_image is None or schema_helpers is None or client is None:
        return {
            "schema": canonical_fallback,
            "raw": "",
            "source": "fallback_canonical" if canonical_fallback else "no_image_or_client",
            "error": None,
        }

    system = schema_helpers["build_system_prompt"]("browser", max_entities=max_entities)

    # Prefer the Set-of-Marks overlay when available — it's strictly more
    # informative than the raw screenshot for bid-targeted actions.
    primary_image = pil_image
    som_used = False
    if use_som:
        som_img = _render_som_screenshot(obs)
        if som_img is not None:
            primary_image = som_img
            som_used = True

    extra_parts: List[str] = []
    url = obs.get("url", "") or ""
    if url:
        extra_parts.append(f"URL: {url}")
    last_action = obs.get("last_action", "") or ""
    if last_action:
        extra_parts.append(f"Last action: {last_action}")
    last_error = obs.get("last_action_error", "") or ""
    if last_error:
        extra_parts.append(f"Last action error: {last_error[:200]}")
    focused = obs.get("focused_element_bid", "") or ""
    if focused:
        extra_parts.append(f"Focused element bid: {focused}")
    if som_used:
        extra_parts.append(
            "The screenshot has SET-OF-MARKS overlays: every interactable "
            "element is wrapped in a dashed bounding box and labelled with "
            "its bid in a black tag. Read bids directly off the image."
        )
    if goal_images:
        extra_parts.append(
            f"NOTE: {len(goal_images)} GOAL reference image(s) follow the "
            "page screenshot. Use them to ground product / item /"
            " visual-similarity references in the goal text."
        )
    axtree_text = _flatten_axtree(obs, max_chars=3000)
    if axtree_text:
        extra_parts.append(
            "AXTree (for element bid grounding, truncated):\n" + axtree_text
        )
    if candidate_actions:
        extra_parts.append(
            "Candidate actions for this page (you MUST copy one of these "
            "verbatim into <actions>; do NOT rename or reformat):\n"
            + "\n".join(f"  - {a}" for a in candidate_actions[:_MAX_CANDIDATE_ACTIONS])
        )
    extra_context = "\n\n".join(extra_parts)

    user_content = schema_helpers["build_user_message"](
        primary_image,
        domain="browser",
        task_id=task_id,
        goal=goal,
        step=step,
        extra_context=extra_context,
    )
    # Append goal-side reference images (VWA tasks). build_user_message
    # returns ``[image, text]``; we extend to ``[image, text, goal_img_1,
    # goal_img_2, ...]`` so each goal image rides as an OpenAI-format
    # image_url part.
    if goal_images:
        from vlm_wrapper.schema import encode_image_b64  # local import: optional dep
        for gi in goal_images[:4]:  # cap to avoid blowing up the prompt
            try:
                b64 = encode_image_b64(gi)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                })
            except Exception as exc:
                logger.debug("goal image encode failed: %s", exc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    parsed: Optional[str] = None
    err: Optional[str] = None
    finish_reason: Optional[str] = None
    recovery: str = ""
    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if resp.choices:
            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
        parsed, recovery = _lenient_parse_schema(
            raw, schema_helpers["parse_schema_output"]
        )
    except Exception as exc:
        err = repr(exc)
        logger.warning("[schema-VLM] step %d failed: %s", step, exc)

    base_meta: Dict[str, Any] = {
        "raw": raw,
        "raw_full_len": len(raw),
        "finish_reason": finish_reason,
        "recovery": recovery,
        "error": err,
    }

    if parsed:
        if recovery and recovery != "strict":
            logger.info(
                "[schema-VLM] step %d salvaged via '%s' (finish_reason=%s, raw_len=%d)",
                step, recovery, finish_reason, len(raw),
            )
        return {"schema": parsed, "source": "vlm", **base_meta}

    if raw and finish_reason == "length":
        logger.warning(
            "[schema-VLM] step %d response truncated (finish_reason=length, "
            "raw_len=%d) — consider raising _SCHEMA_MAX_TOKENS",
            step, len(raw),
        )

    return {
        "schema": canonical_fallback,
        "source": "fallback_canonical" if canonical_fallback else "vlm_no_schema",
        **base_meta,
    }


# ---------------------------------------------------------------------------
# Action vocabulary construction
# ---------------------------------------------------------------------------

# Standard navigation actions BrowserGym always accepts.
_GLOBAL_BROWSER_ACTIONS: List[str] = [
    "scroll(0, 300)",
    "scroll(0, -300)",
    "go_back()",
    "go_forward()",
    "noop()",
]

# Heuristic match for *information-extraction* goals (questions whose
# scoring requires the agent to call ``send_msg_to_user("<answer>")``
# rather than affect page state). Used by ``_build_candidate_actions``
# to seed a placeholder ``send_msg_to_user`` candidate so the action LLM
# always sees the terminal action as an option.
#
# The classifier is split into THREE regimes by task-id prefix because
# VWA "Find a TV listing in Maryland" (page-state task) collides
# textually with AssistantBench "Find the GDP of Japan" (QA task). The
# task-id is the only reliable disambiguator:
#
#   * QA suites → always seed unconditionally (covers the 11 % of AB
#     goals that don't start with a wh-word or end with ``?``, e.g.
#     "Compute the average annual temperature in Arizona ...").
#   * Page-state suites → seed *only* if the goal is grammatically a
#     question (ends with ``?`` or starts with a wh-word). Avoids
#     mis-seeding side-effect tasks where ``send_msg_to_user`` would
#     short-circuit a multi-step page interaction.
#   * Unknown / no task-id → fall back to the page-state rule. False
#     positives are cheap (one extra candidate slot); false negatives
#     are expensive (the May-1 AB regression).
#
# Empirical calibration on the 176 AB goals from the 2026-05-01 run:
#   135/176 (77 %) end with ``?``
#   124/176 (70 %) start with a wh-word
#   157/176 (89 %) match either rule
#   The remaining 19 are caught by the QA-suite prefix.
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(what|who|when|where|how|why|which|whose|whom)\b",
    re.IGNORECASE,
)
_QA_TASK_ID_PREFIXES = (
    "assistantbench.",
    "browsergym/assistantbench.",
)
_PAGE_STATE_TASK_ID_PREFIXES = (
    "visualwebarena.", "browsergym/visualwebarena.",
    "webarena.", "browsergym/webarena.",
    "miniwob.", "browsergym/miniwob.",
    "workarena.", "browsergym/workarena.",
)


def _looks_like_question(goal: Optional[str]) -> bool:
    """Cheap textual classifier: ``?`` ending or wh-word lead."""
    if not goal:
        return False
    g = str(goal).strip()
    return bool(g.endswith("?") or _QUESTION_LEAD_RE.match(g))


def _is_information_extraction_task(
    *, task_id: Optional[str], goal: Optional[str],
) -> bool:
    """Return True when the goal looks like a free-form QA task that
    requires the agent to emit a ``send_msg_to_user("<answer>")``
    terminal action to score (vs. a side-effect-on-page task scored
    by URL / DOM state).
    """
    if task_id:
        tid = str(task_id).lower()
        if any(tid.startswith(p) for p in _QA_TASK_ID_PREFIXES):
            return True
        if any(tid.startswith(p) for p in _PAGE_STATE_TASK_ID_PREFIXES):
            return _looks_like_question(goal)
    return _looks_like_question(goal)


# Placeholder string the candidate-list seeder uses for the answer slot.
# The action LLM is expected to overwrite the placeholder with the
# real grounded answer (via the ``action_string`` field copied verbatim,
# OR via the structured ``action_type=send_msg_to_user`` + ``answer``
# slot). We pick a placeholder long enough to pass the validator regex
# (which requires ``send_msg_to_user(.+)``) but obviously-wrong enough
# that an agent that copies it verbatim still scores 0 and we can
# notice in telemetry.
_SEND_MSG_PLACEHOLDER = 'send_msg_to_user("<your answer here>")'
_REPORT_INFEASIBLE_PLACEHOLDER = (
    'report_infeasible("<reason this task cannot be answered>")'
)
# Matches the angle-bracket placeholder fragments that appear inside the
# seed candidates above. ``_validate_action_string`` rejects any action
# string containing these so the LLM cannot accidentally submit the
# hint as its real answer.
_PLACEHOLDER_LITERAL_RE = re.compile(
    r"<your\s+answer\s+here>"
    r"|<reason\s+this\s+task\s+cannot\s+be\s+answered>"
    r"|<concise\s+answer>"
    r"|<final\s+answer>",
    re.IGNORECASE,
)

# Validates the action string we intend to send to ``env.step(...)``.
#
# ``send_msg_to_user(...)`` and ``report_infeasible(...)`` are the two
# *terminal* actions used by AssistantBench (and accepted by webarena /
# visualwebarena / workarena under the same names — see
# ``BrowserGym/browsergym/core/src/browsergym/core/action/highlevel.py``).
# AssistantBench scores **only** when the agent calls
# ``send_msg_to_user("<final answer>")``: the env extracts the message
# argument and matches it against the reference answer with F1 / exact
# match. Without this terminal action the episode hits ``max_steps``
# with reward=0 — which is exactly what we saw in the May-1 181-task
# AssistantBench run (0/2816 actions were ``send_msg_to_user``, 100 %
# reward=0). Allowing them through the validator is step 1; the
# action-tool enum, structured-fallback path, candidate seed, and
# system prompt also have to be taught about them in tandem.
_BROWSERGYM_ACTION_RE = re.compile(
    r"^\s*("
    r"click\([^)]*\)"
    r"|fill\([^,]+,\s*\".*\"\s*\)"
    r"|check\([^)]*\)"
    r"|press\([^)]*\)"
    r"|hover\([^)]*\)"
    r"|select_option\([^)]*\)"
    r"|focus\([^)]*\)"
    r"|clear\([^)]*\)"
    r"|scroll\(\s*-?\d+\s*,\s*-?\d+\s*\)"
    r"|go_back\(\s*\)"
    r"|go_forward\(\s*\)"
    r"|new_tab\(\s*\)"
    r"|tab_close\(\s*\)"
    r"|tab_focus\([^)]*\)"
    r"|goto\([^)]*\)"
    r"|noop\([^)]*\)"
    r"|send_msg_to_user\(.+\)"
    r"|report_infeasible\(.+\)"
    # ``search_web("query")`` is a *synthetic* action: the actor harness
    # intercepts it before ``env.step()`` (see ``_intercept_search_web``),
    # runs the search server-side via ``search_backends.search``, and
    # injects a synthetic results page into the live page via
    # ``page.goto("data:text/html;...")``. The substitute action passed
    # to BrowserGym is ``noop()`` so the env's observation pipeline
    # picks up the freshly-injected DOM. We need to allow it through
    # the validator so the LLM can emit it.
    r"|search_web\(.+\)"
    r")\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _list_clickable_bids(
    registry, max_results: int = _MAX_CANDIDATE_ACTIONS,
) -> List[Dict[str, Any]]:
    """Return ``list_valid_actions`` results from the browser tool registry."""
    if registry is None:
        return []
    try:
        result = registry.call("list_valid_actions", {})
    except Exception as exc:
        logger.debug("list_valid_actions failed: %s", exc)
        return []
    actions = result.get("actions") if isinstance(result, dict) else None
    return list(actions or [])[:max_results]


def _build_candidate_actions(
    *, obs: Dict[str, Any], registry,
    task_id: Optional[str] = None, goal: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build a list of candidate action strings + structured metadata.

    Combines:
      - ``list_valid_actions`` from :mod:`browsergym_wrapper.tools`
        (each interactive bid → ``click(bid)`` / ``fill(bid, "...")`` /
        ``check(bid)``) when the registry resolves.
      - A small set of standard navigation actions (always available).
      - For *information-extraction* tasks (assistantbench.* and any
        question-form goal — see ``_is_information_extraction_task``):
        a placeholder ``send_msg_to_user("<your answer here>")``
        candidate plus a ``report_infeasible(...)`` companion. Without
        this seed the action LLM never learns the terminal action
        exists, hits ``max_steps``, and the env scores 0 (verified
        empirically on the May-1 181-task AssistantBench run, 0/2816
        actions were ``send_msg_to_user``).

    Returns ``(strings, meta_list)`` where ``meta_list`` carries the parsed
    role/name/bid for each interactive entry (used by the actor LLM prompt).
    """
    meta: List[Dict[str, Any]] = []
    strings: List[str] = []

    seen: set = set()
    raw_actions = _list_clickable_bids(registry)
    for entry in raw_actions:
        a = (entry or {}).get("action") if isinstance(entry, dict) else None
        if not a or a in seen:
            continue
        seen.add(a)
        strings.append(a)
        meta.append({
            "action": a,
            "role": (entry or {}).get("role"),
            "name": (entry or {}).get("name"),
        })

    # Seed terminal actions for QA-style tasks. These come BEFORE the
    # global navigation set so the action LLM sees them next to the
    # interactive candidates rather than buried at the end (the
    # candidate list is truncated to ``_MAX_CANDIDATE_ACTIONS`` — we
    # don't want the answer action to fall off the cliff on a heavy
    # page with 25+ interactive bids).
    if _is_information_extraction_task(task_id=task_id, goal=goal):
        for a, role in (
            (_SEND_MSG_PLACEHOLDER, "terminal_answer"),
            (_REPORT_INFEASIBLE_PLACEHOLDER, "terminal_infeasible"),
        ):
            if a not in seen:
                seen.add(a)
                strings.append(a)
                meta.append({"action": a, "role": role, "name": None})

    for a in _GLOBAL_BROWSER_ACTIONS:
        if a not in seen:
            seen.add(a)
            strings.append(a)
            meta.append({"action": a, "role": "navigation", "name": None})

    return strings[:_MAX_CANDIDATE_ACTIONS], meta[:_MAX_CANDIDATE_ACTIONS]


# ---------------------------------------------------------------------------
# Stage 2 — schema-driven action selection (gpt-5.5)
# ---------------------------------------------------------------------------

_ACTOR_SYSTEM_PROMPT = (
    "You are an Actor Agent for the COS-PLAY web-agent pipeline, driving a "
    "BrowserGym environment.\n"
    "On every step you receive a structured ``<state>...</state>`` schema "
    "describing the visual state of the page (entities have ``bid`` ids "
    "matching the AXTree's set-of-marks), plus a list of candidate "
    "BrowserGym action strings.\n\n"
    "Your job:\n"
    "1. Reason briefly (≤3 sentences) about the schema: which entity matters, "
    "what is the current sub-goal, and why one action best advances it.\n"
    "2. Pick EXACTLY ONE action by calling the ``choose_action`` function. "
    "You may either:\n"
    "   - Echo a candidate string verbatim (STRONGLY preferred), OR\n"
    "   - Specify ``action_type`` + ``bid`` (and ``text`` for ``fill``) so the "
    "harness can construct a typed BrowserGym action.\n\n"
    "CRITICAL: BrowserGym requires bids to be QUOTED string literals. Always "
    "write ``click(\"a17\")`` and ``fill(\"a23\", \"hello\")`` — NEVER "
    "``click(a17)`` or ``click(12)``. The candidate strings are already "
    "correctly quoted; copy them verbatim.\n\n"
    "If recent action history shows an action had NO EFFECT (URL/state did "
    "not change and no error), choose a DIFFERENT action this turn.\n\n"
    "SEARCH-FIRST HEURISTIC: When the goal asks you to FIND, LOCATE, GET, "
    "or NAVIGATE TO a specific item / listing / post / product on a page "
    "that has a search box, your FIRST action should almost always be "
    "``fill(\"<search_box_bid>\", \"<query>\")`` followed by ``press("
    "\"<bid>\", \"Enter\")`` — NOT ``scroll`` or ``go_back``. Scrolling "
    "blindly through a marketplace / listings page rarely converges; "
    "typing the goal's noun-phrase into the search box converges in 1–2 "
    "actions. Only fall back to scroll/click navigation when no search "
    "box or filter control is visible.\n\n"
    "ANTI-REPETITION HINT: If a fill / click signature in your recent "
    "history made no page progress (no URL change, no new content), "
    "try a DIFFERENT bid or DIFFERENT query value rather than "
    "repeating the same one. Repeating the same fill query yields the "
    "same empty page; clicking the same dead-end listing yields the "
    "same dead-end page. Vary the query (synonyms, shorter phrase) or "
    "pick a different visible candidate.\n\n"
    "TERMINAL ACTIONS (CRITICAL for question-answering benchmarks like "
    "AssistantBench): if the goal is a QUESTION (starts with what / "
    "who / when / where / how / why / which, or ends with ``?``) and "
    "you have GROUNDED EVIDENCE on the current page for the answer, "
    "your final action MUST be "
    "``send_msg_to_user(\"<concise answer>\")`` — this is what the "
    "eval harness scores against the reference. Without it the "
    "episode hits max_steps with reward=0 even if you found the "
    "answer in the page text.\n"
    "  • DO NOT call ``report_infeasible(...)`` on step 0 or 1. The "
    "first thing you should do on any QA task is fill the focused "
    "search box (or visible textbox) with a query derived from the "
    "goal and submit it (``press(\"<bid>\", \"Enter\")`` or click the "
    "search button). ``fill()`` is a valid action even if it doesn't "
    "appear in the candidate list — the candidate list is a *hint*, "
    "not an exhaustive whitelist. You can always type into a focused "
    "textbox, and you can always ``goto(<url>)`` to navigate to a "
    "different site. Treat ``report_infeasible`` as the last resort "
    "after at least 4-5 real navigation/search attempts have failed.\n"
    "  • For web research, **STRONGLY prefer ``search_web(\"...\")``** "
    "over any goto-to-Google / goto-to-DDG strategy. ``search_web`` is "
    "a synthetic action whose backing search runs SERVER-SIDE in the "
    "harness (HTTP from Python, NOT via the browser), so it bypasses "
    "the anti-bot walls (Google ``/sorry/index`` CAPTCHA, DDG "
    "``static-pages/418`` teapot, consent dialogs) that BOTH Google "
    "and DuckDuckGo throw at Playwright. After the call, the page "
    "shows real result links you can ``click(...)``. Examples:\n"
    "    – ``search_web(\"beluga whale GFF3 Ensembl 2020\")`` to "
    "find the right bioinformatics database (Ensembl vs NCBI vs "
    "UCSC) before navigating.\n"
    "    – ``search_web(\"gyms Tompkins Square Park morning class "
    "schedule\")`` to find listings without scrolling Yelp.\n"
    "    – ``search_web(\"paintball karting Cologne walking distance"
    "\")`` to find both venues in one shot rather than navigating "
    "Google Maps manually.\n"
    "  Use the structured ``action_type=search_web`` + ``query=...`` "
    "slot when convenient. Use ``search_web`` as your FIRST action "
    "on any AssistantBench-style 'find X about Y' goal — DO NOT "
    "start with ``goto(google.com)`` or ``goto(duckduckgo.com)``: "
    "those will hit anti-bot walls and waste 4–6 steps.\n"
    "  • If you hit a CAPTCHA / 'unusual traffic' / 'I'm a teapot' "
    "/ '418' / 'static-pages/' page (URLs containing ``/sorry/``, "
    "``recaptcha``, ``consent.``, ``static-pages/418``), do NOT report "
    "infeasible. Switch to ``search_web(\"<query>\")`` (the harness "
    "intercepts this and bypasses the anti-bot wall) — or, if the "
    "goal explicitly names a source, navigate DIRECTLY by URL:\n"
    "    – \"on Wikipedia\" → ``goto(\"https://en.wikipedia.org/wiki/Special:Search?search=<terms>\")``\n"
    "    – \"on TripAdvisor\" → ``goto(\"https://www.tripadvisor.com/Search?q=<terms>\")``\n"
    "    – \"on Google Maps\" → ``goto(\"https://www.openstreetmap.org/search?query=<terms>\")``\n"
    "    – \"on TripAdvisor / Yelp / Google reviews\" → go to the canonical site\n"
    "  When in doubt, prefer ``search_web`` over any general-purpose "
    "search-engine goto: search_web is server-side and never blocked.\n"
    "  • When you DO have the answer, be MINIMAL in the payload: "
    "just the requested fact (a number, a name, a date, a list "
    "separated by commas) — NOT a sentence wrapping it. If the goal "
    "asks 'how many X?', answer ``42`` not ``There are 42 X.``\n"
    "  • Use the EXACT format the goal asks (units, capitalisation, "
    "ISO date if implied).\n"
    "  • Use the structured ``action_type=send_msg_to_user`` + "
    "``answer=<your answer>`` slot when convenient — the harness will "
    "build the action string for you and properly escape quotes.\n\n"
    "Always respond by calling the ``choose_action`` function."
)


def _build_action_tools(candidate_actions: List[str]) -> list:
    """OpenAI function-calling tool definition for browser action selection.

    Two ways to specify the action:
      1. ``action_string`` — a verbatim BrowserGym action.  Validated by
         ``_BROWSERGYM_ACTION_RE``; usually the best match for the
         ``candidate_actions`` list.
      2. ``action_type`` + ``bid`` (+ ``text`` / ``key``) — a structured
         description. Useful when the model wants to ``fill`` an input box
         with custom text not present in the candidate list.
    """
    enum_types = [
        "click", "fill", "check", "press", "hover", "select_option",
        "focus", "clear", "scroll_down", "scroll_up", "go_back",
        "go_forward", "new_tab", "tab_close", "noop",
        # Direct URL navigation. Critical for AssistantBench when the
        # default Google start hits the /sorry/index CAPTCHA — agent can
        # ``goto("https://duckduckgo.com/")`` to switch search engines.
        "goto",
        # Server-side web search. The harness intercepts this action,
        # runs the search via ``search_backends.search`` (HTTP from
        # Python — bypasses Playwright TLS-fingerprint anti-bot), and
        # injects the result page into the live Playwright page so the
        # next obs shows clickable result links. STRONGLY preferred
        # over ``goto(google.com)`` / ``goto(html.duckduckgo.com)``
        # which both hit anti-bot walls. Consumes the ``query`` param.
        "search_web",
        # Terminal actions (AssistantBench / webarena scoring).
        # ``send_msg_to_user`` returns the final answer to the user and
        # triggers the eval harness; ``report_infeasible`` is the
        # explicit "task is impossible" stop equivalent to STOP "N/A"
        # in the visualwebarena paper. Both consume the ``answer``
        # parameter (see below).
        "send_msg_to_user", "report_infeasible",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": (
                    "Choose a single BrowserGym action for this turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Brief chain-of-thought (≤3 sentences) "
                                "grounded in the schema entities."
                            ),
                        },
                        "action_string": {
                            "type": "string",
                            "description": (
                                "Verbatim BrowserGym action string. Bids "
                                "MUST be quoted string literals, e.g. "
                                "'click(\"a17\")' or "
                                "'fill(\"a23\", \"hello\")'. NEVER write "
                                "'click(a17)' or 'click(12)' — they will be "
                                "parsed as int/identifier and rejected. "
                                "Prefer copying one of these candidates "
                                "verbatim: "
                                + ", ".join(candidate_actions[:_MAX_CANDIDATE_ACTIONS])
                            ),
                        },
                        "action_type": {
                            "type": "string",
                            "enum": enum_types,
                            "description": (
                                "Structured action type. Used when "
                                "action_string is missing or invalid."
                            ),
                        },
                        "bid": {
                            "type": "string",
                            "description": (
                                "Element bid for click/fill/check/press/"
                                "hover/select_option/focus/clear."
                            ),
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Text payload for fill (input value) or "
                                "select_option (option label)."
                            ),
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Key to press for the press action (e.g. "
                                "'Enter', 'Tab', 'ArrowDown')."
                            ),
                        },
                        "scroll_dy": {
                            "type": "integer",
                            "description": (
                                "Pixels to scroll vertically. Positive = "
                                "down. Used with action_type=scroll_down/up "
                                "if action_string is missing."
                            ),
                        },
                        "answer": {
                            "type": "string",
                            "description": (
                                "Final answer string for "
                                "action_type=send_msg_to_user (the answer "
                                "you want the user / eval harness to "
                                "receive — keep it concise, grounded in "
                                "page evidence, and exactly the form the "
                                "goal asks for, e.g. a number, a name, a "
                                "date) or the reason for "
                                "action_type=report_infeasible (a brief "
                                "explanation of why the goal cannot be "
                                "achieved on the current page set)."
                            ),
                        },
                        "url": {
                            "type": "string",
                            "description": (
                                "Absolute URL for action_type=goto. "
                                "Useful to switch search engines (e.g. "
                                "https://duckduckgo.com/) when the default "
                                "Google start hits a CAPTCHA / consent "
                                "wall, or to jump directly to a known "
                                "destination instead of clicking through."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Free-form search query for "
                                "action_type=search_web. The harness runs "
                                "the search server-side (HTTP from Python, "
                                "not via the browser — bypasses anti-bot) "
                                "and injects the result page into the "
                                "live page. Prefer this over "
                                "goto(google.com) / goto(duckduckgo.com): "
                                "those URLs hit CAPTCHA / 418 walls and "
                                "the agent ends up stuck. Examples: "
                                "'beluga whale GFF3 Ensembl', 'gyms near "
                                "Tompkins Square Park morning classes', "
                                "'paintball karting Cologne'."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }
    ]


_BID_RE = re.compile(r"\(\s*([A-Za-z0-9_-]+)")

# Matches the synthetic ``search_web("query")`` action emitted by the LLM
# when it wants to run a server-side web search (see
# ``_intercept_search_web`` for the runtime intercept that turns this
# into a real search + page injection). Captures the query argument
# verbatim so the harness can pass it to ``search_backends.search``.
# Accepts either double or single quotes around the query so the
# autoquote shim doesn't break.
_SEARCH_WEB_CALL_RE = re.compile(
    r'^\s*search_web\(\s*(?:"([^"]*)"|\'([^\']*)\')\s*\)\s*$',
    re.IGNORECASE | re.DOTALL,
)


def _parse_search_web_query(action: str) -> Optional[str]:
    """Return the query argument of a ``search_web("...")`` action, or
    ``None`` if ``action`` is not a search_web call.

    Used by the step-loop intercept to pull the query out before
    handing the substitute ``noop()`` action to ``env.step()``.
    """
    if not action:
        return None
    m = _SEARCH_WEB_CALL_RE.match(action.strip())
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2)


def _bid_from_action(action: str) -> Optional[str]:
    """Return the first arg of the action call, e.g. ``click(123)`` -> ``"123"``."""
    if not action:
        return None
    m = _BID_RE.search(action)
    return m.group(1) if m else None


def _format_history_block(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Recent action history (newest last):"]
    for entry in history[-_HISTORY_WINDOW:]:
        effect = (
            "ERROR" if entry.get("error")
            else ("NO EFFECT" if entry.get("noop") else "ok")
        )
        lines.append(
            f"  - {entry.get('action')!r} -> {effect}"
            + (f" (err: {entry.get('error_text','')[:80]})" if entry.get("error") else "")
        )
    recent = history[-_HISTORY_WINDOW:]
    noop_actions = sorted({e["action"] for e in recent if e.get("noop")})
    if noop_actions:
        lines.append(
            f"WARNING: Action(s) {noop_actions} had no effect. Pick a DIFFERENT action."
        )
    # Surface recently-errored bids so the model stops retrying the same dead
    # bid. Especially important for fill() where a stale bid will keep
    # erroring after the page re-renders.
    bad_bids = sorted({
        b for b in (
            _bid_from_action(e["action"]) for e in recent if e.get("error")
        ) if b is not None
    })
    if bad_bids:
        lines.append(
            f"AVOID these bids in the next action — they errored recently: {bad_bids}. "
            "Pick a DIFFERENT bid (re-read the schema for the current bid of the target element) "
            "or switch to a different action_type (e.g. click before fill)."
        )
    return "\n".join(lines) + "\n"


def _structured_to_action_string(args: Dict[str, Any]) -> Optional[str]:
    """Build a BrowserGym action string from structured function-call args.

    Used as a fallback when the ``action_string`` field is empty/invalid.
    """
    atype = (args.get("action_type") or "").lower().strip()
    bid = (args.get("bid") or "").strip()
    text = args.get("text", "")
    key = (args.get("key") or "").strip()
    answer = args.get("answer", "")
    dy = args.get("scroll_dy")
    if isinstance(dy, str):
        try:
            dy = int(dy)
        except Exception:
            dy = None

    # NOTE: BrowserGym's high-level action grammar (parsers.py) requires every
    # bid argument to be a STRING literal — bare numerics like ``click(12)`` get
    # parsed as int and ``get_elem_by_bid`` raises ValueError. We always emit
    # bids inside double quotes so they round-trip cleanly through
    # ``highlevel_action_parser`` → ``repr(arg)`` → ``exec()``.
    if atype == "click" and bid:
        return f'click("{bid}")'
    if atype == "check" and bid:
        return f'check("{bid}")'
    if atype == "focus" and bid:
        return f'focus("{bid}")'
    if atype == "clear" and bid:
        return f'clear("{bid}")'
    if atype == "hover" and bid:
        return f'hover("{bid}")'
    if atype == "fill" and bid:
        # Escape backslashes + double quotes for the action string.
        text_str = (text or "").replace("\\", "\\\\").replace("\"", "\\\"")
        return f'fill("{bid}", "{text_str}")'
    if atype == "select_option" and bid:
        text_str = (text or "").replace("\"", "\\\"")
        return f'select_option("{bid}", "{text_str}")'
    if atype == "press" and bid and key:
        key_str = key.replace("\"", "\\\"")
        return f'press("{bid}", "{key_str}")'
    if atype == "scroll_down":
        d = dy if isinstance(dy, int) else 300
        return f"scroll(0, {abs(d)})"
    if atype == "scroll_up":
        d = dy if isinstance(dy, int) else 300
        return f"scroll(0, -{abs(d)})"
    if atype == "go_back":
        return "go_back()"
    if atype == "go_forward":
        return "go_forward()"
    if atype == "new_tab":
        return "new_tab()"
    if atype == "tab_close":
        return "tab_close()"
    if atype == "noop":
        return "noop()"
    # Direct URL navigation. ``url`` parameter takes priority but we
    # fall back to ``text`` for LLM-call-sites that reuse it for any
    # string payload (matching the send_msg_to_user pattern below).
    if atype == "goto":
        url = (args.get("url") or text or "").strip()
        if not url:
            return None
        url_str = url.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'goto("{url_str}")'
    # Server-side web search. The actor harness intercepts this action
    # in the step loop (see ``_intercept_search_web``) and rewrites it
    # to ``noop()`` after injecting the search-results page into the
    # live Playwright page. Accepts ``query`` (preferred) and falls
    # back to ``text`` for callers that reuse ``text`` for any free-
    # form string payload.
    if atype == "search_web":
        query = (args.get("query") or text or "").strip()
        if not query:
            return None
        query_str = query.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'search_web("{query_str}")'
    # Terminal actions — the answer / reason is wrapped in double quotes
    # with embedded quotes/backslashes escaped so the BrowserGym parser
    # round-trips it through ``exec`` cleanly. We accept ``text`` as a
    # fallback name for ``answer`` because some LLM call sites reuse
    # ``text`` for any free-form string payload.
    if atype in ("send_msg_to_user", "report_infeasible"):
        payload = answer if answer else (text or "")
        if not payload:
            return None
        payload_str = (
            str(payload)
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
        )
        return f'{atype}("{payload_str}")'
    return None


# Functions whose first positional argument is a ``bid`` and MUST be a
# Python string literal in the action source (BrowserGym enforces this in
# ``get_elem_by_bid``). Used by ``_autoquote_bids`` and
# ``_validate_action_string``.
_BID_FIRST_ARG_FNS = frozenset({
    "click", "fill", "check", "uncheck", "hover", "focus",
    "clear", "press", "select_option",
})

# Match a function call whose first positional arg is a *bare* identifier
# (not already quoted) — e.g. ``click(12)``, ``fill(a23, "hi")``,
# ``select_option(e7, "Croatia")``. Captures (whitespace, fname, bare_arg,
# delimiter) so the autoquote shim can rewrap the bid in double quotes
# while leaving the rest of the call intact.
_BARE_BID_CALL_RE = re.compile(
    r"^(\s*)(" + "|".join(sorted(_BID_FIRST_ARG_FNS)) + r")"
    r"\(\s*([A-Za-z_][A-Za-z0-9_-]*|\d+)\s*([,)])"
)


def _autoquote_bids(action: str) -> str:
    """Defense-in-depth: rewrite ``click(12)`` → ``click("12")`` etc.

    Only touches the FIRST positional argument of a known bid-taking
    function call, and only when that argument is a bare identifier or
    bare integer (i.e. not already quoted). Idempotent: already-quoted
    bids round-trip unchanged. Useful when the LLM still emits unquoted
    bids despite the prompt telling it to quote them.
    """
    if not action:
        return action
    m = _BARE_BID_CALL_RE.match(action)
    if not m:
        return action
    return f'{m[1]}{m[2]}("{m[3]}"{m[4]}' + action[m.end():]


def _validate_action_string(action: str) -> bool:
    """Strict structural check via BrowserGym's own parser.

    Rejects strings that:
      - the high-level parser cannot parse at all, OR
      - call a bid-taking function with a non-string first argument, OR
      - contain a literal candidate-list placeholder (e.g.
        ``send_msg_to_user("<your answer here>")``). Without this third
        check, a tired LLM that hits ``max_steps`` will sometimes copy
        the placeholder verbatim from the candidate list as its final
        action — which scores 0 against any real answer reference and
        is one of the "obvious failure modes" we want telemetry to
        catch instead of silently submit.

    This is what catches ``click(12)`` *before* it reaches ``env.step()``.
    The earlier regex-only validator was permissive: any ``click([^)]*)``
    matched, including the broken-int-bid form.
    """
    if not action:
        return False
    s = action.strip()
    # Reject literal placeholder contents copied verbatim from the
    # candidate-list seed entries (``_SEND_MSG_PLACEHOLDER`` /
    # ``_REPORT_INFEASIBLE_PLACEHOLDER``). The placeholder text
    # ``<your answer here>`` / ``<reason ...>`` should NEVER appear in
    # a real submission — its presence means the model treated the
    # hint as the answer.
    if _PLACEHOLDER_LITERAL_RE.search(s):
        return False
    # Quick legacy regex pre-filter to weed out obviously malformed strings
    # (keeps the failure mode user-friendly when the parser is missing).
    if not _BROWSERGYM_ACTION_RE.match(s):
        return False
    # Real parse — single source of truth on whether ``env.step()`` will
    # accept the string.
    try:
        from browsergym.core.action.parsers import highlevel_action_parser
    except Exception:  # pragma: no cover — parser ships with browsergym.core
        return True
    try:
        calls = highlevel_action_parser.parse_string(s, parse_all=True).as_list()
    except Exception:
        return False
    if not calls:
        return False
    fname, fargs = calls[0]
    if fname in _BID_FIRST_ARG_FNS:
        if not fargs or not isinstance(fargs[0], str):
            return False
    return True


def _canonicalize_to_candidate(
    raw: str, candidates: List[str],
) -> Optional[str]:
    """Map ``raw`` LLM output back to one of the candidate strings.

    Mirrors ``_canonicalize_action`` in ``generate_cold_start_actor_gymv``
    but operates on full BrowserGym call strings rather than action names.
    Returns ``None`` if no plausible match is found.

    Strategy:
      1. exact match
      2. case-insensitive match
      3. numeric index ("3" or "3." -> candidates[2])
      4. canonical form (collapse whitespace, strip outer parens spaces)
    """
    if not raw or not candidates:
        return None
    s = raw.strip().strip("`").strip()
    if not s:
        return None
    if s in candidates:
        return s
    lower_map = {c.lower(): c for c in candidates}
    if s.lower() in lower_map:
        return lower_map[s.lower()]
    m = re.match(r"^\s*(\d+)\s*[\.\)\-:]?\s*$", s)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    # Whitespace-collapse compare.
    norm = re.sub(r"\s+", "", s)
    for c in candidates:
        if re.sub(r"\s+", "", c) == norm:
            return c
    return None


def select_action(
    *,
    schema_text: Optional[str],
    obs: Dict[str, Any],
    candidate_actions: List[str],
    candidate_meta: List[Dict[str, Any]],
    task: str,
    step: int,
    history: List[Dict[str, Any]],
    client: Any,
    routed_model: str,
    temperature: float = 0.4,
    max_tokens: int = _ACTION_MAX_TOKENS,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Call gpt-5.5 with the schema → ``(action, reasoning, raw, error)``."""
    if not candidate_actions:
        candidate_actions = list(_GLOBAL_BROWSER_ACTIONS)
        candidate_meta = [
            {"action": a, "role": "navigation", "name": None}
            for a in _GLOBAL_BROWSER_ACTIONS
        ]
    if client is None:
        return None, None, "", "no_client"

    history_block = _format_history_block(history)
    schema_block = (
        schema_text.strip() if schema_text else
        "(no schema available — fall back to the URL + AXTree text below)"
    )

    candidate_lines = []
    for i, m in enumerate(candidate_meta[:_MAX_CANDIDATE_ACTIONS], start=1):
        suffix = ""
        if m.get("role") and m.get("name"):
            suffix = f"  # {m['role']}: {str(m['name'])[:60]}"
        elif m.get("role"):
            suffix = f"  # {m['role']}"
        candidate_lines.append(f"  {i}. {m['action']}{suffix}")
    candidate_block = "Candidate actions:\n" + "\n".join(candidate_lines)

    user_parts = [
        f"Task: {task}",
        f"URL: {obs.get('url', '') or ''}",
        f"Step: {step}",
        "",
        "Structured state schema:",
        schema_block,
        "",
        candidate_block,
    ]
    if not schema_text:
        ax_text = _flatten_axtree(obs, max_chars=2000)
        if ax_text:
            user_parts.extend([
                "",
                "AXTree (since no schema was parsed):",
                ax_text,
            ])
    user_parts.extend([
        "",
        history_block.strip(),
        "",
        "Pick the BEST action and call ``choose_action``. Prefer to copy a "
        "candidate verbatim into ``action_string``; otherwise specify "
        "``action_type`` + ``bid`` (+ ``text``).",
    ])
    user_content = "\n".join(p for p in user_parts if p is not None)

    tools = _build_action_tools(candidate_actions)

    raw = ""
    err: Optional[str] = None
    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=[
                {"role": "system", "content": _ACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            reasoning_effort=reasoning_effort,
        )
        choice = resp.choices[0]
        msg = choice.message
        raw = msg.content or ""

        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            raw_args = (
                getattr(tc, "arguments", None)
                or getattr(getattr(tc, "function", None), "arguments", None)
                or "{}"
            )
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            reasoning = args.get("reasoning") or None
            action_string = (args.get("action_string") or "").strip()

            # 1. Verbatim action_string path. Try to snap to a candidate first
            #    (exact match / case-insensitive / index), then fall through to
            #    the autoquote shim + parser-strict validator.
            if action_string:
                snapped = _canonicalize_to_candidate(action_string, candidate_actions)
                if snapped:
                    return snapped, reasoning, raw or json.dumps(args), None
                fixed = _autoquote_bids(action_string)
                if _validate_action_string(fixed):
                    return fixed, reasoning, raw or json.dumps(args), None

            # 2. Structured fallback: build the call from action_type + bid.
            structured = _structured_to_action_string(args)
            if structured and _validate_action_string(structured):
                return structured, reasoning, raw or json.dumps(args), None

        # 3. No tool call — try to extract a verbatim action from the raw text.
        for cand in candidate_actions:
            if cand in raw:
                return cand, None, raw, None

    except Exception as exc:
        err = repr(exc)
        logger.warning("[action-LLM] step %d failed: %s", step, exc)

    return None, None, raw, err


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _is_noop(prev_obs: Dict[str, Any], next_obs: Dict[str, Any]) -> bool:
    """Best-effort no-op detection for a browser env.

    URL + focused element + axtree-node count all unchanged AND no
    ``last_action_error`` ⇒ no-op.
    """
    if (next_obs.get("last_action_error") or "").strip():
        return False
    if (prev_obs.get("url") or "") != (next_obs.get("url") or ""):
        return False
    if (prev_obs.get("focused_element_bid") or "") != (next_obs.get("focused_element_bid") or ""):
        return False

    def _node_count(o: Dict[str, Any]) -> int:
        ax = o.get("axtree_object")
        if not isinstance(ax, dict):
            return -1
        return len(ax.get("nodes") or [])

    return _node_count(prev_obs) == _node_count(next_obs)


# Actions that *navigate away* from the current page. MiniWoB tasks
# terminate the moment the iframe leaves the task URL, so an override
# escalating to ``go_back()`` would destroy any chance of reward. We
# therefore exclude these from the override candidate pool.
_DESTRUCTIVE_NAV_ACTIONS = frozenset({
    "go_back()", "go_forward()", "new_tab()", "tab_close()",
})


def _pick_different(action: str, candidates: List[str]) -> str:
    """Pick a candidate that is neither ``action`` nor a destructive nav."""
    alts = [
        a for a in candidates
        if a != action and a not in _DESTRUCTIVE_NAV_ACTIONS
    ]
    if alts:
        return random.choice(alts)
    # Fall back to a non-destructive default; ``noop()`` is always safe.
    return "noop()"


def _is_nav_only_action(action: str) -> bool:
    """Return True iff ``action`` belongs to the navigation-only set.

    Used by the anti-thrash override (#6d) to detect the
    ``scroll/go_back/go_forward/noop`` loops that gpt-5.x low keeps falling
    into on VWA after a single failed search submission.
    """
    if not action:
        return False
    stripped = action.strip()
    return stripped.startswith(_NAV_ONLY_PREFIXES)


_FILL_ACTION_RE = re.compile(r'^fill\(\s*"([^"]+)"\s*,')
_STOP_GOAL_TOKENS = frozenset({
    "find", "get", "locate", "show", "tell", "navigate", "subscribe",
    "buy", "search", "help", "the", "a", "an", "of", "to", "for", "from",
    "on", "in", "at", "by", "with", "and", "or", "is", "are", "be", "this",
    "that", "me", "my", "our", "your", "their", "his", "her", "its",
    "please", "list", "all", "most", "least", "best", "worst",
})


def _extract_search_query(goal: str, max_words: int = 4) -> str:
    """Heuristic keyword extraction from the task goal for anti-thrash.

    Order of preference:
      1. The first quoted phrase ("..." or '...') — VWA goals frequently
         quote the literal item name ("the most expensive TV").
      2. The longest run of capitalised content words (e.g. ``Maryland NFL``).
      3. The 2-4 longest non-stopword tokens.

    Returns an empty string when nothing usable is extractable; callers
    should treat that as "anti-thrash cannot help, fall through".
    """
    if not goal:
        return ""

    quoted = re.search(r'["\'\u201c\u2018]([^"\'\u201d\u2019]{2,80})["\'\u201d\u2019]', goal)
    if quoted:
        candidate = quoted.group(1).strip()
        if 2 <= len(candidate) <= 80:
            return candidate

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", goal)
    cap_run: List[str] = []
    best_run: List[str] = []
    for tok in tokens:
        if tok[0].isupper() and tok.lower() not in _STOP_GOAL_TOKENS:
            cap_run.append(tok)
            if len(cap_run) > len(best_run):
                best_run = list(cap_run)
        else:
            cap_run = []
    if best_run:
        return " ".join(best_run[:max_words])

    content = [
        t for t in tokens
        if t.lower() not in _STOP_GOAL_TOKENS and len(t) >= 4
    ]
    content.sort(key=len, reverse=True)
    if content:
        return " ".join(content[:max_words])
    return ""


def _action_signature(action: str) -> str:
    """Return the canonical signature for repeat detection.

    Normalises whitespace + quotes inside the argument list so that
    ``click("211")``, ``click( "211" )`` and ``click(211)`` all hash to
    the same key. Argument-less actions like ``go_back()`` collapse to
    ``go_back``.
    """
    if not action:
        return ""
    s = action.strip()
    if s.endswith("()"):
        return s[:-2]
    m = re.match(r"^(\w+)\((.*)\)$", s)
    if not m:
        return s
    op, args = m.group(1), m.group(2)
    args_norm = re.sub(r"\s+", "", args).replace('"', "").replace("'", "")
    return f"{op}({args_norm})"


def _is_repeat_protected(action: str) -> bool:
    """Return True for action types that are exempt from #6e discouragement.

    ``go_back / go_forward / noop`` are kept available even when over-used
    so the agent always has a recovery escape hatch. ``scroll`` is NOT
    protected here — back-to-back identical scrolls are already caught by
    the consecutive-NOOP override (#6); the windowed mechanism adds the
    interleaved-scroll case (e.g. ``scroll → click → scroll → click``).
    """
    if not action:
        return False
    return action.strip().startswith(_REPEAT_PROTECTED_PREFIXES)


def _build_discouraged_signatures(
    sig_history: List[str], window: int = _REPEAT_WINDOW,
    threshold: int = _MAX_REPEATS_BEFORE_DISCOURAGE,
) -> Dict[str, int]:
    """Return ``{sig: count}`` for signatures with count >= ``threshold`` in
    the last ``window`` history entries. The count helps callers print
    informative override reasons (e.g. ``[anti-repeat: tried 3x]``).
    """
    if not sig_history:
        return {}
    recent = sig_history[-window:]
    counts: Dict[str, int] = {}
    for s in recent:
        counts[s] = counts.get(s, 0) + 1
    return {s: c for s, c in counts.items() if c >= threshold}


def _filter_repeat_candidates(
    candidate_actions: List[str], discouraged: Dict[str, int],
) -> List[str]:
    """Drop discouraged action signatures from the candidate list while
    preserving protected types (go_back/go_forward/noop) and backing off
    if the resulting list would be too small to give the LLM choice.

    Strategy:
      1. Mark each candidate ``a`` as discouraged iff
         ``_action_signature(a) in discouraged AND not _is_repeat_protected(a)``.
      2. If the surviving candidates ≥ ``_MIN_CANDIDATES_AFTER_FILTER``,
         use the filtered list (the agent must pick something else).
      3. Otherwise, fall back to the original list — running out of
         candidates is worse than letting the agent retry an unproductive
         click.
    """
    if not discouraged:
        return candidate_actions
    survivors = [
        a for a in candidate_actions
        if _is_repeat_protected(a) or _action_signature(a) not in discouraged
    ]
    if len(survivors) >= _MIN_CANDIDATES_AFTER_FILTER:
        return survivors
    return candidate_actions


def _build_anti_thrash_action(
    candidate_actions: List[str], goal: str,
) -> Optional[str]:
    """Return a ``fill(<bid>, <query>)`` action when one is feasible.

    Picks the *first* fill candidate (matches the order produced by
    ``browsergym_wrapper.tools._h_list_valid_actions`` which surfaces
    searchbox/textbox roles before generic textboxes), then substitutes
    a goal-derived query for the placeholder string.
    """
    query = _extract_search_query(goal)
    if not query:
        return None
    for cand in candidate_actions:
        m = _FILL_ACTION_RE.match(cand)
        if m:
            bid = m.group(1)
            safe = query.replace("\\", "\\\\").replace('"', '\\"')
            return f'fill("{bid}", "{safe}")'
    return None


_SCHEMA_ENTITY_RE = re.compile(
    r"^(e\d+)\[[^]]*\blabel=([^,\]]+?)(?:,[^]]*\bbid=([A-Za-z0-9_-]+))?",
    re.MULTILINE,
)


def _detect_consent_button_bid(
    obs: Dict[str, Any], canonical_schema: Optional[str] = None,
) -> Optional[str]:
    """Return the bid of a cookie/consent ACCEPT button if visible, else None.

    Strategy (most reliable first):
      1. Parse the canonical/heuristic schema text — every entity line is
         ``e<n>[type=element, label=<role> '<name>', bid=<bid>, ...]``,
         which is exactly the bid BrowserGym expects in ``click(bid)``.
      2. Fall back to ``extra_element_properties`` + AXTree walking.

    Only bids whose label matches a localized ``accept all`` keyword are
    returned, ranked by keyword priority (``accept all`` beats plain
    ``accept``).
    """
    candidates: List[Tuple[int, str, str]] = []  # (rank, bid, label)

    # 1. Schema-text parse (the safest source — these bids are guaranteed
    #    valid because the heuristic only emits entities for elements present
    #    in extra_element_properties).
    if canonical_schema:
        for m in _SCHEMA_ENTITY_RE.finditer(canonical_schema):
            label = (m.group(2) or "").lower()
            bid = m.group(3)
            if not bid:
                continue
            for rank, kw in enumerate(_CONSENT_ACCEPT_KEYWORDS):
                if kw in label:
                    candidates.append((rank, bid, label))
                    break

    # 2. extra_element_properties fallback — scan node text for keywords.
    if not candidates:
        extras = obs.get("extra_element_properties") or {}
        if isinstance(extras, dict):
            for bid, props in extras.items():
                if not isinstance(props, dict):
                    continue
                text_parts = []
                for k in ("name", "ariaLabel", "innerText", "value", "title"):
                    v = props.get(k)
                    if isinstance(v, str):
                        text_parts.append(v)
                blob = " ".join(text_parts).strip().lower()
                if not blob:
                    continue
                # Only auto-click clickable elements.
                if not (props.get("clickable")
                        or props.get("role") in ("button", "link", "menuitem")):
                    continue
                for rank, kw in enumerate(_CONSENT_ACCEPT_KEYWORDS):
                    if kw in blob:
                        candidates.append((rank, str(bid), blob[:80]))
                        break

    if not candidates:
        return None

    # Prefer the strongest keyword match (lowest rank).
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Playwright stealth + Google-consent pre-injection
# ---------------------------------------------------------------------------
# Realistic Chromium UA string (matches a stable Chrome 126 desktop build).
# Setting this defeats the most common headless-browser detector — Google's
# ``/sorry/index`` CAPTCHA flips on for the default Playwright UA but stays
# off for a vanilla Chrome desktop UA.
_STEALTH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
# JS init script run on every navigation in the BrowserGym context.
# Removes the ``navigator.webdriver`` flag that anti-bot scripts
# (incl. Google's /sorry CAPTCHA) sniff for. Equivalent in effect to
# ``--disable-blink-features=AutomationControlled``, but applied at
# the page level instead of via a launch flag — necessary because
# BrowserGym hard-rejects overrides on ``chromium.launch(args=...)``.
# The script is idempotent + harmless on non-Google sites.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
// Spoof a small chrome.runtime so simple checks stop probing further.
window.chrome = window.chrome || {runtime: {}};
// Plug a frequently-checked plugins-array hole.
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});
""".strip()
# Google's consent dialog appears on the first visit to any ``*.google.com``
# domain in EU jurisdictions and increasingly worldwide. Pre-seeding the
# ``SOCS`` and ``CONSENT`` cookies (the "I have already chosen, don't ask
# again" record) skips the dialog entirely. Without these, AssistantBench
# tasks (which all start at https://www.google.com/) burn 1-3 steps clicking
# the dialog — and the existing ``_detect_consent_button_bid`` heuristic
# only catches Western-locale "Accept all" / "Reject all" labels, so any
# locale flip silently wedges the agent. The cookies are well-known,
# documented values; setting them does not commit the user to anything
# server-side beyond "no, do not show me the consent dialog again".
_GOOGLE_CONSENT_COOKIES = (
    {
        "name": "SOCS",
        "value": "CAESHAgBEhJnd3NfMjAyMzAyMTYtMF9SQzIaAmVuIAEaBgiA_LyfBg",
        "domain": ".google.com",
        "path": "/",
        "expires": 2147483647,  # ~2038, max signed-int32 epoch second
        "httpOnly": False,
        "secure": True,
        "sameSite": "Lax",
    },
    {
        "name": "CONSENT",
        "value": "YES+srp.gws-20240101-0-RC1.en+FX+667",
        "domain": ".google.com",
        "path": "/",
        "expires": 2147483647,
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax",
    },
)


def _build_pw_stealth_kwargs(payload: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(pw_chromium_kwargs, pw_context_kwargs)`` for ``gym.make``.

    Always sets a realistic UA (cheap, works universally). For payloads
    that are likely to land on a Google property — anything in
    ``browsergym/assistantbench.*`` plus any ``openended`` URL pointing
    at ``*.google.*`` — also pre-seeds the SOCS/CONSENT cookies via
    ``storage_state`` so the consent dialog never appears.

    NOTE: We deliberately do NOT pass ``args=[...]`` to chromium.launch
    via ``pw_chromium_kwargs`` — BrowserGym's ``BrowserEnv.reset()``
    already passes ``args=`` and explicitly rejects overrides. The
    JS-level stealth (``navigator.webdriver`` removal) is instead
    applied via ``_apply_stealth_init_script`` after env construction.
    """
    pw_chromium_kwargs: Dict[str, Any] = {}
    pw_context_kwargs: Dict[str, Any] = {
        "user_agent": _STEALTH_USER_AGENT,
    }

    if _payload_likely_hits_google(payload):
        pw_context_kwargs["storage_state"] = {
            "cookies": [dict(c) for c in _GOOGLE_CONSENT_COOKIES],
            "origins": [],
        }

    return pw_chromium_kwargs, pw_context_kwargs


def _apply_stealth_init_script(env: Any, payload: str) -> None:
    """Inject ``navigator.webdriver = undefined`` (and friends) into the
    BrowserGym env's Playwright context, so anti-bot scripts on Google's
    /sorry CAPTCHA path can't fingerprint Playwright.

    Should be called AFTER ``env.reset()`` (the context is created
    during reset). The script registers as an *init script* — Playwright
    re-runs it on every subsequent page navigation in the same context,
    so even though it doesn't apply to the very first page load, it
    covers the search-submission step where the CAPTCHA actually
    triggers.

    Only fires when ``payload`` looks Google-bound; for MiniWoB / WebArena
    / WorkArena there's no anti-bot wall to bypass and we keep the
    runtime clean.
    """
    if not _payload_likely_hits_google(payload):
        return
    # Walk the gymnasium wrapper chain to reach the BrowserEnv that has .context.
    cur = env
    for _ in range(10):  # bound to avoid infinite loop on cyclic wrappers
        ctx = getattr(cur, "context", None)
        if ctx is not None:
            try:
                ctx.add_init_script(_STEALTH_INIT_SCRIPT)
            except Exception:
                # Best-effort — don't fail the rollout if the context
                # rejects (e.g. already closed).
                pass
            return
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt


def _get_active_browsergym_page(env: Any):
    """Walk the gymnasium wrapper chain to find the active Playwright
    page on the underlying ``BrowserEnv``. Returns ``None`` if no page
    is reachable (env not yet reset, env type unknown, or chain bottoms
    out before we hit a BrowserEnv).

    Used by ``_intercept_search_web`` to inject the synthetic search-
    results page directly into the live page (so the agent's next
    observation contains the result links). Mirrors the wrapper-walk
    in ``_apply_stealth_init_script`` but returns the page handle
    instead of the context.
    """
    cur = env
    for _ in range(10):  # bound to avoid infinite loop on cyclic wrappers
        page = getattr(cur, "page", None)
        if page is not None:
            return page
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return None


def _intercept_search_web(
    env: Any, action: str, *, k: int = 8, timeout: float = 8.0,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """If ``action`` is a ``search_web("...")`` call, run the search
    server-side and inject the result page into the live Playwright
    page; return ``("noop()", search_meta)`` so the caller hands the
    no-op to ``env.step()`` and the env's observation pipeline picks
    up the freshly-injected DOM on the next step.

    If ``action`` is NOT a search_web call, returns
    ``(action, None)`` unchanged — the normal env.step path.

    On any failure (no page handle, search backends all dead, page
    injection rejected) we fall back to a real ``goto(...)`` against
    DDG-HTML's NO-JS endpoint so the agent at least gets a chance to
    parse a real (if rate-limited) results page. Both the ``noop()``
    success path and the ``goto`` fallback are valid BrowserGym
    actions that the parser accepts; the substitution is invisible
    from the agent's history (it still sees ``search_web("...")``
    as the chosen action).

    Returns
    -------
    (substitute_action, search_meta)
        ``substitute_action`` is the string to pass to ``env.step``;
        ``search_meta`` is a dict with telemetry
        (``query``, ``n_results``, ``backend_used``, ``intercepted``,
        ``fallback``) to write into the experience's metadata, or
        ``None`` if no interception happened.
    """
    query = _parse_search_web_query(action)
    if query is None:
        return action, None
    # Lazy import keeps the search_backends module out of the import
    # graph for non-AssistantBench runs.
    try:
        from cold_start import search_backends
    except Exception:
        try:
            import search_backends  # type: ignore
        except Exception as e:
            logger.warning("search_backends import failed: %s", e)
            search_backends = None  # type: ignore
    page = _get_active_browsergym_page(env)
    meta: Dict[str, Any] = {
        "query": query,
        "intercepted": False,
        "n_results": 0,
        "fallback": None,
    }
    # Run the server-side search regardless of page availability — the
    # results dict is still useful telemetry, even if we end up
    # falling back to a goto().
    results: List[Dict[str, str]] = []
    if search_backends is not None:
        try:
            results = search_backends.search(query, k=k, timeout=timeout)
        except Exception as e:
            logger.warning("search_backends.search(%r) failed: %s", query[:60], e)
            results = []
    meta["n_results"] = len(results)
    if results:
        meta["backend_used"] = sorted({r.get("source", "?") for r in results})
    if page is None:
        # No live page handle (shouldn't happen if env was reset, but
        # handle defensively). Fall back to a real DDG-HTML goto so
        # the agent sees *some* result page.
        meta["fallback"] = "no_page_handle"
        url = (
            "https://html.duckduckgo.com/html/?q="
            + urllib.parse.quote(query)
        )
        return f'goto("{url}")', meta
    # Inject the synthetic results page. We use ``page.goto`` with a
    # data: URL (rather than ``page.set_content``) so the URL field
    # in the next observation reads ``data:text/html;...`` — a clear
    # signal to the agent (and to log readers) that this is a
    # synthetic page.
    if search_backends is not None and results:
        data_url = search_backends.results_to_data_url(query, results)
    elif search_backends is not None:
        # Empty result list — render an "empty" page anyway so the
        # agent learns that the search ran but didn't produce hits.
        data_url = search_backends.results_to_data_url(query, [])
        meta["fallback"] = "empty_results"
    else:
        # search_backends import failed; fall back to a real goto.
        meta["fallback"] = "import_failed"
        url = (
            "https://html.duckduckgo.com/html/?q="
            + urllib.parse.quote(query)
        )
        return f'goto("{url}")', meta
    try:
        # Short navigation timeout — data: URLs load instantly so this
        # only really times out if Chromium is stuck on a previous
        # page transition. Failures fall through to the goto() path.
        page.goto(data_url, wait_until="domcontentloaded", timeout=5000)
        meta["intercepted"] = True
        # After a successful injection we hand a noop() to env.step
        # so the env's standard observation extraction runs against
        # the new DOM without firing any extra browser action.
        return "noop()", meta
    except Exception as e:
        logger.warning("search_web injection failed: %s", e)
        meta["fallback"] = f"injection_failed: {type(e).__name__}"
        url = (
            "https://html.duckduckgo.com/html/?q="
            + urllib.parse.quote(query)
        )
        return f'goto("{url}")', meta


def _payload_likely_hits_google(payload: str) -> bool:
    """True if the env created from ``payload`` will probably load a Google
    property in the first few steps. Used to gate the consent-cookie
    pre-injection so we don't inject cookies into envs that don't need
    them (MiniWoB / WebArena / WorkArena)."""
    if not isinstance(payload, str):
        return False
    p = payload.lower()
    if "assistantbench" in p:
        return True
    if p.startswith(("http://", "https://")):
        # ``openended`` URL — match google.com / google.co.uk / etc.
        if "google." in p.split("/", 3)[2]:
            return True
    return False


def _capture_initial_obs(
    *, kind: str, payload: str, headless: bool, seed: Optional[int] = None,
):
    """Boot a real BrowserGym env and reset it.

    Parameters
    ----------
    kind : "task" | "url"
        ``"task"``: ``payload`` is a registered BrowserGym env id
        (e.g. ``"browsergym/miniwob.click-button"``); the task ships with
        its own goal + reward function, so no ``task_kwargs`` are needed.
        ``"url"``: ``payload`` is a live URL; we boot
        ``browsergym/openended`` with ``task_kwargs={"start_url": ...}``.
    headless : bool
        Whether to run Chromium headless. ``False`` is useful for
        interactive debugging.
    seed : int, optional
        Forwarded to ``env.reset(seed=seed)`` so episodes are reproducible.

    Returns ``(env, obs, info)`` where ``env`` is the live BrowserEnv
    (Playwright + Chromium), ``obs`` is the observation dict from
    ``env.reset()`` (carries ``screenshot``, ``axtree_object``,
    ``extra_element_properties``, ``focused_element_bid``, ``url``,
    ``last_action`` / ``last_action_error`` …), and ``info`` is the
    Gymnasium info dict.

    Raises ``ImportError`` if BrowserGym / Playwright is not installed,
    or ``ValueError`` if the requested env id is not registered.
    """
    import gymnasium as gym
    import browsergym.core  # noqa: F401  -- registers openended

    pw_chromium_kwargs, pw_context_kwargs = _build_pw_stealth_kwargs(payload)

    if kind == "task":
        env_id = payload
        if env_id not in gym.envs.registry:
            raise ValueError(
                f"BrowserGym env id {env_id!r} is not registered. "
                f"Did you forget to import the suite (e.g. browsergym.miniwob)? "
                f"Use --list_tasks to see registered ids."
            )
        env = gym.make(
            env_id,
            headless=headless,
            pw_chromium_kwargs=pw_chromium_kwargs,
            pw_context_kwargs=pw_context_kwargs,
        )
    elif kind == "url":
        env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": payload},
            headless=headless,
            pw_chromium_kwargs=pw_chromium_kwargs,
            pw_context_kwargs=pw_context_kwargs,
        )
    else:
        raise ValueError(f"Unknown target kind: {kind!r}")

    try:
        obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        obs, info = env.reset()
    # Inject JS-level stealth post-reset so subsequent navigations
    # (e.g. search-result pages) escape Google's anti-bot CAPTCHA. No-op
    # for non-Google payloads (MiniWoB / WebArena / WorkArena).
    _apply_stealth_init_script(env, payload)
    return env, dict(obs), dict(info or {})


def _step_env(
    env: Any, action: str,
) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
    """Step the live BrowserGym env once with ``action`` (a high-level string)."""
    obs, reward, term, trunc, info = env.step(action)
    return dict(obs), float(reward or 0.0), bool(term), bool(trunc), dict(info or {})


def run_actor_episode(
    *,
    target_kind: str,
    target_payload: str,
    target_safe_id: str,
    headless: bool,
    max_steps: int,
    client: Any,
    routed_model: str,
    fallback_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    use_vision: bool,
    temperature_action: float,
    temperature_schema: float,
    max_entities: int,
    frames_dir: Optional[Path],
    seed: Optional[int],
    verbose: bool,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one BrowserGym episode end-to-end and return ``(Episode, stats)``."""
    bg_obs_to_schema = _import_browsergym_heuristic()
    build_browser_registry = _import_browser_tools()

    if seed is not None:
        random.seed(seed)

    env, obs, _info = _capture_initial_obs(
        kind=target_kind, payload=target_payload, headless=headless, seed=seed,
    )

    if target_kind == "task":
        task_id = target_payload
        task_human = (
            f"Solve the BrowserGym task {target_payload}. "
            + (_extract_goal(obs) or "")
        ).strip()
    else:
        task_id = f"browsergym/openended/{target_safe_id}"
        task_human = (
            f"Browse {target_payload} to satisfy the page goal. "
            + (_extract_goal(obs) or "")
        ).strip()
    task = task_human
    goal = _extract_goal(obs)

    experiences: List[Experience] = []
    history: List[Dict[str, Any]] = []
    consecutive_noops = 0
    last_noop_action: Optional[str] = None
    consecutive_errors = 0
    last_error_bid: Optional[str] = None
    bad_bids: List[str] = []        # bids that errored at any point this ep
    consecutive_nav_actions = 0    # for anti-thrash override (#6d)
    anti_thrash_fires = 0          # diagnostic counter
    action_sig_history: List[str] = []  # for anti-repetition override (#6e)
    anti_repeat_fires = 0           # diagnostic counter
    anti_repeat_drops = 0           # # of candidates dropped from prompts
    consent_dismissed = False  # only auto-click cookie accept once per episode
    schema_calls = 0
    schema_ok = 0
    action_llm_ok = 0
    action_llm_fail = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    som_telemetry: Dict[str, int] = {
        "n_extras": 0, "n_set_of_marks": 0, "n_clickable": 0, "n_input_role": 0,
    }

    t0 = time.time()
    try:
        for step in range(max_steps):
            # Step-0: snapshot SoM telemetry from the initial obs so the
            # parent run can detect SoM-blind episodes (extras populated
            # but no ``set_of_marks=True`` flags) without scanning the
            # whole rollout dump. See ``_count_som_telemetry`` doc.
            if step == 0:
                som_telemetry = _count_som_telemetry(obs)

            # 1. Pull the screenshot for the VLM and (optionally) save it.
            pil = _to_pil(obs.get("screenshot"))
            img_path: Optional[str] = None
            if pil is not None and frames_dir is not None:
                img_path = _save_frame(pil, frames_dir / f"step_{step:03d}.png")

            # 2. Heuristic visual grounding (deterministic, AXTree-walked).
            try:
                canonical_schema = bg_obs_to_schema(
                    obs, step=step, task_id=task_id, max_entities=max_entities,
                )
            except Exception as exc:
                logger.debug("heuristic obs_to_schema failed: %s", exc)
                canonical_schema = None

            # 3. Candidate-action vocabulary from the browser tool registry.
            registry = None
            if build_browser_registry is not None:
                try:
                    registry = build_browser_registry(obs)
                except Exception as exc:
                    logger.debug("build_browser_registry failed: %s", exc)
            candidate_actions, candidate_meta = _build_candidate_actions(
                obs=obs, registry=registry,
                task_id=task_id, goal=goal,
            )

            # 3b. Anti-REPETITION filter (#6e) — drop candidate actions whose
            #     signature has already been tried >= ``_MAX_REPEATS_BEFORE_DISCOURAGE``
            #     times in the last ``_REPEAT_WINDOW`` steps. The filtered list
            #     is what the action LLM and schema VLM see. ``go_back`` /
            #     ``go_forward`` / ``noop`` are always kept (recovery path).
            #     If the filter would leave fewer than
            #     ``_MIN_CANDIDATES_AFTER_FILTER`` choices, it backs off to the
            #     unfiltered list. Catches the May-3 ``visualwebarena.96``
            #     ``click("211")`` 7x loop and the ``.433`` ``fill+press``
            #     loops that the per-step anti-NOOP / anti-error overrides
            #     missed.
            discouraged_signatures = _build_discouraged_signatures(
                action_sig_history,
            )
            if discouraged_signatures:
                pre_filter_n = len(candidate_actions)
                candidate_actions = _filter_repeat_candidates(
                    candidate_actions, discouraged_signatures,
                )
                dropped_now = pre_filter_n - len(candidate_actions)
                if dropped_now > 0:
                    anti_repeat_drops += dropped_now
                    if verbose:
                        avoid_str = ", ".join(
                            f"{sig}({c}x)" for sig, c in
                            sorted(discouraged_signatures.items())
                        )
                        print(
                            f"  step {step}: anti-repeat dropped "
                            f"{dropped_now}/{pre_filter_n} candidates — "
                            f"discouraged: {avoid_str}"
                        )

            # 4. Visual grounding (vision call): screenshot → schema.
            schema_text: Optional[str] = None
            schema_meta: Dict[str, Any] = {
                "schema": None, "raw": "", "source": "skipped", "error": None,
            }
            if (
                use_vision and pil is not None
                and schema_helpers is not None and client is not None
            ):
                schema_budget = (
                    _SCHEMA_MAX_TOKENS_REASONING
                    if _is_reasoning_model(routed_model)
                    else _SCHEMA_MAX_TOKENS
                )
                # VWA tasks ride goal images on goal_object — surface them
                # to the schema VLM so it can ground "this product / this
                # chair / this image" references in the goal text.
                goal_imgs = _extract_goal_images(obs)
                schema_meta = generate_schema_from_image(
                    pil_image=pil,
                    obs=obs,
                    task_id=task_id,
                    goal=goal,
                    step=step,
                    candidate_actions=candidate_actions,
                    client=client,
                    routed_model=routed_model,
                    schema_helpers=schema_helpers,
                    canonical_fallback=canonical_schema,
                    temperature=temperature_schema,
                    max_tokens=schema_budget,
                    max_entities=max_entities,
                    use_som=True,
                    goal_images=goal_imgs,
                    reasoning_effort=reasoning_effort,
                )
                schema_calls += 1
                if schema_meta.get("source") == "vlm":
                    schema_ok += 1
                schema_text = schema_meta.get("schema")
            else:
                schema_text = canonical_schema
                schema_meta = {
                    "schema": canonical_schema,
                    "raw": "",
                    "source": "canonical" if canonical_schema else "text_only",
                    "error": None,
                }

            # 4b. Auto-dismiss cookie / consent dialogs ONCE before handing
            #     control to the LLM. Especially valuable for AssistantBench
            #     which boots on google.com behind a localized consent wall.
            preempted_action: Optional[str] = None
            if not consent_dismissed:
                cb = _detect_consent_button_bid(obs, canonical_schema)
                if cb is not None:
                    preempted_action = f'click("{cb}")'
                    consent_dismissed = True
                    if verbose:
                        print(f"  step {step}: consent-dismiss override -> {preempted_action!r}")

            # 5. Action selection (text-only call: schema → action). Skipped
            #    when we're pre-empting with a consent click, but we still
            #    want a placeholder reasoning trace for the Experience record.
            if preempted_action is not None:
                action = preempted_action
                reasoning = ("Auto-clicking the cookie/consent accept button to "
                             "unblock the page (deterministic pre-LLM heuristic).")
                action_raw = ""
                action_err = None
                action_llm_ok += 1
            else:
                action, reasoning, action_raw, action_err = select_action(
                    schema_text=schema_text,
                    obs=obs,
                    candidate_actions=candidate_actions,
                    candidate_meta=candidate_meta,
                    task=task,
                    step=step,
                    history=history,
                    client=client,
                    routed_model=routed_model,
                    temperature=temperature_action,
                    reasoning_effort=reasoning_effort,
                )
                if action is not None:
                    action_llm_ok += 1
                else:
                    action_llm_fail += 1
                    action = candidate_actions[0] if candidate_actions else "noop()"
                    if last_noop_action == action and len(candidate_actions) > 1:
                        action = _pick_different(action, candidate_actions)

            # 6. Anti-NOOP override.
            if (
                consecutive_noops >= _MAX_CONSECUTIVE_NOOPS
                and action == last_noop_action
                and len(candidate_actions) > 1
            ):
                old_action = action
                action = _pick_different(action, candidate_actions)
                reasoning = (
                    (reasoning or "")
                    + f" [auto-override: '{old_action}' was no-op {consecutive_noops}x]"
                )
                if verbose:
                    print(f"  step {step}: anti-noop override {old_action!r} -> {action!r}")

            # 6b. Anti-ERROR override — if the chosen action targets a
            #     bid that has already errored at any point this episode,
            #     force a different bid. We deliberately avoid escalating
            #     to ``go_back()``: in MiniWoB it leaves the task page and
            #     terminates the episode with reward 0, hiding model bugs.
            chosen_bid = _bid_from_action(action)
            if (
                chosen_bid is not None
                and chosen_bid in bad_bids
                and len(candidate_actions) > 1
            ):
                old_action = action
                alt_pool = [
                    a for a in candidate_actions
                    if (_bid_from_action(a) or "") not in bad_bids
                    and a not in _DESTRUCTIVE_NAV_ACTIONS
                ]
                if not alt_pool:
                    # Stay on the page rather than navigating away.
                    alt_pool = ["scroll(0, 300)", "scroll(0, -300)", "noop()"]
                action = random.choice(alt_pool)
                reasoning = (
                    (reasoning or "")
                    + f" [auto-override: bid={chosen_bid} previously errored — "
                    + f"choosing {action!r} (bad_bids={bad_bids})]"
                )
                if verbose:
                    print(f"  step {step}: anti-error override {old_action!r} -> {action!r} (bad bid {chosen_bid})")

            # 6c2. Anti-REPETITION override (post-LLM swap) — if the LLM
            #      picked an action whose signature is in the discouraged
            #      set despite the candidate-list filtering at #3b (some
            #      models ignore the suggested list), swap to a non-
            #      discouraged candidate. ``go_back/go_forward/noop`` are
            #      protected — never swapped away. Skipped silently if no
            #      non-discouraged alternative exists.
            chosen_sig = _action_signature(action)
            if (
                chosen_sig in discouraged_signatures
                and not _is_repeat_protected(action)
                and candidate_actions
            ):
                alt_pool = [
                    a for a in candidate_actions
                    if (_action_signature(a) not in discouraged_signatures
                        or _is_repeat_protected(a))
                    and a != action
                ]
                if alt_pool:
                    old_action = action
                    repeat_count = discouraged_signatures.get(chosen_sig, 0)
                    action = _pick_different(action, alt_pool)
                    anti_repeat_fires += 1
                    reasoning = (
                        (reasoning or "")
                        + f" [auto-override: anti-repeat — {chosen_sig} "
                        + f"already tried {repeat_count}x in last "
                        + f"{_REPEAT_WINDOW} steps, switched to {action!r}]"
                    )
                    if verbose:
                        print(
                            f"  step {step}: anti-repeat override "
                            f"{old_action!r} -> {action!r} "
                            f"({chosen_sig} tried {repeat_count}x)"
                        )

            # 6d. Anti-THRASH override — if the agent has just emitted
            #     ``_MAX_CONSECUTIVE_NAV`` navigation-only actions in a row
            #     (scroll/go_back/go_forward/noop) AND the *current* chosen
            #     action is again nav-only AND the page surfaces a
            #     ``fill(...)`` candidate, force a fill with a goal-derived
            #     query. Catches the post-search blocked-page recovery
            #     pattern surfaced in the May-3 visualwebarena.92 diagnostic
            #     where gpt-5.5 low looped through 28 nav-only steps.
            if (
                consecutive_nav_actions >= _MAX_CONSECUTIVE_NAV
                and _is_nav_only_action(action)
            ):
                forced = _build_anti_thrash_action(candidate_actions, goal)
                if forced is not None and forced != action:
                    old_action = action
                    prior_nav_run = consecutive_nav_actions
                    action = forced
                    anti_thrash_fires += 1
                    consecutive_nav_actions = 0
                    reasoning = (
                        (reasoning or "")
                        + f" [auto-override: anti-thrash after "
                        + f"{prior_nav_run} nav-only actions, "
                        + f"forcing {forced!r}]"
                    )
                    if verbose:
                        print(
                            f"  step {step}: anti-thrash override "
                            f"{old_action!r} -> {action!r} "
                            f"(consecutive_nav was {prior_nav_run})"
                        )

            # 6c. Last-mile defense: quote any unquoted bid that slipped
            #     through (no-op for already-quoted candidate strings).
            #     ``click(12)`` becomes ``click("12")`` automatically.
            normalized_action = _autoquote_bids(action)
            if normalized_action != action and verbose:
                print(f"  step {step}: autoquote {action!r} -> {normalized_action!r}")
            action = normalized_action

            # 6e. Synthetic ``search_web("...")`` interception. If the
            #     LLM emitted a search_web call, run the search server-
            #     side via ``search_backends.search`` (HTTP from Python,
            #     bypasses the Playwright TLS-fingerprint anti-bot wall),
            #     inject the result page into the live Playwright page,
            #     and substitute ``noop()`` (or a goto fallback) for
            #     ``env.step()``. The agent's history still records the
            #     original ``search_web("...")`` so the SFT/GRPO
            #     consumer sees the agent's logical action, not the
            #     implementation detail.
            search_meta: Optional[Dict[str, Any]] = None
            if _parse_search_web_query(action) is not None:
                action_for_history = action
                substitute_action, search_meta = _intercept_search_web(env, action)
                if verbose:
                    print(
                        f"  step {step}: search_web intercept "
                        f"{action!r} -> step({substitute_action!r}) "
                        f"meta={search_meta}"
                    )
                action_for_step = substitute_action
            else:
                action_for_history = action
                action_for_step = action

            # 7. Step the env.
            try:
                next_obs, reward, terminated, truncated, _next_info = _step_env(
                    env, action_for_step,
                )
            except Exception as exc:
                logger.error(
                    "[%s] step %d env.step(%r) failed: %s",
                    target_payload, step, action_for_step, exc,
                )
                if verbose:
                    traceback.print_exc()
                break
            # After env.step, pivot ``action`` back to the agent's
            # *logical* action string so all downstream bookkeeping
            # (history, anti-thrash, anti-repeat, sidecar JSON, the
            # Experience record consumed by SFT/GRPO) sees what the
            # agent chose — not the implementation-detail substitute
            # we passed to env.step. For non-search_web actions this
            # is a no-op (action_for_history == action_for_step).
            action = action_for_history
            total_reward += reward
            done = bool(terminated) or bool(truncated)
            error_text = (next_obs.get("last_action_error") or "").strip()

            is_noop = _is_noop(obs, next_obs)
            history.append({
                "action": action,
                "reward": reward,
                "noop": is_noop,
                "error": bool(error_text),
                "error_text": error_text,
            })

            if is_noop and action == last_noop_action:
                consecutive_noops += 1
            elif is_noop:
                consecutive_noops = 1
                last_noop_action = action
            else:
                consecutive_noops = 0
                last_noop_action = None

            had_error = bool(error_text)
            this_bid = _bid_from_action(action)
            if had_error:
                # Record this bid as bad so we never re-target it.
                if this_bid is not None and this_bid not in bad_bids:
                    bad_bids.append(this_bid)
                # Cap memory so the avoid-list doesn't grow unbounded.
                bad_bids = bad_bids[-12:]
                if this_bid is not None and this_bid == last_error_bid:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 1
                    last_error_bid = this_bid
            else:
                consecutive_errors = 0
                last_error_bid = None

            # Anti-thrash counter: count the *executed* action against the
            # nav-only run. A successful fill/click/check resets it; an
            # anti-thrash override already reset to 0 above and the synth
            # action is a fill, so this branch keeps the reset stable.
            if _is_nav_only_action(action):
                consecutive_nav_actions += 1
            else:
                consecutive_nav_actions = 0

            # Anti-repetition history: append the *executed* action signature
            # (post-overrides) so the next step's #3b filter and #6c2 swap
            # see the action that really happened, not the LLM's first
            # picks. Cap memory at 3x window to bound the per-episode
            # tracking footprint.
            action_sig_history.append(_action_signature(action))
            action_sig_history = action_sig_history[-(_REPEAT_WINDOW * 3):]

            # 8. Build the Experience record (use compact text observations
            #    that summarise the page; raw obs is too large to serialize).
            obs_summary = (
                f"url={obs.get('url','')} "
                f"focused_bid={obs.get('focused_element_bid','') or 'null'} "
                f"goal={goal[:120]}"
            )
            next_obs_summary = (
                f"url={next_obs.get('url','')} "
                f"focused_bid={next_obs.get('focused_element_bid','') or 'null'} "
                f"err={(error_text or 'null')[:120]}"
            )
            exp = Experience(
                state=obs_summary,
                action=str(action),
                reward=reward,
                next_state=next_obs_summary,
                done=done,
                intentions=reasoning,
                tasks=task,
            )
            exp.idx = step
            exp.action_type = "primitive"
            exp.raw_state = obs_summary[:4000]
            exp.raw_next_state = next_obs_summary[:4000]
            exp.available_actions = list(candidate_actions)
            exp.interface = {
                "env_name": "browsergym",
                "game_name": target_safe_id,
                "target_kind": target_kind,
                "target_payload": target_payload,
                "url": obs.get("url", target_payload if target_kind == "url" else ""),
                "wrapper": (
                    target_payload if target_kind == "task" else "browsergym/openended"
                ),
            }

            # Stash schema + VLM outputs both on Experience.extras (in-memory)
            # AND on Experience.metadata (which Experience.to_dict serialises).
            extras: Dict[str, Any] = {
                "schema": schema_text,
                "schema_source": schema_meta.get("source"),
                "schema_error": schema_meta.get("error"),
                "schema_canonical": canonical_schema,
                "candidate_actions": list(candidate_actions),
                "candidate_meta": candidate_meta[:_MAX_CANDIDATE_ACTIONS],
                "is_noop": is_noop,
                "error_text": error_text or None,
                "url": obs.get("url"),
                "focused_element_bid": obs.get("focused_element_bid") or None,
            }
            if schema_meta.get("finish_reason") is not None:
                extras["schema_finish_reason"] = schema_meta.get("finish_reason")
            if schema_meta.get("recovery"):
                extras["schema_recovery"] = schema_meta.get("recovery")
            schema_raw = schema_meta.get("raw") or ""
            if schema_raw:
                extras["schema_raw_excerpt"] = schema_raw[:4000]
                extras["schema_raw_full_len"] = schema_meta.get(
                    "raw_full_len", len(schema_raw)
                )
            if action_raw:
                extras["action_raw"] = action_raw[:1000]
            if action_err:
                extras["action_error"] = action_err
            if img_path:
                extras["frame_path"] = img_path
            # search_web telemetry: only present on intercepted steps.
            # Captures the query, backend that returned results, count
            # of results, and any fallback path taken — invaluable for
            # post-hoc debugging of "did the search actually fire?".
            if search_meta is not None:
                extras["search_web_meta"] = dict(search_meta)
            exp.extras = extras
            existing_meta = getattr(exp, "metadata", None) or {}
            if isinstance(existing_meta, dict):
                existing_meta = dict(existing_meta)
            else:
                existing_meta = {}
            existing_meta.update(extras)
            exp.metadata = existing_meta
            experiences.append(exp)

            # Dump a sidecar JSON next to each frame so the PNG is
            # self-describing on disk (action / reward / schema / url …).
            if img_path and frames_dir is not None:
                try:
                    sidecar = {
                        "step": step,
                        "url": obs.get("url"),
                        "next_url": next_obs.get("url"),
                        "action": action,
                        "action_raw": (action_raw or "")[:1000],
                        "action_error": action_err,
                        "reward": reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "is_noop": is_noop,
                        "schema_source": schema_meta.get("source"),
                        "schema_error": schema_meta.get("error"),
                        "schema": schema_text,
                        "candidate_actions": list(candidate_actions),
                        "focused_element_bid": obs.get("focused_element_bid"),
                        "frame_path": img_path,
                        "target_kind": target_kind,
                        "target_payload": target_payload,
                    }
                    sidecar_path = frames_dir / f"step_{step:03d}.json"
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar, f, indent=2, ensure_ascii=False, default=str)
                except Exception as exc:
                    logger.debug("frame sidecar write failed at step %d: %s", step, exc)

            if verbose:
                r_short = (
                    (reasoning[:80] + "...")
                    if reasoning and len(reasoning) > 80 else reasoning
                )
                tag = (
                    " [ERR]" if error_text else
                    (" [NOOP]" if is_noop else "")
                )
                print(
                    f"  step {step:>3}: action={action!r:<32} "
                    f"reward={reward:+.2f} cum={total_reward:+.2f}{tag} "
                    f"schema={schema_meta.get('source')} reason={r_short}"
                )

            obs = next_obs
            if done:
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    elapsed = time.time() - t0

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="browsergym",
        game_name=target_safe_id,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "target_kind": target_kind,
        "target_payload": target_payload,
        "target_safe_id": target_safe_id,
        "wrapper": (
            target_payload if target_kind == "task" else "browsergym/openended"
        ),
        "macro": False,
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "elapsed_seconds": round(elapsed, 2),
        "model": fallback_model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_browsergym",
        "use_vision": use_vision,
        "schema_calls": schema_calls,
        "schema_ok": schema_ok,
        "action_llm_ok": action_llm_ok,
        "action_llm_fail": action_llm_fail,
        "noop_steps": sum(1 for h in history if h["noop"]),
        "error_steps": sum(1 for h in history if h.get("error")),
        "som_telemetry": som_telemetry,
        "anti_thrash_fires": anti_thrash_fires,
        "anti_repeat_fires": anti_repeat_fires,
        "anti_repeat_drops": anti_repeat_drops,
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _count_existing_episodes(env_dir: Path) -> int:
    if not env_dir.exists():
        return 0
    return sum(
        1 for f in env_dir.glob("episode_*.json")
        if f.name != "episode_buffer.json"
    )


def _save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_target_rollouts(
    target_kind: str,
    target_payload: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all episodes for one BrowserGym target and persist outputs.

    ``target_kind`` is ``"task"`` (BrowserGym env id) or ``"url"`` (live URL
    on top of ``browsergym/openended``).
    """
    safe = (
        _task_safe(target_payload) if target_kind == "task"
        else _url_safe(target_payload)
    )
    target_dir = output_dir / safe
    target_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = target_dir / "rollouts.jsonl"

    target_episodes = args.episodes
    effective_max_steps = args.max_steps
    label = f"{target_kind}:{target_payload}"

    start_idx = 0
    if args.resume:
        start_idx = _count_existing_episodes(target_dir)
        if start_idx >= target_episodes:
            print(f"  [SKIP] {label}: {start_idx}/{target_episodes} episodes already done")
            return {
                "target_kind": target_kind,
                "target_payload": target_payload,
                "target_safe_id": safe,
                "skipped": True,
                "existing": start_idx,
                "target_episodes": target_episodes,
            }
        if start_idx > 0:
            print(f"  [RESUME] {label}: starting from episode {start_idx}")

    buffer = Episode_Buffer(buffer_size=target_episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t_target = time.time()

    for ep_idx in range(start_idx, target_episodes):
        print(f"\n  [{label}] Episode {ep_idx + 1}/{target_episodes}")
        try:
            frames_dir = (
                target_dir / "frames" / f"ep_{ep_idx:03d}"
                if args.save_frames else None
            )

            episode, stats = run_actor_episode(
                target_kind=target_kind,
                target_payload=target_payload,
                target_safe_id=safe,
                headless=not args.no_headless,
                max_steps=effective_max_steps,
                client=client,
                routed_model=routed_model,
                fallback_model=args.model,
                schema_helpers=schema_helpers,
                use_vision=not args.no_vision,
                temperature_action=args.temperature_action,
                temperature_schema=args.temperature_schema,
                max_entities=args.max_entities,
                frames_dir=frames_dir,
                seed=42 + ep_idx,
                verbose=args.verbose,
                reasoning_effort=getattr(args, "reasoning_effort", None),
            )
            stats["episode_index"] = ep_idx
            print(
                f"    steps={stats['steps']:>3} "
                f"reward={stats['total_reward']:+.2f} "
                f"schema_ok={stats['schema_ok']}/{stats['schema_calls']} "
                f"action_ok={stats['action_llm_ok']} (fail={stats['action_llm_fail']}) "
                f"noops={stats['noop_steps']} errs={stats['error_steps']}"
            )

            buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            with open(target_dir / f"episode_{ep_idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
            _save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as exc:
            print(f"    [ERROR] episode {ep_idx + 1} failed: {exc}")
            traceback.print_exc()
            all_stats.append({
                "target_kind": target_kind,
                "target_payload": target_payload,
                "episode_index": ep_idx,
                "error": str(exc),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed_target = time.time() - t_target
    buffer.save_to_json(str(target_dir / "episode_buffer.json"))
    print(f"\n  Saved {len(buffer)} episodes for {label} in {elapsed_target:.1f}s")

    # ── Watchdog (added 2026-05-03) ──────────────────────────────────────
    # Print per-task timing in a single grep-able line so a parent shell
    # can ``grep '\[WATCHDOG\]' run.log | awk ...`` without scraping the
    # whole rollout_summary.json. Compare against the OSWorld baseline
    # ~24 s/step (gpt-5.4 default reasoning) measured 2026-04-29 from
    # ``Cold-start-out-osworld/gpt5.4_3per_domain``. VWA's Playwright
    # observation path is ~3× lighter than OSWorld's KVM-VM step, so
    # >30 s/step on VWA almost always means we accidentally re-enabled
    # a steering / high-reasoning knob — exactly the inflation pattern
    # that pushed the May-3 ``full7-smoke`` to 95.6 s/step (see the
    # OSWorld `bucketA-smoke` / `full7-smoke` run-comparison table).
    _wd_total_steps = sum(
        s.get("steps", 0) or 0 for s in all_stats if "error" not in s
    )
    _wd_total_eps = sum(1 for s in all_stats if "error" not in s)
    if _wd_total_steps > 0 and elapsed_target > 0:
        _wd_per_step = elapsed_target / _wd_total_steps
        _wd_status = (
            "SLOW " if _wd_per_step > 30.0
            else "FAST " if _wd_per_step < 12.0
            else "ok   "
        )
        print(
            f"  [WATCHDOG {_wd_status}] {label}: "
            f"eps={_wd_total_eps} steps={_wd_total_steps} "
            f"elapsed={elapsed_target:.1f}s "
            f"sec_per_step={_wd_per_step:.1f}s "
            f"(reasoning_effort={getattr(args, 'reasoning_effort', None) or 'unset'})"
        )
        if _wd_per_step > 30.0:
            print(
                f"  [WATCHDOG WARN] {_wd_per_step:.1f}s/step > 30s red-line. "
                f"Likely culprits: reasoning_effort=high/medium, steering "
                f"modules enabled, or LLM provider routing latency. "
                f"Cross-check ``rollout_summary.json:elapsed_seconds`` and "
                f"compare with the May-3 OSWorld watchdog table in the "
                f"implementation_notes/ memo."
            )

    # ── SoM watchdog (added 2026-05-03) ──────────────────────────────────
    # Surface SoM-blind episodes: cases where ``--use_som`` is on (the
    # default) and ``extra_element_properties`` came back populated, BUT
    # zero elements actually carried ``set_of_marks=True``. In that case
    # the overlay renders a passthrough with no bid-tagged boxes, which
    # silently downgrades the actor to a "raw screenshot + AXTree" agent
    # — the same configuration that under-performs the GPT-4V SoM
    # baseline (16.4 %) in the VWA paper. See
    # ``legacy/visualwebarena/vwa-improvement-plan.md`` Tier-1 D.
    # SoM overlay is unconditionally on when vision is on (driver hard-codes
    # ``use_som=True`` in its action-LLM call site). Skip the watchdog when
    # ``--no_vision`` short-circuited the whole VLM path.
    use_som_active = not getattr(args, "no_vision", False)
    if use_som_active:
        _som_eps = [s for s in all_stats if "som_telemetry" in s]
        if _som_eps:
            _som_blind = [
                s for s in _som_eps
                if s["som_telemetry"].get("n_extras", 0) > 0
                and s["som_telemetry"].get("n_set_of_marks", 0) == 0
            ]
            _avg_extras = sum(
                s["som_telemetry"].get("n_extras", 0) for s in _som_eps
            ) / max(1, len(_som_eps))
            _avg_som = sum(
                s["som_telemetry"].get("n_set_of_marks", 0) for s in _som_eps
            ) / max(1, len(_som_eps))
            _avg_input = sum(
                s["som_telemetry"].get("n_input_role", 0) for s in _som_eps
            ) / max(1, len(_som_eps))
            print(
                f"  [SOM] {label}: eps={len(_som_eps)} "
                f"avg_extras={_avg_extras:.1f} "
                f"avg_set_of_marks={_avg_som:.1f} "
                f"avg_input_role={_avg_input:.1f} "
                f"blind_eps={len(_som_blind)}/{len(_som_eps)}"
            )
            if _som_blind:
                print(
                    f"  [SOM WARN] {len(_som_blind)}/{len(_som_eps)} episodes "
                    f"had populated extras but zero set_of_marks flags. "
                    f"The actor saw a passthrough screenshot rather than a "
                    f"bid-tagged overlay; expect VWA-style multi-constraint "
                    f"tasks to thrash. Investigate "
                    f"``BrowserGym.utils.obs.overlay_som`` integration with "
                    f"this BrowserGym version."
                )

    # Anti-thrash watchdog: how often did override #6d fire across the run?
    # Useful for tuning ``_MAX_CONSECUTIVE_NAV`` and confirming the search-
    # first heuristic is no longer needed once the agent reliably picks
    # ``fill(...)`` on its own. ``fires=0`` on a run that included
    # search-heavy tasks indicates either the agent is already breaking out
    # of nav-loops on its own, or the threshold is too high.
    _at_eps = [s for s in all_stats if "anti_thrash_fires" in s]
    if _at_eps:
        _total_fires = sum(s.get("anti_thrash_fires", 0) for s in _at_eps)
        _eps_with_fire = sum(
            1 for s in _at_eps if s.get("anti_thrash_fires", 0) > 0
        )
        print(
            f"  [ANTI-THRASH] {label}: eps={len(_at_eps)} "
            f"total_fires={_total_fires} "
            f"eps_with_fire={_eps_with_fire}/{len(_at_eps)}"
        )

    # Anti-repetition watchdog (#6e): how many candidates were filtered
    # out across the run, and how often did the post-LLM swap fire?
    # ``drops`` measures how often the agent saw a *smaller* candidate list
    # because of repeat history; ``fires`` only counts the off-list-pick
    # fallback. Healthy run on a complex task: drops > 0 (filter is doing
    # work), fires near 0 (LLM respects the suggested list). Pathological:
    # drops > 0, fires > 0 (LLM is ignoring the filter — suggests the
    # suggested-list mechanism in ``select_action`` should be made
    # stricter).
    _ar_eps = [s for s in all_stats if "anti_repeat_fires" in s]
    if _ar_eps:
        _total_fires = sum(s.get("anti_repeat_fires", 0) for s in _ar_eps)
        _total_drops = sum(s.get("anti_repeat_drops", 0) for s in _ar_eps)
        _eps_with_drop = sum(
            1 for s in _ar_eps if s.get("anti_repeat_drops", 0) > 0
        )
        print(
            f"  [ANTI-REPEAT] {label}: eps={len(_ar_eps)} "
            f"total_drops={_total_drops} "
            f"total_swaps={_total_fires} "
            f"eps_with_drop={_eps_with_drop}/{len(_ar_eps)}"
        )

    summary: Dict[str, Any] = {
        "target_kind": target_kind,
        "target_payload": target_payload,
        "target_safe_id": safe,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_browsergym",
        "wrapper": (
            target_payload if target_kind == "task" else "browsergym/openended"
        ),
        "target_episodes": target_episodes,
        "completed_episodes": len([s for s in all_stats if "error" not in s]),
        "use_vision": not args.no_vision,
        "max_steps": effective_max_steps,
        "elapsed_seconds": round(elapsed_target, 2),
        "episode_stats": all_stats,
    }
    rewards = [s["total_reward"] for s in all_stats if "error" not in s]
    steps_list = [s["steps"] for s in all_stats if "error" not in s]
    if rewards:
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
    if steps_list:
        summary["mean_steps"] = sum(steps_list) / len(steps_list)

    with open(target_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start actor-agent rollouts using gpt-5.5 visual grounding "
            "+ schema-driven action selection over BrowserGym openended."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help=(
            "BrowserGym task ids to run, e.g. "
            "'browsergym/miniwob.click-button browsergym/webarena.42'. "
            "Each task carries its own goal + reward function. Mutually "
            "additive with --urls."
        ),
    )
    parser.add_argument(
        "--urls", type=str, nargs="+", default=None,
        help=(
            "Open-ended BrowserGym URLs (uses ``browsergym/openended`` with "
            "``task_kwargs={'start_url': <url>}``; reward is always 0). "
            f"Default if neither --tasks nor --urls is given: "
            f"{' '.join(DEFAULT_URLS)}"
        ),
    )
    parser.add_argument(
        "--list_tasks", action="store_true",
        help="Probe registered BrowserGym task ids per suite and exit.",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Episodes per target URL (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max outer steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--max_entities", type=int, default=_DEFAULT_MAX_ENTITIES,
        help=f"Cap on entities per schema (default: {_DEFAULT_MAX_ENTITIES})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Backbone model for visual grounding + actor (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature_action", type=float, default=0.4,
        help="Sampling temperature for the action call (default: 0.4)",
    )
    parser.add_argument(
        "--temperature_schema", type=float, default=0.2,
        help="Sampling temperature for the visual schema call (default: 0.2)",
    )
    parser.add_argument(
        "--no_vision", action="store_true",
        help="Skip the vision call; use the deterministic AXTree-walked "
             "heuristic schema for action selection.",
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help=(
            "Persist the PNG frames sent to the VLM under "
            "<safe_id>/frames/ep_NNN/step_NNN.png plus a sidecar "
            "<step_NNN>.json carrying the action / reward / schema / "
            "url / candidate actions for that step. Off by default; "
            "the rollouts.jsonl + episode_NNN.json files always include "
            "a frame_path field, but the PNGs themselves are only "
            "written when this flag is set."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip episodes that already have an episode_NNN.json on disk.",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Override OPENAI_API_KEY / OPENROUTER_API_KEY for this run.",
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="Override the OpenAI base URL (e.g. for a custom proxy).",
    )
    parser.add_argument(
        "--no_headless", action="store_true",
        help=(
            "Render the browser visibly. By default the actor runs "
            "headless (Xvfb-backed Chromium); only pass --no_headless "
            "when you have a real X display and want to watch the run."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <codebase_root>/Cold-start-out-browsergym)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details (action, reward, schema source).",
    )
    parser.add_argument(
        "--reasoning_effort", "--reasoning-effort",
        type=str, default="low",
        choices=list(_VALID_REASONING_EFFORTS),
        help=(
            "OpenAI reasoning_effort knob for gpt-5.x / o1 / o3 / o4. "
            "One of {minimal, low, medium, high}. Default: 'low' "
            "(2026-05-03). Why ``low`` and not ``minimal``: OpenAI "
            "direct ``/v1/chat/completions`` rejects ``minimal`` for "
            "gpt-5.x with HTTP-400 (``Unsupported value: 'reasoning_"
            "effort' does not support 'minimal' with this model``); "
            "``low`` is accepted on direct + OpenRouter + silently "
            "dropped by the driver for non-OpenAI-reasoning models "
            "(Claude / Gemini / Qwen3-VL). For SFT cold-start data "
            "generation ``low`` is the right knob — the student never "
            "consumes the teacher's hidden thinking, so anything above "
            "``low`` is wasted budget on structured-extraction + "
            "constrained-action tasks. Reserve ``medium`` / ``high`` "
            "for the leaderboard chase where teacher answer correctness "
            "is the bottleneck (visual reasoning MCQ, hard multi-hop)."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Probe BrowserGym imports up front so we fail fast with a clear message
    # instead of crashing inside the first episode.
    try:
        import gymnasium  # noqa: F401
        import browsergym.core  # noqa: F401
    except Exception as exc:
        print(
            "[FATAL] BrowserGym + Playwright are required for the actor "
            "pipeline (no synthetic / offline mode is supported).\n"
            f"        ImportError: {exc}\n"
            "        Install via: bash install/install_browsergym.sh"
        )
        sys.exit(2)

    suite_ok, suite_fail = _import_optional_task_suites()

    if args.list_tasks:
        buckets = _list_registered_task_ids()
        total = sum(len(v) for v in buckets.values())
        print(f"Registered browsergym/* env ids: {total}")
        for prefix in sorted(buckets):
            ids = buckets[prefix]
            print(f"  {prefix:<18s} {len(ids):>5d} envs   e.g. {ids[0]}")
        if suite_fail:
            print()
            print("Optional suites that failed to import (skipped):")
            for f in suite_fail:
                print(f"  - {f}")
        return

    # Build the (kind, payload) target list. --tasks takes priority; --urls
    # is additive. If neither is given, fall back to DEFAULT_URLS so the
    # script still runs out of the box.
    raw_tasks: List[str] = list(args.tasks or [])
    raw_urls: List[str] = list(args.urls or [])
    if not raw_tasks and not raw_urls:
        raw_urls = list(DEFAULT_URLS)

    # Validate task ids against the live registry.
    bad_tasks: List[str] = []
    if raw_tasks:
        registered = set(gymnasium.envs.registry.keys())
        for t in raw_tasks:
            if t not in registered:
                bad_tasks.append(t)
    if bad_tasks:
        print("[ERROR] These --tasks env ids are not registered:")
        for t in bad_tasks:
            print(f"  - {t}")
        print(
            "Did you forget to install the task suite? "
            "Use --list_tasks to inspect what's available."
        )
        if suite_fail:
            print("Failed-to-import suites that may be needed:")
            for f in suite_fail:
                print(f"  - {f}")
        sys.exit(2)

    infra_errors = _preflight_task_infra(raw_tasks)
    if infra_errors:
        print("[ERROR] Task suite infrastructure is missing:")
        for e in infra_errors:
            print(e)
        print(
            "\nThe launcher script (run_coldstart_actor_browsergym.sh) "
            "auto-resolves MINIWOB_URL when miniwob-plusplus is checked out at "
            "/workspace/BrowserGym/miniwob-plusplus. Set the listed env vars "
            "and re-run, or drop the affected --tasks."
        )
        sys.exit(2)

    targets: List[Tuple[str, str]] = (
        [("task", t) for t in raw_tasks] + [("url", u) for u in raw_urls]
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else CODEBASE_ROOT / "Cold-start-out-browsergym"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    has_key = bool(
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not has_key and not args.no_vision:
        print("[WARNING] No API key set. Both VLM and actor calls will fail.")
        print("  Set: export OPENAI_API_KEY='sk-...' or OPENROUTER_API_KEY='sk-or-...'")

    client, routed_model = _build_client_and_route(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    if client is None:
        print(
            "[WARNING] No OpenAI/OpenRouter client could be built — "
            "actor will fall through to deterministic defaults."
        )

    schema_helpers = _import_schema_helpers()

    print("=" * 78)
    print("  Cold-Start Actor Agent — BrowserGym + gpt-5.5")
    print("=" * 78)
    if _API_KEYS_FILE_USED is not None:
        print(f"  API keys file:      {_API_KEYS_FILE_USED}")
    print(f"  Targets ({len(targets)}):")
    for kind, payload in targets:
        print(f"    [{kind:<4s}] {payload}")
    if suite_ok:
        print(f"  Suites loaded:      {', '.join(suite_ok)}")
    if suite_fail:
        print(f"  Suites unavailable: {len(suite_fail)} (use --list_tasks for details)")
    print(f"  Episodes (per target): {args.episodes}")
    print(f"  Max steps:          {args.max_steps}")
    print(f"  Max entities:       {args.max_entities}")
    print(f"  Model (configured): {args.model}")
    print(f"  Model (routed):     {routed_model}")
    print(f"  Vision schema:      {'OFF (--no_vision)' if args.no_vision else 'ON'}")
    print(f"  Headless:           {not args.no_headless}")
    if args.no_headless:
        print("    [WARN] --no_headless requested: a real X display is required. "
              "Headless is the recommended default.")
    print(f"  Save frames:        {args.save_frames}"
          + ("  (PNG + JSON sidecar per step)" if args.save_frames else ""))
    print(f"  Resume:             {args.resume}")
    print(f"  Output:             {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    target_summaries: List[Dict[str, Any]] = []
    for kind, payload in targets:
        print(f"\n{'━' * 78}")
        print(f"  TARGET ({kind}): {payload}")
        print(f"{'━' * 78}")
        try:
            summary = run_target_rollouts(
                kind, payload,
                args=args, output_dir=output_dir,
                client=client, routed_model=routed_model,
                schema_helpers=schema_helpers,
            )
        except Exception as exc:
            traceback.print_exc()
            summary = {
                "target_kind": kind, "target_payload": payload, "error": str(exc),
            }
        target_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_browsergym",
        "use_vision": not args.no_vision,
        "tasks": [p for k, p in targets if k == "task"],
        "urls": [p for k, p in targets if k == "url"],
        "targets": [{"kind": k, "payload": p} for k, p in targets],
        "episodes_per_target": args.episodes,
        "max_steps": args.max_steps,
        "max_entities": args.max_entities,
        "temperature_action": args.temperature_action,
        "temperature_schema": args.temperature_schema,
        "suites_loaded": suite_ok,
        "suites_unavailable": suite_fail,
        "elapsed_seconds": round(overall_elapsed, 2),
        "per_target_summaries": target_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ACTOR COLD-START (BROWSERGYM) — BATCH COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Targets processed: {len(target_summaries)}")
    completed = [
        s for s in target_summaries
        if not s.get("skipped") and "completed_episodes" in s
    ]
    total_eps = sum(s["completed_episodes"] for s in completed)
    print(f"  Total episodes:   {total_eps}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    print(f"  Output:           {output_dir}")
    print(f"  Master summary:   {master_path}")
    if completed:
        means = [s["mean_reward"] for s in completed if "mean_reward" in s]
        steps_means = [s["mean_steps"] for s in completed if "mean_steps" in s]
        if means:
            print(f"  Avg reward:       {sum(means) / len(means):.2f}")
        if steps_means:
            print(f"  Avg steps:        {sum(steps_means) / len(steps_means):.1f}")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<url_safe>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
