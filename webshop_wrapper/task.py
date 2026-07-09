"""WebShopTask — a thin BrowserGym task that points Playwright at a
running WebShop (or stub) Flask server and reads reward over a small
side-channel HTTP endpoint.

Why a custom task instead of plain ``browsergym/openended``
----------------------------------------------------------
``OpenEndedTask`` (browsergym/core/task.py:77) takes a ``start_url`` +
``goal`` and always returns ``reward=0`` from ``validate()``.  WebShop
has a real per-episode reward (rule-based attribute matching, evaluated
on the ``/done/...`` page).  We subclass ``OpenEndedTask`` so we can:

1. Inject a per-task ``goal_idx`` -> ``/fixed_<idx>`` URL.
2. On ``validate()``, detect the current page is ``/done/...`` and pull
   the reward from the bridge endpoint (``/__bridge/session/<id>``).

This keeps everything else — ``cold_start/generate_cold_start_actor_browsergym.py``,
``browsergym_wrapper.tools``, anti-thrash, anti-repeat, all 116 regression
tests — untouched, because the task is just another ``browsergym/<id>``
gym env from the agent's perspective.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Tuple

import gymnasium as gym
import playwright.sync_api
from browsergym.core.registration import register_task
from browsergym.core.task import OpenEndedTask


_DEFAULT_BASE_URL_FALLBACK = "http://127.0.0.1:3000"


def _default_base_url() -> str:
    """Read WEBSHOP_BASE_URL at call time, not import time, so callers
    can spawn a server on a non-default port and register tasks against
    it in the same process."""
    return os.environ.get("WEBSHOP_BASE_URL", _DEFAULT_BASE_URL_FALLBACK)


def _default_num_goals() -> int:
    return int(os.environ.get("WEBSHOP_NUM_GOALS", "5"))


class WebShopTask(OpenEndedTask):
    """A WebShop episode pinned to a single goal index.

    Parameters
    ----------
    seed
        Forwarded to ``AbstractBrowserTask`` (used only for goal sampling
        randomisation, which we bypass by pinning ``goal_idx``).
    goal_idx
        Index into the WebShop goal list.  In stub mode there are
        ``_DEFAULT_NUM_GOALS`` deterministic goals; in full mode the
        goal list is whatever WebShop's ``get_goals(...)`` returns
        (~12k for the 1k-product split, ~50k for the full split).
    base_url
        Where the WebShop Flask server is running.  Default
        ``http://127.0.0.1:3000`` matches both ``stub_app.py`` and the
        real ``web_agent_site/app.py``.
    """

    @classmethod
    def get_task_id(cls) -> str:
        return "webshop"

    def __init__(
        self,
        seed: int,
        goal_idx: int = 0,
        base_url: str | None = None,
    ) -> None:
        self.goal_idx = int(goal_idx)
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.session_id = f"fixed_{self.goal_idx}"
        start_url = f"{self.base_url}/{self.session_id}"

        # OpenEndedTask sets self.start_url + self.goal; the goal text is
        # filled in by setup() once we can hit the bridge endpoint.
        super().__init__(seed=seed, start_url=start_url, goal="")

        # Bump the navigation timeout: WebShop's first page-load can be
        # slow if the search-engine warm-up is still running.
        self.timeout = 30000  # ms

    def setup(self, page: playwright.sync_api.Page) -> Tuple[str, dict]:
        # Pull the human-readable goal text from the bridge endpoint
        # before navigating, so the agent prompt has it on step 0.
        goal_text = self._fetch_goal_text() or "Buy a product matching the page instructions."
        self.goal = goal_text
        page.goto(self.start_url, timeout=self.timeout)
        return goal_text, {
            "goal_idx": self.goal_idx,
            "session_id": self.session_id,
            "base_url": self.base_url,
        }

    def validate(
        self,
        page: playwright.sync_api.Page,
        chat_messages: list[str],
    ) -> Tuple[float, bool, str, dict]:
        # User-typed exit override (mirrors OpenEndedTask).
        for message in chat_messages:
            if message.get("role") == "user" and message.get("message") == "exit":
                return 0.0, True, "", {"reason": "user_exit"}

        # Episode is "done" iff the agent has reached /done/<session>/...
        url = page.url or ""
        is_done_page = f"/done/{self.session_id}" in url
        if not is_done_page:
            return 0.0, False, "", {}

        sess = self._fetch_session_state()
        reward = float(sess.get("reward", 0.0))
        return reward, True, "", {"webshop_session": sess}

    # ------------------------------------------------------------------ #
    # Bridge endpoint helpers — both stub_app.py and the patched
    # web_agent_site/app.py expose ``/__bridge/session/<id>`` (the patch
    # is applied by install/install_webshop.sh in full mode).
    # ------------------------------------------------------------------ #
    def _fetch_session_state(self) -> dict:
        try:
            url = f"{self.base_url}/__bridge/session/{self.session_id}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}

    def _fetch_goal_text(self) -> str:
        sess = self._fetch_session_state()
        goal = sess.get("goal") or {}
        return str(goal.get("instruction_text", ""))


def register_webshop_tasks(num_goals: int | None = None) -> list[str]:
    """Register ``browsergym/webshop.<idx>`` envs with gymnasium.

    Idempotent — re-registering the same id is a no-op.  Returns the
    list of registered env ids.

    Parameters
    ----------
    num_goals
        How many fixed goals to register.  Default ``WEBSHOP_NUM_GOALS``
        env var (``5``).  Stub mode supports up to 5; full mode can go
        much higher once the real WebShop dataset is loaded.
    """
    n = num_goals if num_goals is not None else _default_num_goals()
    registered: list[str] = []
    for idx in range(n):
        env_id = f"browsergym/webshop.{idx}"
        if env_id in gym.envs.registry:
            registered.append(env_id)
            continue
        register_task(
            id=f"webshop.{idx}",
            task_class=WebShopTask,
            task_kwargs={"goal_idx": idx},
            nondeterministic=False,
        )
        registered.append(env_id)
    return registered


__all__ = ["WebShopTask", "register_webshop_tasks"]
