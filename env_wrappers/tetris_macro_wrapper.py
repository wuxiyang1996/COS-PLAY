"""Macro-action wrapper for Tetris.

Replaces primitive move actions (left, right, rotate, drop) with
placement-level actions.  Each action = one piece placement
(rotation + column), so every LLM decision commits a piece and
produces meaningful reward signal for GRPO training.

The underlying TetrisEnv already accepts action sequences via string
parsing, so this wrapper:
  1. Enumerates valid (rotation, column) placements for the current piece
  2. Simulates each to compute outcome metrics (lines, holes, height)
  3. Presents them as numbered choices to the agent
  4. Translates the chosen placement into a move-sequence string

Usage (in episode_runner.py)::

    base_env = make_gaming_env("tetris", max_steps=200)
    env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))
    obs, info = env.reset()
    # info["action_names"] now contains placement descriptions
    obs, reward, term, trunc, info = env.step("T-right col5 (1line, 0holes, h=4)")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
MAX_ROTATIONS = {0: 2, 1: 1, 2: 4, 3: 2, 4: 2, 5: 4, 6: 4}

ROTATION_LABELS: Dict[str, Dict[int, str]] = {
    "I": {0: "horizontal", 1: "vertical"},
    "O": {0: ""},
    "T": {0: "up", 1: "right", 2: "down", 3: "left"},
    "S": {0: "flat", 1: "vertical"},
    "Z": {0: "flat", 1: "vertical"},
    "J": {0: "up", 1: "right", 2: "down", 3: "left"},
    "L": {0: "up", 1: "right", 2: "down", 3: "left"},
}

MAX_PLACEMENTS = 25


class TetrisMacroActionWrapper:
    """Wraps a Tetris NL environment to present placement-level actions.

    Compatible with the co-evolution episode runner: exposes
    ``reset()``, ``step(action_str)``, ``close()``, and populates
    ``info["action_names"]`` with human-readable placement descriptions.
    """

    _is_macro_action = True

    def __init__(self, env: Any):
        self._env = env
        self._placements: List[dict] = []
        self._action_names: List[str] = []
        self._step_count = 0

    # ------------------------------------------------------------------
    # Internal env access
    # ------------------------------------------------------------------

    @property
    def _tetris_env(self):
        """Navigate wrapper chain to reach the underlying TetrisEnv."""
        env = self._env
        while True:
            inner = getattr(env, "_env", None)
            if inner is None:
                break
            env = inner
        return env

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[str, Dict[str, Any]]:
        obs_nl, info = self._env.reset(**kwargs)
        self._step_count = 0
        self._refresh_placements(info)
        obs_nl = self._build_observation()
        return obs_nl, info

    def step(
        self, action: Union[str, int]
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        idx = self._resolve_action_index(action)

        chosen_placement: Optional[dict] = None
        if idx is not None and idx < len(self._placements):
            chosen_placement = self._placements[idx]
            action_seq = self._build_action_sequence(chosen_placement)
        else:
            action_seq = "hard_drop"

        obs_nl, reward, terminated, truncated, info = self._env.step(action_seq)
        self._step_count += 1

        info["raw_env_reward"] = float(reward)
        if chosen_placement and not chosen_placement.get("fallback"):
            info["placement_metrics"] = {
                "piece": chosen_placement["piece_name"],
                "rotation": chosen_placement["rotation"],
                "column": chosen_placement["game_col"],
                "lines_cleared": chosen_placement["lines_cleared"],
                "new_holes": chosen_placement["new_holes"],
                "max_height": chosen_placement["max_height"],
            }
        else:
            info["placement_metrics"] = None

        tenv = self._tetris_env
        empty_val = tenv.base_pixels[0].value
        info["board_stats"] = {
            "stack_height": self._max_height(
                tenv.board, tenv.padding, tenv.width, empty_val,
            ),
            "holes": self._count_holes(
                tenv.board, tenv.padding, tenv.width, empty_val,
            ),
            "lines_total": tenv.lines_cleared_total,
            "level": tenv.level,
            "score": tenv.current_score,
        }

        if not (terminated or truncated):
            self._refresh_placements(info)
            obs_nl = self._build_observation()

        return obs_nl, reward, terminated, truncated, info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    # ------------------------------------------------------------------
    # Placement enumeration
    # ------------------------------------------------------------------

    def _refresh_placements(self, info: Dict[str, Any]) -> None:
        self._placements = self._enumerate_placements()
        if not self._placements:
            self._placements = [{"fallback": True}]
            self._action_names = ["hard_drop (no valid placement)"]
        else:
            self._action_names = [
                self._describe_placement(p) for p in self._placements
            ]
        info["action_names"] = self._action_names

    def _enumerate_placements(self) -> List[dict]:
        tenv = self._tetris_env
        if tenv.active_tetromino is None or tenv.game_over:
            return []

        original_idx = tenv.active_tetromino_original_idx
        original_def = tenv.TETROMINOES[original_idx]
        board_id = tenv.active_tetromino.id
        piece_id = original_def.id
        piece_name = PIECE_NAMES[piece_id]
        max_rots = MAX_ROTATIONS[piece_id]

        empty_val = tenv.base_pixels[0].value
        board = tenv.board
        pad = tenv.padding
        width = tenv.width
        height = tenv.height
        h_pad = tenv.height_padded
        w_pad = tenv.width_padded

        holes_before = self._count_holes(board, pad, width, empty_val)

        # spawn_x is where the env placed the piece (for dx calculation)
        spawn_x = w_pad // 2 - tenv.active_tetromino.matrix.shape[1] // 2

        placements: List[dict] = []
        matrix = original_def.matrix.copy()

        for rot in range(max_rots):
            rot_matrix = matrix.copy()
            rot_matrix[rot_matrix > 0] = board_id
            ph, pw = rot_matrix.shape

            # Check rotation feasibility at spawn (y=0)
            if rot > 0 and self._collides(board, rot_matrix, spawn_x, 0,
                                           h_pad, w_pad, empty_val):
                matrix = np.rot90(matrix, k=-1)
                continue

            for x in range(w_pad):
                if x + pw > w_pad:
                    break

                land_y = self._find_landing_y(
                    board, rot_matrix, x, empty_val, h_pad, w_pad,
                )
                if land_y is None:
                    continue

                # Game column = leftmost occupied cell in game coords
                occ_cols = [
                    x + c
                    for r in range(ph) for c in range(pw)
                    if rot_matrix[r, c] != 0
                ]
                if not occ_cols:
                    continue
                game_col = min(occ_cols) - pad
                if game_col < 0 or max(occ_cols) - pad >= width:
                    continue

                # Simulate placement on a copy
                temp_board = board.copy()
                for r in range(ph):
                    for c in range(pw):
                        if rot_matrix[r, c] != 0:
                            temp_board[land_y + r, x + c] = rot_matrix[r, c]

                lines = self._count_clearable(temp_board, pad, width,
                                               height, empty_val)
                holes_after = self._count_holes(temp_board, pad, width,
                                                 empty_val)
                max_h = self._max_height(temp_board, pad, width, empty_val)

                placements.append({
                    "rotation": rot,
                    "game_col": game_col,
                    "lines_cleared": lines,
                    "new_holes": holes_after - holes_before,
                    "max_height": max_h,
                    "piece_name": piece_name,
                    "dx": x - spawn_x,
                })

            matrix = np.rot90(matrix, k=-1)

        # Sort: lines cleared ↓, new holes ↑, height ↑
        placements.sort(
            key=lambda p: (-p["lines_cleared"], p["new_holes"], p["max_height"])
        )

        # Deduplicate by (rotation, game_col) — keep first (best landing)
        seen = set()
        unique: List[dict] = []
        for p in placements:
            key = (p["rotation"], p["game_col"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        placements = unique

        if len(placements) > MAX_PLACEMENTS:
            placements = placements[:MAX_PLACEMENTS]

        return placements

    # ------------------------------------------------------------------
    # Board simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collides(board, piece_matrix, x, y, h_pad, w_pad, empty_val) -> bool:
        ph, pw = piece_matrix.shape
        for r in range(ph):
            for c in range(pw):
                if piece_matrix[r, c] == 0:
                    continue
                br, bc = y + r, x + c
                if br < 0 or br >= h_pad or bc < 0 or bc >= w_pad:
                    return True
                if board[br, bc] != empty_val:
                    return True
        return False

    @staticmethod
    def _find_landing_y(board, piece_matrix, x, empty_val, h_pad, w_pad):
        ph, pw = piece_matrix.shape
        prev_y = None
        for test_y in range(h_pad):
            hit = False
            for r in range(ph):
                for c in range(pw):
                    if piece_matrix[r, c] == 0:
                        continue
                    br, bc = test_y + r, x + c
                    if br >= h_pad or bc >= w_pad or br < 0 or bc < 0:
                        hit = True
                        break
                    if board[br, bc] != empty_val:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                return prev_y
            prev_y = test_y
        return prev_y

    @staticmethod
    def _count_clearable(board, pad, width, height, empty_val) -> int:
        count = 0
        for row_idx in range(height):
            row = board[row_idx, pad: pad + width]
            if np.all(row != empty_val):
                count += 1
        return count

    @staticmethod
    def _count_holes(board, pad, width, empty_val) -> int:
        holes = 0
        game_area = board[:, pad: pad + width]
        for c in range(game_area.shape[1]):
            col = game_area[:, c]
            block_seen = False
            for r in range(col.shape[0]):
                if col[r] != empty_val:
                    block_seen = True
                elif block_seen:
                    holes += 1
        return holes

    @staticmethod
    def _max_height(board, pad, width, empty_val) -> int:
        game_area = board[:, pad: pad + width]
        max_h = 0
        for c in range(game_area.shape[1]):
            col = game_area[:, c]
            for r in range(col.shape[0]):
                if col[r] != empty_val:
                    max_h = max(max_h, col.shape[0] - r)
                    break
        return max_h

    # ------------------------------------------------------------------
    # Action translation
    # ------------------------------------------------------------------

    def _resolve_action_index(self, action: Union[str, int]) -> Optional[int]:
        action_str = str(action).strip()
        if action_str in self._action_names:
            return self._action_names.index(action_str)
        try:
            idx = int(action_str) - 1
            if 0 <= idx < len(self._action_names):
                return idx
        except (ValueError, TypeError):
            pass
        for i, name in enumerate(self._action_names):
            if action_str in name or name in action_str:
                return i
        return 0

    @staticmethod
    def _build_action_sequence(placement: dict) -> str:
        if placement.get("fallback"):
            return "hard_drop"
        parts: List[str] = []
        for _ in range(placement["rotation"]):
            parts.append("rotate_right,1")
        dx = placement["dx"]
        if dx < 0:
            parts.extend(["left,1"] * abs(dx))
        elif dx > 0:
            parts.extend(["right,1"] * dx)
        parts.append("hard_drop,1")
        return "; ".join(parts)

    @staticmethod
    def _describe_placement(p: dict) -> str:
        name = p["piece_name"]
        rot = p["rotation"]
        rot_label = ROTATION_LABELS.get(name, {}).get(rot, f"r{rot}")

        desc = name
        if rot_label:
            desc += f"-{rot_label}"
        desc += f" col{p['game_col']}"

        parts: List[str] = []
        lc = p["lines_cleared"]
        if lc > 0:
            parts.append(f"{lc}line{'s' if lc > 1 else ''}")
        nh = p["new_holes"]
        if nh > 0:
            parts.append(f"+{nh}hole{'s' if nh > 1 else ''}")
        elif nh < 0:
            parts.append(f"fills {abs(nh)}")
        parts.append(f"h={p['max_height']}")

        return f"{desc} ({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> str:
        """Build an observation that includes the board grid and game stats.

        The board is rendered WITHOUT the active piece so the agent sees
        the settled state and chooses where to place the current piece.
        The format is compatible with ``_extract_tetris_facts`` in
        ``decision_agents.agent_helper`` (10-char rows of ``.IOTSZJL``).
        """
        tenv = self._tetris_env
        board = tenv.board
        pad = tenv.padding
        width = tenv.width
        height = tenv.height
        empty_val = tenv.base_pixels[0].value

        symbols = {empty_val: "."}
        for i, sym in enumerate(tenv.TETROMINO_SYMBOLS):
            symbols[i + len(tenv.base_pixels)] = sym

        game_area = board[0:height, pad: pad + width]
        rows: List[str] = []
        for row in game_area:
            rows.append("".join(symbols.get(int(c), "#") for c in row))

        # Trim leading empty rows, keep 1-2 above content
        first_content = height
        for i, row in enumerate(rows):
            if any(c != "." for c in row):
                first_content = i
                break
        show_from = max(0, first_content - 2)
        board_text = "\n".join(rows[show_from:])

        # Piece info
        piece_name = "?"
        if tenv.active_tetromino_original_idx is not None:
            piece_name = PIECE_NAMES[tenv.active_tetromino_original_idx]
        next_names = [PIECE_NAMES[idx] for idx in tenv.piece_queue[:4]]

        # Stats — derived from the padded board via the wrapper's own
        # static helpers so we don't depend on private TetrisEnv methods
        # (which exist in some GamingAgent forks but not the release branch).
        stack_h = self._max_height(tenv.board, tenv.padding, tenv.width, empty_val)
        holes = self._count_holes(tenv.board, tenv.padding, tenv.width, empty_val)

        # Column heights
        col_heights: List[int] = []
        for c in range(game_area.shape[1]):
            col = game_area[:, c]
            h = 0
            for r in range(col.shape[0]):
                if col[r] != empty_val:
                    h = col.shape[0] - r
                    break
            col_heights.append(h)

        obs = (
            f"Board:\n{board_text}\n"
            f"Next Pieces: {','.join(next_names)}\n"
            f"Game Stats: stack_h={stack_h} holes={holes} "
            f"lines={tenv.lines_cleared_total} Lv:{tenv.level}\n"
            f"Current piece: {piece_name}\n"
            f"Column heights: {','.join(str(h) for h in col_heights)}\n"
        )
        return obs
