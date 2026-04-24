#!/usr/bin/env python3
"""
Generate replay videos (MP4) from best-performing episodes using the
official game environment renderers.

Usage (from game-ai-agent conda env):
    python generate_replay_gifs.py [--source all] [--format mp4|gif]
"""

import json
import glob
import io
import os
import re
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGLET_HEADLESS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent / "output"
RUNS_DIR = Path(__file__).parent / "runs"
REPLAY_DIR = Path(__file__).parent / "replays"

# ═══════════════════════════════════════════════════════════════════════════
# Board state parsers  (text observation → numpy array the env expects)
# ═══════════════════════════════════════════════════════════════════════════

TETRIS_SYM_TO_ID = {".": 0, "#": 1, "I": 2, "O": 3, "T": 4, "S": 5, "Z": 6, "J": 7, "L": 8}
TETRIS_W, TETRIS_H = 10, 20          # visible playfield (real state text is 20 rows)

# TetrisEnv builds its internal board as np.pad(board, ((0, padding),
# (padding, padding)), constant_values=WALL). So the `padding` rows sit
# BELOW the playfield as a bottom wall (not above it), and the env's
# `_max_col_height()` reports the tallest column over the *padded* board,
# which means it *includes* those 4 wall rows. Consequently the raw
# `stack_height` value in `board_stats` ranges from 4 (empty playfield,
# walls only) to 24 (playfield full to the spawn row). To render faithfully
# we must subtract this offset before clamping to the visible 20-row grid.
TETRIS_PAD_H = 4                     # bottom wall rows included in stack_height
TETRIS_STACK_MIN = TETRIS_PAD_H      # minimum raw stack_height (empty playfield)


def parse_tetris_board(state_text: str) -> Optional[np.ndarray]:
    rows = []
    for line in state_text.split("\n"):
        s = line.strip()
        if s and len(s) == TETRIS_W and all(c in ".IOTSZJL#" for c in s):
            rows.append([TETRIS_SYM_TO_ID.get(c, 1) for c in s])
    if not rows:
        return None
    while len(rows) < TETRIS_H:
        rows.insert(0, [0] * TETRIS_W)
    return np.array(rows[:TETRIS_H], dtype=np.uint8)


def parse_tetris_meta(state_text: str) -> dict:
    meta = {}
    m = re.search(r"Next Pieces:\s*([A-Z,]+)", state_text)
    if m:
        meta["next_symbols"] = m.group(1).split(",")
    for key, pat in [("perf", r"PerfS:(\S+)"), ("score", r" S:(\S+)"),
                     ("lines", r" L:(\d+)"), ("level", r"Lv:(\d+)")]:
        m = re.search(pat, state_text)
        if m:
            meta[key] = m.group(1)
    # Training format: "stack_h=4 holes=0 lines=33 Lv:4"
    if "lines" not in meta:
        m = re.search(r"lines=(\d+)", state_text)
        if m:
            meta["lines"] = m.group(1)
    if "score" not in meta:
        m = re.search(r"Score:\s*(\d+)", state_text)
        if m:
            meta["score"] = m.group(1)
    return meta


def parse_2048_board(state_text: str) -> Optional[np.ndarray]:
    m = re.search(r"'board':\s*\[(\[.*?\])\]", state_text, re.DOTALL)
    if not m:
        return None
    raw = re.sub(r"np\.\w+\((\d+)\)", r"\1", m.group(1))
    rows_raw = re.findall(r"\[([^\]]+)\]", raw)
    board = []
    for row in rows_raw:
        board.append([int(x.strip()) for x in row.split(",")])
    if not board:
        return None
    vals = np.array(board, dtype=np.int64)
    powers = np.zeros_like(vals, dtype=np.uint8)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            if v > 0:
                powers[r, c] = int(np.log2(v))
    return powers


SOKOBAN_ITEM_TO_STATE = {
    "Wall": 0, "Empty": 1, "Dock": 2, "Box on Dock": 3,
    "Box": 4, "Worker": 5, "Worker on Dock": 6,
}
SOKOBAN_STATE_COLORS = {
    0: (80, 80, 100),    # Wall   – dark blue-grey
    1: (220, 210, 190),  # Floor  – warm beige
    2: (200, 80, 80),    # Dock   – red target
    3: (180, 140, 50),   # Box on target – gold
    4: (160, 120, 60),   # Box    – brown
    5: (60, 120, 220),   # Worker – blue
    6: (100, 60, 200),   # Worker on target – purple
}


def parse_sokoban_table(state_text: str) -> Optional[np.ndarray]:
    items = []
    max_col, max_row = 0, 0
    for line in state_text.split("\n"):
        m = re.match(r"\s*\d+\s*\|\s*(\S[\w ]*\S|\S)\s*\|\s*\((\d+),\s*(\d+)\)", line)
        if not m:
            continue
        item_type = m.group(1).strip()
        col, row = int(m.group(2)), int(m.group(3))
        state_val = SOKOBAN_ITEM_TO_STATE.get(item_type)
        if state_val is None:
            continue
        items.append((row, col, state_val))
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    if not items:
        return None
    grid = np.ones((max_row + 1, max_col + 1), dtype=np.uint8)  # default floor
    for r, c, v in items:
        grid[r, c] = v
    return grid


CANDY_CHAR_TO_IDX = {"R": 4, "C": 2, "G": 1, "P": 3, "Y": 5, "B": 6, "O": 7, " ": 0}


def parse_candy_board(state_text: str) -> Optional[np.ndarray]:
    rows = []
    for line in state_text.split("\n"):
        m = re.match(r"\d+\|\s*(.*)", line.strip())
        if m:
            chars = m.group(1).strip().split()
            rows.append([CANDY_CHAR_TO_IDX.get(ch, 0) for ch in chars])
    if not rows:
        return None
    return np.array(rows, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# Environment-based renderers
# ═══════════════════════════════════════════════════════════════════════════

_tetris_env = None
_2048_env = None
_sokoban_env = None
_candy_env = None
_candy_renderer = None


def _get_tetris_env():
    global _tetris_env
    if _tetris_env is None:
        from gamingagent.envs.custom_04_tetris.tetrisEnv import TetrisEnv
        _tetris_env = TetrisEnv(
            render_mode="rgb_array",
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_tetris_"),
        )
        _tetris_env.reset(seed=0)
    return _tetris_env


def _get_2048_env():
    global _2048_env
    if _2048_env is None:
        from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv
        _2048_env = TwentyFortyEightEnv(
            render_mode="rgb_array",
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_2048_"),
        )
        _2048_env.reset(seed=0)
    return _2048_env


def _get_sokoban_env():
    global _sokoban_env
    if _sokoban_env is None:
        from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv
        _sokoban_env = SokobanEnv(
            render_mode="rgb_array",
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_sokoban_"),
        )
        _sokoban_env.reset(seed=0)
    return _sokoban_env


def _get_candy_env():
    global _candy_env, _candy_renderer
    if _candy_env is None:
        from gamingagent.envs.custom_03_candy_crush.candyCrushEnv import CandyCrushEnv
        from tile_match_gym.renderer import Renderer
        _candy_env = CandyCrushEnv(
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_candy_"),
        )
        _candy_renderer = Renderer(
            _candy_env.num_rows, _candy_env.num_cols,
            _candy_env.num_colours, _candy_env.num_moves,
            render_fps=2, render_mode="rgb_array",
        )
        _candy_env.renderer = _candy_renderer
        _candy_env.internal_render_mode = "rgb_array"
        _candy_env.reset(seed=0)
    return _candy_env


TETRIS_PIECE_COLORS = {
    0: (30, 30, 40),       # empty
    1: (60, 60, 70),       # wall / garbage (#)
    2: (0, 240, 240),      # I - cyan
    3: (240, 240, 0),      # O - yellow
    4: (160, 0, 240),      # T - purple
    5: (0, 240, 0),        # S - green
    6: (240, 0, 0),        # Z - red
    7: (0, 0, 240),        # J - blue
    8: (240, 160, 0),      # L - orange
}

TETRIS_PIECE_SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}

TETRIS_ORIENTED_SHAPES = {
    ("I", "horizontal"): [[1, 1, 1, 1]],
    ("I", "vertical"):   [[1], [1], [1], [1]],
    ("O", ""):           [[1, 1], [1, 1]],
    ("T", "up"):         [[0, 1, 0], [1, 1, 1]],
    ("T", "down"):       [[1, 1, 1], [0, 1, 0]],
    ("T", "left"):       [[1, 0], [1, 1], [1, 0]],
    ("T", "right"):      [[0, 1], [1, 1], [0, 1]],
    ("S", "flat"):       [[0, 1, 1], [1, 1, 0]],
    ("S", "vertical"):   [[1, 0], [1, 1], [0, 1]],
    ("Z", "flat"):       [[1, 1, 0], [0, 1, 1]],
    ("Z", "vertical"):   [[0, 1], [1, 1], [1, 0]],
    ("J", "up"):         [[1, 0, 0], [1, 1, 1]],
    ("J", "down"):       [[1, 1, 1], [0, 0, 1]],
    ("J", "left"):       [[1, 1], [1, 0], [1, 0]],
    ("J", "right"):      [[0, 1], [0, 1], [1, 1]],
    ("L", "up"):         [[0, 0, 1], [1, 1, 1]],
    ("L", "down"):       [[1, 1, 1], [1, 0, 0]],
    ("L", "left"):       [[1, 0], [1, 0], [1, 1]],
    ("L", "right"):      [[1, 1], [0, 1], [0, 1]],
}

TETRIS_SYM_TO_COLOR_ID = {"I": 2, "O": 3, "T": 4, "S": 5, "Z": 6, "J": 7, "L": 8}


class TetrisBoardSim:
    """Simulates a tetris board by placing macro-action pieces one at a time.

    The simulator is intentionally simplistic (no garbage rows, no spawn
    constraints, no SRS kicks) and will diverge from the real TetrisEnv
    after many steps. Use :meth:`sync_with_real` after each placement to
    clamp the simulator's visible state to the real env's `board_stats`
    so the rendered gif stays faithful to what actually happened.
    """

    def __init__(self):
        self.grid = np.zeros((TETRIS_H, TETRIS_W), dtype=int)
        self.sim_lines_cleared = 0  # cumulative lines cleared inside the sim
        self.sim_placements = 0     # cumulative pieces placed by the sim

    def place(self, piece: str, orient: str, col: int) -> int:
        shape = TETRIS_ORIENTED_SHAPES.get((piece, orient))
        if shape is None:
            return 0
        cid = TETRIS_SYM_TO_COLOR_ID.get(piece, 1)
        rows, cols = len(shape), len(shape[0])

        # Drop: find lowest valid row
        land_row = 0
        for r in range(TETRIS_H - rows + 1):
            can_place = True
            for ri, srow in enumerate(shape):
                for ci, val in enumerate(srow):
                    if val:
                        c = col + ci
                        if c < 0 or c >= TETRIS_W or self.grid[r + ri, c] != 0:
                            can_place = False
                            break
                if not can_place:
                    break
            if can_place:
                land_row = r
            else:
                break

        placed_any = False
        for ri, srow in enumerate(shape):
            for ci, val in enumerate(srow):
                if val:
                    c = col + ci
                    if 0 <= c < TETRIS_W and 0 <= land_row + ri < TETRIS_H:
                        self.grid[land_row + ri, c] = cid
                        placed_any = True

        if placed_any:
            self.sim_placements += 1

        cleared = 0
        new_rows = []
        for r in range(TETRIS_H):
            if np.all(self.grid[r] != 0):
                cleared += 1
            else:
                new_rows.append(self.grid[r].copy())
        if cleared > 0:
            empty = [np.zeros(TETRIS_W, dtype=int) for _ in range(cleared)]
            self.grid = np.array(empty + new_rows)
        self.sim_lines_cleared += cleared
        return cleared

    # ─────────────────────────── stats-first sync ───────────────────────────

    def _sim_max_height(self) -> int:
        """Current max column height in the simulator (0..TETRIS_H)."""
        for r in range(TETRIS_H):
            if np.any(self.grid[r] != 0):
                return TETRIS_H - r
        return 0

    def _drop_densest_rows(self, n: int) -> None:
        """Remove up to `n` rows that are most likely to have been cleared by
        the real game but were missed by the simulator. Preference: higher
        cell count first, then lower (bottom-most) row first as a tiebreak.
        Remaining rows shift down so empty rows appear on top.
        """
        if n <= 0:
            return
        fills = [(int(np.sum(self.grid[r] != 0)), r) for r in range(TETRIS_H)]
        # Only consider rows that have at least one filled cell; clearing empty
        # rows is a no-op that wastes the quota.
        fills = [fr for fr in fills if fr[0] > 0]
        if not fills:
            return
        # Densest first; among equals prefer lower rows (r larger).
        fills.sort(key=lambda fr: (-fr[0], -fr[1]))
        to_remove = {r for _, r in fills[:n]}
        kept = [self.grid[r].copy() for r in range(TETRIS_H) if r not in to_remove]
        empty = [np.zeros(TETRIS_W, dtype=int) for _ in range(len(to_remove))]
        self.grid = np.array(empty + kept)

    def _truncate_top(self, max_visible_height: int) -> None:
        """Keep only the bottom `max_visible_height` rows of content; zero out
        everything above.
        """
        max_visible_height = max(0, min(TETRIS_H, max_visible_height))
        keep_from_row = TETRIS_H - max_visible_height
        for r in range(keep_from_row):
            self.grid[r, :] = 0

    def sync_with_real(self, real_lines_total: int,
                       real_stack_height: Optional[int] = None) -> None:
        """Align the simulator's visible state with the real env's stats.

        The env's raw ``stack_height`` is reported over its *padded* board,
        so it includes ``TETRIS_PAD_H`` bottom-wall rows. We subtract that
        offset to recover the piece-only stack height, which is what the
        rendered 20-row playfield should match.

        Two anchors are applied (in order):
          1. **Max column height**: if the sim's tallest column exceeds the
             real piece-stack, truncate the top so the rendered board never
             looks taller than reality.
          2. **Filled-cell budget**: the real env has left roughly
             ``placements*4 - lines_total*10`` cells on the padded board
             after play. If the sim has significantly more than that
             (sim's simple drop logic tends to miss line clears), drop its
             densest rows until filled-cell count is within tolerance.
        """
        # Always mirror the real env's cumulative line-clear count upward
        # (never downward). The sim may miss real clears so this keeps it
        # monotonically consistent with what the env actually saw.
        self.sim_lines_cleared = max(self.sim_lines_cleared,
                                     int(real_lines_total))

        if real_stack_height is not None:
            piece_stack = max(0, int(real_stack_height) - TETRIS_PAD_H)
            visible_h = min(piece_stack, TETRIS_H)
            if self._sim_max_height() > visible_h:
                self._truncate_top(visible_h)

        # Filled-cell budget: expected remaining cells on the 20-row
        # playfield ≈ sim_placements*4 - real_lines_total*10, clamped to
        # [0, TETRIS_H * TETRIS_W]. Allow a tolerance of 10 cells (one
        # full row) before we start pruning, so occasional clear-mismatches
        # don't visibly wipe the board.
        sim_filled = int(np.sum(self.grid != 0))
        expected = int(self.sim_placements) * 4 - int(real_lines_total) * 10
        expected = max(0, min(TETRIS_H * TETRIS_W, expected))
        excess_cells = sim_filled - (expected + 10)
        if excess_cells > 0:
            # Estimate how many rows to drop: assume ~6 cells per row of
            # actual play (partial fills). Drop at most enough rows to
            # bring excess back under tolerance.
            rows_to_drop = max(1, excess_cells // 6)
            self._drop_densest_rows(rows_to_drop)


_tetris_sim: Optional[TetrisBoardSim] = None


def render_tetris_stats(board_stats: dict, step: int, reward: float,
                        total_reward: float, action: str) -> Image.Image:
    """Render a tetris frame by simulating piece placement on a persistent board."""
    global _tetris_sim
    if _tetris_sim is None:
        _tetris_sim = TetrisBoardSim()

    m = re.match(r'^([A-Z])-?(\w*)\s+col(\d+)', action)
    placed_cells = []
    if m:
        piece, orient, col_str = m.group(1), m.group(2), int(m.group(3))
        cid = TETRIS_SYM_TO_COLOR_ID.get(piece, 1)
        shape = TETRIS_ORIENTED_SHAPES.get((piece, orient))
        if shape:
            rows_s = len(shape)
            land_row = 0
            for r in range(TETRIS_H - rows_s + 1):
                ok = True
                for ri, srow in enumerate(shape):
                    for ci, val in enumerate(srow):
                        if val:
                            c = col_str + ci
                            if c < 0 or c >= TETRIS_W or _tetris_sim.grid[r + ri, c] != 0:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    land_row = r
                else:
                    break
            for ri, srow in enumerate(shape):
                for ci, val in enumerate(srow):
                    if val:
                        placed_cells.append((land_row + ri, col_str + ci))
        _tetris_sim.place(piece, orient, col_str)

    # Stats-first sync: clamp the visible board so it never contradicts the
    # real env's line-clear count / stack height. Without this the simulator's
    # drop/line-clear logic diverges and we get "64 lines cleared but the
    # board is still full" or stack_height > 20 stacks that look illegal.
    real_lines = int(board_stats.get("lines_total", 0))
    real_stack = board_stats.get("stack_height")
    _tetris_sim.sync_with_real(
        real_lines_total=real_lines,
        real_stack_height=int(real_stack) if real_stack is not None else None,
    )
    placed_set_sync = set()
    for (r, c) in placed_cells:
        if 0 <= r < TETRIS_H and 0 <= c < TETRIS_W and _tetris_sim.grid[r, c] != 0:
            placed_set_sync.add((r, c))
    placed_cells = list(placed_set_sync)

    cell = 24
    pad = 8
    board_w = TETRIS_W * cell
    board_h = TETRIS_H * cell
    sidebar_w = 150
    header_h = 40
    W = pad + board_w + pad + sidebar_w + pad
    H = header_h + pad + board_h + pad

    bg = (25, 25, 35)
    board_bg = (15, 15, 22)
    grid_color = (35, 35, 48)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    fnt_title = _pil_font(14)
    fnt_label = _pil_font(11)
    fnt_value = _pil_font(13)
    fnt_small = _pil_font_regular(10)

    draw.text((pad, 10), "TETRIS", fill=(0, 240, 240), font=fnt_title)
    draw.text((pad + 90, 12), f"Piece {step}", fill=(180, 180, 200), font=fnt_label)

    bx, by = pad, header_h + pad
    draw.rectangle([(bx - 2, by - 2), (bx + board_w + 1, by + board_h + 1)],
                   outline=(80, 80, 100), width=2)
    draw.rectangle([(bx, by), (bx + board_w - 1, by + board_h - 1)], fill=board_bg)

    for r in range(TETRIS_H + 1):
        y = by + r * cell
        draw.line([(bx, y), (bx + board_w, y)], fill=grid_color, width=1)
    for c in range(TETRIS_W + 1):
        x = bx + c * cell
        draw.line([(x, by), (x, by + board_h)], fill=grid_color, width=1)

    placed_set = set(placed_cells)
    for r in range(TETRIS_H):
        for c in range(TETRIS_W):
            cid = _tetris_sim.grid[r, c]
            if cid == 0:
                continue
            base = TETRIS_PIECE_COLORS.get(cid, (100, 100, 100))
            glow = (r, c) in placed_set
            if glow:
                base = tuple(min(255, int(v * 1.3)) for v in base)
            x0 = bx + c * cell + 1
            y0 = by + r * cell + 1
            x1 = x0 + cell - 2
            y1 = y0 + cell - 2
            draw.rectangle([(x0, y0), (x1, y1)], fill=base)
            hi = tuple(min(255, int(v * 1.5)) for v in base)
            draw.line([(x0, y0), (x1, y0)], fill=hi, width=1)
            draw.line([(x0, y0), (x0, y1)], fill=hi, width=1)
            lo = tuple(max(0, int(v * 0.5)) for v in base)
            draw.line([(x0, y1), (x1, y1)], fill=lo, width=1)
            draw.line([(x1, y0), (x1, y1)], fill=lo, width=1)

    sx = bx + board_w + pad + 4
    sy = by

    score = board_stats.get("score", 0)
    draw.text((sx, sy), "SCORE", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    draw.text((sx, sy), str(int(score)), fill=(255, 255, 255), font=fnt_value)
    sy += 24

    lines = board_stats.get("lines_total", 0)
    draw.text((sx, sy), "LINES", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    draw.text((sx, sy), str(lines), fill=(255, 255, 255), font=fnt_value)
    sy += 24

    level = board_stats.get("level", 1)
    draw.text((sx, sy), "LEVEL", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    draw.text((sx, sy), str(level), fill=(255, 255, 255), font=fnt_value)
    sy += 30

    draw.text((sx, sy), "ACTION", fill=(160, 160, 180), font=fnt_label)
    sy += 15
    act_lines = textwrap.wrap(action, width=16)
    for aline in act_lines[:2]:
        draw.text((sx, sy), aline, fill=(180, 220, 255), font=fnt_small)
        sy += 14
    sy += 10

    # STACK: the env reports `stack_height` over its *padded* board, so the
    # raw number includes TETRIS_PAD_H (=4) bottom wall rows. Show both the
    # raw value (same as what the agent/env sees) and the derived
    # piece-only stack height (which is what's actually rendered on the
    # 20-row playfield) so the viewer can't misread '23' as 'impossible'.
    raw_stack = int(board_stats.get("stack_height", 0))
    piece_stack = max(0, raw_stack - TETRIS_PAD_H)
    near_topout = piece_stack >= TETRIS_H - 1          # top spawn row about to block
    stack_color = (255, 120, 120) if near_topout else (255, 255, 255)
    draw.text((sx, sy), "STACK", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    draw.text((sx, sy), f"{piece_stack}/{TETRIS_H}",
              fill=stack_color, font=fnt_value)
    sy += 15
    draw.text((sx, sy), f"raw {raw_stack} (+{TETRIS_PAD_H} wall)",
              fill=(140, 140, 160), font=fnt_small)
    sy += 18

    draw.text((sx, sy), "HOLES", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    holes = board_stats.get("holes", 0)
    draw.text((sx, sy), str(holes),
              fill=(255, 200, 100) if holes > 0 else (255, 255, 255), font=fnt_value)
    sy += 24

    if near_topout:
        draw.text((sx, sy), "TOPOUT!", fill=(255, 90, 90), font=fnt_label)
        sy += 14
        draw.text((sx, sy), "spawn blocked soon",
                  fill=(220, 150, 150), font=fnt_small)
        sy += 16

    sy = by + board_h - 50
    draw.text((sx, sy), "REWARD", fill=(160, 160, 180), font=fnt_label)
    sy += 15
    draw.text((sx, sy), f"+{reward:.0f}", fill=(255, 220, 100), font=fnt_small)
    sy += 15
    draw.text((sx, sy), f"Total: {total_reward:.0f}", fill=(100, 255, 100), font=fnt_small)

    return img


def render_tetris_env(state_text: str, step: int, reward: float,
                      total_reward: float, action: str) -> Optional[Image.Image]:
    board_ids = parse_tetris_board(state_text)
    if board_ids is None:
        return None

    meta = parse_tetris_meta(state_text)
    next_syms = meta.get("next_symbols", [])

    cell = 24
    pad = 8
    board_w = TETRIS_W * cell
    board_h = TETRIS_H * cell
    sidebar_w = 150
    header_h = 40
    W = pad + board_w + pad + sidebar_w + pad
    H = header_h + pad + board_h + pad

    bg = (25, 25, 35)
    board_bg = (15, 15, 22)
    grid_color = (35, 35, 48)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    fnt_title = _pil_font(14)
    fnt_label = _pil_font(11)
    fnt_value = _pil_font(13)
    fnt_small = _pil_font_regular(10)

    # Header
    draw.text((pad, 10), "TETRIS", fill=(0, 240, 240), font=fnt_title)
    draw.text((pad + 90, 12), f"Step {step}", fill=(180, 180, 200), font=fnt_label)

    # Board background + border
    bx, by = pad, header_h + pad
    draw.rectangle([(bx - 2, by - 2), (bx + board_w + 1, by + board_h + 1)],
                   outline=(80, 80, 100), width=2)
    draw.rectangle([(bx, by), (bx + board_w - 1, by + board_h - 1)], fill=board_bg)

    # Grid lines
    for r in range(TETRIS_H + 1):
        y = by + r * cell
        draw.line([(bx, y), (bx + board_w, y)], fill=grid_color, width=1)
    for c in range(TETRIS_W + 1):
        x = bx + c * cell
        draw.line([(x, by), (x, by + board_h)], fill=grid_color, width=1)

    # Filled cells with 3D-style shading
    for r in range(TETRIS_H):
        for c in range(TETRIS_W):
            cid = board_ids[r, c]
            if cid == 0:
                continue
            base = TETRIS_PIECE_COLORS.get(cid, (100, 100, 100))
            x0 = bx + c * cell + 1
            y0 = by + r * cell + 1
            x1 = x0 + cell - 2
            y1 = y0 + cell - 2
            draw.rectangle([(x0, y0), (x1, y1)], fill=base)
            # Highlight (top-left edge)
            hi = tuple(min(255, int(v * 1.5)) for v in base)
            draw.line([(x0, y0), (x1, y0)], fill=hi, width=1)
            draw.line([(x0, y0), (x0, y1)], fill=hi, width=1)
            # Shadow (bottom-right edge)
            lo = tuple(max(0, int(v * 0.5)) for v in base)
            draw.line([(x0, y1), (x1, y1)], fill=lo, width=1)
            draw.line([(x1, y0), (x1, y1)], fill=lo, width=1)

    # Sidebar
    sx = bx + board_w + pad + 4
    sy = by

    # Score
    draw.text((sx, sy), "SCORE", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    score = meta.get("score", int(total_reward))
    draw.text((sx, sy), str(score), fill=(255, 255, 255), font=fnt_value)
    sy += 24

    # Lines
    draw.text((sx, sy), "LINES", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    draw.text((sx, sy), str(meta.get("lines", 0)), fill=(255, 255, 255), font=fnt_value)
    sy += 24

    # Level
    draw.text((sx, sy), "LEVEL", fill=(160, 160, 180), font=fnt_label)
    sy += 16
    draw.text((sx, sy), str(meta.get("level", 1)), fill=(255, 255, 255), font=fnt_value)
    sy += 30

    # Next pieces
    draw.text((sx, sy), "NEXT", fill=(160, 160, 180), font=fnt_label)
    sy += 18
    mini_cell = 10
    for sym in next_syms[:4]:
        cid = TETRIS_SYM_TO_COLOR_ID.get(sym, 1)
        color = TETRIS_PIECE_COLORS.get(cid, (100, 100, 100))
        shape = TETRIS_PIECE_SHAPES.get(sym, [[1]])
        for ri, row in enumerate(shape):
            for ci, val in enumerate(row):
                if val:
                    px = sx + ci * mini_cell
                    py = sy + ri * mini_cell
                    draw.rectangle([(px, py), (px + mini_cell - 1, py + mini_cell - 1)],
                                   fill=color)
        sy += (len(shape) + 1) * mini_cell + 4

    # Action + reward at bottom of sidebar
    sy = by + board_h - 80
    draw.text((sx, sy), "ACTION", fill=(160, 160, 180), font=fnt_label)
    sy += 15
    act_disp = str(action)[:16]
    draw.text((sx, sy), act_disp, fill=(180, 220, 255), font=fnt_small)
    sy += 18
    draw.text((sx, sy), f"Reward: {reward:+.0f}", fill=(255, 220, 100), font=fnt_small)
    sy += 15
    draw.text((sx, sy), f"Total:  {total_reward:.0f}", fill=(100, 255, 100), font=fnt_small)

    return img


def render_2048_env(state_text: str, step: int, reward: float,
                    total_reward: float, action: str) -> Optional[Image.Image]:
    board_powers = parse_2048_board(state_text)
    if board_powers is None:
        return None
    env = _get_2048_env()
    env.board = board_powers
    env.total_score = int(total_reward)

    frame = env.render()
    if frame is None:
        return None

    img = Image.fromarray(frame.astype(np.uint8))

    fnt = _pil_font(14)
    info_h = 30
    new_img = Image.new("RGB", (img.width, img.height + info_h), (250, 248, 238))
    new_img.paste(img, (0, info_h))
    draw = ImageDraw.Draw(new_img)
    draw.text((10, 6), f"Step {step}  Action: {action}  Reward: {reward:+.0f}  Total: {total_reward:.0f}",
              fill=(119, 110, 101), font=fnt)
    return new_img


def render_sokoban_env(state_text: str, step: int, reward: float,
                       total_reward: float, action: str) -> Optional[Image.Image]:
    grid = parse_sokoban_table(state_text)
    if grid is None:
        return None

    env = _get_sokoban_env()
    env.room_state = grid
    env.dim_room = grid.shape
    player_pos = np.argwhere(np.isin(grid, [5, 6]))
    if player_pos.size:
        env.player_position = player_pos[0]

    frame = env.render()
    if frame is not None and hasattr(frame, "shape"):
        img = Image.fromarray(frame.astype(np.uint8))
    else:
        tile = 36
        rows, cols = grid.shape
        img = Image.new("RGB", (cols * tile, rows * tile), (40, 40, 50))
        draw = ImageDraw.Draw(img)
        fnt_t = _pil_font(14)
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * tile, r * tile
                color = SOKOBAN_STATE_COLORS.get(int(grid[r, c]), (100, 100, 100))
                draw.rounded_rectangle([x0 + 1, y0 + 1, x0 + tile - 2, y0 + tile - 2],
                                       radius=4, fill=color)
                char = {0: "#", 1: "", 2: ".", 3: "*", 4: "$", 5: "@", 6: "+"}.get(int(grid[r, c]), "?")
                if char:
                    tw, th = _text_size(draw, char, fnt_t)
                    draw.text((x0 + (tile - tw) // 2, y0 + (tile - th) // 2),
                              char, fill=(255, 255, 255), font=fnt_t)

    fnt = _pil_font(13)
    info_h = 40
    new_img = Image.new("RGB", (img.width, img.height + info_h), (30, 30, 40))
    new_img.paste(img, (0, info_h))
    draw = ImageDraw.Draw(new_img)
    draw.text((6, 4), f"SOKOBAN  Step {step}  Action: {action}", fill=(255, 255, 255), font=fnt)
    draw.text((6, 22), f"Reward: {reward:+.1f}  Total: {total_reward:.1f}", fill=(255, 220, 100), font=fnt)
    return new_img


# ═══════════════════════════════════════════════════════════════════════════
# PIL fallback renderers  (Candy Crush, Super Mario, Avalon, Diplomacy)
# ═══════════════════════════════════════════════════════════════════════════

def render_candy_env(state_text: str, step: int, reward: float,
                     total_reward: float, action: str) -> Optional[Image.Image]:
    """Render Candy Crush board using GamingAgent-style circles with info sidebar."""
    board_colors = parse_candy_board(state_text)
    if board_colors is None:
        return None

    color_to_letter = {0: "_", 1: "G", 2: "C", 3: "P", 4: "R", 5: "Y", 6: "B"}
    rows, cols = board_colors.shape

    cell_size = 52
    spacing = 3
    board_pad = 24
    board_w = cols * cell_size + (cols - 1) * spacing
    board_h = rows * cell_size + (rows - 1) * spacing
    sidebar_w = 280
    total_w = board_pad + board_w + board_pad + sidebar_w + board_pad
    total_h = board_pad + board_h + board_pad
    total_h = max(total_h, 500)

    img = Image.new("RGB", (total_w, total_h), (20, 16, 30))
    draw = ImageDraw.Draw(img)

    bx0 = board_pad
    by0 = (total_h - board_h) // 2

    draw.rounded_rectangle(
        [bx0 - 6, by0 - 6, bx0 + board_w + 6, by0 + board_h + 6],
        radius=10, outline=(100, 80, 140), width=3,
    )

    for r in range(rows):
        for c in range(cols):
            val = int(board_colors[r, c])
            letter = color_to_letter.get(val, "_")
            color = CANDY_COLORS.get(letter, (64, 64, 64))
            x = bx0 + c * (cell_size + spacing)
            y = by0 + r * (cell_size + spacing)
            cx, cy = x + cell_size // 2, y + cell_size // 2
            radius = cell_size // 2 - 3

            if letter != "_":
                hi = tuple(min(255, int(ch * 1.35)) for ch in color)
                lo = tuple(max(0, int(ch * 0.65)) for ch in color)
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                             fill=color, outline=hi, width=2)
                draw.ellipse([cx - radius + 3, cy - radius + 3,
                              cx - radius + radius, cy - radius + radius],
                             fill=None, outline=(*hi, 70), width=1)
                fnt_c = _pil_font(16)
                tw, th = _text_size(draw, letter, fnt_c)
                draw.text((cx - tw // 2, cy - th // 2), letter,
                          fill=(255, 255, 255, 200), font=fnt_c)
            else:
                draw.rounded_rectangle([x + 2, y + 2, x + cell_size - 2, y + cell_size - 2],
                                       radius=6, fill=(40, 36, 50))

    sx = bx0 + board_w + board_pad + 10
    sy = by0

    fnt_title = _pil_font(22)
    fnt_label = _pil_font(14)
    fnt_value = _pil_font(17)

    draw.text((sx, sy), "CANDY CRUSH", fill=(255, 220, 60), font=fnt_title)
    sy += 38

    score_m = re.search(r"Score:\s*(\d+)", state_text)
    score_str = score_m.group(1) if score_m else f"{int(total_reward)}"
    moves_m = re.search(r"Moves Left:\s*(\d+)", state_text)
    moves_str = moves_m.group(1) if moves_m else "?"

    fields = [
        ("Step", str(step)),
        ("Moves Left", moves_str),
        ("Score", score_str),
        ("Reward", f"{reward:+.1f}"),
        ("Total Reward", f"{total_reward:.1f}"),
        ("Action", action[:32]),
    ]
    for label, value in fields:
        draw.text((sx, sy), label, fill=(160, 150, 180), font=fnt_label)
        sy += 17
        draw.text((sx, sy), value, fill=(255, 255, 255), font=fnt_value)
        sy += 28

    return img


CANDY_COLORS = {
    "R": (220, 40, 40), "C": (60, 160, 240), "G": (40, 200, 60),
    "P": (180, 60, 200), "Y": (240, 200, 40), "O": (240, 140, 40),
}

AVALON_GOOD_ROLES = {"Merlin", "Percival", "Servant"}
AVALON_EVIL_ROLES = {"Assassin", "Morgana", "Mordred", "Oberon", "Minion"}

_avalon_pw = None
_avalon_browser = None
_avalon_page = None

DIPLO_POWER_COLORS = {
    "AUSTRIA": (200, 50, 50),
    "ENGLAND": (50, 80, 180),
    "FRANCE":  (80, 160, 220),
    "GERMANY": (100, 100, 100),
    "ITALY":   (60, 180, 80),
    "RUSSIA":  (180, 180, 180),
    "TURKEY":  (220, 180, 50),
}


def _pil_font(size: int):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _pil_font_regular(size: int):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ── Avalon renderer (AgentEvolver pixel-art style via Playwright) ─────────

def _get_avalon_page():
    global _avalon_pw, _avalon_browser, _avalon_page
    if _avalon_page is None:
        from playwright.sync_api import sync_playwright
        _avalon_pw = sync_playwright().start()
        _avalon_browser = _avalon_pw.chromium.launch(headless=True)
        _avalon_page = _avalon_browser.new_page(viewport={"width": 960, "height": 540})
        template_path = Path(__file__).parent / "avalon_template.html"
        _avalon_page.goto(f"file://{template_path.resolve()}")
        _avalon_page.wait_for_load_state("networkidle")
    return _avalon_page


def _parse_avalon_state(state_text: str):
    try:
        state_dict = json.loads(state_text)
    except (json.JSONDecodeError, TypeError):
        state_dict = None

    if state_dict and isinstance(state_dict, dict):
        ref = state_dict.get("0", list(state_dict.values())[0] if state_dict else "")
    else:
        ref = state_text

    # Compact training format: "game=avalon | phase=opening | step=0/50 | quest=1 | round=1 | role=Merlin | team_size=2"
    compact_m = re.search(r"game\s*=\s*avalon", ref)
    if compact_m:
        phase_m = re.search(r"phase\s*=\s*(\w+)", ref)
        phase = phase_m.group(1).strip().title() if phase_m else "Unknown"
        quest_m = re.search(r"quest\s*=\s*(\d+)", ref)
        cur_quest = int(quest_m.group(1)) if quest_m else 1
        round_m = re.search(r"round\s*=\s*(\d+)", ref)
        cur_round = int(round_m.group(1)) if round_m else 1
        role_m = re.search(r"role\s*=\s*(\w+)", ref)
        role_name = role_m.group(1).strip() if role_m else None
        team_size_m = re.search(r"team_size\s*=\s*(\d+)", ref)
        num_players_m = re.search(r"players\s*=\s*(\d+)", ref)
        num_players = 5
        if num_players_m:
            num_players = int(num_players_m.group(1))
        roles = []
        for pid in range(num_players):
            if pid == 0 and role_name:
                is_good = role_name in AVALON_GOOD_ROLES
                roles.append({"name": role_name, "is_good": is_good})
            else:
                roles.append(None)
        return {
            "phase": phase,
            "quest": cur_quest,
            "round": cur_round,
            "quest_results": [],
            "leader": 0,
            "proposed_team": [],
            "num_players": num_players,
            "roles": roles,
        }

    phase_m = re.search(r"Avalon Game\s*[—–-]\s*([\w ]+)\s*===", ref)
    phase = phase_m.group(1).strip() if phase_m else "Unknown"

    quest_m = re.search(r"Current quest:\s*(\d+)\s*of\s*(\d+)", ref)
    cur_quest = int(quest_m.group(1)) if quest_m else 1

    round_m = re.search(r"Current round:\s*(\d+)\s*of\s*(\d+)", ref)
    cur_round = int(round_m.group(1)) if round_m else 1

    results_m = re.search(r"Quest results so far:\s*\[([^\]]*)\]", ref)
    quest_results = []
    if results_m:
        for tok in results_m.group(1).split(","):
            t = tok.strip().lower()
            if "pass" in t or "success" in t:
                quest_results.append("pass")
            elif "fail" in t:
                quest_results.append("fail")

    leader_m = re.search(r"Quest leader:\s*Player\s*(\d+)", ref)
    leader = int(leader_m.group(1)) if leader_m else None

    team_m = re.search(r"Proposed team:\s*\[([^\]]*)\]", ref)
    proposed_team = []
    if team_m:
        for tok in team_m.group(1).split(","):
            m = re.search(r"Player\s*(\d+)", tok.strip())
            if m:
                proposed_team.append(int(m.group(1)))

    num_players = len(state_dict) if state_dict else 1
    roles = []
    for pid in range(num_players):
        obs = state_dict.get(str(pid), "") if state_dict else ref
        rm = re.search(r"Your role:\s*(\w[\w ]*)", obs)
        if rm:
            name = rm.group(1).strip()
            is_good = name in AVALON_GOOD_ROLES
            roles.append({"name": name, "is_good": is_good})
        else:
            roles.append(None)

    return {
        "phase": phase,
        "quest": cur_quest,
        "round": cur_round,
        "quest_results": quest_results,
        "leader": leader,
        "proposed_team": proposed_team,
        "num_players": num_players if num_players >= 5 else 5,
        "roles": roles,
    }


_avalon_quest_tracker: Dict[str, Any] = {}
_diplomacy_controlled_power: Optional[str] = None


def _reset_avalon_tracker():
    _avalon_quest_tracker.clear()
    _avalon_quest_tracker.update({
        "prev_quest": 0,
        "quest_results": [],
        "last_proposed_team": [],
        "leader_rotation": 0,
    })


def render_avalon_frame(state_text, step, reward, total_reward, action):
    if not _avalon_quest_tracker:
        _reset_avalon_tracker()

    parsed = _parse_avalon_state(state_text)

    action_str = str(action).strip()
    is_compact = "game=avalon" in state_text or "game = avalon" in state_text

    if is_compact:
        cur_quest = parsed["quest"]
        prev_quest = _avalon_quest_tracker.get("prev_quest", 0)
        quest_results = list(_avalon_quest_tracker.get("quest_results", []))

        if prev_quest > 0 and cur_quest > prev_quest:
            prev_reward = _avalon_quest_tracker.get("prev_reward", 1.0)
            if prev_reward >= 1.05:
                quest_results.append("pass")
            else:
                quest_results.append("fail")
            _avalon_quest_tracker["quest_results"] = quest_results

        _avalon_quest_tracker["prev_quest"] = cur_quest
        _avalon_quest_tracker["prev_reward"] = reward
        parsed["quest_results"] = list(quest_results)

        team_re = re.match(r"^[\d,\s]+$", action_str)
        if team_re:
            parsed["phase"] = "Team Selection"
            team_members = [int(x.strip()) for x in action_str.split(",") if x.strip().isdigit()]
            _avalon_quest_tracker["last_proposed_team"] = team_members
            parsed["proposed_team"] = team_members
            _avalon_quest_tracker["leader_rotation"] = (
                _avalon_quest_tracker.get("leader_rotation", 0) + 1
            )
            parsed["leader"] = _avalon_quest_tracker["leader_rotation"] % parsed["num_players"]
        elif action_str.lower() in ("approve", "reject"):
            parsed["phase"] = "Team Voting"
            parsed["proposed_team"] = _avalon_quest_tracker.get("last_proposed_team", [])
        elif action_str.lower() in ("pass", "fail"):
            parsed["phase"] = "Quest Voting"
            parsed["proposed_team"] = _avalon_quest_tracker.get("last_proposed_team", [])
        else:
            parsed["phase"] = "Unknown"

    actions = {}
    roles = parsed.get("roles", [])
    try:
        action_dict = json.loads(action)
        if isinstance(action_dict, dict):
            for pid, act in action_dict.items():
                idx = int(pid) if str(pid).isdigit() else 0
                role_info = roles[idx] if idx < len(roles) and roles[idx] else None
                role_label = f"[{role_info['name']}] " if role_info else ""
                actions[str(pid)] = f"{role_label}{str(act).strip()}"[:200]
    except (json.JSONDecodeError, TypeError):
        role_name = roles[0]["name"] if roles and roles[0] else "Player 0"
        actions["0"] = f"[{role_name}] {action_str}"[:200]

    frame_data = {
        **parsed,
        "actions": actions,
        "step": step,
        "reward": float(reward),
        "total_reward": float(total_reward),
    }

    page = _get_avalon_page()
    page.evaluate(f"renderFrame({json.dumps(frame_data)})")
    screenshot_bytes = page.screenshot()
    return Image.open(io.BytesIO(screenshot_bytes))


# ── Diplomacy renderer (official diplomacy package map) ──────────────────

def _parse_diplomacy_powers(state_text: str) -> tuple:
    try:
        state_dict = json.loads(state_text)
    except (json.JSONDecodeError, TypeError):
        state_dict = None

    if state_dict and isinstance(state_dict, dict):
        ref = list(state_dict.values())[0] if state_dict else ""
    else:
        ref = state_text

    # Compact training format: "game=diplomacy | step=0/20 | phase=S1901M | power=GERMANY | centers=3 | units=F KIE, A BER, A MUN"
    compact_m = re.search(r"game\s*=\s*diplomacy", ref)
    if compact_m:
        phase_m = re.search(r"phase\s*=\s*(\S+)", ref)
        phase = phase_m.group(1).strip() if phase_m else "?"
        power_m = re.search(r"power\s*=\s*(\w+)", ref)
        power = power_m.group(1).strip() if power_m else None
        units_m = re.search(r"units\s*=\s*([^|]+)", ref)
        unit_list = []
        if units_m:
            unit_list = [u.strip() for u in units_m.group(1).split(",") if u.strip()]
        powers_info = {}
        for pname in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
            if pname == power:
                powers_info[pname] = {"units": unit_list, "centers": []}
            else:
                powers_info[pname] = {"units": [], "centers": []}
        return phase, powers_info

    phase_m = re.search(r"Phase:\s*(\S+)", ref)
    phase = phase_m.group(1) if phase_m else "?"

    powers_info = {}
    for pname in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
        ptext = state_dict.get(pname, ref) if state_dict else ref
        u_m = re.search(rf"{pname}[^:]*:\s*\d+\s*centers,\s*units=\[([^\]]*)\]", ref)
        units = []
        if u_m and u_m.group(1).strip():
            units = [u.strip().strip("'\"") for u in u_m.group(1).split(",") if u.strip()]
        sc_m = re.search(r"Your supply centers:\s*\[([^\]]*)\]", ptext)
        centers = []
        if sc_m and sc_m.group(1).strip():
            centers = [c.strip().strip("'\"") for c in sc_m.group(1).split(",") if c.strip()]
        powers_info[pname] = {"units": units, "centers": centers}
    return phase, powers_info


DIPLO_POWER_COLORS = {
    "AUSTRIA": (220, 50, 47),
    "ENGLAND": (38, 139, 210),
    "FRANCE":  (108, 113, 196),
    "GERMANY": (133, 153, 0),
    "ITALY":   (42, 161, 152),
    "RUSSIA":  (211, 54, 130),
    "TURKEY":  (203, 75, 22),
}


def render_diplomacy_map(state_text, step, reward, total_reward, action):
    from diplomacy import Game
    from diplomacy.engine.renderer import Renderer as DipRenderer

    phase, powers_info = _parse_diplomacy_powers(state_text)

    # Detect which power our model controls
    try:
        state_dict = json.loads(state_text)
    except (json.JSONDecodeError, TypeError):
        state_dict = None

    our_power_name = _diplomacy_controlled_power
    if not our_power_name:
        pm = re.search(r"power\s*=\s*(\w+)", state_text)
        our_power_name = pm.group(1) if pm else None

    # Parse orders from action for map arrows
    all_orders: Dict[str, list] = {}
    try:
        action_dict = json.loads(action) if isinstance(action, str) else action
        if isinstance(action_dict, dict):
            for pname, orders in action_dict.items():
                if isinstance(orders, list):
                    all_orders[pname] = orders
                else:
                    all_orders[pname] = [str(orders)]
    except (json.JSONDecodeError, TypeError):
        if isinstance(action, str) and action.strip():
            if our_power_name:
                all_orders[our_power_name] = [action.strip()]

    game = Game()
    for pname in game.powers:
        game.powers[pname].units = []
        game.powers[pname].centers = []
        game.powers[pname].retreats = {}
    for pname, info in powers_info.items():
        if pname in game.powers:
            game.powers[pname].units = info["units"]
            game.powers[pname].centers = info["centers"]

    # Set phase so the renderer validates orders correctly
    if phase and phase != "?":
        try:
            game.phase_type = phase[-1] if phase[-1] in ("M", "R", "A") else "M"
        except Exception:
            pass

    # Set orders on the game for map arrow rendering
    for pname, orders in all_orders.items():
        if pname in game.powers and orders:
            try:
                game.set_orders(pname, orders)
            except Exception:
                pass

    renderer = DipRenderer(game)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        renderer.render(output_path=f.name, incl_abbrev=True, incl_orders=True)
        svg_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png_path = f.name

    try:
        subprocess.run(
            ["rsvg-convert", "-w", "720", svg_path, "-o", png_path],
            check=True, capture_output=True,
        )
        map_img = Image.open(png_path).convert("RGB")
    except Exception:
        os.unlink(svg_path)
        os.unlink(png_path)
        return None
    finally:
        for p in (svg_path, png_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


    sidebar_w = 320
    header_h = 36
    canvas_h = max(map_img.height + header_h, 680)
    canvas_w = map_img.width + sidebar_w
    out = Image.new("RGB", (canvas_w, canvas_h), (15, 20, 35))
    out.paste(map_img, (0, header_h))
    draw = ImageDraw.Draw(out)

    # Header
    draw.rectangle([(0, 0), (canvas_w, header_h)], fill=(20, 30, 50))
    draw.text((10, 5), f"DIPLOMACY", fill=(255, 255, 255), font=_pil_font(14))
    draw.text((140, 8), f"Phase: {phase}   Step {step}",
              fill=(180, 200, 240), font=_pil_font(12))
    if our_power_name and our_power_name in powers_info:
        our_sc = len(powers_info[our_power_name].get("centers", []))
        draw.text((420, 8), f"★ {our_power_name}: {our_sc} Supply Centers",
                  fill=(255, 220, 100), font=_pil_font(12))
    else:
        draw.text((420, 8), f"Step {step}",
                  fill=(255, 220, 100), font=_pil_font(12))
    draw.line([(0, header_h - 1), (canvas_w, header_h - 1)], fill=(40, 60, 90), width=1)

    # Sidebar with all powers
    sx = map_img.width
    draw.rectangle([(sx, 0), (canvas_w, canvas_h)], fill=(18, 24, 42))
    draw.line([(sx, 0), (sx, canvas_h)], fill=(40, 60, 90), width=2)

    draw.text((sx + 10, header_h + 4), "ALL POWERS — ORDERS",
              fill=(180, 200, 240), font=_pil_font(10))
    y = header_h + 24

    power_order = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    fnt_name = _pil_font(10)
    fnt_order = _pil_font_regular(9)
    fnt_sc = _pil_font_regular(8)

    for pname in power_order:
        info = powers_info.get(pname, {"units": [], "centers": []})
        color = DIPLO_POWER_COLORS.get(pname, (180, 180, 180))
        is_ours = (pname == our_power_name)

        if is_ours:
            draw.rectangle([(sx + 2, y - 2), (canvas_w - 2, y + 14)],
                           fill=(color[0] // 4, color[1] // 4, color[2] // 4))

        label = f"{'★ ' if is_ours else ''}{pname}"
        sc_count = len(info.get("centers", []))
        unit_count = len(info.get("units", []))
        draw.text((sx + 8, y), label, fill=color, font=fnt_name)
        draw.text((sx + 175, y + 1), f"{unit_count}u {sc_count}sc",
                  fill=(160, 175, 200), font=fnt_sc)
        y += 16

        orders = all_orders.get(pname, [])
        if orders:
            for order in orders[:6]:
                order_text = str(order).strip()
                if len(order_text) > 38:
                    order_text = order_text[:35] + "..."
                draw.text((sx + 16, y), order_text,
                          fill=(200, 210, 230), font=fnt_order)
                y += 13
            if len(orders) > 6:
                draw.text((sx + 16, y), f"... +{len(orders) - 6} more",
                          fill=(130, 145, 170), font=fnt_order)
                y += 13
        else:
            draw.text((sx + 16, y), "(no orders)", fill=(100, 115, 140), font=fnt_order)
            y += 13

        y += 6

    return out


# ── Candy Crush renderer ─────────────────────────────────────────────────

def render_candy_frame(state_text, step, reward, total_reward, action):
    board = []
    for line in state_text.split("\n"):
        m = re.match(r"\d+\|\s*(.*)", line.strip())
        if m:
            board.append(m.group(1).strip().split())
    if not board:
        return None
    rows, cols = len(board), len(board[0])
    cell = 52
    pad = 12
    info_h = 80
    W = pad + cols * cell + pad
    H = info_h + pad + rows * cell + pad
    img = Image.new("RGB", (W, H), (40, 30, 50))
    draw = ImageDraw.Draw(img)
    fnt, fnt_cell = _pil_font(15), _pil_font(20)
    draw.text((pad, 8), f"CANDY CRUSH   Step {step}", fill=(255, 255, 255), font=fnt)
    draw.text((pad, 30), f"Action: {action}", fill=(200, 180, 255), font=fnt)
    score_m = re.search(r"Score:\s*(\d+)", state_text)
    moves_m = re.search(r"Moves Left:\s*(\d+)", state_text)
    draw.text((pad, 52),
              f"Score: {score_m.group(1) if score_m else int(total_reward)}  "
              f"Moves: {moves_m.group(1) if moves_m else '?'}  Reward: {reward:+.0f}",
              fill=(255, 220, 100), font=fnt)
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            x0 = pad + c * cell
            y0 = info_h + pad + r * cell
            color = CANDY_COLORS.get(ch, (120, 120, 120))
            draw.rounded_rectangle([x0 + 2, y0 + 2, x0 + cell - 3, y0 + cell - 3], radius=8, fill=color)
            tw, th = _text_size(draw, ch, fnt_cell)
            draw.text((x0 + (cell - tw) // 2, y0 + (cell - th) // 2), ch, fill=(255, 255, 255), font=fnt_cell)
    return img


# ── Super Mario fallback renderer (used when NES env unavailable) ────────

MARIO_ELEM_COLORS = {
    "Mario": (228, 50, 50), "Bricks": (160, 100, 40),
    "Question Blocks": (255, 200, 50), "Goomba": (140, 80, 40),
    "Koopas": (40, 160, 40), "Warp Pipe": (40, 180, 40),
    "Mushrooms": (200, 60, 60), "Stair Blocks": (120, 120, 120),
    "Flag": (255, 255, 100),
}


def render_mario_frame(state_text, step, reward, total_reward, action):
    elements = {}
    mario = re.search(r"Position of Mario:\s*\((\d+),\s*(\d+)\)", state_text)
    if mario:
        elements["Mario"] = [(int(mario.group(1)), int(mario.group(2)))]
    for label in ["Bricks", "Question Blocks", "Inactivated Blocks", "Monster Goomba",
                  "Monster Koopas", "Warp Pipe", "Item Mushrooms", "Stair Blocks", "Flag"]:
        m = re.search(rf"- {label}:\s*(.*)", state_text)
        if m and "None" not in m.group(1):
            coords = re.findall(r"\((\d+),\s*(\d+)(?:,\s*\d+)?\)", m.group(1))
            if coords:
                short = label.replace("Monster ", "").replace("Item ", "")
                elements[short] = [(int(x), int(y)) for x, y in coords]
    if not elements:
        return None
    W, H, ground_y = 560, 300, 210
    img = Image.new("RGB", (W, H), (107, 140, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, ground_y, W, H], fill=(139, 90, 43))
    draw.rectangle([0, ground_y, W, ground_y + 4], fill=(80, 180, 80))
    fnt, fnt_sm = _pil_font(13), _pil_font(11)
    draw.text((8, 6), f"SUPER MARIO  Step {step}", fill=(255, 255, 255), font=_pil_font(15))
    draw.text((8, 28), f"Action: {action}  Score: {reward:.0f}",
              fill=(255, 220, 100), font=fnt)
    scale, ox = 1.2, 20
    for label, coords in elements.items():
        color = MARIO_ELEM_COLORS.get(label, (200, 200, 200))
        for (x, y) in coords:
            sx, sy = int(x * scale) + ox, ground_y - int(y * scale)
            sz = 20 if label == "Mario" else 14
            if label == "Mario":
                draw.ellipse([sx - sz // 2, sy - sz, sx + sz // 2, sy], fill=color)
                draw.text((sx - 4, sy - sz - 12), "M", fill=(255, 255, 255), font=fnt_sm)
            elif label == "Flag":
                draw.rectangle([sx, sy - 40, sx + 4, sy], fill=(200, 200, 200))
                draw.polygon([(sx + 4, sy - 40), (sx + 24, sy - 30), (sx + 4, sy - 20)], fill=(0, 200, 0))
            else:
                draw.rectangle([sx - sz // 2, sy - sz, sx + sz // 2, sy], fill=color)
    return img


# ── Generic text fallback renderer ───────────────────────────────────────

def render_text_frame(game, state_text, step, reward, total_reward, action):
    W, H = 700, 500
    img = Image.new("RGB", (W, H), (25, 25, 35))
    draw = ImageDraw.Draw(img)
    fnt, fnt_sm = _pil_font(14), _pil_font(11)
    draw.text((12, 8), f"{game.upper()}  Step {step}  Reward: {reward:+.1f}  Total: {total_reward:.1f}",
              fill=(255, 255, 255), font=fnt)
    draw.text((12, 30), f"Action: {action[:80]}", fill=(180, 220, 255), font=fnt_sm)
    lines = textwrap.fill(state_text[:2000], width=90).split("\n")[:32]
    y = 56
    for line in lines:
        draw.text((12, y), line[:100], fill=(180, 180, 180), font=fnt_sm)
        y += 14
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Renderer dispatch
# ═══════════════════════════════════════════════════════════════════════════

RENDERERS = {
    "tetris":             render_tetris_env,
    "twenty_forty_eight": render_2048_env,
    "sokoban":            render_sokoban_env,
    "candy_crush":        render_candy_env,
    "super_mario":        render_mario_frame,
    "avalon":             render_avalon_frame,
    "diplomacy":          render_diplomacy_map,
}


# ═══════════════════════════════════════════════════════════════════════════
# Episode discovery + GIF assembly
# ═══════════════════════════════════════════════════════════════════════════

def find_best_episodes() -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for sp in glob.glob(str(OUTPUT_DIR / "**" / "rollout_summary.json"), recursive=True):
        with open(sp) as f:
            data = json.load(f)
        game = data.get("game", "unknown")
        stats = data.get("episode_stats", [])
        if not stats:
            continue
        top = max(stats, key=lambda s: s["total_reward"])
        if game not in best or top["total_reward"] > best[game]["reward"]:
            ep_dir = Path(sp).parent
            ep_file = ep_dir / f"episode_{top['episode_index']:03d}.json"
            if ep_file.exists():
                best[game] = {
                    "run": str(Path(sp).relative_to(OUTPUT_DIR)),
                    "episode_index": top["episode_index"],
                    "reward": top["total_reward"],
                    "steps": top["steps"],
                    "episode_path": str(ep_file),
                }
    return best


def load_episode(path: str):
    # utf-8-sig transparently strips a BOM if one is present (some recovered
    # baseline episode JSONs on Windows were written with a BOM by PowerShell).
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    # Training / inference rollouts store experiences; baseline scripts store step_details.
    if "experiences" in data:
        return data["experiences"]
    if "step_details" in data:
        return [
            {
                "state": s.get("state", ""),
                "action": s.get("action", "?"),
                "reward": s.get("reward", 0.0),
                "board_stats": s.get("board_stats"),
            }
            for s in data["step_details"]
        ]
    raise KeyError(f"{path} has neither 'experiences' nor 'step_details'")


# ─────────────────────────────────────────────────────────────────────────
# Baseline (avg_<model>_<game>) replay helper
# ─────────────────────────────────────────────────────────────────────────

def find_avg_baseline_episode(baseline_dir: Path) -> Optional[Dict[str, Any]]:
    """Pick the episode whose total_reward is closest to the mean in
    `<baseline_dir>/reward_summary.json`. Returns dict with 'episode_path',
    'episode_index', 'reward', 'steps'."""
    summary = baseline_dir / "reward_summary.json"
    if not summary.exists():
        return None
    with open(summary, encoding="utf-8-sig") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    rewards = [e.get("total_reward", 0.0) for e in episodes if "error" not in e]
    if not rewards:
        return None
    mean_r = sum(rewards) / len(rewards)
    chosen = min(
        (e for e in episodes if "error" not in e),
        key=lambda e: abs(e.get("total_reward", 0.0) - mean_r),
    )
    ep_idx = chosen.get("episode_index", 0)
    ep_path = baseline_dir / f"episode_{ep_idx:03d}.json"
    if not ep_path.exists():
        return None
    return {
        "episode_path": str(ep_path),
        "episode_index": ep_idx,
        "reward": chosen.get("total_reward", 0.0),
        "steps": chosen.get("steps", 0),
        "mean_reward": mean_r,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Extract episodes from training runs/ checkpoints (grpo_data)
# ═══════════════════════════════════════════════════════════════════════════

RUNS_GAME_MAP = {
    "Qwen3-8B_tetris_20260322_170438":              "tetris",
    "Qwen3-8B_20260321_213813_(Candy_crush)":       "candy_crush",
    "Qwen3-8B_2048_20260322_071227":                "twenty_forty_eight",
    "Qwen3-8B_avalon_20260322_200424":              "avalon",
    "Qwen3-8B_diplomacy_20260327_042539":           "diplomacy",
    "Qwen3-8B_super_mario_20260323_030839":         "super_mario",
}


def _extract_state_from_prompt(prompt: str) -> str:
    """Extract the game state observation from the training prompt."""
    start = prompt.find("Game state:\n\n")
    if start < 0:
        start = prompt.find("Game state:\n")
    if start < 0:
        return prompt
    state_start = prompt.index("\n", start) + 1
    if prompt[state_start] == "\n":
        state_start += 1
    end = prompt.find("\nAssigned subgoal:", state_start)
    if end < 0:
        end = prompt.find("\nAvailable actions", state_start)
    if end < 0:
        end = len(prompt)
    return prompt[state_start:end].strip()


def _extract_action_from_prompt_completion(prompt: str, completion: str) -> str:
    """Map the ACTION: N in the completion to the actual action name from the prompt."""
    m = re.search(r"ACTION:\s*(\d+)", completion)
    if not m:
        return completion.strip()
    action_num = int(m.group(1))
    action_map = {}
    for am in re.finditer(r"^\s+(\d+)\.\s+(.+)$", prompt, re.MULTILINE):
        action_map[int(am.group(1))] = am.group(2).strip()
    return action_map.get(action_num, f"action_{action_num}")


def _load_grpo_episodes(grpo_path: str) -> Dict[str, List[dict]]:
    """Load and group action_taking.jsonl by episode_id, returning sorted experiences."""
    episodes: Dict[str, List[dict]] = {}
    with open(grpo_path) as f:
        for line in f:
            d = json.loads(line)
            eid = d["episode_id"]
            if eid not in episodes:
                episodes[eid] = []
            episodes[eid].append(d)
    for eid in episodes:
        episodes[eid].sort(key=lambda x: x["step"])
    return episodes


def find_best_episodes_from_runs() -> Dict[str, Dict[str, Any]]:
    """Find best episodes across ALL training checkpoint steps."""
    best: Dict[str, Dict[str, Any]] = {}

    for run_name, game in RUNS_GAME_MAP.items():
        run_dir = RUNS_DIR / run_name
        if not run_dir.exists():
            continue

        grpo_dir = run_dir / "grpo_data"
        if not grpo_dir.exists():
            continue

        # Search ALL checkpoint steps for the globally best episode
        global_best_reward = float("-inf")
        global_best_steps = None
        global_best_eid = None
        global_best_step_name = None

        for grpo_file in sorted(grpo_dir.glob("step_*/action_taking.jsonl")):
            episodes = _load_grpo_episodes(str(grpo_file))
            if not episodes:
                continue
            for eid, steps in episodes.items():
                total = sum(s["reward"] for s in steps)
                if total > global_best_reward:
                    global_best_reward = total
                    global_best_steps = steps
                    global_best_eid = eid
                    global_best_step_name = grpo_file.parent.name

        if global_best_steps is None:
            continue

        experiences = []
        for s in global_best_steps:
            state = _extract_state_from_prompt(s["prompt"])
            action = _extract_action_from_prompt_completion(s["prompt"], s["completion"])
            experiences.append({
                "state": state,
                "action": action,
                "reward": s["reward"],
            })

        best[game] = {
            "run": run_name,
            "episode_id": global_best_eid,
            "reward": global_best_reward,
            "steps": len(global_best_steps),
            "grpo_step": global_best_step_name,
            "experiences": experiences,
        }

    return best


def _grpo_steps_to_experiences(steps: List[dict]) -> List[dict]:
    """Convert grpo_data steps into the experience format used by generate_gif."""
    experiences = []
    for s in steps:
        state = _extract_state_from_prompt(s["prompt"])
        action = _extract_action_from_prompt_completion(s["prompt"], s["completion"])
        experiences.append({"state": state, "action": action, "reward": s["reward"]})
    return experiences


def find_per_player_episodes_from_inference() -> Dict[str, Dict[str, Any]]:
    """Find best episode per role (Avalon) / per power (Diplomacy) from inference output.

    These episodes contain full multi-player state and action data for all
    participants, enabling rich visualization of all players.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # ── Avalon: one per role from fine-tuned inference ──
    role_best: Dict[str, tuple] = {}
    for sp in sorted(glob.glob(str(OUTPUT_DIR / "infer_avalon_da_*" / "avalon" / "*" / "rollout_summary.json"))):
        ep_dir = Path(sp).parent
        with open(sp) as f:
            data = json.load(f)
        for s in data.get("episode_stats", []):
            ep_file = ep_dir / f"episode_{s['episode_index']:03d}.json"
            if not ep_file.exists():
                continue
            with open(ep_file) as f:
                ep = json.load(f)
            try:
                state_dict = json.loads(ep["experiences"][0]["state"])
                rm = re.search(r"Your role:\s*(\w+)", state_dict.get("0", ""))
                role = rm.group(1) if rm else "Unknown"
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            reward = s["total_reward"]
            steps = s["steps"]
            if role not in role_best or reward > role_best[role][0] or \
               (reward == role_best[role][0] and steps < role_best[role][1]):
                role_best[role] = (reward, steps, str(ep_file), ep)

    for role in sorted(role_best):
        reward, steps, ep_path, ep = role_best[role]
        key = f"avalon_{role.lower()}"
        results[key] = {
            "game": "avalon",
            "episode_path": ep_path,
            "role": role,
            "reward": reward,
            "steps": steps,
        }

    # ── Diplomacy: one per power controlled by our model ──
    power_best: Dict[str, tuple] = {}
    for sp in sorted(glob.glob(str(OUTPUT_DIR / "infer_diplomacy_da_*" / "diplomacy" / "*" / "rollout_summary.json"))):
        ep_dir = Path(sp).parent
        with open(sp) as f:
            data = json.load(f)
        for s in data.get("episode_stats", []):
            ep_file = ep_dir / f"episode_{s['episode_index']:03d}.json"
            if not ep_file.exists():
                continue
            with open(ep_file) as f:
                ep = json.load(f)
            meta = ep.get("metadata", {})
            our_power = meta.get("controlled_power", "")
            if not our_power:
                continue
            sc_rewards = meta.get("final_sc_rewards", {})
            our_sc = sc_rewards.get(our_power, 0)
            reward = s["total_reward"]
            steps = s["steps"]
            if our_power not in power_best or our_sc > power_best[our_power][0] or \
               (our_sc == power_best[our_power][0] and reward > power_best[our_power][1]):
                power_best[our_power] = (our_sc, reward, steps, str(ep_file))

    for power in sorted(power_best):
        our_sc, reward, steps, ep_path = power_best[power]
        key = f"diplomacy_{power.lower()}"
        results[key] = {
            "game": "diplomacy",
            "episode_path": ep_path,
            "power": power,
            "sc_score": our_sc,
            "reward": reward,
            "steps": steps,
        }

    return results


def find_per_player_episodes_from_runs() -> Dict[str, Dict[str, Any]]:
    """Find the best successful episode per role (Avalon) and per power (Diplomacy)
    from training runs (compact single-player state format)."""
    results: Dict[str, Dict[str, Any]] = {}

    # ── Avalon: one per role ──
    avalon_run = "Qwen3-8B_avalon_20260322_200424"
    avalon_dir = RUNS_DIR / avalon_run / "grpo_data" / "step_0005" / "action_taking.jsonl"
    if avalon_dir.exists():
        episodes = _load_grpo_episodes(str(avalon_dir))
        role_best: Dict[str, tuple] = {}
        for eid, steps in episodes.items():
            rewards = [s["reward"] for s in steps]
            total = sum(rewards)
            if rewards[-1] < 1.5:
                continue
            rm = re.search(r"role\s*=\s*(\w+)", steps[0]["prompt"])
            role = rm.group(1) if rm else "Unknown"
            if role not in role_best or total > role_best[role][1]:
                role_best[role] = (eid, total, steps)
        for role, (eid, total, steps) in sorted(role_best.items()):
            key = f"avalon_{role.lower()}"
            results[key] = {
                "game": "avalon",
                "run": avalon_run,
                "episode_id": eid,
                "role": role,
                "reward": total,
                "steps": len(steps),
                "experiences": _grpo_steps_to_experiences(steps),
            }

    # ── Diplomacy: one per power (search ALL steps for global best) ──
    diplo_run = "Qwen3-8B_diplomacy_20260327_042539"
    diplo_grpo = RUNS_DIR / diplo_run / "grpo_data"
    if diplo_grpo.exists():
        power_best: Dict[str, tuple] = {}
        for grpo_file in sorted(diplo_grpo.glob("step_*/action_taking.jsonl")):
            episodes = _load_grpo_episodes(str(grpo_file))
            for eid, steps in episodes.items():
                total = sum(s["reward"] for s in steps)
                pm = re.search(r"power\s*=\s*(\w+)", steps[-1]["prompt"])
                power = pm.group(1) if pm else "Unknown"
                cm = re.search(r"centers\s*=\s*(\d+)", steps[-1]["prompt"])
                centers = int(cm.group(1)) if cm else 0
                if power not in power_best or total > power_best[power][1]:
                    power_best[power] = (eid, total, steps, centers)
        for power, (eid, total, steps, centers) in sorted(power_best.items()):
            key = f"diplomacy_{power.lower()}"
            results[key] = {
                "game": "diplomacy",
                "run": diplo_run,
                "episode_id": eid,
                "power": power,
                "reward": total,
                "centers": centers,
                "steps": len(steps),
                "experiences": _grpo_steps_to_experiences(steps),
            }

    return results


def _render_frames(game: str, experiences: list, max_frames: int = 0,
                    controlled_power: Optional[str] = None) -> List[Image.Image]:
    """Render all experience steps into PIL frames."""
    global _diplomacy_controlled_power, _tetris_sim
    if game == "avalon":
        _reset_avalon_tracker()
    if game == "diplomacy" and controlled_power:
        _diplomacy_controlled_power = controlled_power
    if game == "tetris":
        _tetris_sim = None
    frames: List[Image.Image] = []
    cumulative = 0.0
    items = experiences if max_frames <= 0 else experiences[:max_frames]

    # Tetris: collapse individual moves into macro placements (show only hard_drop frames)
    if game == "tetris":
        has_individual_moves = any(
            exp.get("action", "") in ("left", "right", "rotate_left", "rotate_right", "soft_drop", "no_op")
            for exp in items[:10]
        )
        if has_individual_moves:
            macro_items = []
            for exp in items:
                a = exp.get("action", "")
                cumulative += exp.get("reward", 0.0)
                if a == "hard_drop" or exp is items[-1]:
                    macro_items.append({**exp, "_cumulative": cumulative})
            items = macro_items

    # Super Mario: rewards are already cumulative game scores, not deltas
    mario_cumulative = (game == "super_mario")
    prev_mario_score = 0.0

    piece_num = 0
    for i, exp in enumerate(items):
        state = exp.get("state", "")
        action = exp.get("action", "?")
        reward = exp.get("reward", 0.0)
        if "_cumulative" not in exp:
            if mario_cumulative:
                cumulative += max(0, reward - prev_mario_score)
                prev_mario_score = reward
            else:
                cumulative += reward
        else:
            cumulative = exp["_cumulative"]

        if game == "tetris" and "_cumulative" in exp:
            piece_num += 1
            action = f"Piece {piece_num}"

        # Tetris with board_stats (macro action format, no board text)
        board_stats = exp.get("board_stats")
        if game == "tetris" and board_stats and not state:
            frame = render_tetris_stats(board_stats, i + 1, reward, cumulative, action)
        else:
            renderer = RENDERERS.get(game)
            frame = None
            if renderer:
                try:
                    frame = renderer(state, i, reward, cumulative, action)
                except Exception:
                    frame = None
            if frame is None:
                frame = render_text_frame(game, state, i, reward, cumulative, action)

        if frame is not None:
            frames.append(frame)
    return frames


def _uniform_frames(frames: List[Image.Image]) -> List[Image.Image]:
    """Pad all frames to same dimensions."""
    if not frames:
        return frames
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    # H.264 needs even dimensions
    max_w += max_w % 2
    max_h += max_h % 2
    result = []
    for f in frames:
        if f.size != (max_w, max_h):
            new = Image.new("RGB", (max_w, max_h), (0, 0, 0))
            new.paste(f, (0, 0))
            result.append(new)
        else:
            result.append(f)
    return result


def generate_replay(game: str, experiences: list, out_path: Path,
                    fps: float = 2.0, fmt: str = "mp4",
                    controlled_power: Optional[str] = None):
    frames = _render_frames(game, experiences, controlled_power=controlled_power)
    if not frames:
        print(f"  [!] No frames rendered for {game}")
        return

    uniform = _uniform_frames(frames)

    if fmt == "gif":
        duration_ms = int(1000 / fps)
        uniform[0].save(out_path, save_all=True, append_images=uniform[1:],
                        duration=duration_ms, loop=0)
    else:
        import imageio
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p",
                                    macro_block_size=2)
        for f in uniform:
            writer.append_data(np.array(f))
        writer.close()

    size_kb = out_path.stat().st_size // 1024
    print(f"  -> {out_path.name}  ({len(uniform)} frames, {fps} fps, {size_kb}KB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate replay videos from best episodes")
    parser.add_argument("--source", choices=["output", "runs", "per-player", "all", "baseline"], default="all",
                        help="Episode source: 'output' (inference), 'runs' (training checkpoints), "
                             "'per-player' (one replay per Avalon role / Diplomacy power), "
                             "'baseline' (avg-reward episode from baselines/output/<model>_<game>/), "
                             "or 'all' (inference output for visual games + per-player for avalon/diplomacy)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                        help="Output format: mp4 (default, better quality/size) or gif")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override OUTPUT_DIR for inference episode discovery")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Override RUNS_DIR for training checkpoint discovery")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Baseline episode directory (e.g. baselines/output/gpt54_tetris). "
                             "Used together with --source baseline.")
    parser.add_argument("--baseline-game", type=str, default=None,
                        help="Game name for baseline replay (e.g. tetris, twenty_forty_eight).")
    parser.add_argument("--baseline-tag", type=str, default="gpt54",
                        help="Tag used in output filename avg_<tag>_<game>.<ext> (default 'gpt54').")
    args = parser.parse_args()

    global OUTPUT_DIR, RUNS_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    if args.runs_dir:
        RUNS_DIR = Path(args.runs_dir)

    REPLAY_DIR.mkdir(exist_ok=True)
    ext = args.format  # "mp4" or "gif"

    if args.source == "all":
        # Visual games: pick best from EITHER inference output OR training runs
        best_output = find_best_episodes()
        best_runs = find_best_episodes_from_runs()
        visual_games = {"tetris", "twenty_forty_eight", "candy_crush", "super_mario", "sokoban"}

        # Merge: for each visual game pick whichever source has higher reward,
        # but only use runs if the state contains actual board data for the GUI renderer
        _board_parsers = {
            "tetris": parse_tetris_board,
            "candy_crush": parse_candy_board,
            "twenty_forty_eight": parse_2048_board,
        }
        best_visual: Dict[str, Dict[str, Any]] = {}
        for game in visual_games:
            out_info = best_output.get(game)
            run_info = best_runs.get(game)
            out_r = out_info["reward"] if out_info else float("-inf")
            run_r = run_info["reward"] if run_info else float("-inf")

            # Check if runs data has parseable board for GUI rendering
            run_renderable = False
            if run_info and game in _board_parsers:
                sample_state = run_info["experiences"][0]["state"]
                run_renderable = _board_parsers[game](sample_state) is not None
            elif run_info:
                run_renderable = True

            if run_r > out_r and run_info and run_renderable:
                best_visual[game] = {**run_info, "_source": "runs"}
            elif out_info:
                best_visual[game] = {**out_info, "_source": "output"}
            elif run_info:
                best_visual[game] = {**run_info, "_source": "runs"}

        # Avalon/Diplomacy from inference output (full multi-player state+actions)
        per_player = find_per_player_episodes_from_inference()

        print("=" * 72)
        print("ALL REPLAYS  (best source per game)")
        print("=" * 72)
        for game in sorted(best_visual):
            info = best_visual[game]
            src = info["_source"]
            print(f"  {game:22s}  reward {info['reward']:10.1f}  steps {info['steps']:4d}  [{src}]")
        for key in sorted(per_player):
            info = per_player[key]
            extra = info.get("role") or info.get("power") or ""
            sc = f"  SC={info['sc_score']:.3f}" if "sc_score" in info else ""
            print(f"  {key:22s}  reward {info['reward']:6.1f}  steps {info['steps']:3d}  "
                  f"{extra}{sc}  [inference]")
        print()

        for game in sorted(best_visual):
            info = best_visual[game]
            print(f"Rendering {game} ...")
            try:
                if info["_source"] == "output":
                    experiences = load_episode(info["episode_path"])
                else:
                    experiences = info["experiences"]
                out = REPLAY_DIR / f"best_{game}.{ext}"
                fps = 3.0 if game in ("tetris", "candy_crush", "twenty_forty_eight") else 1.5
                generate_replay(game, experiences, out, fps=fps, fmt=ext)
            except Exception as exc:
                import traceback
                print(f"  [!] {game} failed: {exc}")
                traceback.print_exc()

        for key in sorted(per_player):
            info = per_player[key]
            game = info["game"]
            print(f"Rendering {key} ...")
            try:
                if "episode_path" in info:
                    experiences = load_episode(info["episode_path"])
                else:
                    experiences = info["experiences"]
                out = REPLAY_DIR / f"best_{key}.{ext}"
                fps = 1.5
                power = info.get("power")
                generate_replay(game, experiences, out, fps=fps, fmt=ext,
                                controlled_power=power)
            except Exception as exc:
                import traceback
                print(f"  [!] {key} failed: {exc}")
                traceback.print_exc()

    elif args.source == "per-player":
        per_player = find_per_player_episodes_from_runs()
        print("=" * 72)
        print("PER-PLAYER REPLAYS  (Avalon by role, Diplomacy by power)")
        print("=" * 72)
        for key in sorted(per_player):
            info = per_player[key]
            extra = info.get("role") or info.get("power", "")
            centers = f"  centers={info['centers']}" if "centers" in info else ""
            print(f"  {key:25s}  {info['episode_id']:30s}  "
                  f"reward {info['reward']:6.1f}  steps {info['steps']:3d}  "
                  f"{extra}{centers}")
        print()

        for key in sorted(per_player):
            info = per_player[key]
            game = info["game"]
            print(f"Rendering {key} ...")
            try:
                out = REPLAY_DIR / f"best_{key}.{ext}"
                fps = 1.5
                power = info.get("power")
                generate_replay(game, info["experiences"], out, fps=fps, fmt=ext,
                                controlled_power=power)
            except Exception as exc:
                import traceback
                print(f"  [!] {key} failed: {exc}")
                traceback.print_exc()

    elif args.source == "baseline":
        if not args.baseline_dir or not args.baseline_game:
            parser.error("--source baseline requires --baseline-dir and --baseline-game")
        baseline_dir = Path(args.baseline_dir)
        if not baseline_dir.exists():
            parser.error(f"baseline-dir does not exist: {baseline_dir}")
        game = args.baseline_game
        info = find_avg_baseline_episode(baseline_dir)
        if info is None:
            parser.error(f"No usable episodes found in {baseline_dir}")

        print("=" * 72)
        print(f"BASELINE REPLAY  (avg episode from {baseline_dir})")
        print("=" * 72)
        print(f"  mean reward: {info['mean_reward']:.1f}")
        print(f"  chosen  ep : #{info['episode_index']}  reward {info['reward']:.1f}  "
              f"steps {info['steps']}")
        print(f"  episode    : {info['episode_path']}")
        print()

        experiences = load_episode(info["episode_path"])
        out = REPLAY_DIR / f"avg_{args.baseline_tag}_{game}.{ext}"
        fps = 3.0 if game in ("tetris", "candy_crush", "twenty_forty_eight") else 1.5
        print(f"Rendering {out.name} ...")
        try:
            generate_replay(game, experiences, out, fps=fps, fmt=ext)
        except Exception as exc:
            import traceback
            print(f"  [!] failed: {exc}")
            traceback.print_exc()

    elif args.source == "runs":
        best = find_best_episodes_from_runs()
        print("=" * 72)
        print("BEST EPISODES PER GAME  (from training runs/)")
        print("=" * 72)
        for game in sorted(best):
            info = best[game]
            print(f"  {game:22s}  {info['episode_id']:30s}  "
                  f"reward {info['reward']:10.1f}  steps {info['steps']:4d}  "
                  f"({info['grpo_step']})")
        print()

        for game in sorted(best):
            info = best[game]
            print(f"Rendering {game} ...")
            try:
                out = REPLAY_DIR / f"best_{game}.{ext}"
                fps = 3.0 if game in ("tetris", "candy_crush", "twenty_forty_eight") else 1.5
                generate_replay(game, info["experiences"], out, fps=fps, fmt=ext)
            except Exception as exc:
                import traceback
                print(f"  [!] {game} failed: {exc}")
                traceback.print_exc()
    else:
        best = find_best_episodes()
        print("=" * 72)
        print("BEST EPISODES PER GAME  (from inference output/)")
        print("=" * 72)
        for game in sorted(best):
            info = best[game]
            print(f"  {game:22s}  ep {info['episode_index']:3d}  "
                  f"reward {info['reward']:10.1f}  steps {info['steps']:4d}")
        print()

        for game in sorted(best):
            info = best[game]
            print(f"Rendering {game} ...")
            try:
                experiences = load_episode(info["episode_path"])
                out = REPLAY_DIR / f"best_{game}.{ext}"
                fps = 3.0 if game in ("tetris", "candy_crush", "twenty_forty_eight") else 1.5
                generate_replay(game, experiences, out, fps=fps, fmt=ext)
            except Exception as exc:
                print(f"  [!] {game} failed: {exc}")

    # Clean up env adapters
    for env in [_tetris_env, _2048_env, _sokoban_env, _candy_env]:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    # Clean up Playwright browser
    if _avalon_browser is not None:
        try:
            _avalon_browser.close()
        except Exception:
            pass
    if _avalon_pw is not None:
        try:
            _avalon_pw.stop()
        except Exception:
            pass

    # Try NES emulator replay for Super Mario (needs orak-mario env)
    mario_script = Path(__file__).parent / "generate_mario_replay.py"
    orak_python = Path("/workspace/anaconda3/envs/orak-mario/bin/python")
    has_mario = False
    if args.source in ("all", "output"):
        has_mario = True
    elif args.source == "runs":
        try:
            has_mario = "super_mario" in best
        except (UnboundLocalError, NameError):
            pass
    if has_mario and mario_script.exists() and orak_python.exists():
        print("\nRunning NES emulator replay for Super Mario ...")
        import subprocess
        result = subprocess.run(
            [str(orak_python), str(mario_script), "--format", ext],
            capture_output=True, text=True,
            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  [!] NES replay failed: {result.stderr[-300:]}")

    print(f"\nAll replays saved to {REPLAY_DIR}/")


if __name__ == "__main__":
    main()
