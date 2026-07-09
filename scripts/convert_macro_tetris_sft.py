"""Convert GPT-5.4 macro-action Tetris rollouts to action_taking SFT format.

Source : /workspace/SFT_Data/.../gpt54/tetris/rollouts.jsonl
         (20 episodes × ~79 macro steps = ~1573 examples)

Target : Game-AI-Agent/labeling/output/gpt54_skill_labeled/grpo_coldstart/
         tetris/action_taking.jsonl

Each record matches the schema of the existing primitive tetris file
(``type, game, episode, step, prompt, completion, chosen_action,
available_actions, reward, summary_state, intention, active_skill``) so
``trainer.SFT.data_loader._align_action_taking_to_coevolution`` can
transform it into the co-evolution episode_runner format at training
time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SYSTEM_PROMPT = """You are an expert game-playing agent. You receive a game state and must choose exactly one action by its NUMBER.

Rules:
- Study the state carefully before choosing.
- Consider which action makes the most progress toward winning.
- NEVER repeat the same action more than 2 times in a row.
- If recent actions got zero reward, change strategy.

Output format (strict):
REASONING: <1-2 sentences>
ACTION: <number>

"""

# Heuristic tag classification — keep parallel to primitive tetris SFT.
def _classify_tag(intent: str, lines_cleared: int) -> str:
    s = (intent or "").lower()
    if lines_cleared >= 1 or "clear" in s or "line" in s:
        return "CLEAR"
    if "survive" in s or "top out" in s or "danger" in s:
        return "SURVIVE"
    if "flat" in s or "stack" in s or "level" in s or "smooth" in s:
        return "SETUP"
    if "hole" in s or "fill" in s:
        return "BUILD"
    return "SETUP"


def _shorten_intent(intent: str, max_words: int = 14) -> str:
    text = re.split(r"[;.\n]", intent or "", maxsplit=1)[0].strip()
    if not text:
        return "place piece to keep stack flat"
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).rstrip(".,;")


def _build_subgoal_text(tag: str, chosen_action: str, lines_cleared: int) -> str:
    """Generate a concise, action-grounded subgoal phrase.

    Examples:
      "[CLEAR] hard-drop I col9 to clear 1 line"
      "[SETUP] place S-flat at col4 to keep stack flat"
      "[BUILD] place T col2 to fill hole"
    """
    # Action string like "S-flat col4 (+1hole, h=6)" or "I col9 (1line, h=2)"
    m = re.match(r"^(\S+)\s+col(\d+)", chosen_action)
    piece = m.group(1) if m else chosen_action.split()[0]
    col = m.group(2) if m else "?"
    if tag == "CLEAR":
        return f"hard-drop {piece} at col{col} to clear {lines_cleared} line(s)"
    if tag == "SURVIVE":
        return f"drop {piece} at col{col} to avoid topping out"
    if tag == "BUILD":
        return f"place {piece} at col{col} to fill hole"
    return f"place {piece} at col{col} to keep stack flat"


def _build_state_block(state: str) -> str:
    return (state or "").strip()


def _build_summary_state(state: str, available: list, action: str) -> str:
    # Best-effort: extract stack_h / holes / piece from state text
    fields = {}
    m = re.search(r"stack_h=(\d+)", state)
    if m: fields["stack_h"] = m.group(1)
    m = re.search(r"holes=(\d+)", state)
    if m: fields["holes"] = m.group(1)
    m = re.search(r"lines=(\d+)", state)
    if m: fields["lines"] = m.group(1)
    m = re.search(r"Current piece:\s*(\w+)", state)
    if m: fields["piece"] = m.group(1)
    m = re.search(r"Next Pieces:\s*([\w,]+)", state)
    if m: fields["next"] = m.group(1)
    fields["game"] = "tetris"
    return " | ".join(f"{k}={v}" for k, v in fields.items())


def convert_episode(ep: dict, ep_idx: int):
    """Yield one SFT record per experience in an episode rollout."""
    game = ep.get("game_name", "tetris")
    eid = ep.get("episode_id", f"ep_{ep_idx:03d}")
    for step_idx, exp in enumerate(ep.get("experiences", [])):
        state = exp.get("state", "")
        avail = exp.get("available_actions", [])
        chosen = exp.get("action", "")
        reward = float(exp.get("reward", 0.0))
        intent_long = exp.get("intentions", exp.get("subgoal", ""))

        if not avail or not chosen:
            continue
        # Find chosen index (1-indexed); skip if mismatch
        try:
            chosen_idx = avail.index(chosen) + 1
        except ValueError:
            # Try fuzzy match if action string changed slightly
            chosen_no_metrics = re.sub(r"\s*\(.*\)\s*$", "", chosen).strip()
            chosen_idx = None
            for i, a in enumerate(avail, start=1):
                if re.sub(r"\s*\(.*\)\s*$", "", a).strip() == chosen_no_metrics:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                continue

        # lines_cleared from action string (e.g., "Z-flat col6 (1line, h=7)")
        lc_match = re.search(r"\((\d+)line", chosen)
        lines_cleared = int(lc_match.group(1)) if lc_match else 0

        tag = _classify_tag(intent_long, lines_cleared)
        intent_short = _build_subgoal_text(tag, chosen, lines_cleared)
        intention = f"[{tag}] {intent_short}"

        action_lines = "\n".join(
            f"  {i+1}. {a}" for i, a in enumerate(avail)
        )

        prompt = (
            SYSTEM_PROMPT
            + "Game state:\n\n"
            + _build_state_block(state)
            + "\n\nAvailable actions (pick ONE by number):\n"
            + action_lines
            + "\n\nChoose the best action. Output REASONING then ACTION number."
        )

        completion = f"REASONING: Expert play.\nACTION: {chosen_idx}"

        yield {
            "type": "action_taking",
            "game": game,
            "episode": eid,
            "step": step_idx,
            "prompt": prompt,
            "completion": completion,
            "chosen_action": chosen,
            "available_actions": avail,
            "reward": reward,
            "summary_state": _build_summary_state(state, avail, chosen),
            "intention": intention,
            "active_skill": "",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/workspace/SFT_Data/game_sft/env_wrapper/gpt54/tetris/rollouts.jsonl",
    )
    ap.add_argument(
        "--dst",
        default="/workspace/Game-AI-Agent/labeling/output/gpt54_skill_labeled/grpo_coldstart/tetris/action_taking.jsonl",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"ERROR: source missing {src}", file=sys.stderr)
        sys.exit(1)

    out_records = []
    n_episodes = 0
    for line in open(src):
        if not line.strip():
            continue
        ep = json.loads(line)
        n_episodes += 1
        out_records.extend(convert_episode(ep, n_episodes))

    print(f"Source       : {src}")
    print(f"Destination  : {dst}")
    print(f"Episodes     : {n_episodes}")
    print(f"Output recs  : {len(out_records)}")
    if out_records:
        sample = out_records[0]
        print(f"\n--- Sample prompt (first record, last 600 chars) ---")
        print(sample["prompt"][-600:])
        print(f"\n--- Sample completion ---")
        print(sample["completion"])
        print(f"\n--- intention: {sample['intention']!r}")
        print(f"--- chosen_action: {sample['chosen_action']!r}")

    if args.dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ Wrote {len(out_records)} records → {dst}")


if __name__ == "__main__":
    main()
