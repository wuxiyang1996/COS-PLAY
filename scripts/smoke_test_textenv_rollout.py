"""Offline smoke test for the text-env rollout path in episode_runner.

Runs one ALFWorld and one WebShop episode with a fake vLLM client that
returns canned responses, and prints the exact action prompt sent at an
early step so it can be diffed against the SFT cold-start format.

Usage:  python scripts/smoke_test_textenv_rollout.py [alfworld|webshop]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

captured_prompts = {"action": [], "skill": [], "other": []}


class FakeResult:
    def __init__(self, text):
        self.text = text
        self.prompt_tokens = 0
        self.completion_tokens = 0


class FakeVLLMClient:
    async def generate_chat(self, messages, adapter=None, temperature=0.3,
                            max_tokens=128, stop=None, **kw):
        content = messages[-1]["content"]
        if adapter == "action_taking":
            captured_prompts["action"].append(content)
            return FakeResult("REASONING: Try the second option.\nACTION: 2")
        if adapter == "skill_selection":
            captured_prompts["skill"].append(content)
            return FakeResult("REASONING: fits.\nSKILL: 1")
        captured_prompts["other"].append(content)
        return FakeResult("[EXECUTE] make progress")

    async def generate(self, prompt, **kw):
        return FakeResult("[EXECUTE] make progress")


async def main(game: str):
    from trainer.coevolution.episode_runner import run_episode_async
    result = await run_episode_async(
        game, max_steps=6, vllm_client=FakeVLLMClient(),
        skill_bank=None, temperature=0.3,
        alfworld_split="train",
    )
    print(f"\n=== {game}: episode finished ===")
    print(f"steps={getattr(result, 'steps', '?')} reward={result.total_reward} "
          f"grpo_records={len(result.grpo_records)}")
    if captured_prompts["action"]:
        idx = min(1, len(captured_prompts["action"]) - 1)
        print(f"\n=== ACTION PROMPT (step {idx}) ===")
        print(captured_prompts["action"][idx])
    at = [r for r in result.grpo_records if r.adapter == "action_taking"]
    if at:
        print("\n=== GRPO action_taking record 0: reward =", at[0].reward, "===")


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "alfworld"
    asyncio.run(main(game))
