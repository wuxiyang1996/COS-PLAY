"""GRPO buffer should store real model text, not repr(parsed)."""

from skill_agents_grpo.grpo.grpo_outputs import (
    SkillBankLLMOutput,
    default_grpo_training_completion,
)


def test_skill_bank_llm_output_is_dict_without_raw_key():
    o = SkillBankLLMOutput({"eff_add": ["x"], "eff_del": []}, raw_completion='{"eff_add":["x"]}')
    assert isinstance(o, dict)
    assert o.get("eff_add") == ["x"]
    assert "_grpo_raw_completion" not in o
    assert o._grpo_raw_completion == '{"eff_add":["x"]}'
    assert list(dict(o).keys()) == ["eff_add", "eff_del"]


def test_default_training_completion_prefers_raw():
    o = SkillBankLLMOutput({"a": 1}, raw_completion='hello json')
    assert default_grpo_training_completion(o) == "hello json"


def test_default_training_completion_segment_rollouts():
    from skill_agents_grpo.infer_segmentation.preference import PreferenceListWithRollouts

    pl = PreferenceListWithRollouts([], raw_rollouts=['{"a":1}', '{"b":2}'])
    assert default_grpo_training_completion(pl) == '{"a":1}\n---\n{"b":2}'


def test_wrapper_uses_training_completion_fn():
    from skill_agents_grpo.grpo.buffer import GRPOBuffer
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction

    buf = GRPOBuffer()

    def orig(**kw):
        return SkillBankLLMOutput({"eff_add": ["z"]}, raw_completion="RAW_BODY")

    rewards = []

    def rf(sample, *a, **k):
        rewards.append(1.0)
        return 1.0

    w = GRPOCallWrapper(
        SkillFunction.CONTRACT,
        reward_fn=rf,
        buffer=buf,
        group_size=2,
        temperature=0.5,
        prompt_extractor=lambda *a, **k: "P",
        metadata_extractor=lambda *a, **k: {},
    )
    fn = w.wrap(orig)
    fn()

    samples = buf.samples_for(SkillFunction.CONTRACT)
    assert len(samples) == 1
    assert samples[0].completions == ["RAW_BODY", "RAW_BODY"]
