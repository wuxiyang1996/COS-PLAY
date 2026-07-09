from env_wrappers.alfworld_nl_wrapper import (
    ALFWorldNLWrapper,
    alfworld_obs_to_natural_language,
)


class FakeALFWorldEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return ["You are in a kitchen."], {
            "admissible_commands": [["look", "open fridge 1"]],
        }

    def step(self, actions):
        self.actions.extend(actions)
        return ["The fridge is open."], [0.5], [False], {
            "admissible_commands": [["take apple 1 from fridge 1"]],
        }

    def close(self):
        self.closed = True


def test_alfworld_obs_to_natural_language_includes_actions():
    text = alfworld_obs_to_natural_language(
        ["You see a sink."],
        {"admissible_commands": [["look", "go north"]]},
    )

    assert "You see a sink." in text
    assert "Admissible actions: look; go north" in text


def test_alfworld_nl_wrapper_reset_step_contract():
    fake = FakeALFWorldEnv()
    env = ALFWorldNLWrapper(fake, max_steps=3)

    obs, info = env.reset()
    assert "You are in a kitchen." in obs
    assert info["env"] == "alfworld"
    assert env.action_names == ["look", "open fridge 1"]

    obs, reward, terminated, truncated, info = env.step("open fridge 1")
    assert fake.actions == ["open fridge 1"]
    assert "The fridge is open." in obs
    assert reward == 0.5
    assert terminated is False
    assert truncated is False
    assert info["last_action"] == "open fridge 1"
    assert env.action_names == ["take apple 1 from fridge 1"]
