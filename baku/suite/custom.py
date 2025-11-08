import numpy as np
from dataclasses import dataclass, field

# Minimal dummy environment for testing with preprocessed data
class DummyEnv:
    def __init__(self, state_dim, action_dim, max_episode_len):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._max_episode_len = max_episode_len
        self.current_step = 0

    def observation_spec(self):
        return {"features": np.zeros(self._state_dim, dtype=np.float32)}

    def action_spec(self):
        return np.zeros(self._action_dim, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.observation_spec()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self._max_episode_len
        return DummyTimeStep(done)

class DummyTimeStep:
    def __init__(self, done):
        self.last_flag = done
        self.reward = 0.0
        self.observation = {"features": np.zeros(1, dtype=np.float32), "goal_achieved": 0}

    def last(self):
        return self.last_flag

# Flexible task creation function
def make_custom_task(dataset, env_cls=DummyEnv, **env_kwargs):
    """
    Returns a list of envs and task descriptions.
    env_cls: either DummyEnv (for dry run) or your real Franka+RUKA env
    env_kwargs: parameters to pass to env_cls constructor
    """
    envs = [env_cls(dataset._max_state_dim, dataset._max_action_dim, dataset._max_episode_len, **env_kwargs)]
    task_descriptions = [dataset.task_emb]
    return envs, task_descriptions

def task_make_fn(dataset, env_cls=DummyEnv, max_episode_len=1000, max_state_dim=50, **env_kwargs):
    envs = [
        env_cls(
            dataset._max_state_dim,
            dataset._max_action_dim,
            max_episode_len,
            **env_kwargs,
        )
    ]
    task_descriptions = [dataset.task_emb]
    return envs, task_descriptions

# Flexible CustomSuite
class CustomSuite:
    def __init__(self, dataset, env_cls=DummyEnv, **env_kwargs):
        self.name = "custom"
        self.dataset = dataset
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs

        # keys expected by train.py
        self.pixel_keys = []          # set ["pixels0"] if using camera
        self.proprio_key = "features"
        self.feature_key = "features"

        # training hyperparams
        self.action_repeat = 1
        self.discount = 0.99
        self.hidden_dim = 256
        self.num_eval_episodes = 1

        # expose task_make_fn to train.py
        self.task_make_fn = lambda: task_make_fn(self.dataset, self.env_cls, **self.env_kwargs)
