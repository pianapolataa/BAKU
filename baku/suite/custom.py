from dataclasses import dataclass, field
import numpy as np
import hydra
from omegaconf import DictConfig

# Minimal dummy environment for testing with preprocessed data
class DummyEnv:
    def __init__(self, state_dim, action_dim, max_episode_len, max_action_dim=None, **kwargs):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._max_episode_len = max_episode_len
        self.current_step = 0

    def observation_spec(self):
        # return dict mapping keys -> objects with .shape (numpy arrays work)
        # ensure matches cfg.suite.proprio_key and cfg.suite.feature_key
        return {
            "features": np.zeros((self._state_dim,), dtype=np.float32),
        }

    def action_spec(self):
        # return an object with .shape
        return np.zeros((self._action_dim,), dtype=np.float32)

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
    env_cls: either DummyEnv (for dry run) or your real env
    env_kwargs: parameters to pass to env_cls constructor
    """
    # If dataset is a config, instantiate it
    if isinstance(dataset, (dict, DictConfig)):
        dataset = hydra.utils.call(dataset)

    envs = [env_cls(dataset._max_state_dim, dataset._max_action_dim, dataset._max_episode_len, **env_kwargs)]
    task_descriptions = [dataset.task_emb]
    return envs, task_descriptions

def task_make_fn(dataset, env_cls=DummyEnv, max_episode_len=1000, max_state_dim=50, max_action_dim=10, **env_kwargs):
    if isinstance(dataset, (dict, DictConfig)):
        dataset = hydra.utils.call(dataset)

    # don't forward config-only keys to env constructor
    env_kwargs = dict(env_kwargs)
    env_kwargs.pop("max_action_dim", None)

    state_dim = getattr(dataset, "_max_state_dim", max_state_dim)
    action_dim = getattr(dataset, "_max_action_dim", None)
    episode_len = getattr(dataset, "_max_episode_len", max_episode_len)

    envs = [env_cls(state_dim, action_dim, episode_len, **env_kwargs)]
    task_descriptions = [getattr(dataset, "task_emb", None)]
    return envs, task_descriptions

@dataclass
class CustomSuite:
    dataset: any
    env_cls: type = DummyEnv
    hidden_dim: int = 256
    action_repeat: int = 1
    discount: float = 0.99
    num_eval_episodes: int = 1
    pixel_keys: list = field(default_factory=list)
    proprio_key: str = "features"
    feature_key: str = "features"
    
    # Add task_make_fn so OmegaConf can see it
    task_make_fn: any = field(default_factory=lambda: task_make_fn)