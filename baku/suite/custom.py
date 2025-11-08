from dataclasses import dataclass, field
import numpy as np
import hydra
from omegaconf import DictConfig
from dm_env import specs

class ObsArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

# Minimal dummy environment for testing with preprocessed data
class DummyEnv:
    def __init__(self, state_dim, action_dim, max_episode_len, image_shape=(3, 84, 84), **kwargs):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._max_episode_len = max_episode_len
        self._image_shape = tuple(image_shape)
        self.current_step = 0
    
    def observation_spec(self):
        return {
            "pixels0": specs.BoundedArray(
                shape=self._image_shape, dtype=np.uint8, minimum=0, maximum=255, name="pixels0"
            ),
            "features": specs.Array(
                shape=(self._state_dim,), dtype=np.float32, name="features"
            ),
        }

    def action_spec(self):
        return specs.BoundedArray(
            shape=(self._action_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="action",
        )

    def reset(self):
        self.current_step = 0
        return {
            "pixels0": np.zeros(self._image_shape, dtype=np.uint8),
            "features": np.zeros((self._state_dim,), dtype=np.float32),
            "task_emb": np.zeros(256, dtype=np.float32),  # match whatever your embedding size is
            "goal_achieved": False,
        }

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self._max_episode_len
        return DummyTimeStep(done, self._state_dim, self._image_shape)

class DummyTimeStep:
    def __init__(self, done, state_dim, image_shape):
        self.last_flag = done
        self.reward = 0.0
        self.observation = {
            "pixels0": np.zeros(image_shape, dtype=np.uint8),
            "features": np.zeros((state_dim,), dtype=np.float32),
            "task_emb": np.zeros(256, dtype=np.float32),
            "goal_achieved": False,
        }

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