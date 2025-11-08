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
    def __init__(
        self,
        state_dim,
        action_dim,
        max_episode_len,
        image_shape=(3, 84, 84),
        history_len=1,
        **kwargs,
    ):
        self._state_dim = int(state_dim)
        # allow None -> will be resolved by task_make_fn fallback
        self._action_dim = int(action_dim) if action_dim is not None else None
        self._max_episode_len = int(max_episode_len)
        self._image_shape = tuple(image_shape)  # (C, H, W)
        self._history_len = int(history_len)
        self.current_step = 0

    def observation_spec(self):
        # ResnetEncoder expects images shaped (C, H, W).
        # Keep pixels0 as (C, H, W) to match agent/network expectations.
        # features still include history dimension if used by the dataset.
        return {
            "pixels0": specs.BoundedArray(
                shape=(self._image_shape[0], self._image_shape[1], self._image_shape[2]),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name="pixels0",
            ),
            "features": specs.Array(
                shape=(self._history_len, self._state_dim), dtype=np.float32, name="features"
            ),
        }

    def action_spec(self):
        dim = self._action_dim if self._action_dim is not None else ()
        return specs.BoundedArray(
            shape=(int(self._action_dim),) if self._action_dim is not None else (0,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="action",
        )

    def reset(self):
        self.current_step = 0
        # return a timestep-like object consistent with step()
        return DummyTimeStep(False, self._state_dim, self._image_shape, self._history_len)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self._max_episode_len
        return DummyTimeStep(done, self._state_dim, self._image_shape, self._history_len)

class DummyTimeStep:
    def __init__(self, done, state_dim, image_shape, history_len=1):
        self.last_flag = bool(done)
        self.reward = 0.0
        # pixels shaped (C, H, W) to match observation_spec and agent expectation
        self.observation = {
            "pixels0": np.zeros((image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8),
            "features": np.zeros((history_len, state_dim), dtype=np.float32),
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

    state_dim = getattr(dataset, "_max_state_dim", None)
    action_dim = getattr(dataset, "_max_action_dim", None)
    episode_len = getattr(dataset, "_max_episode_len", None)
    task_emb = getattr(dataset, "task_emb", None)

    envs = [env_cls(state_dim, action_dim, episode_len, **env_kwargs)]
    task_descriptions = [task_emb]
    return envs, task_descriptions

def task_make_fn(dataset, env_cls=DummyEnv, max_episode_len=1000, max_state_dim=50, max_action_dim=10, **env_kwargs):
    if isinstance(dataset, (dict, DictConfig)):
        dataset = hydra.utils.call(dataset)

    # don't forward config-only keys to env constructor
    env_kwargs = dict(env_kwargs)
    env_kwargs.pop("max_action_dim", None)

    state_dim = int(getattr(dataset, "_max_state_dim", max_state_dim))
    # fallback to config max_action_dim if dataset missing value
    action_dim = getattr(dataset, "_max_action_dim", None)
    if action_dim is None:
        action_dim = int(max_action_dim)
    else:
        action_dim = int(action_dim)
    episode_len = int(getattr(dataset, "_max_episode_len", max_episode_len))
    history_len = int(getattr(dataset, "history_len", env_kwargs.get("history_len", 1)))

    # pass history_len into env so observation_spec aligns with dataset
    env_kwargs["history_len"] = history_len

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
    history_len: int = 1
    
    # Add task_make_fn so OmegaConf can see it
    task_make_fn: any = field(default_factory=lambda: task_make_fn)