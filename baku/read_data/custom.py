import pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset


class CustomTeleopBCDataset(IterableDataset):
    """
    IterableDataset for BAKU training on your teleop data.
    Yields dicts with:
      - 'features' : concatenated proprio (arm + hand)
      - 'actions'  : concatenated commanded states (arm + hand)
      - 'task_emb' : task embedding vector
    """

    def __init__(self, pkl_file):
        self.pkl_file = pkl_file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        self.observations = data["observations"]
        self.task_emb = data["task_emb"]

        # compute max state/action dimensions for BAKU
        self.__max_state_dim = max(
            len(obs["arm_states"]) + len(obs["ruka_states"]) for obs in self.observations
        )
        self.__max_action_dim = max(
            len(obs["commanded_arm_states"]) + len(obs["commanded_ruka_states"])
            for obs in self.observations
        )

        self._num_samples = len(self.observations)

        # action normalization using PKL min/max
        self.stats = {
            "actions": {
                "min": np.concatenate([data["min_arm"], data["min_ruka"]]),
                "max": np.concatenate([data["max_arm"], data["max_ruka"]]),
            }
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"]) / 
                                (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5)
        }


    def _sample(self):
        idx = np.random.randint(0, self._num_samples)
        obs = self.observations[idx]

        # features = concatenated proprio
        features = np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0)

        # action = concatenated commanded states
        actions = np.concatenate(
            [obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0
        )
        actions = self.preprocess["actions"](actions)

        return {
            "features": features.astype(np.float32),
            "actions": actions.astype(np.float32),
            "task_emb": self.task_emb.astype(np.float32),
        }

    def __iter__(self):
        while True:
            yield self._sample()

    @property
    def _max_episode_len(self):
        return self._num_samples

    @property
    def _max_state_dim(self):
        return self.__max_state_dim

    @property
    def _max_action_dim(self):
        return self.__max_action_dim

    @property
    def envs_till_idx(self):
        # single "environment"
        return 1

    def sample_test(self, env_idx, step=None):
        # return dummy prompt for eval
        return {
            "prompt_features": None,
            "prompt_actions": None,
            "task_emb": self.task_emb.astype(np.float32),
        }
