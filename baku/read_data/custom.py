import pickle
import numpy as np
from torch.utils.data import IterableDataset


class CustomTeleopBCDataset(IterableDataset):
    """
    IterableDataset for BAKU training on your teleop data.
    Produces Libero-like formatting for feature-only data:
      - "features": (history_len, max_state_dim)
      - "actions" : (history_len, num_queries, action_dim)  if temporal_agg True
                  : (history_len, action_dim)                if temporal_agg False
      - "pixels0" : dummy (1, 3, H, W) to remain compatible if code expects pixels
      - "task_emb": task embedding vector
    """

    def __init__(
        self, pkl_file, action_repeat: int = 10, history_len: int = 1, temporal_agg: bool = True
    ):
        self.pkl_file = pkl_file
        self.action_repeat = int(action_repeat)
        self.history_len = int(history_len)
        self.temporal_agg = bool(temporal_agg)

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        self.observations = data["observations"]
        self.task_emb = np.asarray(
            data.get("task_emb", data.get("task_embedding", np.zeros(1))), dtype=np.float32
        )

        self.min_arm = np.array(data.get("min_arm", np.zeros(7)), dtype=np.float32)
        self.max_arm = np.array(data.get("max_arm", np.ones(7)), dtype=np.float32)
        self.min_ruka = np.array(data.get("min_ruka", np.zeros(16)), dtype=np.float32)
        self.max_ruka = np.array(data.get("max_ruka", np.ones(16)), dtype=np.float32)

        self.__max_state_dim = max(
            len(obs["arm_states"]) + len(obs["ruka_states"]) for obs in self.observations
        )
        self.__max_action_dim = max(
            len(obs["commanded_arm_states"]) + len(obs["commanded_ruka_states"])
            for obs in self.observations
        )

        self._num_samples = len(self.observations)

        # raw stacked actions
        self.actions = np.stack(
            [
                np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]])
                for obs in self.observations
            ],
            axis=0,
        ).astype(np.float32)

        # normalization using PKL min/max if present, else identity
        if "min_arm" in data and "min_ruka" in data and "max_arm" in data and "max_ruka" in data:
            self.stats = {
                "actions": {
                    "min": np.concatenate([data["min_arm"], data["min_ruka"]]),
                    "max": np.concatenate([data["max_arm"], data["max_ruka"]]),
                },
                "features": {
                    "min": np.concatenate([data["min_arm"], data["min_ruka"]]),
                    "max": np.concatenate([data["max_arm"], data["max_ruka"]]),
                }
            }
        else:
            self.stats = {"actions": {"min": 0.0, "max": 1.0}}
            self.preprocess = {"actions": lambda x: x}

    def _sample(self):
        idx = np.random.randint(0, self._num_samples)
        obs = self.observations[idx]

        # Concatenate proprioceptive features (arm + ruka) + actions
        features = np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0).astype(np.float32)
        actions = np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0).astype(np.float32)

        if self.temporal_agg:
            sampled_actions = np.tile(actions.reshape(1, 1, -1),
                                    (self.history_len, self.action_repeat, 1)).astype(np.float32)
        else:
            sampled_actions = np.tile(actions.reshape(1, 1, -1),
                                    (self.history_len, 1, 1)).astype(np.float32)

        return {
            "pixels0": np.zeros((1, 3, 84, 84), dtype=np.float32),
            "features": features,
            "actions": sampled_actions,
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
        return 1

    def sample_test(self, env_idx, step=None):
        prompt_features = None
        prompt_actions = None
        return {
            "prompt_features": prompt_features,
            "prompt_actions": prompt_actions,
            "task_emb": self.task_emb.astype(np.float32),
        }