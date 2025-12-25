import pickle
import numpy as np
from torch.utils.data import IterableDataset


class CustomTeleopBCDataset(IterableDataset):
    """
    IterableDataset for BAKU training on teleop data.
    Produces Libero-like formatting for feature-only data:
      - "features": (history_len, max_state_dim)
      - "actions" : (history_len, num_queries, action_dim) if temporal_agg True
                  : (history_len, action_dim)               if temporal_agg False
      - "pixels0" : dummy (1, 3, H, W)
      - "task_emb": task embedding vector
    """

    def __init__(self, pkl_file, action_repeat: int = 10, history_len: int = 10, temporal_agg: bool = True):
        self.pkl_file = pkl_file
        self.action_repeat = int(action_repeat)
        self.history_len = int(history_len)
        self.temporal_agg = bool(temporal_agg)

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        self.observations = data["observations"]
        self.task_emb = np.asarray(data.get("task_emb", np.zeros(1)), dtype=np.float32)

        # Extract min/max for normalization
        self.min_arm = np.array(data["min_arm"], dtype=np.float32)
        self.max_arm = np.array(data["max_arm"], dtype=np.float32)
        self.min_ruka = np.array(data["min_ruka"], dtype=np.float32)
        self.max_ruka = np.array(data["max_ruka"], dtype=np.float32)

        self.min_arm_cmd = np.array(data["min_arm_commanded"], dtype=np.float32)
        self.max_arm_cmd = np.array(data["max_arm_commanded"], dtype=np.float32)

        self.min_ruka_cmd = np.array(data["min_ruka_commanded"], dtype=np.float32)
        self.max_ruka_cmd = np.array(data["max_ruka_commanded"], dtype=np.float32)

        self.__max_state_dim = max(len(obs["arm_states"]) + len(obs["ruka_states"]) + 1 for obs in self.observations)
        self.__max_action_dim = max(len(obs["commanded_arm_states"]) + len(obs["commanded_ruka_states"]) for obs in self.observations)
        self._num_samples = len(self.observations)

        # stats dict for normalization
        self.stats = {
            "features": {
                "min": np.concatenate([self.min_arm, self.min_ruka, np.array([0.0])]),
                "max": np.concatenate([self.max_arm, self.max_ruka, np.array([1.0])])
            },
            "actions": {
                "min": np.concatenate([self.min_arm_cmd, self.min_ruka_cmd]),
                "max": np.concatenate([self.max_arm_cmd, self.max_ruka_cmd]),
            }
        }

        print("ruka cmd min/max:", self.min_ruka_cmd.min(), self.max_ruka_cmd.max())

        # --- normalize all features and actions upfront (to [0, 1]) ---
        self.normalized_features = []
        self.normalized_actions = []

        eps = 1e-8  # avoid divide-by-zero

        for obs in self.observations:
            # features: arm + ruka
            feat = np.concatenate(
                [obs["arm_states"], obs["ruka_states"], np.array([obs["progress"]], dtype=np.float32)],
                axis=0
            ).astype(np.float32)
            feat = (feat - self.stats["features"]["min"]) / (
                self.stats["features"]["max"] - self.stats["features"]["min"] + eps
            )
            feat = np.clip(feat, 0.0, 1.0)
            self.normalized_features.append(feat)

            # actions: commanded arm + ruka
            act = np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0).astype(np.float32)
            act = (act - self.stats["actions"]["min"]) / (
                self.stats["actions"]["max"] - self.stats["actions"]["min"] + eps
            )
            act = np.clip(act, 0.0, 1.0)
            self.normalized_actions.append(act)

        self.normalized_features = np.stack(self.normalized_features, axis=0)
        self.normalized_actions = np.stack(self.normalized_actions, axis=0)

        print("sent actions min/max:", np.min(self.stats["actions"]["min"]), np.max(self.stats["actions"]["max"]))

    def _sample(self):
        idx = np.random.randint(0, self._num_samples)
        features = self.normalized_features[idx]
        actions = self.normalized_actions[idx]
        pixels = self.observations[idx]["pixels0"].astype(np.float32)

        # Libero-style features: (history_len, max_state_dim)
        feat = np.zeros((self.history_len, self.__max_state_dim), dtype=np.float32)
        feat[0, :features.shape[0]] = features
        task_emb_t = np.tile(
            self.task_emb[None, :],        # (1, E)
            (self.history_len, 1)          # (T, E)
        )
        pixels_t = np.tile(
            pixels[None, :, :, :],
            (self.history_len, 1, 1, 1)
        )

        # Libero-style actions: (history_len, action_repeat, action_dim)
        if self.temporal_agg:
            sampled_actions = np.tile(actions.reshape(1, 1, -1),
                                      (self.history_len, self.action_repeat, 1)).astype(np.float32)
        else:
            sampled_actions = np.tile(actions.reshape(1, 1, -1),
                                      (self.history_len, 1, 1)).astype(np.float32)

        return {
            # "pixels0": np.zeros((1, 3, 84, 84), dtype=np.float32),
            "pixels0": pixels_t,  # shape becomes (1, 3, 84, 84)
            "features": feat,
            "actions": sampled_actions,
            "task_emb": task_emb_t.astype(np.float32),  # (T, E)
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

