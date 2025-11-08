# import pickle
# import numpy as np
# import torch
# from torch.utils.data import IterableDataset


# class CustomTeleopBCDataset(IterableDataset):
#     """
#     IterableDataset for BAKU training on your teleop data.
#     Yields dicts with:
#       - 'features' : concatenated proprio (arm + hand)
#       - 'actions'  : concatenated commanded states (arm + hand)
#       - 'task_emb' : task embedding vector
#     """

#     def __init__(self, pkl_file, action_repeat: int = 1):
#         self.pkl_file = pkl_file
#         self.action_repeat = int(action_repeat)
#         with open(pkl_file, "rb") as f:
#             data = pickle.load(f)

#         self.observations = data["observations"]
#         self.task_emb = np.asarray(data["task_emb"], dtype=np.float32)

#         # compute max state/action dimensions for BAKU
#         self.__max_state_dim = max(
#             len(obs["arm_states"]) + len(obs["ruka_states"]) for obs in self.observations
#         )
#         self.__max_action_dim = max(
#             len(obs["commanded_arm_states"]) + len(obs["commanded_ruka_states"])
#             for obs in self.observations
#         )

#         self._num_samples = len(self.observations)

#         # create actions array (raw, before preprocess) so agent.discretize can use it
#         self.actions = np.stack(
#             [
#                 np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]])
#                 for obs in self.observations
#             ],
#             axis=0,
#         ).astype(np.float32)

#         # action normalization using PKL min/max
#         self.stats = {
#             "actions": {
#                 "min": np.concatenate([data["min_arm"], data["min_ruka"]]),
#                 "max": np.concatenate([data["max_arm"], data["max_ruka"]]),
#             }
#         }
#         self.preprocess = {
#             "actions": lambda x: (x - self.stats["actions"]["min"]) / 
#                                 (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5)
#         }


#     def _sample(self):
#         idx = np.random.randint(0, self._num_samples)
#         obs = self.observations[idx]

#         # features = concatenated proprio
#         features = np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0)

#         # action = concatenated commanded states
#         actions = np.concatenate(
#             [obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0
#         )
#         actions = self.preprocess["actions"](actions)

#         # pad action vector to dataset max action dim so flattened size matches agent
#         d = actions.shape[0]
#         if d < self.__max_action_dim:
#             padded = np.zeros((self.__max_action_dim,), dtype=actions.dtype)
#             padded[:d] = actions
#             actions = padded

#         # replicate along the action-time axis if agent expects repeated actions
#         if self.action_repeat > 1:
#             # tile into shape (t2, D) then add batch/time dims -> (1, t2, D)
#             tiled = np.tile(actions.reshape(1, -1), (self.action_repeat, 1))
#             actions_out = tiled.reshape(1, self.action_repeat, -1).astype(np.float32)
#         else:
#             actions_out = actions.reshape(1, 1, -1).astype(np.float32)

#         return {
#             # "pixels0": np.zeros((1, 84, 84, 3), dtype=np.float32), 
#             # "features": features.astype(np.float32),
#             "pixels0": np.zeros((1, 3, 84, 84), dtype=np.float32),
#             "features": features.reshape(1, -1).astype(np.float32),  # t=1
#             "actions": actions_out,
#             "task_emb": self.task_emb.astype(np.float32),
#         }

#     def __iter__(self):
#         while True:
#             yield self._sample()

#     @property
#     def _max_episode_len(self):
#         return self._num_samples

#     @property
#     def _max_state_dim(self):
#         return self.__max_state_dim

#     @property
#     def _max_action_dim(self):
#         return self.__max_action_dim

#     @property
#     def envs_till_idx(self):
#         # single "environment"
#         return 1

#     def sample_test(self, env_idx, step=None):
#         # return dummy prompt for eval
#         return {
#             "prompt_features": None,
#             "prompt_actions": None,
#             "task_emb": self.task_emb.astype(np.float32),
#         }

# ...existing code...
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
        # match agent's expected repeat by default; can be overridden via Hydra param
        self.action_repeat = int(action_repeat)
        self.history_len = int(history_len)
        self.temporal_agg = bool(temporal_agg)

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # raw observations (each obs is a single-step dict in this dataset)
        self.observations = data["observations"]
        self.task_emb = np.asarray(data.get("task_emb", data.get("task_embedding", np.zeros(1))), dtype=np.float32)

        # compute max state/action dimensions
        self.__max_state_dim = max(
            len(obs["arm_states"]) + len(obs["ruka_states"]) for obs in self.observations
        )
        self.__max_action_dim = max(
            len(obs["commanded_arm_states"]) + len(obs["commanded_ruka_states"])
            for obs in self.observations
        )

        self._num_samples = len(self.observations)

        # raw stacked actions (keep unchanged for rollouts / discretize)
        self.actions = np.stack(
            [
                np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]])
                for obs in self.observations
            ],
            axis=0,
        ).astype(np.float32)

        # action normalization using PKL min/max if present in file, else identity
        if "min_arm" in data and "min_ruka" in data and "max_arm" in data and "max_ruka" in data:
            self.stats = {
                "actions": {
                    "min": np.concatenate([data["min_arm"], data["min_ruka"]]),
                    "max": np.concatenate([data["max_arm"], data["max_ruka"]]),
                }
            }
            self.preprocess = {
                "actions": lambda x: (x - self.stats["actions"]["min"])
                / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5)
            }
        else:
            # fallback: assume already normalized
            self.stats = {"actions": {"min": 0.0, "max": 1.0}}
            self.preprocess = {"actions": lambda x: x}

    def _sample(self):
        idx = np.random.randint(0, self._num_samples)
        obs = self.observations[idx]

        # features = concatenated proprio (arm + ruka)
        features = np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0).astype(np.float32)

        # action = concatenated commanded states, then normalize
        actions = np.concatenate(
            [obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0
        ).astype(np.float32)
        actions = self.preprocess["actions"](actions)

        # pad action vector to dataset max action dim so flattened size matches agent
        d = actions.shape[0]
        if d < self.__max_action_dim:
            padded = np.zeros((self.__max_action_dim,), dtype=actions.dtype)
            padded[:d] = actions
            actions = padded

        # Build Libero-style features: (history_len, max_state_dim)
        feat = np.zeros((self.history_len, self.__max_state_dim), dtype=np.float32)
        state_dim = features.shape[0]
        feat[0, :state_dim] = features

        # Build Libero-style actions
        if self.temporal_agg:
            # (history_len, action_repeat, action_dim)
            act = np.tile(actions.reshape(1, -1), (self.history_len, self.action_repeat, 1))
            sampled_actions = act.astype(np.float32)
        else:
            # (history_len, action_dim)
            sampled_actions = np.tile(actions.reshape(1, -1), (self.history_len, 1)).astype(np.float32)

        return {
            # keep a dummy pixel tensor for compatibility with code expecting pixels
            "pixels0": np.zeros((1, 3, 84, 84), dtype=np.float32),
            "features": feat,
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
        # single "environment"
        return 1

    def sample_test(self, env_idx, step=None):
        # Keep simple: return prompt placeholders similar to Libero features branch
        # This dataset is single-step per entry, so we return None prompts unless explicit step logic needed.
        prompt_features = None
        prompt_actions = None
        return {
            "prompt_features": prompt_features,
            "prompt_actions": prompt_actions,
            "task_emb": self.task_emb.astype(np.float32),
        }