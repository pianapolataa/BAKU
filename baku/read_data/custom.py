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

#     def __init__(self, pkl_file):
#         self.pkl_file = pkl_file
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

#         return {
#             "features": features.astype(np.float32),
#             "actions": actions.astype(np.float32),
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
from typing import Any, Dict


class CustomTeleopBCDataset(IterableDataset):
    """
    IterableDataset for BAKU training on teleop data saved by preprocess.py.

    Yields dicts with keys:
      - 'features' : concatenated proprio (arm + hand)
      - 'actions'  : concatenated commanded states (arm + hand) (normalized)
      - 'task_emb' : task embedding vector (np.float32)

    Exposed attributes required by train.py / suite.task_make_fn:
      - _max_episode_len (int)
      - _max_state_dim (int)
      - _max_action_dim (int)
      - stats (dict)
      - preprocess (dict of callables)
      - actions (np.ndarray)    # raw actions (before preprocess) for discretizer
      - task_emb (np.ndarray)
      - envs_till_idx (int)
      - sample_test(env_idx, step=None) -> prompt
      - __iter__ yields training samples (infinite)
    """

    def __init__(self, pkl_file: str):
        self.pkl_file = pkl_file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # core data
        self.observations = data["observations"]
        self.task_emb = np.asarray(data["task_emb"], dtype=np.float32)

        # create raw actions array (before normalization) for discretization
        self.actions = np.stack(
            [
                np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]])
                for obs in self.observations
            ],
            axis=0,
        ).astype(np.float32)

        # compute max state/action dimensions
        self.__max_state_dim = max(
            len(obs["arm_states"]) + len(obs["ruka_states"]) for obs in self.observations
        )
        self.__max_action_dim = max(
            len(obs["commanded_arm_states"]) + len(obs["commanded_ruka_states"])
            for obs in self.observations
        )

        self._num_samples = len(self.observations)

        # action normalization using PKL min/max (expects keys min_arm/max_arm etc.)
        self.stats: Dict[str, Any] = {
            "actions": {
                "min": np.concatenate([data["min_arm"], data["min_ruka"]]),
                "max": np.concatenate([data["max_arm"], data["max_ruka"]]),
            }
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
                                / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5)
        }

    def _sample(self) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self._num_samples)
        obs = self.observations[idx]

        features = np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0).astype(np.float32)
        actions = np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0).astype(np.float32)
        actions = self.preprocess["actions"](actions)

        return {
            "features": features,
            "actions": actions,
            "task_emb": self.task_emb,  # already np.float32
        }

    def __iter__(self):
        while True:
            yield self._sample()

    @property
    def _max_episode_len(self) -> int:
        return self._num_samples

    @property
    def _max_state_dim(self) -> int:
        return self.__max_state_dim

    @property
    def _max_action_dim(self) -> int:
        return self.__max_action_dim

    @property
    def envs_till_idx(self) -> int:
        # single "environment" in this dataset
        return 1

    def sample_test(self, env_idx: int, step: int = None) -> Any:
        # return a prompt for evaluation; adapt shape if your agent expects something specific
        # Here we return just the task embedding (agent should handle dict/array)
        return self.task_emb

    def __len__(self):
        # helpful for debugging (not required by IterableDataset)
        return self._num_samples


# helper factory if you want hydra to call a function instead of instantiating the class
def create_dataset(cfg: Dict[str, Any] = None) -> CustomTeleopBCDataset:
    """
    cfg can be a config node with a 'pkl_file' field or the raw pkl path.
    Example dataloader config with Hydra:
      _target_: read_data.custom.create_dataset
      pkl_file: /path/to/data.pkl
    """
    if cfg is None:
        raise ValueError("cfg required (must include pkl_file)")
    pkl = cfg.get("pkl_file") if isinstance(cfg, dict) else getattr(cfg, "pkl_file")
    return CustomTeleopBCDataset(pkl)