# suite/custom_suite.py
from pathlib import Path
from typing import List
import pickle
import torch
from torch.utils.data import Dataset


class PKLDataset(Dataset):
    """
    Dataset for loading pickled demonstrations for BC training.
    Compatible with WorkspaceIL and BCDataset loader.
    """

    def __init__(
        self,
        demo_paths: List[Path],
        obs_type: str = "pixels",
        history_len: int = 1,
    ):
        self.demo_paths = demo_paths
        self.obs_type = obs_type
        self.history_len = history_len
        self.episodes = []

        for path in demo_paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
                observations = data["observations"] if obs_type == "pixels" else data["states"]
                actions = data["actions"]
                task_emb = data.get("task_emb", None)
                for i in range(len(observations)):
                    self.episodes.append({
                        "observation": observations[i],
                        "action": actions[i],
                        "task_emb": task_emb
                    })

        self.num_samples = len(self.episodes)

        # Infer specs
        sample_obs = self.episodes[0]["observation"]
        sample_action = self.episodes[0]["action"]
        if isinstance(sample_obs, dict):  # pixel observations
            self.obs_spec = {
                "pixels": sample_obs["pixels"].shape,
                "pixels_egocentric": sample_obs.get("pixels_egocentric", sample_obs["pixels"]).shape,
                "features": (100,),  # dummy features
                "proprioceptive": (sample_obs.get("proprioceptive", sample_obs["pixels"].flatten()).shape[0],),
            }
        else:  # features only
            self.obs_spec = {"features": sample_obs.shape}

        self.action_spec = sample_action.shape
        self._max_episode_len = self.num_samples
        self._max_state_dim = self.obs_spec.get("features", (100,))[0]
        self._max_action_dim = self.action_spec[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        obs = ep["observation"]
        act = ep["action"]
        task_emb = ep["task_emb"]

        if self.obs_type == "pixels" and isinstance(obs, dict):
            pixels = obs["pixels"][-self.history_len:]
            pixels_tensor = torch.stack([torch.tensor(p, dtype=torch.float32) for p in pixels])
            return {
                "pixels": pixels_tensor,
                "actions": torch.tensor(act, dtype=torch.float32),
                "task_emb": task_emb
            }
        elif self.obs_type == "features":
            obs_array = obs[-self.history_len:] if isinstance(obs, (list, tuple)) else [obs]
            return {
                "features": torch.tensor(obs_array, dtype=torch.float32),
                "actions": torch.tensor(act, dtype=torch.float32),
                "task_emb": task_emb
            }
        else:
            # fallback for raw pixels
            return {
                "pixels": torch.tensor(obs, dtype=torch.float32),
                "actions": torch.tensor(act, dtype=torch.float32),
                "task_emb": task_emb
            }
