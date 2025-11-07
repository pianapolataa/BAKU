# read_data/custom.py
import pickle
from pathlib import Path
import random
import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as T
import numpy as np


class BCDataset(IterableDataset):
    """
    Iterable dataset for BC training on pickled demos.
    Compatible with WorkspaceIL.
    """

    def __init__(
        self,
        path: str,
        suite: str,
        scenes: list,
        tasks: list,
        num_demos_per_task: int = 100,
        obs_type: str = "pixels",
        history: bool = False,
        history_len: int = 1,
        prompt: str = "text",
        temporal_agg: bool = False,
        num_queries: int = 10,
        img_size: int = 128,
        store_actions: bool = False,
    ):
        self.obs_type = obs_type
        self.history_len = history_len if history else 1
        self.prompt = prompt
        self.temporal_agg = temporal_agg
        self.num_queries = num_queries
        self.img_size = img_size
        self.store_actions = store_actions

        # Simple augmentation: convert to tensor
        self.aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        # Load demo paths
        self.paths = []
        base_path = Path(path) / suite
        for scene in scenes:
            scene_path = base_path / scene
            if scene_path.exists():
                self.paths.extend(list(scene_path.glob("*.pkl")))

        # Load episodes
        self.episodes = []
        for p in self.paths:
            with open(p, "rb") as f:
                data = pickle.load(f)
                obs = data["observations"] if obs_type == "pixels" else data["states"]
                actions = data["actions"]
                task_emb = data.get("task_emb", None)
                for i in range(min(num_demos_per_task, len(obs))):
                    self.episodes.append({
                        "observation": obs[i],
                        "action": actions[i],
                        "task_emb": task_emb
                    })

        self.num_samples = len(self.episodes)
        if self.store_actions:
            self.all_actions = [ep["action"] for ep in self.episodes]

        # Infer specs
        sample_obs = self.episodes[0]["observation"]
        sample_action = self.episodes[0]["action"]
        if isinstance(sample_obs, dict):  # pixel observations
            self.obs_spec = {
                "pixels": sample_obs["pixels"].shape,
                "pixels_egocentric": sample_obs.get("pixels_egocentric", sample_obs["pixels"]).shape,
                "features": (100,),
                "proprioceptive": (sample_obs.get("proprioceptive", sample_obs["pixels"].flatten()).shape[0],),
            }
        else:
            self.obs_spec = {"features": sample_obs.shape}

        self.action_spec = sample_action.shape
        self._max_episode_len = self.num_samples
        self._max_state_dim = self.obs_spec.get("features", (100,))[0]
        self._max_action_dim = self.action_spec[0]

    def _sample(self):
        ep = random.choice(self.episodes)
        obs = ep["observation"]
        act = ep["action"]
        task_emb = ep["task_emb"]

        if self.obs_type == "pixels" and isinstance(obs, dict):
            pixels = obs["pixels"][-self.history_len:]
            pixels_tensor = torch.stack([self.aug(p) for p in pixels])
            return {
                "pixels": pixels_tensor,
                "actions": torch.tensor(act, dtype=torch.float32),
                "task_emb": task_emb
            }
        elif self.obs_type == "features":
            obs_array = np.array(obs[-self.history_len:] if isinstance(obs, (list, tuple)) else [obs])
            return {
                "features": torch.tensor(obs_array, dtype=torch.float32),
                "actions": torch.tensor(act, dtype=torch.float32),
                "task_emb": task_emb
            }
        else:
            return {
                "pixels": torch.tensor(obs, dtype=torch.float32),
                "actions": torch.tensor(act, dtype=torch.float32),
                "task_emb": task_emb
            }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self.num_samples
