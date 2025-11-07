# custom_suite.py

import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class PKLDataset(Dataset):
    """
    Iterable dataset for BAKU, works with multiple demo PKLs.
    Each PKL should be preprocessed like your previous script:
    {
        "observations": [...],
        "timestamps": [...],
        "max_cartesian": ...,
        "min_cartesian": ...,
        "max_gripper": ...,
        "min_gripper": ...,
        "task_emb": ...
    }
    """

    def __init__(self, demo_folder: str):
        self.demo_folder = Path(demo_folder)
        self.pkl_files = sorted(self.demo_folder.glob("*.pkl"))
        self.frames = []
        self.load_all()

    def load_all(self):
        """Load all PKLs and flatten frames into a single list."""
        self.frames = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            for obs in data["observations"]:
                frame = {
                    "cartesian_states": np.array(obs["cartesian_states"], dtype=np.float32),
                    "commanded_cartesian_states": np.array(obs["commanded_cartesian_states"], dtype=np.float32),
                    "gripper_states": np.array(obs["gripper_states"], dtype=np.float32),
                    "commanded_gripper_states": np.array(obs["commanded_gripper_states"], dtype=np.float32),
                    "task_emb": np.array(data["task_emb"], dtype=np.float32),
                    "timestamp": obs["timestamp"],
                }
                # Combine commanded_arm + commanded_gripper into a single action vector
                frame["action"] = np.concatenate([
                    frame["commanded_cartesian_states"],
                    frame["commanded_gripper_states"]
                ]).astype(np.float32)

                self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


def make(demo_folder, num_frames=1, eval=False):
    """
    Minimal suite function for BAKU.
    Returns:
        envs: list of env-like objects
        task_descriptions: list of task embeddings
    """
    dataset = PKLDataset(demo_folder)

    class DummyEnv:
        def __init__(self, dataset):
            self.dataset = dataset
            self.idx = 0
            self._obs_spec = {
                "features": np.zeros_like(dataset[0]["cartesian_states"]),
                "proprioceptive": np.zeros_like(dataset[0]["gripper_states"]),
                "pixels": np.zeros((3, 128, 128), dtype=np.uint8),
                "task_emb": np.zeros_like(dataset[0]["task_emb"]),
            }
            self._action_spec = np.zeros_like(dataset[0]["action"])

        def reset(self):
            self.idx = 0
            return self.dataset[self.idx]

        def step(self, action):
            self.idx += 1
            done = self.idx >= len(self.dataset)
            obs = self.dataset[self.idx - 1] if not done else self.dataset[-1]
            reward = 0.0
            return obs, reward, done, {}

        def observation_spec(self):
            return self._obs_spec

        def action_spec(self):
            return self._action_spec

    envs = [DummyEnv(dataset)]
    task_descriptions = [dataset[0]["task_emb"]]
    return envs, task_descriptions
