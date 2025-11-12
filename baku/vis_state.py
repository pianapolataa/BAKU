#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # -----------------------------
    # 1. Load processed demo PKL
    # -----------------------------
    pkl_file = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl")
    with open(pkl_file, "rb") as f:
        demo_data = pickle.load(f)
    demo_obs = demo_data["observations"]
    print(f"Loaded {len(demo_obs)} demo steps.")

    # -----------------------------
    # 2. Select demo index
    # -----------------------------
    idx = 3  # which demo (observation) index to visualize
    if idx >= len(demo_obs):
        raise IndexError(f"Demo index {idx} is out of range (num demos: {len(demo_obs)})")

    obs_dict = demo_obs[idx]

    # -----------------------------
    # 3. Extract arm + ruka states
    # -----------------------------
    arm_states = np.array(obs_dict["arm_states"], dtype=np.float32)
    ruka_states = np.array(obs_dict["ruka_states"], dtype=np.float32)
    combined_states = np.concatenate([arm_states, ruka_states])

    print(f"Arm state dim: {arm_states.shape[0]}, Ruka state dim: {ruka_states.shape[0]}")
    print(f"Combined state shape: {combined_states.shape}")

    # -----------------------------
    # 4. Plot each state dimension
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(combined_states, marker="o", linewidth=1.5)
    plt.title(f"Combined Arm + Ruka States (Demo Index {idx})")
    plt.xlabel("State Dimension")
    plt.ylabel("Raw State Value")
    plt.grid(True)
    plt.tight_layout()

    save_path = f"/home_shared/grail_sissi/BAKU/demo_index{idx}_states.png"
    plt.savefig(save_path)
    print(f"Saved state plot to {save_path}")


if __name__ == "__main__":
    main()
