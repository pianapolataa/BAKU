#!/usr/bin/env python3
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # -----------------------------
    # 1. Load processed demo PKL
    # -----------------------------
    pkl_file = Path("/home_shared/grail_sissi/BAKU/test/demo_task.pkl")
    with open(pkl_file, "rb") as f:
        demo_data = pickle.load(f)
    demo_obs = demo_data["observations"]
    print(f"Loaded {len(demo_obs)} demo steps.")

    # -----------------------------
    # 2. Extract state matrix (steps × 23)
    # -----------------------------
    state_list = []
    for obs_dict in demo_obs:
        combined_state = np.concatenate(
            [obs_dict["arm_states"], obs_dict["ruka_states"]]
        ).astype(np.float32)
        state_list.append(combined_state)

    states = np.stack(state_list, axis=0)  # shape: (T, 23)
    T, D = states.shape
    print(f"States shape = {states.shape} (should be T x 23)")

    # -----------------------------
    # 3. Plot 23 state dims in 5×5 grid
    # -----------------------------
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()

    time = np.arange(T)

    for i in range(25):
        ax = axes[i]
        if i < D:
            ax.plot(time, states[:, i])
            ax.set_title(f"State[{i}]")
        else:
            ax.axis("off")  # unused subplot

    plt.tight_layout()
    save_path = "/home_shared/grail_sissi/BAKU/state_grid.png"
    plt.savefig(save_path)
    print(f"Saved 5x5 state grid plot to {save_path}")

if __name__ == "__main__":
    main()
