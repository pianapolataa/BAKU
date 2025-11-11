#!/usr/bin/env python3
import pickle
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from train import WorkspaceIL  # your existing import
from suite.custom import task_make_fn
import matplotlib.pyplot as plt

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg: DictConfig):
    # -----------------------------
    # 1. Load processed demo PKL
    # -----------------------------
    pkl_file = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl")
    with open(pkl_file, "rb") as f:
        demo_data = pickle.load(f)
    demo_obs = demo_data["observations"]
    print(f"Loaded {len(demo_obs)} demo steps.")

    # -----------------------------
    # 2. Create environment via suite
    # -----------------------------
    envs, _ = task_make_fn(demo_data)
    env = envs[0]

    # -----------------------------
    # 3. Setup Workspace & Agent
    # -----------------------------
    workspace = WorkspaceIL(cfg)
    workspace.env = [env]

    # Load trained BC snapshot
    bc_snapshot_path = Path(
        "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.11_train/deterministic/001855/snapshot/9000.pt"
    )
    workspace.load_snapshot({"bc": bc_snapshot_path})
    workspace.agent.train(False)  # set agent to eval mode

    # -----------------------------
    # 4. Build normalization stats
    # -----------------------------
    norm_stats = {
        "features": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]])
        },
        "actions": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]])
        }
    }

    # -----------------------------
    # 5. Rollout and compare raw actions
    # -----------------------------
    total_mse = 0.0
    max_abs_diff = np.zeros(23, dtype=np.float32)

    # Collect differences for action index 7
    diff_idx7 = []

    for step_idx, obs_dict in enumerate(demo_obs):
        agent_obs = {
            "features": np.concatenate([obs_dict["arm_states"], obs_dict["ruka_states"]]).astype(np.float32),
            "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
            "task_emb": np.asarray(demo_data["task_emb"], dtype=np.float32),
        }

        with torch.no_grad():
            agent_action_raw = workspace.agent.act(
                agent_obs,
                prompt=None,
                norm_stats=norm_stats,
                step=step_idx,
                global_step=workspace.global_step,
                eval_mode=True,
            )

        if isinstance(agent_action_raw, torch.Tensor):
            agent_action_raw = agent_action_raw.cpu().numpy()

        demo_action_raw = np.concatenate([
            obs_dict["commanded_arm_states"],
            obs_dict["commanded_ruka_states"]
        ]).astype(np.float32)

        if isinstance(demo_action_raw, torch.Tensor):
            demo_action_raw = demo_action_raw.cpu().numpy()

        # compute absolute difference
        abs_diff = np.abs(agent_action_raw - demo_action_raw)
        max_abs_diff = np.maximum(max_abs_diff, abs_diff)

        diff_idx7.append(abs_diff[9])  # collect index 7 difference

        mse = (abs_diff ** 2).mean()
        total_mse += mse

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")
    print(f"Max absolute difference per action index: {max_abs_diff}")

    # -----------------------------
    # 6. Plot histogram for action index 7
    # -----------------------------
    diff_idx7 = np.array(diff_idx7)
    plt.figure(figsize=(7,4))
    plt.hist(diff_idx7, bins=50, color="skyblue", edgecolor="black")
    plt.title("Absolute Differences for Action Index 7")
    plt.xlabel("Absolute difference")
    plt.ylabel("Frequency")

    # Add mean, median, max lines
    mean_val = diff_idx7.mean()
    median_val = np.median(diff_idx7)
    max_val = diff_idx7.max()
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}")
    plt.axvline(median_val, color="green", linestyle="--", label=f"Median: {median_val:.4f}")
    plt.axvline(max_val, color="purple", linestyle="--", label=f"Max: {max_val:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home_shared/grail_sissi/BAKU/diff_index7_hist.png")
    print("Saved histogram to diff_index7_hist.png")

if __name__ == "__main__":
    main()
