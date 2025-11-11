#!/usr/bin/env python3
import pickle
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from train import WorkspaceIL
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
        "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.11_train/deterministic/010941/snapshot/80000.pt"
    )
    workspace.load_snapshot({"bc": bc_snapshot_path})
    workspace.agent.train(False)

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
    # 5. Rollout and record raw differences (not abs)
    # -----------------------------
    total_mse = 0.0
    diff_idx7 = []  # signed differences

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

        # compute signed difference
        diff = agent_action_raw - demo_action_raw
        diff_idx7.append(diff[11])  # index 7 only

        mse = (diff ** 2).mean()
        total_mse += mse

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")

    # -----------------------------
    # 6. Plot difference over time
    # -----------------------------
    diff_idx7 = np.array(diff_idx7)
    steps = np.arange(len(diff_idx7))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, diff_idx7, label="Action index 7 difference", color="dodgerblue")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Action Index 7: Agent - Demo Difference Over Time")
    plt.xlabel("Step")
    plt.ylabel("Difference (Agent - Demo)")
    plt.legend()
    plt.tight_layout()

    save_path = "/home_shared/grail_sissi/BAKU/diff_index7_timeline.png"
    plt.savefig(save_path)
    print(f"Saved time-series difference plot to {save_path}")

if __name__ == "__main__":
    main()
