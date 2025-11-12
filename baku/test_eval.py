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
        "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.12_train/deterministic/120228/snapshot/59000.pt"
    )
    workspace.load_snapshot({"bc": bc_snapshot_path})
    workspace.agent.train(False)

    # -----------------------------
    # 4. Build normalization stats
    # -----------------------------
    norm_stats = {
        "features": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]]),
        },
        "actions": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]]),
        },
    }

    # -----------------------------
    # 5. Rollout and collect raw values
    # -----------------------------
    idx = 1  # the action index to visualize
    agent_values, demo_values = [], []
    total_mse = 0.0

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

        demo_action_raw = np.concatenate(
            [obs_dict["commanded_arm_states"], obs_dict["commanded_ruka_states"]]
        ).astype(np.float32)

        # record raw values for the chosen action index
        agent_values.append(agent_action_raw[idx])
        demo_values.append(demo_action_raw[idx])

        # accumulate MSE for sanity check
        diff = agent_action_raw - demo_action_raw
        total_mse += (diff**2).mean()

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")

    # -----------------------------
    # 6. Plot raw values over time
    # -----------------------------
    steps = np.arange(len(agent_values))

    plt.figure(figsize=(9, 4))
    plt.plot(steps, demo_values, label="Demo", color="black", linewidth=1.5)
    plt.plot(steps, agent_values, label="Predicted", color="dodgerblue", linestyle="--")
    plt.title(f"Action Index {idx}: Raw Commanded Values Over Time")
    plt.xlabel("Step")
    plt.ylabel("Raw Action Value")
    plt.legend()
    plt.tight_layout()

    save_path = f"/home_shared/grail_sissi/BAKU/action_index{idx}_raw_vs_demo.png"
    plt.savefig(save_path)
    print(f"Saved raw vs demo plot to {save_path}")


if __name__ == "__main__":
    main()
