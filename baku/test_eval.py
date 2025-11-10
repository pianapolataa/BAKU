#!/usr/bin/env python3
import pickle
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from train import WorkspaceIL  # assumes WorkspaceIL is importable

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
    from suite.custom import task_make_fn
    envs, _ = task_make_fn(demo_data)
    env = envs[0]

    # -----------------------------
    # 3. Setup Workspace & Agent
    # -----------------------------
    workspace = WorkspaceIL(cfg)
    workspace.env = [env]

    # Load BC weights
    bc_snapshot = Path("/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.10_train/deterministic/131852/snapshot/500.pt")
    workspace.load_snapshot({"bc": bc_snapshot})

    workspace.agent.train(False)  # eval mode

    # Build normalization stats
    norm_stats = {
        "features": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]])
        },
        "actions": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),  # assuming actions match states
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]])
        }
    }

    # -----------------------------
    # 4. Rollout and compare raw actions
    # -----------------------------
    total_mse = 0.0
    for step_idx, obs_dict in enumerate(demo_obs):
        # Build agent observation as numpy arrays (match training / env outputs)
        agent_obs = {
            "features": np.concatenate([obs_dict["arm_states"], obs_dict["ruka_states"]]).astype(np.float32),
            "pixels0": np.zeros((84, 84, 3), dtype=np.float32),
            "task_emb": np.asarray(demo_data["task_emb"], dtype=np.float32),
        }

        with torch.no_grad():
            # Get agent action (agent may accept numpy obs)
            agent_action_raw = workspace.agent.act(
                agent_obs,
                prompt=None,
                norm_stats=norm_stats,
                step=step_idx,
                global_step=workspace.global_step,
                eval_mode=True,
            )

        # convert agent output to numpy if it's a tensor
        if isinstance(agent_action_raw, torch.Tensor):
            agent_action_raw = agent_action_raw.cpu().numpy()

        # Build raw demo action (concatenate arm + ruka)
        demo_action_raw = np.concatenate([
            obs_dict["commanded_arm_states"],
            obs_dict["commanded_ruka_states"]
        ]).astype(np.float32)

        # Compute MSE on raw actions
        mse = ((agent_action_raw - demo_action_raw) ** 2).mean()
        total_mse += mse

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")

if __name__ == "__main__":
    main()
