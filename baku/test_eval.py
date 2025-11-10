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
    pkl_file = Path("proc_data/default_scene/demo_task/demo_0.pkl")  # e.g., processed_data_pkl/demo_task.pkl
    with open(pkl_file, "rb") as f:
        demo_data = pickle.load(f)
    demo_obs = demo_data["observations"]
    print(f"Loaded {len(demo_obs)} demo steps.")

    # -----------------------------
    # 2. Create environment via suite
    # -----------------------------
    from suite import task_make_fn
    envs, _ = task_make_fn(demo_data)
    env = envs[0]

    # -----------------------------
    # 3. Setup Workspace & Agent
    # -----------------------------
    workspace = WorkspaceIL(cfg)
    workspace.env = [env]  # replace with single env

    # Load BC weights
    bc_snapshot = Path("exp_local/2025.11.10_train/deterministic/131852/snapshot/500.pt")
    workspace.load_snapshot({"bc": bc_snapshot})
        

    workspace.agent.train(False)  # eval mode

    # -----------------------------
    # 4. Rollout and compare actions
    # -----------------------------
    total_mse = 0.0
    for step_idx, obs_dict in enumerate(demo_obs):
        # Build agent observation
        agent_obs = {
            "features": torch.tensor(
                np.concatenate([obs_dict["arm_states"], obs_dict["ruka_states"]])[None, :],
                dtype=torch.float32,
                device=workspace.device
            ),
            "pixels0": torch.zeros((1, 3, 84, 84), dtype=torch.float32, device=workspace.device),
            "task_emb": torch.tensor(demo_data["task_emb"][None, :], dtype=torch.float32, device=workspace.device),
        }

        with torch.no_grad():
            action = workspace.agent.act(
                agent_obs,
                prompt=None,
                stats=workspace.stats,
                step=step_idx,
                global_step=workspace.global_step,
                eval_mode=True,
            )

        # Get corresponding demo action (normalized)
        demo_action = np.concatenate([obs_dict["commanded_arm_states"], obs_dict["commanded_ruka_states"]])
        demo_action = workspace.expert_replay_loader.dataset.preprocess["actions"](demo_action)

        # Compute MSE
        mse = ((action.cpu().numpy().ravel() - demo_action) ** 2).mean()
        total_mse += mse

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean action MSE: {mean_mse:.8f}")

if __name__ == "__main__":
    main()
