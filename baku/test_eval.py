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

    # Load BC snapshot
    bc_snapshot_path = Path(
        "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.10_train/deterministic/131852/snapshot/500.pt"
    )
    bc_state = torch.load(bc_snapshot_path, map_location=workspace.agent.device)
    workspace.agent.load_snapshot(bc_state, eval=True)

    workspace.agent.train(False)  # ensure eval mode

    # -----------------------------
    # 4. Build normalization stats
    # -----------------------------
    # keys must match agent's proprio key
    norm_stats = {
        workspace.agent.proprio_key: {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]]),
        },
        "actions": {
            "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
            "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]]),
        },
    }

    # -----------------------------
    # 5. Rollout and compute raw action MSE
    # -----------------------------
    total_mse = 0.0

    for step_idx, obs_dict in enumerate(demo_obs):
        # --- Build agent observation ---
        agent_obs = {}
        if workspace.agent.obs_type == "features":
            agent_obs[workspace.agent.feature_key] = np.concatenate([
                obs_dict["arm_states"],
                obs_dict["ruka_states"]
            ]).astype(np.float32)
        else:
            # pixels case
            for key in workspace.agent.pixel_keys:
                agent_obs[key] = np.zeros((3, 84, 84), dtype=np.uint8)
            if workspace.agent.use_proprio:
                agent_obs[workspace.agent.proprio_key] = np.concatenate([
                    obs_dict["arm_states"],
                    obs_dict["ruka_states"]
                ]).astype(np.float32)

        # --- Build prompt ---
        prompt = {"task_emb": np.asarray(demo_data["task_emb"], dtype=np.float32)}

        # --- Get agent action ---
        with torch.no_grad():
            agent_action_raw = workspace.agent.act(
                agent_obs,
                prompt=prompt,
                norm_stats=norm_stats,
                step=step_idx,
                global_step=workspace.global_step,
                eval_mode=True,
            )

        agent_action_raw = np.array(agent_action_raw).ravel()

        # --- Get demo action ---
        demo_action_raw = np.concatenate([
            obs_dict["commanded_arm_states"],
            obs_dict["commanded_ruka_states"]
        ]).astype(np.float32).ravel()

        # --- Compute MSE ---
        mse = ((agent_action_raw - demo_action_raw) ** 2).mean()
        total_mse += mse

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")


if __name__ == "__main__":
    main()
