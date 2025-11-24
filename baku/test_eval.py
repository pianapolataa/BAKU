import pickle
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from train import WorkspaceIL
from suite.custom import task_make_fn
import matplotlib.pyplot as plt


def align_quaternion_signs(quat, ref_quat):
    """Ensure quaternion has same hemisphere as reference."""
    if np.dot(quat, ref_quat) < 0:
        quat = -quat
    return quat


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
    # "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.19_train/deterministic/193903/snapshot/61000.pt" # 3 demo policy
        "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.24_train/deterministic/100720/snapshot/31000.pt" # new 11 demo w noise
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
            "min": np.concatenate([demo_data["min_arm_commanded"], demo_data["min_ruka_commanded"]]),
            "max": np.concatenate([demo_data["max_arm_commanded"], demo_data["max_ruka_commanded"]]),
        },
    }

    # -----------------------------
    # 5. Prepare quaternion reference
    # -----------------------------
    ref_quat = demo_obs[0]["arm_states"][3:7].copy()
    ref_quat_cmd = demo_obs[0]["commanded_arm_states"][3:7].copy()

    # -----------------------------
    # 6. Rollout and collect all action dims
    # -----------------------------
    num_actions = 23
    agent_actions = []
    demo_actions = []
    total_mse = 0.0

    for step_idx, obs_dict in enumerate(demo_obs):
        arm_state = obs_dict["arm_states"].copy()
        cmd_state = obs_dict["commanded_arm_states"].copy()

        # --- Quaternion sign fix ---
        arm_state[3:7] = align_quaternion_signs(arm_state[3:7], ref_quat)
        cmd_state[3:7] = align_quaternion_signs(cmd_state[3:7], ref_quat_cmd)

        # --- Build obs and actions ---
        agent_obs = {
            "features": np.concatenate([arm_state, obs_dict["ruka_states"]]).astype(np.float32),
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
            [cmd_state, obs_dict["commanded_ruka_states"]]
        ).astype(np.float32)

        # --- Store + compare ---
        agent_actions.append(agent_action_raw)
        demo_actions.append(demo_action_raw)

        diff = agent_action_raw - demo_action_raw
        total_mse += (diff**2).mean()

    mean_mse = total_mse / len(demo_obs)
    print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")

    agent_actions = np.array(agent_actions)  # [T, 23]
    demo_actions = np.array(demo_actions)    # [T, 23]
    steps = np.arange(len(demo_obs))

    # -----------------------------
    # 7. Plot all 23 dimensions
    # -----------------------------
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if i < num_actions:
            ax.plot(steps, demo_actions[:, i], label="Demo", color="black", linewidth=1)
            ax.plot(steps, agent_actions[:, i], label="Pred", color="dodgerblue", linestyle="--")
            ax.set_title(f"Action {i}", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
        else:
            ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
    fig.suptitle("Predicted vs Demo Raw Action Values", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = "/home_shared/grail_sissi/BAKU/all_action_dims_raw_vs_demo.png"
    plt.savefig(save_path, dpi=200)
    print(f"Saved combined raw vs demo plot to {save_path}")


if __name__ == "__main__":
    main()