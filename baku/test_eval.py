# import pickle
# import torch
# import numpy as np
# from pathlib import Path
# import hydra
# from omegaconf import DictConfig
# from train import WorkspaceIL
# from suite.custom import task_make_fn
# import matplotlib.pyplot as plt


# @hydra.main(config_path="cfgs", config_name="config")
# def main(cfg: DictConfig):
#     # -----------------------------
#     # 1. Load processed demo PKL
#     # -----------------------------
#     pkl_file = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl")
#     with open(pkl_file, "rb") as f:
#         demo_data = pickle.load(f)
#     demo_obs = demo_data["observations"]
#     print(f"Loaded {len(demo_obs)} demo steps.")

#     # -----------------------------
#     # 2. Create environment via suite
#     # -----------------------------
#     envs, _ = task_make_fn(demo_data)
#     env = envs[0]

#     # -----------------------------
#     # 3. Setup Workspace & Agent
#     # -----------------------------
#     workspace = WorkspaceIL(cfg)
#     workspace.env = [env]

#     # Load trained BC snapshot
#     bc_snapshot_path = Path(
#         "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.12_train/deterministic/172024/snapshot/4000.pt"
#     )
#     workspace.load_snapshot({"bc": bc_snapshot_path})
#     workspace.agent.train(False)

#     # -----------------------------
#     # 4. Build normalization stats
#     # -----------------------------
#     norm_stats = {
#         "features": {
#             "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
#             "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]]),
#         },
#         "actions": {
#             "min": np.concatenate([demo_data["min_arm_commanded"], demo_data["min_ruka_commanded"]]),
#             "max": np.concatenate([demo_data["max_arm_commanded"], demo_data["max_ruka_commanded"]]),
#         },
#     }

#     # -----------------------------
#     # 5. Rollout and collect all action dims
#     # -----------------------------
#     num_actions = 23
#     agent_actions = []
#     demo_actions = []
#     total_mse = 0.0

#     for step_idx, obs_dict in enumerate(demo_obs):
#         agent_obs = {
#             "features": np.concatenate([obs_dict["arm_states"], obs_dict["ruka_states"]]).astype(np.float32),
#             "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
#             "task_emb": np.asarray(demo_data["task_emb"], dtype=np.float32),
#         }

#         with torch.no_grad():
#             agent_action_raw = workspace.agent.act(
#                 agent_obs,
#                 prompt=None,
#                 norm_stats=norm_stats,
#                 step=step_idx,
#                 global_step=workspace.global_step,
#                 eval_mode=True,
#             )

#         if isinstance(agent_action_raw, torch.Tensor):
#             agent_action_raw = agent_action_raw.cpu().numpy()

#         demo_action_raw = np.concatenate(
#             [obs_dict["commanded_arm_states"], obs_dict["commanded_ruka_states"]]
#         ).astype(np.float32)

#         agent_actions.append(agent_action_raw)
#         demo_actions.append(demo_action_raw)

#         diff = agent_action_raw - demo_action_raw
#         total_mse += (diff**2).mean()

#     mean_mse = total_mse / len(demo_obs)
#     print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")

#     agent_actions = np.array(agent_actions)  # [T, 23]
#     demo_actions = np.array(demo_actions)    # [T, 23]
#     steps = np.arange(len(demo_obs))

#     # -----------------------------
#     # 6. Plot all 23 dimensions
#     # -----------------------------
#     rows, cols = 5, 5
#     fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
#     axes = axes.flatten()

#     for i in range(rows * cols):
#         ax = axes[i]
#         if i < num_actions:
#             ax.plot(steps, demo_actions[:, i], label="Demo", color="black", linewidth=1)
#             ax.plot(steps, agent_actions[:, i], label="Pred", color="dodgerblue", linestyle="--")
#             ax.set_title(f"Action {i}", fontsize=10)
#             ax.tick_params(axis="both", labelsize=8)
#         else:
#             ax.axis("off")

#     # Add a single legend outside subplots
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

#     fig.suptitle("Predicted vs Demo Raw Action Values", fontsize=16)
#     fig.tight_layout(rect=[0, 0, 1, 0.97])

#     save_path = "/home_shared/grail_sissi/BAKU/all_action_dims_raw_vs_demo.png"
#     plt.savefig(save_path, dpi=200)
#     print(f"Saved combined raw vs demo plot to {save_path}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from train import WorkspaceIL
from suite.custom import task_make_fn


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
        "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.12_train/deterministic/173751/snapshot/2000.pt"
    )
    workspace.load_snapshot({"bc": bc_snapshot_path})
    workspace.agent.train(False)
    print(f"Loaded policy snapshot from {bc_snapshot_path}")

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
    # 5. Run model inference and store predicted actions
    # -----------------------------
    predicted_arm_states = []
    predicted_hand_states = []
    timestamps = []
    observations = []

    for step_idx, obs_dict in enumerate(tqdm(demo_obs, desc="Generating predictions")):
        agent_obs = {
            "features": np.concatenate([obs_dict["arm_states"], obs_dict["ruka_states"]]).astype(np.float32),
            "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
            "task_emb": np.asarray(demo_data["task_emb"], dtype=np.float32),
        }
        # print(obs_dict["arm_states"])
        # print(obs_dict["ruka_states"])

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

        # Split predicted full vector (arm + hand)
        pred_arm = agent_action_raw[:7]
        pred_hand = agent_action_raw[7:]

        print(pred_arm)
        print(pred_hand)

        predicted_arm_states.append(pred_arm)
        predicted_hand_states.append(pred_hand)

        # Construct observation entry (similar to preprocessed data)
        obs = {
            "pixels0": obs_dict["pixels0"],  # dummy image
            "timestamp": obs_dict["timestamp"],
            "arm_states": obs_dict["arm_states"],  # keep original current state
            "commanded_arm_states": pred_arm.astype(np.float32),
            "ruka_states": obs_dict["ruka_states"],
            "commanded_ruka_states": pred_hand.astype(np.float32),
        }
        observations.append(obs)
        timestamps.append(obs_dict["timestamp"])

    # -----------------------------
    # 6. Compute new min/max bounds for predicted commands
    # -----------------------------
    predicted_arm_stack = np.stack(predicted_arm_states, axis=0)
    predicted_hand_stack = np.stack(predicted_hand_states, axis=0)

    max_arm_command = np.max(predicted_arm_stack, axis=0)
    min_arm_command = np.min(predicted_arm_stack, axis=0)
    max_ruka_command = np.max(predicted_hand_stack, axis=0)
    min_ruka_command = np.min(predicted_hand_stack, axis=0)

    # -----------------------------
    # 7. Save to PKL
    # -----------------------------
    save_data = {
        "observations": observations,
        "timestamps": np.array(timestamps, dtype=np.float64),
        "max_arm": demo_data["max_arm"],
        "min_arm": demo_data["min_arm"],
        "max_ruka": demo_data["max_ruka"],
        "min_ruka": demo_data["min_ruka"],
        "max_arm_commanded": max_arm_command,
        "min_arm_commanded": min_arm_command,
        "max_ruka_commanded": max_ruka_command,
        "min_ruka_commanded": min_ruka_command,
        "task_emb": demo_data["task_emb"],
    }

    save_path = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl/predicted_task.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n✅ Saved predicted data to {save_path}")
    print(f"Total predictions: {len(predicted_arm_states)} frames")
    print(f"Arm command range: {min_arm_command} to {max_arm_command}")
    print(f"Hand command range: {min_ruka_command} to {max_ruka_command}")


if __name__ == "__main__":
    main()
