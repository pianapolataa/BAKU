import torch
import numpy as np
from train import WorkspaceIL  # your BC workspace
from suite.custom import task_make_fn
import pickle
from pathlib import Path

# -----------------------------
# 1. Load processed demo PKL
# -----------------------------
pkl_file = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl")
with open(pkl_file, "rb") as f:
    demo_data = pickle.load(f)
demo_obs = demo_data["observations"]
print(f"Loaded {len(demo_obs)} demo steps.")

# -----------------------------
# 2. Create environment
# -----------------------------
envs, _ = task_make_fn(demo_data)
env = envs[0]

# -----------------------------
# 3. Setup Workspace & Agent
# -----------------------------
workspace = WorkspaceIL(cfg)  # assume cfg is already defined or pass a minimal cfg
workspace.env = [env]

# Load trained BC snapshot
bc_snapshot = Path("/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.10_train/deterministic/131852/snapshot/500.pt")
workspace.load_snapshot({"bc": bc_snapshot})
workspace.agent.train(False)

# -----------------------------
# 4. Build normalization stats used during training
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
# 5. Rollout with exact training preprocessing
# -----------------------------
total_mse = 0.0

# helper to normalize features exactly as training
def normalize_features(obs, key="features"):
    return (obs - norm_stats[key]["min"]) / (norm_stats[key]["max"] - norm_stats[key]["min"] + 1e-8)

for step_idx, obs_dict in enumerate(demo_obs):
    # Concatenate arm + ruka features
    raw_features = np.concatenate([obs_dict["arm_states"], obs_dict["ruka_states"]]).astype(np.float32)
    agent_obs = {
        "features": normalize_features(raw_features),
        "pixels0": np.zeros((1, 3, 84, 84), dtype=np.uint8),  # dummy, unused if obs_type='features'
        "task_emb": np.asarray(demo_data["task_emb"], dtype=np.float32),
    }

    with torch.no_grad():
        agent_action = workspace.agent.act(
            agent_obs,
            prompt=None,
            norm_stats=norm_stats,
            step=step_idx,
            global_step=workspace.global_step,
            eval_mode=True,
        )

    # Convert to numpy if tensor
    if isinstance(agent_action, torch.Tensor):
        agent_action = agent_action.cpu().numpy()

    # Ground truth action
    demo_action = np.concatenate([obs_dict["commanded_arm_states"], obs_dict["commanded_ruka_states"]]).astype(np.float32)
    
    # Compute MSE
    mse = ((agent_action - demo_action) ** 2).mean()
    total_mse += mse

mean_mse = total_mse / len(demo_obs)
print(f"Demo rollout finished. Steps: {len(demo_obs)}, Mean raw action MSE: {mean_mse:.8f}")
