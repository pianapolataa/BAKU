#!/usr/bin/env python3
import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_ROOT = Path("/home_shared/grail_sissi/BAKU/baku/vr-hand-tracking/Franka-Teach/data")
IMG_SIZE = (128, 128)
SAVE_PATH = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl")
# SAVE_PATH = Path("/home_shared/grail_sissi/BAKU/test")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
TASK_NAME = "demo_task"

def flip_to_reference(quat, ref):
    """Flip quaternion sign to match reference hemisphere."""
    return quat if np.dot(quat, ref) >= 0 else -quat


def extract_quat(x):
    """Assumes 7-DoF arm: pos[0:3], quat[3:7]."""
    return x[3:7]


# ----------------------------
# GATHER ALL DEMO FOLDERS
# ----------------------------
demo_dirs = sorted(
    [p for p in DATA_ROOT.iterdir() if p.is_dir() and "demonstration_9" in p.name]
)
print(f"Found {len(demo_dirs)} demos")

if len(demo_dirs) == 0:
    raise RuntimeError("No demonstration folders found!")

# ----------------------------
# GLOBAL STORAGE
# ----------------------------
all_observations = []
all_timestamps = []

reference_quat = None  # from first frame of first demo
reference_cmd_quat = None

fixed_counts = {"global_flips": 0, "continuity_flips": 0}

# ----------------------------
# PROCESS EACH DEMO
# ----------------------------
for d_idx, DEMO in enumerate(demo_dirs):
    print(f"\n=== Processing {DEMO.name} ===")

    # Load pkl data
    with open(DEMO / "states.pkl", "rb") as f:
        arm_states = pickle.load(f)
    with open(DEMO / "commanded_states.pkl", "rb") as f:
        arm_commanded_states = pickle.load(f)
    with open(DEMO / "ruka_states.pkl", "rb") as f:
        hand_states = pickle.load(f)
    with open(DEMO / "ruka_commanded_states.pkl", "rb") as f:
        hand_commanded_states = pickle.load(f)

    print(f"Loaded {len(arm_states)} arm frames, {len(hand_states)} hand frames")

    # Extract timestamps
    arm_times = np.array([s["timestamp"] for s in arm_states])
    hand_times = np.array([s["timestamp"] for s in hand_states])

    # Loop through hand frames
    for i, t in enumerate(tqdm(hand_times, desc=f"Sync {DEMO.name}")):
        obs = {}
        obs["pixels0"] = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        obs["timestamp"] = float(t)

        # find nearest arm frame
        arm_idx = np.argmin(np.abs(arm_times - t))
        if abs(arm_times[arm_idx] - t) > 0.05:
            continue

        # raw arm state
        arm_state = np.concatenate(
            [arm_states[arm_idx]["state"].pos, arm_states[arm_idx]["state"].quat]
        ).astype(np.float32)

        cmd_state = np.concatenate(
            [
                arm_commanded_states[arm_idx]["state"].pos,
                arm_commanded_states[arm_idx]["state"].quat,
            ]
        ).astype(np.float32)

        # Set reference quaternions using very first frame of first demo
        if reference_quat is None:
            reference_quat = extract_quat(arm_state).copy()
            reference_cmd_quat = extract_quat(cmd_state).copy()
            print("Set global reference quaternions")

        # Apply sign flip to match global reference hemisphere
        arm_quat = extract_quat(arm_state).copy()
        cmd_quat = extract_quat(cmd_state).copy()

        # global hemisphere alignment
        arm_quat_aligned = flip_to_reference(arm_quat, reference_quat)
        if np.dot(arm_quat_aligned, arm_quat) < 0:
            fixed_counts["global_flips"] += 1
        arm_quat = arm_quat_aligned

        cmd_quat_aligned = flip_to_reference(cmd_quat, reference_cmd_quat)
        if np.dot(cmd_quat_aligned, cmd_quat) < 0:
            fixed_counts["global_flips"] += 1
        cmd_quat = cmd_quat_aligned

        # --- Continuity with previously appended frame (global continuity)
        # If we already have frames in all_observations, ensure this frame is continuous
        if len(all_observations) > 0:
            prev_arm_quat = all_observations[-1]["arm_states"][3:7]
            prev_cmd_quat = all_observations[-1]["commanded_arm_states"][3:7]

            if np.dot(prev_arm_quat, arm_quat) < 0:
                arm_quat = -arm_quat
                fixed_counts["continuity_flips"] += 1

            if np.dot(prev_cmd_quat, cmd_quat) < 0:
                cmd_quat = -cmd_quat
                fixed_counts["continuity_flips"] += 1

        # Replace adjusted quats back
        arm_state[3:7] = arm_quat
        cmd_state[3:7] = cmd_quat

        # Hand states
        obs["arm_states"] = arm_state
        obs["commanded_arm_states"] = cmd_state
        obs["ruka_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
        obs["commanded_ruka_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)

        all_observations.append(obs)
        all_timestamps.append(float(t))

# ----------------------------
# GLOBAL MIN/MAX
# ----------------------------
print("\nComputing global min/max across all demos...")

arm_stack = np.stack([o["arm_states"] for o in all_observations], axis=0)
hand_stack = np.stack([o["ruka_states"] for o in all_observations], axis=0)
arm_cmd_stack = np.stack([o["commanded_arm_states"] for o in all_observations], axis=0)
hand_cmd_stack = np.stack([o["commanded_ruka_states"] for o in all_observations], axis=0)

data = {
    "observations": all_observations,
    "timestamps": np.array(all_timestamps, dtype=np.float64),
    "max_arm": arm_stack.max(axis=0),
    "min_arm": arm_stack.min(axis=0),
    "max_ruka": hand_stack.max(axis=0),
    "min_ruka": hand_stack.min(axis=0),
    "max_arm_commanded": arm_cmd_stack.max(axis=0),
    "min_arm_commanded": arm_cmd_stack.min(axis=0),
    "max_ruka_commanded": hand_cmd_stack.max(axis=0),
    "min_ruka_commanded": hand_cmd_stack.min(axis=0),
    "task_emb": np.zeros(256, dtype=np.float32),
}

# ----------------------------
# SAVE MERGED PKL
# ----------------------------
save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
with open(save_file, "wb") as f:
    pickle.dump(data, f)

print(f"\nSaved {len(all_observations)} merged frames to {save_file}")
print(f"Global hemisphere flips applied: {fixed_counts['global_flips']}")
print(f"Continuity flips applied: {fixed_counts['continuity_flips']}")