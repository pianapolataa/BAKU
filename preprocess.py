# import pickle
# import cv2
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_1")  # folder containing your PKLs/videos
# CAM_INDEX = 0                                    # index of the camera you want to use
# IMG_SIZE = (128, 128)                            # resize images to this
# SAVE_PATH = Path("processed_data_1_pkl")
# SAVE_PATH.mkdir(parents=True, exist_ok=True)
# TASK_NAME = "demo_task"                          # arbitrary task name for BAKU
# NUM_FRAMES = None                                # optionally limit number of frames

# # ----------------------------
# # LOAD DATA
# # ----------------------------
# print("Loading PKL data...")

# with open(DATA_FOLDER / "states.pkl", "rb") as f:
#     arm_states = pickle.load(f)  # list of dicts: {'state': [x7,...], 'timestamp': t}

# with open(DATA_FOLDER / "commanded_states.pkl", "rb") as f:
#     arm_commanded_states = pickle.load(f)

# with open(DATA_FOLDER / "ruka_states.pkl", "rb") as f:
#     hand_states = pickle.load(f)

# with open(DATA_FOLDER / "ruka_commanded_states.pkl", "rb") as f:
#     hand_commanded_states = pickle.load(f)

# print(f"Number of arm_states: {len(arm_states)}")
# print(f"Number of commanded arm_states: {len(arm_commanded_states)}")
# print(f"Number of hand_states: {len(hand_states)}")
# print(f"Number of commanded hand_states: {len(hand_commanded_states)}")

# # ----------------------------
# # SYNCHRONIZE DATA (hand timestamps as reference)
# # ----------------------------
# print("Synchronizing data...")
# observations = []
# timestamps = []  # store timestamps for all synchronized frames

# # Extract timestamps
# arm_times = np.array([s["timestamp"] for s in arm_states])
# hand_times = np.array([s["timestamp"] for s in hand_states])

# # Loop over hand frames
# for i, t in enumerate(tqdm(hand_times if NUM_FRAMES is None else hand_times[:NUM_FRAMES])):
#     obs = {}
#     obs["pixels0"] = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)  # dummy image
#     obs["timestamp"] = float(t)  # store timestamp for this frame

#     # Find closest arm frame to this hand timestamp
#     arm_idx = np.argmin(np.abs(arm_times - t))
#     time_diff = abs(arm_times[arm_idx] - t)

#     # Skip if timestamps differ by more than 0.05s
#     if time_diff > 0.05:
#         continue

#     # Extract Franka joint positions
#     arm_state = arm_states[arm_idx]['state']
#     commanded_arm_state = arm_commanded_states[arm_idx]['state']

#     obs["cartesian_states"] = np.concatenate([arm_state.pos, arm_state.quat]).astype(np.float32)
#     obs["commanded_cartesian_states"] = np.concatenate([commanded_arm_state.pos, commanded_arm_state.quat]).astype(np.float32)

#     # Hand/gripper states
#     obs["gripper_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
#     obs["commanded_gripper_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)

#     observations.append(obs)
#     timestamps.append(float(t))

# # ----------------------------
# # COMPUTE MIN/MAX BOUNDS
# # ----------------------------
# print("Computing min/max bounds...")

# cartesian_stack = np.stack([o["cartesian_states"] for o in observations], axis=0)
# hand_stack = np.stack([o["gripper_states"] for o in observations], axis=0)

# max_cartesian = np.max(cartesian_stack, axis=0)
# min_cartesian = np.min(cartesian_stack, axis=0)

# max_gripper = np.max(hand_stack, axis=0)
# min_gripper = np.min(hand_stack, axis=0)

# # ----------------------------
# # TASK EMBEDDING (dummy)
# # ----------------------------
# # If you want a real embedding, you can use sentence-transformers
# task_emb = np.zeros(256, dtype=np.float32)

# # ----------------------------
# # SAVE PKL
# # ----------------------------
# data = {
#     "observations": observations,
#     "timestamps": np.array(timestamps, dtype=np.float64),
#     "max_cartesian": max_cartesian,
#     "min_cartesian": min_cartesian,
#     "max_gripper": max_gripper,
#     "min_gripper": min_gripper,
#     "task_emb": task_emb
# }

# save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
# with open(save_file, "wb") as f:
#     pickle.dump(data, f)

# print(f"Saved processed data (with timestamps) to {save_file}")

import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_1")  # folder containing your PKLs/videos
CAM_INDEX = 0                                    # index of the camera you want to use
IMG_SIZE = (128, 128)                            # resize images to this
SAVE_PATH = Path("processed_data_1_pkl")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
TASK_NAME = "demo_task"                          # arbitrary task name for BAKU
NUM_FRAMES = None                                # optionally limit number of frames

# ----------------------------
# LOAD DATA
# ----------------------------
print("Loading PKL data...")

with open(DATA_FOLDER / "states.pkl", "rb") as f:
    arm_states = pickle.load(f)  # list of dicts: {'state': [x7,...], 'timestamp': t}

with open(DATA_FOLDER / "commanded_states.pkl", "rb") as f:
    arm_commanded_states = pickle.load(f)

with open(DATA_FOLDER / "ruka_states.pkl", "rb") as f:
    hand_states = pickle.load(f)

with open(DATA_FOLDER / "ruka_commanded_states.pkl", "rb") as f:
    hand_commanded_states = pickle.load(f)

print(f"Number of arm_states: {len(arm_states)}")
print(f"Number of commanded arm_states: {len(arm_commanded_states)}")
print(f"Number of hand_states: {len(hand_states)}")
print(f"Number of commanded hand_states: {len(hand_commanded_states)}")

# ----------------------------
# SYNCHRONIZE DATA (hand timestamps as reference)
# ----------------------------
print("Synchronizing data...")
observations = []
timestamps = []  # store timestamps for all synchronized frames

# Extract timestamps
arm_times = np.array([s["timestamp"] for s in arm_states])
hand_times = np.array([s["timestamp"] for s in hand_states])

# Loop over hand frames
for i, t in enumerate(tqdm(hand_times if NUM_FRAMES is None else hand_times[:NUM_FRAMES])):
    obs = {}
    # Dummy image (you can replace with actual image loading)
    obs["pixels"] = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    # If you have egocentric camera you can add obs["pixels_egocentric"] similarly
    
    obs["timestamp"] = float(t)

    # Find closest arm frame to this hand timestamp
    arm_idx = np.argmin(np.abs(arm_times - t))
    time_diff = abs(arm_times[arm_idx] - t)

    # Skip if timestamps differ by more than 0.05s
    if time_diff > 0.05:
        continue

    # Extract Franka joint positions
    arm_state = arm_states[arm_idx]['state']
    commanded_arm_state = arm_commanded_states[arm_idx]['state']

    # store state and commanded state
    obs["joint_states"] = np.concatenate([arm_state.pos, arm_state.quat]).astype(np.float32)
    obs["commanded_joint_states"] = np.concatenate([commanded_arm_state.pos, commanded_arm_state.quat]).astype(np.float32)

    # Hand/gripper states
    obs["gripper_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
    obs["commanded_gripper_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)

    observations.append(obs)
    timestamps.append(float(t))

# ----------------------------
# COMPUTE MIN/MAX BOUNDS
print("Computing min/max bounds...")

joint_stack = np.stack([o["joint_states"] for o in observations], axis=0)
gripper_stack = np.stack([o["gripper_states"] for o in observations], axis=0)

max_joint = np.max(joint_stack, axis=0)
min_joint = np.min(joint_stack, axis=0)

max_gripper = np.max(gripper_stack, axis=0)
min_gripper = np.min(gripper_stack, axis=0)

# ----------------------------
# TASK EMBEDDING (dummy)
task_emb = np.zeros(256, dtype=np.float32)

# ----------------------------
# CONVERT TO LIBERO-COMPATIBLE FORMAT
print("Converting to LIBERO-compatible format…")

# In LIBERO loader:  
#   data["observations"] = (if obs_type == "pixels") something like dict with keys “pixels”, possibly "pixels_egocentric", "joint_states", "gripper_states"
#   data["actions"] = np.ndarray shaped (T, action_dim)
#   data["task_emb"] = task embedding

# Here we treat each demonstration as a single episode (list) of steps.
# But for simplicity we'll wrap all steps as one episode.

# Build one episode
episode_obs = {
    "pixels": np.stack([o["pixels"] for o in observations], axis=0),
    # Optional: if you had egocentric images you’d do "pixels_egocentric"
    "joint_states": np.stack([o["joint_states"] for o in observations], axis=0),
    "gripper_states": np.stack([o["gripper_states"] for o in observations], axis=0),
}

# Define actions as commanded_joint_states + commanded_gripper_states for each timestep
actions = np.stack(
    [
        np.concatenate([o["commanded_joint_states"], o["commanded_gripper_states"]])
        for o in observations
    ],
    axis=0
).astype(np.float32)

data = {
    "observations": [episode_obs],   # list of episodes
    "actions": [actions],            # list of corresponding action sequences
    "task_emb": [task_emb],          # list of task embeddings, one per episode
    # You can optionally include "timestamps", "max_joint", etc. but not required
}

# ----------------------------
# SAVE PKL (LIBERO format)
save_dir = Path("expert_demos/libero")
save_dir.mkdir(parents=True, exist_ok=True)
save_file = save_dir / f"{TASK_NAME}.pkl"

with open(save_file, "wb") as f:
    pickle.dump(data, f)

print(f"Saved LIBERO-compatible demo to {save_file}")
