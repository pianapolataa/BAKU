# import pickle
# import cv2
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_0")  # folder containing your PKLs/videos
# CAM_INDEX = 0                                    # index of the camera you want to use
# IMG_SIZE = (128, 128)                            # resize images to this
# SAVE_PATH = Path("processed_data_pkl")
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
from scipy.spatial.transform import Rotation as R, Slerp

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_0")  # folder containing your PKLs/videos
CAM_INDEX = 0
IMG_SIZE = (128, 128)
SAVE_PATH = Path("processed_data_pkl")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
TASK_NAME = "demo_task"
NUM_FRAMES = None  # optionally limit number of frames

# ----------------------------
# LOAD DATA
# ----------------------------
print("Loading PKL data...")

with open(DATA_FOLDER / "states.pkl", "rb") as f:
    arm_states = pickle.load(f)

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
timestamps = []

arm_times = np.array([s["timestamp"] for s in arm_states])
hand_times = np.array([s["timestamp"] for s in hand_states])

# Loop over hand frames
for i, t in enumerate(tqdm(hand_times if NUM_FRAMES is None else hand_times[:NUM_FRAMES])):
    # Find closest arm frame
    arm_idx = np.argmin(np.abs(arm_times - t))
    time_diff = abs(arm_times[arm_idx] - t)
    if time_diff > 0.05:
        continue

    obs = {}
    obs["pixels0"] = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    obs["timestamp"] = float(t)

    arm_state = arm_states[arm_idx]['state']
    commanded_arm_state = arm_commanded_states[arm_idx]['state']

    obs["cartesian_states"] = np.concatenate([arm_state.pos, arm_state.quat]).astype(np.float32)
    obs["commanded_cartesian_states"] = np.concatenate([commanded_arm_state.pos, commanded_arm_state.quat]).astype(np.float32)

    obs["gripper_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
    obs["commanded_gripper_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)

    observations.append(obs)
    timestamps.append(float(t))

# ----------------------------
# INTERPOLATE FRAMES
# ----------------------------
print("Interpolating frames...")

interpolated_obs = []
interpolated_timestamps = []

for i in range(len(observations)-1):
    obs1 = observations[i]
    obs2 = observations[i+1]

    # Append original frame
    interpolated_obs.append(obs1)
    interpolated_timestamps.append(timestamps[i])

    # Interpolated frame
    interp_obs = {}
    interp_obs["pixels0"] = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    interp_obs["timestamp"] = (timestamps[i] + timestamps[i+1]) / 2.0

    # Interpolate cartesian states (linear for pos, SLERP for quat)
    pos1, quat1 = obs1["cartesian_states"][:3], obs1["cartesian_states"][3:]
    pos2, quat2 = obs2["cartesian_states"][:3], obs2["cartesian_states"][3:]

    interp_obs["cartesian_states"] = np.zeros(7, dtype=np.float32)
    interp_obs["cartesian_states"][:3] = (pos1 + pos2) / 2.0

    r = R.from_quat([quat1, quat2])
    slerp = Slerp([0, 1], r)
    interp_obs["cartesian_states"][3:] = slerp(0.5).as_quat()

    # Commanded cartesian states
    pos1c, quat1c = obs1["commanded_cartesian_states"][:3], obs1["commanded_cartesian_states"][3:]
    pos2c, quat2c = obs2["commanded_cartesian_states"][:3], obs2["commanded_cartesian_states"][3:]

    interp_obs["commanded_cartesian_states"] = np.zeros(7, dtype=np.float32)
    interp_obs["commanded_cartesian_states"][:3] = (pos1c + pos2c) / 2.0
    r = R.from_quat([quat1c, quat2c])
    slerp = Slerp([0,1], r)
    interp_obs["commanded_cartesian_states"][3:] = slerp(0.5).as_quat()

    # Interpolate gripper states
    interp_obs["gripper_states"] = (obs1["gripper_states"] + obs2["gripper_states"]) / 2.0
    interp_obs["commanded_gripper_states"] = (obs1["commanded_gripper_states"] + obs2["commanded_gripper_states"]) / 2.0

    interpolated_obs.append(interp_obs)
    interpolated_timestamps.append(interp_obs["timestamp"])

# Append the last original frame
interpolated_obs.append(observations[-1])
interpolated_timestamps.append(timestamps[-1])

observations = interpolated_obs
timestamps = interpolated_timestamps

# ----------------------------
# COMPUTE MIN/MAX BOUNDS
# ----------------------------
print("Computing min/max bounds...")

cartesian_stack = np.stack([o["cartesian_states"] for o in observations], axis=0)
hand_stack = np.stack([o["gripper_states"] for o in observations], axis=0)

max_cartesian = np.max(cartesian_stack, axis=0)
min_cartesian = np.min(cartesian_stack, axis=0)

max_gripper = np.max(hand_stack, axis=0)
min_gripper = np.min(hand_stack, axis=0)

# ----------------------------
# TASK EMBEDDING (dummy)
# ----------------------------
task_emb = np.zeros(256, dtype=np.float32)

# ----------------------------
# SAVE PKL
# ----------------------------
data = {
    "observations": observations,
    "timestamps": np.array(timestamps, dtype=np.float64),
    "max_cartesian": max_cartesian,
    "min_cartesian": min_cartesian,
    "max_gripper": max_gripper,
    "min_gripper": min_gripper,
    "task_emb": task_emb
}

save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
with open(save_file, "wb") as f:
    pickle.dump(data, f)

print(f"Saved processed data (with timestamps) to {save_file}")
