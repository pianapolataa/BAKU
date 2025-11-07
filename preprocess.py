import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_0")  # folder containing your PKLs/videos
CAM_INDEX = 0                                    # index of the camera you want to use
IMG_SIZE = (128, 128)                            # resize images to this
SAVE_PATH = Path("processed_data_pkl")
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
    obs["pixels0"] = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)  # dummy image
    obs["timestamp"] = float(t)  # store timestamp for this frame

    # Find closest arm frame to this hand timestamp
    arm_idx = np.argmin(np.abs(arm_times - t))
    time_diff = abs(arm_times[arm_idx] - t)

    # Skip if timestamps differ by more than 0.05s
    if time_diff > 0.05:
        continue

    # Extract Franka joint positions
    arm_state = arm_states[arm_idx]['state']
    commanded_arm_state = arm_commanded_states[arm_idx]['state']

    obs["cartesian_states"] = np.concatenate([arm_state.pos, arm_state.quat]).astype(np.float32)
    obs["commanded_cartesian_states"] = np.concatenate([commanded_arm_state.pos, commanded_arm_state.quat]).astype(np.float32)

    # Hand/gripper states
    obs["gripper_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
    obs["commanded_gripper_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)

    observations.append(obs)
    timestamps.append(float(t))

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
# If you want a real embedding, you can use sentence-transformers
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
