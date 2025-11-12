import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_9")  # folder containing your PKLs/videos
CAM_INDEX = 0                                    # index of the camera you want to use
IMG_SIZE = (128, 128)                            # resize images to this
SAVE_PATH = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl")
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

    obs["arm_states"] = np.concatenate([arm_state.pos, arm_state.quat]).astype(np.float32)
    obs["commanded_arm_states"] = np.concatenate([commanded_arm_state.pos, commanded_arm_state.quat]).astype(np.float32)

    # Hand/gripper states
    obs["ruka_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
    obs["commanded_ruka_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)

    observations.append(obs)
    timestamps.append(float(t))

# ----------------------------
# FIX QUATERNION SIGN FLIPS
# ----------------------------
print("Checking for quaternion sign flips...")

def fix_quaternion_sequence(observations, key, quat_start_idx=3, quat_end_idx=7):
    """
    Ensures quaternion continuity by flipping signs if consecutive quats have negative dot product.
    """
    num_flips = 0
    for i in range(1, len(observations)):
        q_prev = observations[i - 1][key][quat_start_idx:quat_end_idx]
        q_curr = observations[i][key][quat_start_idx:quat_end_idx]
        if np.dot(q_prev, q_curr) < 0:  # opposite hemisphere
            observations[i][key][quat_start_idx:quat_end_idx] *= -1
            num_flips += 1
    return num_flips

num_arm_flips = fix_quaternion_sequence(observations, "arm_states")
num_arm_cmd_flips = fix_quaternion_sequence(observations, "commanded_arm_states")

print(f"Fixed {num_arm_flips} quaternion sign flips in arm_states.")
print(f"Fixed {num_arm_cmd_flips} quaternion sign flips in commanded_arm_states.")


# ----------------------------
# COMPUTE MIN/MAX BOUNDS
# ----------------------------
print("Computing min/max bounds...")

arm_stack = np.stack([o["arm_states"] for o in observations], axis=0)
hand_stack = np.stack([o["ruka_states"] for o in observations], axis=0)
arm_stack_command = np.stack([o["commanded_arm_states"] for o in observations], axis=0)
hand_stack_command = np.stack([o["commanded_ruka_states"] for o in observations], axis=0)

max_arm = np.max(arm_stack, axis=0)
min_arm = np.min(arm_stack, axis=0)
max_ruka = np.max(hand_stack, axis=0)
min_ruka = np.min(hand_stack, axis=0)

max_arm_command = np.max(arm_stack_command, axis=0)
min_arm_command = np.min(arm_stack_command, axis=0)

max_ruka_command = np.max(hand_stack_command, axis=0)
min_ruka_command = np.min(hand_stack_command, axis=0)

print(max_ruka_command)
print(min_ruka_command)

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
    "max_arm": max_arm,
    "min_arm": min_arm,
    "max_ruka": max_ruka,
    "min_ruka": min_ruka,
    "max_arm_commanded": max_arm_command,
    "min_arm_commanded": min_arm_command,
    "max_ruka_commanded": max_ruka_command,
    "min_ruka_commanded": min_ruka_command,
    "task_emb": task_emb
}

save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
with open(save_file, "wb") as f:
    pickle.dump(data, f)

print(len(observations))
print(f"Saved processed data (with timestamps) to {save_file}")