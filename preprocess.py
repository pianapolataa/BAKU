import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = Path("demonstration_0")  # folder containing your PKLs/videos
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

# Load video
video_path = DATA_FOLDER / f"cam_{CAM_INDEX}_rgb_video.avi"
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

frames = []
frame_timestamps = []

print("Reading video frames...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, IMG_SIZE)
    frames.append(frame)
    # approximate timestamp based on frame index / fps
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    frame_timestamps.append(timestamp)
cap.release()

frames = np.array(frames)

# ----------------------------
# SYNCHRONIZE DATA
# ----------------------------
def interpolate_state(states, target_time):
    """Find the closest state to the target timestamp."""
    times = np.array([s["timestamp"] for s in states])
    idx = np.argmin(np.abs(times - target_time))
    return np.array(states[idx]["state"], dtype=np.float32)

print("Synchronizing data...")
observations = []
for i, t in enumerate(tqdm(frame_timestamps if NUM_FRAMES is None else frame_timestamps[:NUM_FRAMES])):
    obs = {}
    obs["pixels0"] = frames[i]

    obs["cartesian_states"] = interpolate_state(arm_states, t)
    obs["commanded_cartesian_states"] = interpolate_state(arm_commanded_states, t)

    obs["gripper_states"] = interpolate_state(hand_states, t)
    obs["commanded_gripper_states"] = interpolate_state(hand_commanded_states, t)

    observations.append(obs)

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
    "max_cartesian": max_cartesian,
    "min_cartesian": min_cartesian,
    "max_gripper": max_gripper,
    "min_gripper": min_gripper,
    "task_emb": task_emb
}

save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
with open(save_file, "wb") as f:
    pickle.dump(data, f)

print(f"Saved processed data to {save_file}")
