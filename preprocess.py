# #!/usr/bin/env python3
# import pickle
# import cv2
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# DATA_ROOT = Path("/home_shared/grail_sissi/BAKU/baku/vr-hand-tracking/Franka-Teach/marker_data")
# IMG_SIZE = (84, 84)
# SAVE_PATH = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl")
# SAVE_PATH.mkdir(parents=True, exist_ok=True)
# TASK_NAME = "demo_task"

# def flip_to_reference(quat, ref):
#     return quat if np.dot(quat, ref) >= 0 else -quat

# def extract_quat(x):
#     return x[3:7]

# # >>> ADDED
# def load_and_resize_rgb(img, size):
#     """Convert BGR->RGB, resize, convert to CHW, float32, shape (1,3,H,W)."""
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#     img = img.astype(np.float32) / 255.0
#     img = np.transpose(img, (2, 0, 1))  # CHW
#     return img  # (3,H,W)


# # ----------------------------
# # GATHER ALL DEMO FOLDERS
# # ----------------------------
# demo_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and "demonstration" in p.name])
# print(f"Found {len(demo_dirs)} demos")

# if len(demo_dirs) == 0:
#     raise RuntimeError("No demonstration folders found!")

# # ----------------------------
# # GLOBAL STORAGE
# # ----------------------------
# all_observations = []
# all_timestamps = []

# reference_quat = None
# reference_cmd_quat = None

# fixed_counts = {"global_flips": 0, "continuity_flips": 0}

# # ----------------------------
# # PROCESS EACH DEMO
# # ----------------------------
# for d_idx, DEMO in enumerate(demo_dirs):
#     print(f"\n=== Processing {DEMO.name} ===")
#     if DEMO.name == "demonstration_64": continue

#     # Load pkl data
#     with open(DEMO / "states.pkl", "rb") as f:
#         arm_states = pickle.load(f)
#     with open(DEMO / "commanded_states.pkl", "rb") as f:
#         arm_commanded_states = pickle.load(f)
#     with open(DEMO / "ruka_states.pkl", "rb") as f:
#         hand_states = pickle.load(f)
#     with open(DEMO / "ruka_commanded_states.pkl", "rb") as f:
#         hand_commanded_states = pickle.load(f)

#     # >>> ADDED: Load RGB metadata + video
#     rgb_meta_path = DEMO / "cam_1_rgb_video.metadata"
#     rgb_video_path = DEMO / "cam_1_rgb_video.avi"

#     with open(rgb_meta_path, "rb") as f:
#         rgb_meta = pickle.load(f)
#     rgb_timestamps = np.array(rgb_meta["timestamps"], dtype=np.float64)

#     rgb_cap = cv2.VideoCapture(str(rgb_video_path))
#     if not rgb_cap.isOpened():
#         raise RuntimeError(f"Failed to open video: {rgb_video_path}")

#     print(f"Loaded {len(arm_states)} arm frames, {len(hand_states)} hand frames, {len(rgb_timestamps)} rgb frames")

#     # Extract timestamps
#     arm_times = np.array([s["timestamp"] for s in arm_states])
#     hand_times = np.array([s["timestamp"] for s in hand_states])

#     num_hand_frames = len(hand_times)
#     first = -1
#     # Loop through hand frames
#     for i, t in enumerate(tqdm(hand_times, desc=f"Sync {DEMO.name}")):
#         obs = {}
#         obs["timestamp"] = float(t)

#         # >>> ADDED: find closest rgb frame
#         rgb_idx = np.argmin(np.abs(rgb_timestamps - t))
#         if abs(rgb_timestamps[rgb_idx] - t) > 0.05:
#             continue

#         # >>> ADDED: read frame from video
#         rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
#         ret, frame = rgb_cap.read()
#         if not ret:
#             continue

#         obs["pixels0"] = load_and_resize_rgb(frame, IMG_SIZE)  # (3,H,W)
#         # obs["pixels0"] = np.zeros((3, 84, 84))

#         # find nearest arm frame
#         arm_idx = np.argmin(np.abs(arm_times - t))
#         if abs(arm_times[arm_idx] - t) > 0.05:
#             continue
#         if (first == -1): first = i

#         # raw arm state
#         arm_state = np.concatenate(
#             [arm_states[arm_idx]["state"].pos, arm_states[arm_idx]["state"].quat]
#         ).astype(np.float32)

#         cmd_state = np.concatenate(
#             [arm_commanded_states[arm_idx]["state"].pos,
#              arm_commanded_states[arm_idx]["state"].quat]
#         ).astype(np.float32)

#         if reference_quat is None:
#             reference_quat = extract_quat(arm_state).copy()
#             reference_cmd_quat = extract_quat(cmd_state).copy()
#             print("Set global reference quaternions")

#         # Apply sign flip to match global reference hemisphere
#         arm_quat = extract_quat(arm_state).copy()
#         cmd_quat = extract_quat(cmd_state).copy()

#         arm_quat_aligned = flip_to_reference(arm_quat, reference_quat)
#         if np.dot(arm_quat_aligned, arm_quat) < 0:
#             fixed_counts["global_flips"] += 1
#         arm_quat = arm_quat_aligned

#         cmd_quat_aligned = flip_to_reference(cmd_quat, reference_cmd_quat)
#         if np.dot(cmd_quat_aligned, cmd_quat) < 0:
#             fixed_counts["global_flips"] += 1
#         cmd_quat = cmd_quat_aligned

#         # continuity logic
#         if len(all_observations) > 0:
#             prev_arm_quat = all_observations[-1]["arm_states"][3:7]
#             prev_cmd_quat = all_observations[-1]["commanded_arm_states"][3:7]

#             if np.dot(prev_arm_quat, arm_quat) < 0:
#                 arm_quat = -arm_quat
#                 fixed_counts["continuity_flips"] += 1

#             if np.dot(prev_cmd_quat, cmd_quat) < 0:
#                 cmd_quat = -cmd_quat
#                 fixed_counts["continuity_flips"] += 1

#         arm_state[3:7] = arm_quat
#         cmd_state[3:7] = cmd_quat

#         # Hand states
#         obs["arm_states"] = arm_state
#         obs["commanded_arm_states"] = cmd_state
#         obs["ruka_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
#         obs["commanded_ruka_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)
#         obs["progress"] = (i - first) / (num_hand_frames - first)
#         all_observations.append(obs)
#         all_timestamps.append(float(t))

#     rgb_cap.release()
# # ----------------------------
# # GLOBAL MIN/MAX
# # ----------------------------
# print("\nComputing global min/max across all demos...")

# arm_stack = np.stack([o["arm_states"] for o in all_observations], axis=0)
# hand_stack = np.stack([o["ruka_states"] for o in all_observations], axis=0)
# arm_cmd_stack = np.stack([o["commanded_arm_states"] for o in all_observations], axis=0)
# hand_cmd_stack = np.stack([o["commanded_ruka_states"] for o in all_observations], axis=0)

# data = {
#     "observations": all_observations,
#     "timestamps": np.array(all_timestamps, dtype=np.float64),
#     "max_arm": arm_stack.max(axis=0),
#     "min_arm": arm_stack.min(axis=0),
#     "max_ruka": hand_stack.max(axis=0),
#     "min_ruka": hand_stack.min(axis=0),
#     "max_arm_commanded": arm_cmd_stack.max(axis=0),
#     "min_arm_commanded": arm_cmd_stack.min(axis=0),
#     "max_ruka_commanded": hand_cmd_stack.max(axis=0),
#     "min_ruka_commanded": hand_cmd_stack.min(axis=0),
#     "task_emb": np.zeros(256, dtype=np.float32),
# }

# save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
# with open(save_file, "wb") as f:
#     pickle.dump(data, f)

# print(f"\nSaved {len(all_observations)} merged frames to {save_file}")
# print(f"Global hemisphere flips applied: {fixed_counts['global_flips']}")
# print(f"Continuity flips applied: {fixed_counts['continuity_flips']}")

#!/usr/bin/env python3
import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_ROOT = Path("/home_shared/grail_sissi/BAKU/baku/vr-hand-tracking/Franka-Teach/marker_data_2")
IMG_SIZE = (84, 84)
SAVE_PATH = Path("/home_shared/grail_sissi/BAKU/processed_data_pkl")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
TASK_NAME = "demo_task"

# UPSAMPLING CONFIG
PRECISION_START_FRAME = 50
UPSAMPLE_FACTOR = 2  # Each precision frame will appear 5 times total

def flip_to_reference(quat, ref):
    return quat if np.dot(quat, ref) >= 0 else -quat

def extract_quat(x):
    return x[3:7]

def load_and_resize_rgb(img, size):
    """Convert BGR->RGB, resize, convert to CHW, float32, shape (3,H,W)."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

# ----------------------------
# GATHER ALL DEMO FOLDERS
# ----------------------------
demo_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and "demonstration" in p.name])
print(f"Found {len(demo_dirs)} demos")

if len(demo_dirs) == 0:
    raise RuntimeError("No demonstration folders found!")

# ----------------------------
# GLOBAL STORAGE
# ----------------------------
all_observations = []
all_timestamps = []

reference_quat = None
reference_cmd_quat = None

fixed_counts = {"global_flips": 0, "continuity_flips": 0, "upsampled_frames": 0}

# ----------------------------
# PROCESS EACH DEMO
# ----------------------------
for d_idx, DEMO in enumerate(demo_dirs):
    print(f"\n=== Processing {DEMO.name} ===")
    if DEMO.name == "demonstration_74": continue

    # Load pkl data
    with open(DEMO / "states.pkl", "rb") as f:
        arm_states = pickle.load(f)
    with open(DEMO / "commanded_states.pkl", "rb") as f:
        arm_commanded_states = pickle.load(f)
    with open(DEMO / "ruka_states.pkl", "rb") as f:
        hand_states = pickle.load(f)
    with open(DEMO / "ruka_commanded_states.pkl", "rb") as f:
        hand_commanded_states = pickle.load(f)

    rgb_meta_path = DEMO / "cam_1_rgb_video.metadata"
    rgb_video_path = DEMO / "cam_1_rgb_video.avi"

    with open(rgb_meta_path, "rb") as f:
        rgb_meta = pickle.load(f)
    rgb_timestamps = np.array(rgb_meta["timestamps"], dtype=np.float64)

    rgb_cap = cv2.VideoCapture(str(rgb_video_path))
    if not rgb_cap.isOpened():
        raise RuntimeError(f"Failed to open video: {rgb_video_path}")

    arm_times = np.array([s["timestamp"] for s in arm_states])
    hand_times = np.array([s["timestamp"] for s in hand_states])

    num_hand_frames = len(hand_times)
    first = -1
    
    # Counter for the current demonstration to track the precision phase
    demo_frame_count = 0

    for i, t in enumerate(tqdm(hand_times, desc=f"Sync {DEMO.name}")):
        obs = {}
        obs["timestamp"] = float(t)

        rgb_idx = np.argmin(np.abs(rgb_timestamps - t))
        if abs(rgb_timestamps[rgb_idx] - t) > 0.05:
            continue

        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
        ret, frame = rgb_cap.read()
        if not ret:
            continue

        obs["pixels0"] = load_and_resize_rgb(frame, IMG_SIZE)

        arm_idx = np.argmin(np.abs(arm_times - t))
        if abs(arm_times[arm_idx] - t) > 0.05:
            continue
        if (first == -1): first = i

        arm_state = np.concatenate(
            [arm_states[arm_idx]["state"].pos, arm_states[arm_idx]["state"].quat]
        ).astype(np.float32)

        cmd_state = np.concatenate(
            [arm_commanded_states[arm_idx]["state"].pos,
             arm_commanded_states[arm_idx]["state"].quat]
        ).astype(np.float32)

        if reference_quat is None:
            reference_quat = extract_quat(arm_state).copy()
            reference_cmd_quat = extract_quat(cmd_state).copy()

        arm_quat = extract_quat(arm_state).copy()
        cmd_quat = extract_quat(cmd_state).copy()

        arm_quat_aligned = flip_to_reference(arm_quat, reference_quat)
        if np.dot(arm_quat_aligned, arm_quat) < 0:
            fixed_counts["global_flips"] += 1
        arm_quat = arm_quat_aligned

        cmd_quat_aligned = flip_to_reference(cmd_quat, reference_cmd_quat)
        if np.dot(cmd_quat_aligned, cmd_quat) < 0:
            fixed_counts["global_flips"] += 1
        cmd_quat = cmd_quat_aligned

        if len(all_observations) > 0:
            prev_arm_quat = all_observations[-1]["arm_states"][3:7]
            prev_cmd_quat = all_observations[-1]["commanded_arm_states"][3:7]

            if np.dot(prev_arm_quat, arm_quat) < 0:
                arm_quat = -arm_quat
                fixed_counts["continuity_flips"] += 1

            if np.dot(prev_cmd_quat, cmd_quat) < 0:
                cmd_quat = -cmd_quat
                fixed_counts["continuity_flips"] += 1

        arm_state[3:7] = arm_quat
        cmd_state[3:7] = cmd_quat

        obs["arm_states"] = arm_state
        obs["commanded_arm_states"] = cmd_state
        obs["ruka_states"] = np.array(hand_states[i]["state"], dtype=np.float32)
        obs["commanded_ruka_states"] = np.array(hand_commanded_states[i]["state"], dtype=np.float32)
        obs["progress"] = (i - first) / (num_hand_frames - first)

        # --- UPSAMPLING LOGIC ---
        # Add the frame once as standard
        all_observations.append(obs)
        all_timestamps.append(float(t))

        # If we are in the precision phase (frame 50+), add extra copies
        if demo_frame_count >= PRECISION_START_FRAME:
            for _ in range(UPSAMPLE_FACTOR - 1):
                all_observations.append(obs.copy())
                all_timestamps.append(float(t))
                fixed_counts["upsampled_frames"] += 1
        
        demo_frame_count += 1

    rgb_cap.release()

# ----------------------------
# GLOBAL MIN/MAX
# ----------------------------
print("\nComputing global min/max across all demos (including upsampled data)...")

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

save_file = SAVE_PATH / f"{TASK_NAME}.pkl"
with open(save_file, "wb") as f:
    pickle.dump(data, f)

print(f"\nSaved {len(all_observations)} total frames to {save_file}")
print(f"Extra upsampled frames added: {fixed_counts['upsampled_frames']}")
print(f"Global hemisphere flips applied: {fixed_counts['global_flips']}")
print(f"Continuity flips applied: {fixed_counts['continuity_flips']}")