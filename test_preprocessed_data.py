import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = Path("/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/demonstration_0")
PROCESSED_FILE = Path("processed_data_pkl/demo_task.pkl")

# ----------------------------
# LOAD RAW DATA
# ----------------------------
print("Loading raw PKL files...")

with open(DATA_FOLDER / "states.pkl", "rb") as f:
    arm_states = pickle.load(f)
with open(DATA_FOLDER / "commanded_states.pkl", "rb") as f:
    arm_commanded_states = pickle.load(f)
with open(DATA_FOLDER / "ruka_states.pkl", "rb") as f:
    hand_states = pickle.load(f)
with open(DATA_FOLDER / "ruka_commanded_states.pkl", "rb") as f:
    hand_commanded_states = pickle.load(f)

print(f"  Arm states: {len(arm_states)}")
print(f"  Hand states: {len(hand_states)}")

# ----------------------------
# LOAD PROCESSED DATA
# ----------------------------
print("\nLoading processed PKL...")
with open(PROCESSED_FILE, "rb") as f:
    data = pickle.load(f)

observations = data["observations"]

print(f"  Processed observations: {len(observations)}")

# ----------------------------
# BASIC SHAPE CHECKS
# ----------------------------
print("\nChecking observation shapes...")

o0 = observations[0]
print("  Pixels shape:", o0["pixels0"].shape)
print("  arm shape:", o0["arm_states"].shape)
print("  Commanded arm shape:", o0["commanded_arm_states"].shape)
print("  ruka shape:", o0["ruka_states"].shape)
print("  Commanded ruka shape:", o0["commanded_ruka_states"].shape)

# ----------------------------
# TIMESTAMP CONSISTENCY CHECK
# ----------------------------
print("\nChecking synchronization (sample of first 100 hand frames)...")

arm_times = np.array([s["timestamp"] for s in arm_states])
hand_times = np.array([s["timestamp"] for s in hand_states])

for i in range(min(100, len(hand_states))):
    t = hand_states[i]["timestamp"]
    closest_arm_idx = np.argmin(np.abs(arm_times - t))
    print(f"  Hand[{i}] time={t:.3f}, matched Arm[{closest_arm_idx}] time={arm_times[closest_arm_idx]:.3f}, Δ={abs(t - arm_times[closest_arm_idx]):.4f}s")

# ----------------------------
# VALUE RANGE CHECK
# ----------------------------
print("\nChecking numeric ranges...")

max_cartesian = data["max_arm"]
min_cartesian = data["min_arm"]
max_gripper = data["max_ruka"]
min_gripper = data["min_ruka"]

print("  arm range:")
print("    min:", np.round(min_cartesian, 4))
print("    max:", np.round(max_cartesian, 4))
print("  ruka range:")
print("    min:", np.round(min_gripper, 4))
print("    max:", np.round(max_gripper, 4))

# ----------------------------
# VALUE SANITY CHECK (no NaNs, infs)
# ----------------------------
print("\nChecking for NaN or inf values...")
for key in ["arm_states", "ruka_states"]:
    vals = np.stack([o[key] for o in observations])
    if np.isnan(vals).any() or np.isinf(vals).any():
        print(f"  ⚠️  {key} contains invalid values!")
    else:
        print(f"  ✅  {key} valid.")

