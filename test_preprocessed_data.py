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
print("  Cartesian shape:", o0["cartesian_states"].shape)
print("  Commanded Cartesian shape:", o0["commanded_cartesian_states"].shape)
print("  Gripper shape:", o0["gripper_states"].shape)
print("  Commanded Gripper shape:", o0["commanded_gripper_states"].shape)

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

max_cartesian = data["max_cartesian"]
min_cartesian = data["min_cartesian"]
max_gripper = data["max_gripper"]
min_gripper = data["min_gripper"]

print("  Cartesian range:")
print("    min:", np.round(min_cartesian, 4))
print("    max:", np.round(max_cartesian, 4))
print("  Gripper range:")
print("    min:", np.round(min_gripper, 4))
print("    max:", np.round(max_gripper, 4))

# ----------------------------
# VALUE SANITY CHECK (no NaNs, infs)
# ----------------------------
print("\nChecking for NaN or inf values...")
for key in ["cartesian_states", "gripper_states"]:
    vals = np.stack([o[key] for o in observations])
    if np.isnan(vals).any() or np.isinf(vals).any():
        print(f"  ⚠️  {key} contains invalid values!")
    else:
        print(f"  ✅  {key} valid.")

# ----------------------------
# OPTIONAL: VISUALIZE SMOOTHNESS
# ----------------------------
print("\nPlotting first Cartesian vs Gripper dimension to visualize sync...")

cartesian_vals = [o["cartesian_states"][0] for o in observations]
gripper_vals = [o["gripper_states"][0] for o in observations]

plt.figure(figsize=(8, 4))
plt.plot(cartesian_vals, label="Arm (first pos component)")
plt.plot(gripper_vals, label="Hand (first finger state)")
plt.legend()
plt.title("Arm vs Hand Value Evolution (first components)")
plt.xlabel("Frame index")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

# ----------------------------
# SUMMARY
# ----------------------------
print("\n✅ Verification complete.")
print("If the timestamp deltas are small (<0.02s typical), and plots are smooth, your preprocessing is correct.")
