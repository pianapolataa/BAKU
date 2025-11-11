import numpy as np
import pickle
pkl_file = "/home_shared/grail_sissi/vr-hand-tracking/Franka-Teach/data/processed_data_pkl/demo_task.pkl"
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

actions = np.stack([
    np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]])
    for obs in data["observations"]
])

print("raw actions min/max:", actions.min(), actions.max())
