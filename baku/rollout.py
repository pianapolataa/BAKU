#!/usr/bin/env python3
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt

# --- Franka imports ---
from frankateach.messages import FrankaAction
from frankateach.network import create_request_socket
from frankateach.constants import *

sys.path.append(os.path.expanduser("/home_shared/grail_sissi/BAKU/baku/vr-hand-tracking/Franka-Teach/RUKA"))
# --- Ruka imports ---
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos
from ruka_hand.control.rukav2_teleop import *

# --- Agent imports ---
from train import WorkspaceIL
from suite.custom import task_make_fn


class AgentRollout:
    def __init__(self, cfg, demo_data_path, snapshot_path, save_log=True):
        self.demo_path = Path(demo_data_path)
        self.snapshot_path = Path(snapshot_path)
        self.save_log = save_log

        # -----------------------------
        # Load demo data
        # -----------------------------
        with open(self.demo_path, "rb") as f:
            demo_data = pickle.load(f)
        self.demo_data = demo_data

        # -----------------------------
        # Load agent + environment
        # -----------------------------
        envs, _ = task_make_fn(demo_data)
        self.workspace = WorkspaceIL(cfg)
        self.workspace.env = envs
        self.workspace.load_snapshot({"bc": self.snapshot_path})
        self.workspace.agent.train(False)
        print(f"Loaded BC snapshot from {self.snapshot_path}")

        # -----------------------------
        # Build normalization stats
        # -----------------------------
        self.norm_stats = {
            "features": {
                "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"]]),
                "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"]]),
            },
            "actions": {
                "min": np.concatenate([demo_data["min_arm_commanded"], demo_data["min_ruka_commanded"]]),
                "max": np.concatenate([demo_data["max_arm_commanded"], demo_data["max_ruka_commanded"]]),
            }
        }

        # -----------------------------
        # Setup Franka communication
        # -----------------------------
        self.arm_socket = create_request_socket(LOCALHOST, CONTROL_PORT)
        print("Connected to Franka arm.")

        # Reset robot
        reset_action = FrankaAction(
            pos=np.zeros(3),
            quat=np.zeros(4),
            gripper=-1,
            reset=True,
            timestamp=time.time(),
        )
        self.arm_socket.send(pickle.dumps(reset_action, protocol=-1))
        _ = pickle.loads(self.arm_socket.recv())
        print("Franka arm reset complete.")

        # Initialize Ruka hand
        self.handler = RUKAv2Handler()
        time.sleep(0.5)
        self.handler.reset()
        time.sleep(1)
        print("Ruka hand initialized.")

        # -----------------------------
        # Logging
        # -----------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logged_data = []  # always store arm actions for plotting
        self.plot_path = Path(f"/home_shared/grail_sissi/BAKU/rollout.png")
        if self.save_log:
            self.log_path = Path(f"live_rollout_{timestamp}.pkl")
            print(f"Logging enabled: {self.log_path}")

    def get_arm_state(self):
        """Query current arm state from Franka."""
        self.arm_socket.send(b"get_state")
        raw_state = pickle.loads(self.arm_socket.recv())
        arm_state = np.concatenate([raw_state.pos, raw_state.quat]).astype(np.float32)
        return arm_state
    
    def norm_quat_vec(self, arr7):
        a = arr7.astype(np.float64).copy()
        q = a[3:7]
        n = np.linalg.norm(q) + 1e-8
        a[3:7] = (q / n).astype(np.float32)
        return a.astype(np.float32)

    def run(self, duration_s: float = 60.0, freq: float = 50.0):
        print("Starting live rollout...")
        dt = 1.0 / freq
        t0 = time.time()
        ref_quat = self.demo_data["observations"][0]["arm_states"][3:7].astype(np.float32)

        try:
            cnt = 0
            while time.time() - t0 < duration_s:
                cnt += 1
                arm_state = self.get_arm_state()
                ruka_state = self.handler.hand.read_pos()
                demo_obs = self.demo_data["observations"][min(cnt+2, len(self.demo_data["observations"]) - 1)]
                arm_state_1 = demo_obs["arm_states"].copy()
                ruka_state_1 = demo_obs["ruka_states"].copy()

                quat = arm_state[3:7].copy()
                if np.dot(ref_quat, quat) < 0:
                    quat *= -1.0
                    arm_state[3:7] = quat

                arm_state[:1] = arm_state_1[:1].copy()
                feat = np.concatenate([arm_state.copy(), ruka_state.copy()], axis=0).astype(np.float32)
                feat_1 = np.concatenate([arm_state_1.copy(), ruka_state_1.copy()], axis=0).astype(np.float32)
                if (cnt == 1):
                    print(feat)
                    print(feat_1)
                    feat = feat_1.copy()

                obs = {
                    "features": feat,
                    "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
                    "task_emb": np.asarray(self.demo_data["task_emb"], dtype=np.float32),
                }

                obs_1 = {
                    "features": feat_1,
                    "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
                    "task_emb": np.asarray(self.demo_data["task_emb"], dtype=np.float32),
                }

                with torch.no_grad():
                    action = self.workspace.agent.act(obs, prompt=None, norm_stats=self.norm_stats, step=0,
                                                     global_step=self.workspace.global_step, eval_mode=True)
                    action_1 = self.workspace.agent.act(obs_1, prompt=None, norm_stats=self.norm_stats, step=0,
                                                     global_step=self.workspace.global_step, eval_mode=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if isinstance(action_1, torch.Tensor):
                    action_1 = action_1.cpu().numpy()

                # --- Always store for plotting ---
                self.logged_data.append({
                    "timestamp": time.time() - t0,
                    "action": feat.copy(),
                    "action_1": feat_1.copy()
                })
                
                if (cnt < 2): action = action_1.copy()
                print(cnt)

                arm_action = self.norm_quat_vec(action[:7])
                arm_action[:3] = np.clip(arm_action[:3], a_min=ROBOT_WORKSPACE_MIN, a_max=ROBOT_WORKSPACE_MAX)
                arm_action_1 = self.norm_quat_vec(action_1[:7])
                arm_action_1[:3] = np.clip(arm_action_1[:3], a_min=ROBOT_WORKSPACE_MIN, a_max=ROBOT_WORKSPACE_MAX)

                franka_action = FrankaAction(
                    pos=arm_action[:3],
                    quat=arm_action[3:7],
                    gripper=-1,
                    reset=False,
                    timestamp=time.time(),
                )
                self.arm_socket.send(pickle.dumps(franka_action, protocol=-1))
                _ = self.arm_socket.recv()

                # 6. Send hand command directly
                hand_action = np.clip(action[7:], self.handler.hand.min_lim, self.handler.hand.max_lim)
                move_to_pos(curr_pos=ruka_state, des_pos=hand_action, hand=self.handler.hand, traj_len=20)

                elapsed = time.time() - t0
                next_time = (len(self.logged_data) + 1) * dt
                time.sleep(0.037)

        except KeyboardInterrupt:
            print("Rollout interrupted by user.")

        finally:
            self.arm_socket.close()
            print("Connections closed.")

            # Save pickle log if requested
            if self.save_log and self.logged_data:
                with open(self.log_path, "wb") as f:
                    pickle.dump(self.logged_data, f)
                print(f"Saved rollout log to {self.log_path}")

            # Always plot full 23D action array
            if self.logged_data:
                timestamps = [d["timestamp"] for d in self.logged_data]
                full_actions = np.stack([d["action"] for d in self.logged_data], axis=0)
                full_actions_1 = np.stack([d["action_1"] for d in self.logged_data], axis=0)

                fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
                axes = axes.flatten()
                labels = [f"dim_{i}" for i in range(23)]

                for i in range(23):
                    axes[i].plot(timestamps, full_actions[:, i], label="rollout")
                    axes[i].plot(timestamps, full_actions_1[:, i], label="demo-based", linestyle="--")
                    axes[i].set_ylabel(labels[i])
                    axes[i].grid(True)
                    axes[i].legend(fontsize=8)

                # Hide extra subplots (last 2)
                for i in range(23, 25):
                    axes[i].axis('off')

                axes[-1].set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(self.plot_path)
                plt.close(fig)
                print(f"Saved full 23D action plot to {self.plot_path}")



# ------------------------------------------------------------
# Hydra entry point
# ------------------------------------------------------------
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="/home_shared/grail_sissi/BAKU/baku/cfgs", config_name="config")
def main(cfg: DictConfig):
    demo_data_path = "/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl"
    # snapshot_path = "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.19_train/deterministic/193903/snapshot/57000.pt" # 3 demo policy best:57000
    snapshot_path = "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.24_train/deterministic/174342/snapshot/28000.pt" # new 11 demo w noise

    rollout = AgentRollout(cfg, demo_data_path, snapshot_path, save_log=True)
    rollout.run(duration_s=180, freq=50)


if __name__ == "__main__":
    main()
