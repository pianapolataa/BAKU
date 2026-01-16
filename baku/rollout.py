#!/usr/bin/env python3
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os
import cv2
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
                "min": np.concatenate([demo_data["min_arm"], demo_data["min_ruka"], np.array([0.0])]),
                "max": np.concatenate([demo_data["max_arm"], demo_data["max_ruka"], np.array([1.0])]),
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
        # Setup IP camera for live RGB
        # -----------------------------
        self.cam_url = "http://10.21.35.3:4747/video"
        self.cam = cv2.VideoCapture(self.cam_url)

        if not self.cam.isOpened():
            raise RuntimeError(f"Cannot open IP camera at {self.cam_url}")

        print(f"Connected to IP camera at {self.cam_url}")

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
    
    def load_and_resize_rgb(self, img, size):
        """Convert BGR->RGB, resize, convert to CHW, float32, shape (1,3,H,W)."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img  # (3,H,W)

    def run(self, duration_s: float = 60.0, freq: float = 50.0):
        print("Starting live rollout...")
        dt = 1.0 / freq
        t0 = time.time()
        ref_quat = self.demo_data["observations"][0]["arm_states"][3:7].astype(np.float32)
        num_steps = 300

        try:
            for cnt in range(num_steps):
                arm_state = self.get_arm_state()
                ruka_state = self.handler.hand.read_pos()
                demo_obs = self.demo_data["observations"][min(cnt, len(self.demo_data["observations"]) - 1)]
                arm_demo = demo_obs["arm_states"].copy()
                ruka_demo = demo_obs["ruka_states"].copy()

                # Rollout progress
                progress_real = np.array([cnt / num_steps], dtype=np.float32)
                progress_demo = np.array([demo_obs.get("progress", 0.0)], dtype=np.float32)

                quat = arm_state[3:7].copy()
                if np.dot(ref_quat, quat) < 0:
                    quat *= -1.0
                    arm_state[3:7] = quat

                # --- Build feature vector ---
                feat = np.concatenate([arm_state, ruka_state, progress_real], axis=0).astype(np.float32)
                feat_demo = np.concatenate([arm_demo, ruka_demo, progress_demo], axis=0).astype(np.float32)

                # Grab camera frame
                ret, frame = self.cam.read()
                if not ret:
                    raise RuntimeError("Failed to read frame from IP camera")
                rgb = self.load_and_resize_rgb(frame, size=(84, 84))  # shape: (C,H,W)

                # Task embedding
                task_emb = np.asarray(self.demo_data["task_emb"], dtype=np.float32)

                obs = {
                    "features": feat,
                    "pixels0": rgb,        # keep (C,H,W) for single-step
                    "task_emb": task_emb
                }

                obs_demo = {
                    "features": feat_demo,
                    "pixels0": rgb.copy(), # same shape
                    "task_emb": task_emb
                }

                # --- Agent act ---
                with torch.no_grad():
                    action = self.workspace.agent.act(obs, prompt=None, norm_stats=self.norm_stats,
                                                    step=0, global_step=self.workspace.global_step,
                                                    eval_mode=True)
                    action_demo = self.workspace.agent.act(obs_demo, prompt=None, norm_stats=self.norm_stats,
                                                        step=0, global_step=self.workspace.global_step,
                                                        eval_mode=True)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    if isinstance(action_demo, torch.Tensor):
                        action_demo = action_demo.cpu().numpy()

                # Logging
                self.logged_data.append({
                    "timestamp": time.time() - t0,
                    "action": action.copy(),
                    "action_demo": action_demo.copy()
                })

                # Use demo first step to stabilize
                if cnt < 2:
                    action = action_demo.copy()

                # Apply to Franka & Ruka
                arm_action = self.norm_quat_vec(action[:7])
                print(cnt)
                arm_action[:3] = np.clip(arm_action[:3], a_min=ROBOT_WORKSPACE_MIN, a_max=ROBOT_WORKSPACE_MAX)
                hand_action = np.clip(action[7:], self.handler.hand.min_lim, self.handler.hand.max_lim)

                franka_action = FrankaAction(pos=arm_action[:3], quat=arm_action[3:7], gripper=-1,
                                            reset=False, timestamp=time.time())
                self.arm_socket.send(pickle.dumps(franka_action, protocol=-1))
                _ = self.arm_socket.recv()

                move_to_pos(curr_pos=ruka_state, des_pos=hand_action, hand=self.handler.hand, traj_len=20)
                time.sleep(dt /3)

        except KeyboardInterrupt:
            print("Rollout interrupted by user.")

        finally:
            self.arm_socket.close()
            self.handler.hand.close()
            if hasattr(self, "cam"):
                self.cam.release()
            if self.save_log and self.logged_data:
                with open(self.log_path, "wb") as f:
                    pickle.dump(self.logged_data, f)
                print(f"Saved rollout log to {self.log_path}")

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="/home_shared/grail_sissi/BAKU/baku/cfgs", config_name="config")
def main(cfg: DictConfig):
    demo_data_path = "/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl"
    snapshot_path = "/home_shared/grail_sissi/BAKU/baku/exp_local/2026.01.16_train/deterministic/105856/snapshot/47000.pt" # working bread pick 2
    # snapshot_path = "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.12.25_train/deterministic/105414/snapshot/43000.pt" # working music box

    rollout = AgentRollout(cfg, demo_data_path, snapshot_path, save_log=True)
    rollout.run(duration_s=180, freq=50)


if __name__ == "__main__":
    main()