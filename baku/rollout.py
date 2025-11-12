#!/usr/bin/env python3
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os
from deoxys.utils import transform_utils
from scipy.spatial.transform import Rotation as R


# --- Franka imports ---
from frankateach.messages import FrankaAction
from frankateach.network import create_request_socket
from frankateach.constants import *

sys.path.append(os.path.expanduser("/home_shared/grail_sissi/BAKU/baku/vr-hand-tracking/Franka-Teach/RUKA"))
# --- Ruka imports ---
from ruka_hand.control.hand import Hand
from ruka_hand.utils.trajectory import move_to_pos

# --- Agent imports ---
from train import WorkspaceIL
from suite.custom import task_make_fn


class AgentRollout:
    def __init__(self, cfg, demo_data_path, snapshot_path, save_log=True):
        self.demo_path = Path(demo_data_path)
        self.snapshot_path = Path(snapshot_path)
        self.save_log = save_log

        # -----------------------------
        # Load demo data (for normalization + task_emb)
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
            }
        }

        # -----------------------------
        # Setup communication with Franka
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

        # -----------------------------
        # Initialize Ruka hand
        # -----------------------------
        self.hand = Hand(hand_type="right")
        self.curr_hand_pos = self.hand.read_pos()
        time.sleep(0.5)
        test_pos = self.hand.tensioned_pos
        move_to_pos(curr_pos=self.curr_hand_pos, des_pos=test_pos, hand=self.hand, traj_len=50)
        time.sleep(1)
        self.curr_hand_pos = self.hand.read_pos()
        print("Ruka hand initialized.")

        # -----------------------------
        # Logging
        # -----------------------------
        if self.save_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = Path(f"live_rollout_{timestamp}.pkl")
            self.logged_data = []
            print(f"Logging enabled: {self.log_path}")

    def get_arm_state(self):
        """Query current arm state from the Franka controller."""
        self.arm_socket.send(b"get_state")
        raw_state = pickle.loads(self.arm_socket.recv())
        return raw_state  # expected to include 'arm_states'

    def run(self, duration_s: float = 60.0, freq: float = 50.0):
        """Run live closed-loop rollout with teleop-like arm retargeting."""
        print("Starting live rollout...")
        dt = 1.0 / freq
        t0 = time.time()

        # -----------------------------
        # Initialize home pose (robot_init_H)
        # -----------------------------
        self.arm_socket.send(b"get_state")
        robot_state = pickle.loads(self.arm_socket.recv())
        self.home_rot, self.home_pos = transform_utils.quat2mat(robot_state.quat), robot_state.pos
        self.robot_init_H = np.eye(4)
        self.robot_init_H[:3, :3] = self.home_rot
        self.robot_init_H[:3, 3] = self.home_pos

        # -----------------------------
        # Initialize hand frame filter
        # -----------------------------
        use_filter = True
        comp_filter = Filter(np.concatenate([self.home_pos, transform_utils.mat2quat(self.home_rot)]), comp_ratio=0.8)

        # -----------------------------
        # Set initial hand frame
        # -----------------------------
        hand_init_H = None
        self.hand_frame_initialized = False

        try:
            while time.time() - t0 < duration_s:
                # -----------------------------
                # 1. Get current arm + hand states
                # -----------------------------
                arm_state = self.get_arm_state()["arm_states"]
                ruka_state = self.hand.read_pos()

                # -----------------------------
                # 2. Initialize hand frame on first step
                # -----------------------------
                if not self.hand_frame_initialized:
                    hand_init_H = np.eye(4)
                    hand_init_H[:3, :3] = np.eye(3)  # or set to initial rotation if known
                    hand_init_H[:3, 3] = arm_state[:3]
                    self.hand_frame_initialized = True

                # -----------------------------
                # 3. Construct observation
                # -----------------------------
                obs = {
                    "features": np.concatenate([arm_state, ruka_state]).astype(np.float32),
                    "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
                    "task_emb": np.asarray(self.demo_data["task_emb"], dtype=np.float32),
                }

                # -----------------------------
                # 4. Predict next action
                # -----------------------------
                with torch.no_grad():
                    action = self.workspace.agent.act(
                        obs,
                        prompt=None,
                        norm_stats=self.norm_stats,
                        step=0,
                        global_step=self.workspace.global_step,
                        eval_mode=True,
                    )

                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                # -----------------------------
                # 5. Map network output to homogenous matrix
                # -----------------------------
                # Split arm + hand
                arm_action = action[:7]
                hand_action = action[7:]

                # Build moving hand frame
                moving_hand_H = np.eye(4)
                moving_hand_H[:3, 3] = arm_action[:3]
                moving_hand_H[:3, :3] = R.from_quat(arm_action[3:7]).as_matrix()

                # -----------------------------
                # 6. Compute robot frame transform
                # -----------------------------
                H_R = self._to_robot_frame(hand_init_H, moving_hand_H)
                relative_pos = H_R[:3, 3]
                relative_rot = H_R[:3, :3]

                # Optional filtering
                if use_filter:
                    filtered = comp_filter(np.concatenate([relative_pos, transform_utils.mat2quat(relative_rot)]))
                    relative_pos = filtered[:3]
                    relative_rot = R.from_quat(filtered[3:]).as_matrix()

                # -----------------------------
                # 7. Apply hand rotation compensation
                # -----------------------------
                hand_axes_mat = hand_init_H[:3, :3]
                angles = self._angles_around_axes(relative_rot, hand_axes_mat)
                hand_angles = angles.copy()
                hand_angles[2] = np.clip(hand_angles[2], -25, 25)
                hand_angles[0] = np.clip(hand_angles[0], -60, 0)
                hand_angles[1] = 0
                R_hand_limited = self._rot_from_hand_axes(hand_angles, hand_axes_mat)
                R_arm_comp = relative_rot @ R_hand_limited.T

                # -----------------------------
                # 8. Compute target arm pose
                # -----------------------------
                target_rot = self.home_rot @ R_arm_comp
                target_pos = self.home_pos + relative_pos
                target_quat = transform_utils.mat2quat(target_rot)

                # Clip to workspace limits
                target_pos = np.clip(target_pos, a_min=ROBOT_WORKSPACE_MIN, a_max=ROBOT_WORKSPACE_MAX)

                # -----------------------------
                # 9. Send arm command
                # -----------------------------
                franka_action = FrankaAction(
                    pos=target_pos.astype(np.float32),
                    quat=target_quat.astype(np.float32),
                    gripper=-1,
                    reset=False,
                    timestamp=time.time(),
                )
                self.arm_socket.send(pickle.dumps(franka_action, protocol=-1))
                _ = self.arm_socket.recv()

                # -----------------------------
                # 10. Send hand command
                # -----------------------------
                move_to_pos(curr_pos=self.curr_hand_pos, des_pos=hand_action, hand=self.hand, traj_len=25)
                self.curr_hand_pos = self.hand.read_pos()

                # -----------------------------
                # 11. Logging
                # -----------------------------
                if self.save_log:
                    self.logged_data.append({
                        "timestamp": time.time(),
                        "arm_state": arm_state,
                        "ruka_state": ruka_state,
                        "arm_action": arm_action,
                        "hand_action": hand_action,
                        "target_pos": target_pos,
                        "target_quat": target_quat,
                    })

                # Maintain frequency
                elapsed = time.time() - t0
                next_time = (len(self.logged_data) + 1) * dt
                time.sleep(max(0, next_time - elapsed))

        except KeyboardInterrupt:
            print("Rollout interrupted by user.")

        finally:
            # Close sockets and hand
            self.arm_socket.close()
            self.hand.close()
            print("Connections closed.")

            if self.save_log and self.logged_data:
                with open(self.log_path, "wb") as f:
                    pickle.dump(self.logged_data, f)
                print(f"Saved rollout log to {self.log_path}")


# ------------------------------------------------------------
# Example Hydra-compatible entry point
# ------------------------------------------------------------
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="/home_shared/grail_sissi/BAKU/baku/cfgs", config_name="config")
def main(cfg: DictConfig):
    demo_data_path = "/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl"
    snapshot_path = "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.12_train/deterministic/142203/snapshot/9000.pt"

    rollout = AgentRollout(cfg, demo_data_path, snapshot_path, save_log=True)
    rollout.run(duration_s=180, freq=50)  # run for 3 minutes at 50 Hz


if __name__ == "__main__":
    main()
