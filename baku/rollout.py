#!/usr/bin/env python3
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# --- Franka imports ---
from frankateach.messages import FrankaAction
from frankateach.network import create_request_socket
from frankateach.constants import LOCALHOST, CONTROL_PORT

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

        # -----------------------------
        # Initialize Ruka hand
        # -----------------------------
        self.hand = Hand(hand_type="right")
        self.curr_hand_pos = self.hand.read_pos()
        time.sleep(0.5)
        move_to_pos(curr_pos=self.curr_hand_pos, des_pos=self.hand.tensioned_pos, hand=self.hand, traj_len=50)
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
        """Query current arm state from Franka."""
        self.arm_socket.send(b"get_state")
        raw_state = pickle.loads(self.arm_socket.recv())
        # Concatenate pos+quat to match agent input
        arm_state = np.concatenate([raw_state.pos, raw_state.quat]).astype(np.float32)
        return arm_state

    def run(self, duration_s: float = 60.0, freq: float = 50.0):
        print("Starting live rollout...")
        dt = 1.0 / freq
        t0 = time.time()

        try:
            while time.time() - t0 < duration_s:
                # 1. Get current arm + hand states
                arm_state = self.get_arm_state()
                ruka_state = self.hand.read_pos()
                print(arm_state)
                break

                # 2. Construct agent observation
                obs = {
                    "features": np.concatenate([arm_state, ruka_state]).astype(np.float32),
                    "pixels0": np.zeros((3, 84, 84), dtype=np.uint8),
                    "task_emb": np.asarray(self.demo_data["task_emb"], dtype=np.float32),
                }

                # 3. Predict next action
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

                # 4. Split into arm + hand commands
                arm_action = action[:7]  # pos(3) + quat(4)
                hand_action = action[7:]

                # 5. Send arm command directly
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
                move_to_pos(curr_pos=self.curr_hand_pos, des_pos=hand_action, hand=self.hand, traj_len=25)
                self.curr_hand_pos = self.hand.read_pos()

                # 7. Logging
                if self.save_log:
                    self.logged_data.append({
                        "timestamp": time.time(),
                        "arm_state": arm_state,
                        "ruka_state": ruka_state,
                        "arm_action": arm_action,
                        "hand_action": hand_action,
                    })

                # Maintain loop frequency
                elapsed = time.time() - t0
                next_time = (len(self.logged_data) + 1) * dt
                time.sleep(max(0, next_time - elapsed))

        except KeyboardInterrupt:
            print("Rollout interrupted by user.")

        finally:
            self.arm_socket.close()
            self.hand.close()
            print("Connections closed.")
            if self.save_log and self.logged_data:
                with open(self.log_path, "wb") as f:
                    pickle.dump(self.logged_data, f)
                print(f"Saved rollout log to {self.log_path}")


# ------------------------------------------------------------
# Hydra entry point
# ------------------------------------------------------------
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="/home_shared/grail_sissi/BAKU/baku/cfgs", config_name="config")
def main(cfg: DictConfig):
    demo_data_path = "/home_shared/grail_sissi/BAKU/processed_data_pkl/demo_task.pkl"
    snapshot_path = "/home_shared/grail_sissi/BAKU/baku/exp_local/2025.11.12_train/deterministic/142203/snapshot/9000.pt"

    rollout = AgentRollout(cfg, demo_data_path, snapshot_path, save_log=True)
    rollout.run(duration_s=180, freq=50)


if __name__ == "__main__":
    main()
