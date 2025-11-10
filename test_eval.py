#!/usr/bin/env python3
import pickle
import torch
import numpy as np
from pathlib import Path
from train import WorkspaceIL, make_agent  # reuse your training code
from dataloader import CustomTeleopBCDataset  # your dataset class

# --- CONFIG ---
SNAPSHOT_PATH = Path("exp_local/2025.11.10_train/deterministic/131852/snapshot/500.pt")  # your saved policy
PKL_FILE = Path("proc_data/default_scene/demo_task/demo_0.pkl")  # your single demo pickle
DEVICE = "cuda"  # or "cpu"

# --- LOAD DATASET ---
dataset = CustomTeleopBCDataset(PKL_FILE, action_repeat=1, history_len=1, temporal_agg=False)
demo_actions = []
for i in range(len(dataset.observations)):
    obs = dataset.observations[i]
    act = np.concatenate([obs["commanded_arm_states"], obs["commanded_ruka_states"]], axis=0)
    demo_actions.append(act.astype(np.float32))
demo_actions = np.stack(demo_actions)

# --- CREATE DUMMY ENV ---
class DummyStep:
    def __init__(self, obs):
        self.observation = obs
    def last(self): 
        return False

class DummyEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx = 0
        self._max_episode_len = len(dataset.observations)
    def reset(self):
        self.idx = 0
        obs = self.dataset.observations[self.idx]
        return DummyStep({"features": np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0)})
    def step(self, action):
        self.idx += 1
        if self.idx >= self._max_episode_len:
            done = True
            obs = self.dataset.observations[-1]
        else:
            done = False
            obs = self.dataset.observations[self.idx]
        ts = DummyStep({"features": np.concatenate([obs["arm_states"], obs["ruka_states"]], axis=0)})
        ts.last_flag = done
        ts.reward = 0
        ts.observation["goal_achieved"] = False
        return ts

# --- LOAD POLICY ---
env = DummyEnv(dataset)
obs_spec = {"features": (dataset._max_state_dim,)}
action_spec = type('ActionSpec', (), {"shape": (dataset._max_action_dim,)})()

cfg = type('Cfg', (), {})()  # minimal config
cfg.agent = type('AgentCfg', (), {"obs_shape": obs_spec, "action_shape": action_spec, "policy_head": "bc"})()
cfg.use_proprio = True
agent = make_agent(obs_spec, action_spec, cfg)
payload = torch.load(SNAPSHOT_PATH)
agent.load_snapshot(payload, eval=True)
agent.train(False)

# --- ROLLOUT & COMPARE ---
obs_step = env.reset()
diffs = []

for step_idx in range(len(demo_actions)):
    obs_tensor = {"features": torch.tensor(obs_step.observation["features"], dtype=torch.float32).unsqueeze(0).to(DEVICE)}
    with torch.no_grad():
        pred_action = agent.act(obs_tensor, prompt=None, stats=dataset.stats, step=step_idx, global_step=0, eval_mode=True)
    demo_action = demo_actions[step_idx]
    diff = pred_action.squeeze(0).cpu().numpy() - demo_action
    diffs.append(diff)
    print(f"Step {step_idx}: max diff = {np.max(np.abs(diff))}, mean diff = {np.mean(diff)}")
    obs_step = env.step(pred_action)

diffs = np.stack(diffs)
print("Overall max difference:", np.max(np.abs(diffs)))
print("Overall mean difference:", np.mean(np.abs(diffs)))
