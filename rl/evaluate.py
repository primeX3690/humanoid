"""
rl/evaluate.py

Evaluates a trained PPO policy (rl/train_ppo.py) against the zero-action
PD-hold baseline, on the same seeds, and reports the metrics that matter
to an investor or a grant reviewer - not a training-reward curve, but:

  * survival rate  : fraction of episodes that finish without a fall
  * mean survival  : mean episode length in seconds (episode = 20 s)
  * velocity error : |forward velocity - commanded velocity| while alive
  * push robustness: survival rate as a function of push magnitude
  * cost of transport (CoT = mean electrical-free mechanical power /
                       (m * g * v)), the standard locomotion-efficiency metric

Usage:
    python -m rl.evaluate --run runs/robust --episodes 20 --out results/rl_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch

from rl.biped_env import BipedEnv
from dynamics.leg_dynamics import RobotMassParams

torch.set_num_threads(1)   # match rl/train_ppo.py: without this, multi-threaded matmul
                            # reduction order is non-deterministic even with deterministic=True
                            # and fixed env seeds - caught because two runs of this exact script
                            # on the same policy reported v=0.067 then v=0.191 for cmd=0.3,push=0.0
                            # (see docs/BUGS_FOUND.md)

EPISODE_STEPS = 1000


@dataclass
class EvalResult:
    label: str
    push_velocity: float
    cmd_vx: float
    episodes: int
    survival_rate: float
    mean_survival_s: float
    mean_vx_error: float
    mean_forward_speed: float
    cost_of_transport: float | None


def _load_policy(run_dir: str):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    dummy = DummyVecEnv([lambda: BipedEnv()])
    vn = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), dummy)
    vn.training = False
    vn.norm_reward = False
    model = PPO.load(os.path.join(run_dir, "model.zip"), device="cpu")

    def act(obs: np.ndarray) -> np.ndarray:
        o = vn.normalize_obs(obs[None, :])
        a, _ = model.predict(o, deterministic=True)
        return a[0]
    return act


def rollout(policy, push_velocity: float, cmd_vx: float, episodes: int, seed0: int = 1000,
            label: str = "") -> EvalResult:
    env = BipedEnv(cmd_vx_range=(cmd_vx, cmd_vx), push_velocity=push_velocity,
                   push_interval_s=(1.5, 3.0), max_episode_steps=EPISODE_STEPS)
    mass = RobotMassParams().total_mass_kg
    survived, lengths, verr, speeds, powers = 0, [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        for k in range(EPISODE_STEPS):
            action = policy(obs) if policy is not None else np.zeros(env.action_space.shape[0])
            obs, _, terminated, truncated, info = env.step(action)
            verr.append(abs(info["vx"] - cmd_vx))
            speeds.append(info["vx"])
            qd = env.data.qvel[6:18]
            powers.append(float(np.sum(np.abs(env.data.actuator_force * qd))))
            if terminated or truncated:
                break
        survived += int(not terminated)
        lengths.append((k + 1) * env.dt)
    mean_speed = float(np.mean(speeds))
    cot = None
    if mean_speed > 0.05:
        cot = float(np.mean(powers) / (mass * 9.81 * mean_speed))
    return EvalResult(label=label, push_velocity=push_velocity, cmd_vx=cmd_vx, episodes=episodes,
                      survival_rate=survived / episodes, mean_survival_s=float(np.mean(lengths)),
                      mean_vx_error=float(np.mean(verr)), mean_forward_speed=mean_speed,
                      cost_of_transport=cot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir with model.zip + vecnormalize.pkl")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="results/rl_eval.json")
    args = ap.parse_args()

    policy = _load_policy(args.run)
    results = []
    for cmd in (0.0, 0.3):
        for push in (0.0, 0.5, 1.0):
            for label, pol in (("PD-hold baseline", None), ("PPO policy", policy)):
                r = rollout(pol, push, cmd, args.episodes, label=label)
                results.append(r)
                print(f"{label:17s} cmd={cmd:.1f} push={push:.1f}  survive={r.survival_rate:5.0%} "
                      f"t={r.mean_survival_s:5.1f}s  vx_err={r.mean_vx_error:.3f}  "
                      f"v={r.mean_forward_speed:.3f}  CoT={r.cost_of_transport}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
