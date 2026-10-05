"""
rl/train_ppo.py

Trains a locomotion policy for the biped in rl/biped_env.py with PPO
(Schulman et al. 2017) using Stable-Baselines3 - the same algorithm and
library family used in most open legged-robot RL work. CPU-only.

Curriculum stages (run in order, each resuming from the previous):

  stand : velocity command fixed at 0, moderate pushes  -> learn to balance and recover
  walk  : forward-velocity commands in [0, 0.5] m/s, light pushes
  robust: same commands, larger pushes                  -> push-robust walking

Example (Windows/Linux laptop, 4 cores):

    python -m rl.train_ppo --stage stand  --timesteps 1000000 --n-envs 4 --out runs/stand
    python -m rl.train_ppo --stage walk   --timesteps 3000000 --n-envs 4 --out runs/walk \\
                           --init-from runs/stand
    python -m rl.train_ppo --stage robust --timesteps 2000000 --n-envs 4 --out runs/robust \\
                           --init-from runs/walk

Each run writes `model.zip` and `vecnormalize.pkl` (observation statistics,
required to run the policy later) into --out, plus periodic checkpoints.
"""
from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rl.biped_env import BipedEnv

STAGES = {
    "stand":  dict(cmd_vx_range=(0.0, 0.0), push_velocity=0.5),
    "walk":   dict(cmd_vx_range=(0.0, 0.5), push_velocity=0.3),
    "robust": dict(cmd_vx_range=(0.0, 0.5), push_velocity=0.8),
}


def make_env_fn(stage: str, seed: int):
    cfg = STAGES[stage]

    def _init():
        env = BipedEnv(cmd_vx_range=cfg["cmd_vx_range"], push_velocity=cfg["push_velocity"], seed=seed)
        return Monitor(env)   # records episode return / length -> rollout/ep_rew_mean, ep_len_mean
    return _init


def build_vec_env(stage: str, n_envs: int, seed: int):
    fns = [make_env_fn(stage, seed + i) for i in range(n_envs)]
    return SubprocVecEnv(fns) if n_envs > 1 and os.cpu_count() and os.cpu_count() > 1 else DummyVecEnv(fns)


def linear_schedule(initial: float):
    return lambda progress_remaining: progress_remaining * initial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), default="stand")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/stand")
    ap.add_argument("--init-from", default=None, help="directory of a previous run to resume from")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1)   # env stepping dominates; avoids oversubscribing small CPUs

    venv = build_vec_env(args.stage, args.n_envs, args.seed)
    if args.init_from:
        venv = VecNormalize.load(os.path.join(args.init_from, "vecnormalize.pkl"), venv)
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(os.path.join(args.init_from, "model.zip"), env=venv, device="cpu",
                         custom_objects={"learning_rate": linear_schedule(args.lr)})
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", venv, device="cpu", seed=args.seed, verbose=1,
            learning_rate=linear_schedule(args.lr),
            n_steps=max(256, 8192 // args.n_envs), batch_size=1024, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
            max_grad_norm=1.0, target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=torch.nn.Tanh, log_std_init=-1.0),
        )

    ckpt = CheckpointCallback(save_freq=max(1, 250_000 // args.n_envs), save_path=args.out,
                              name_prefix="ckpt")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, reset_num_timesteps=not args.init_from,
                progress_bar=False)
    model.save(os.path.join(args.out, "model.zip"))
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
