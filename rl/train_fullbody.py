"""PPO training for the full-body humanoid (curriculum: stand -> push-robust stand -> walk -> walk + pushes + full randomisation).

    python -m rl.train_fullbody --stage stand  --steps 2_000_000
    python -m rl.train_fullbody --stage walk   --steps 20_000_000 --resume runs/fullbody/stand
Needs: gymnasium, stable-baselines3, torch, MuJoCo. FIRST run `python -m rl.fullbody_env --selfcheck` (zero-action PD standing); if the robot
falls there, tune CLASS_GAINS in rl/fullbody_env.py before spending GPU hours. NOT run in the sandbox this was written in.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np

STAGES = {   # kwargs for FullBodyEnv
    "stand": dict(cmd_vx=(0, 0), cmd_vy=(0, 0), cmd_yaw=(0, 0), push_force=0.0, randomize=False),
    "push": dict(cmd_vx=(0, 0), cmd_vy=(0, 0), cmd_yaw=(0, 0), push_force=60.0, randomize=True),
    "walk": dict(cmd_vx=(0.0, 0.4), cmd_vy=(-0.1, 0.1), cmd_yaw=(-0.2, 0.2), push_force=0.0, randomize=True),
    "robust": dict(cmd_vx=(0.0, 0.5), cmd_vy=(-0.15, 0.15), cmd_yaw=(-0.3, 0.3), push_force=80.0, randomize=True),
}


def make_env_fn(stage, seed):
    def _init():
        from rl.fullbody_env import FullBodyEnv
        return FullBodyEnv(seed=seed, **STAGES[stage])
    return _init


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="stand", choices=list(STAGES)); ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", default=None); ap.add_argument("--out", default="runs/fullbody")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    env = VecNormalize(SubprocVecEnv([make_env_fn(a.stage, a.seed + i) for i in range(a.envs)]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    if a.resume:
        model = PPO.load(os.path.join(a.resume, "model"), env=env); env = VecNormalize.load(os.path.join(a.resume, "vecnorm.pkl"), env.venv); model.set_env(env)
    else:
        model = PPO("MlpPolicy", env, n_steps=256, batch_size=2048, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.005, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256, 128]), seed=a.seed, verbose=1)
    model.learn(total_timesteps=a.steps)
    out = os.path.join(a.out, a.stage); os.makedirs(out, exist_ok=True); model.save(os.path.join(out, "model")); env.save(os.path.join(out, "vecnorm.pkl"))
    json.dump(dict(stage=a.stage, steps=a.steps, seed=a.seed), open(os.path.join(out, "meta.json"), "w")); print("saved", out)


if __name__ == "__main__":
    main()
