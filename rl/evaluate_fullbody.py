"""
Evaluate a trained full-body policy (needs stable-baselines3 + MuJoCo):
    python -m rl.evaluate_fullbody --run runs/fullbody/walk --episodes 20 --stage walk
Reports fall rate, mean survival time, mean forward-velocity tracking error, mean tilt - the numbers that decide whether the next curriculum stage is worth starting.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def summarize(eps):
    """eps: list of dicts(fell, steps, vel_err (mean), tilt (mean)). Pure NumPy (unit-tested)."""
    n = len(eps)
    return dict(episodes=n, fall_rate=float(np.mean([e["fell"] for e in eps])), mean_survival_s=float(np.mean([e["steps"] for e in eps])) / 50.0,
                mean_vel_err_m_s=float(np.mean([e["vel_err"] for e in eps])), mean_tilt_rad=float(np.mean([e["tilt"] for e in eps])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--stage", default="walk"); ap.add_argument("--episodes", type=int, default=20); a = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.train_fullbody import STAGES
    from rl.fullbody_env import FullBodyEnv
    venv = DummyVecEnv([lambda: FullBodyEnv(seed=123, max_steps=500, **STAGES[a.stage])])
    vn = VecNormalize.load(os.path.join(a.run, "vecnorm.pkl"), venv); vn.training = False; vn.norm_reward = False
    model = PPO.load(os.path.join(a.run, "model")); env = venv.envs[0]; eps = []
    for _ in range(a.episodes):
        obs = vn.reset(); steps = 0; errs = []; tilts = []; fell = False
        for _ in range(500):
            act, _ = model.predict(obs, deterministic=True); obs, r, done, info = vn.step(act); steps += 1
            s = env.b.state(); errs.append(float(np.linalg.norm(s["lin_vel"][:2] - env.cmd[:2]))); tilts.append(s["tilt"])
            if done[0]: fell = bool(info[0].get("fell", False)); break
        eps.append(dict(fell=fell, steps=steps, vel_err=float(np.mean(errs)), tilt=float(np.mean(tilts))))
    out = summarize(eps); print(json.dumps(out, indent=1)); os.makedirs("results", exist_ok=True); json.dump(out, open(f"results/eval_{a.stage}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
