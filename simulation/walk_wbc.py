"""
Walking: humanoid-loco LIPM + ZMP-preview plan  ->  whole-body QP controller on the full 32-dof MuJoCo humanoid.

The repo's planner supplies (a) the CoM x/y trajectory, (b) swing-foot 3-D trajectories, (c) the contact schedule.
The WBC tracks them with strict priority:  level 1 = CoM + torso attitude (+ contact dynamics/friction/torque limits),
level 2 = swing-foot pose, level 3 = posture (arms hold).  Contact set switches single/double support per the plan.
  python -m simulation.walk_wbc [--actuators repo|strong] [--steps 8]
Requires the humanoid-loco repo root on PYTHONPATH (planning/, dynamics/, trajectory/ ...).
"""
import os
import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, sys, time
import numpy as np, mujoco
from model.humanoid_model import ModelParams, repo_actuator_params, joint_speed_limits, ACTUATED
from simulation.wbc_sim import WBCSim
from planning.footstep_planner import GaitParams, foot_target_trajectories
from dynamics.lipm import LIPMParams
from simulation.walk_simulator import simulate_walk


def build_plan(n_steps=8, zc=0.85, step_length=0.3):
    g = GaitParams(n_steps=n_steps, step_length_m=step_length)
    w = simulate_walk(g, LIPMParams(com_height_m=zc, dt=g.dt))
    t, L, R = foot_target_trajectories(g, w.footsteps)
    n = min(len(t), len(w.com_x))
    t = t[:n]
    com = np.stack([w.com_x[:n], w.com_y[:n]], 1)
    vel = np.gradient(com, t, axis=0); acc = np.gradient(vel, t, axis=0)
    fv = {k: np.gradient(v[:n], t, axis=0) for k, v in (("L", L), ("R", R))}
    fa = {k: np.gradient(fv[k], t, axis=0) for k in "LR"}
    return dict(g=g, w=w, t=t, com=com, com_v=vel, com_a=acc, foot={"L": L[:n], "R": R[:n]}, foot_v=fv, foot_a=fa,
                steps=w.footsteps)


def swing_side(plan, tk):
    for s in plan["steps"]:
        if s.start_time <= tk < s.end_time:
            return "L" if s.side == "left" else "R"
    return None


def run(actuators="repo", n_steps=8, zc=0.85, verbose=True, max_time=None, estimated=False):
    p = repo_actuator_params() if actuators == "repo" else ModelParams()
    # leg flexion that gives CoM height ~= zc (linear fit of the model: bend 0.3 -> 0.906 m, 0.5 -> 0.848 m)
    bend = float(np.clip(0.3 + (0.906 - zc) / (0.906 - 0.848) * 0.2, 0.1, 0.7))
    S = WBCSim(table=False, params=p, leg_bend=bend)
    w = S.wbc
    est = None
    if estimated:
        from simulation.estimated_state import EstimatedState
        est = EstimatedState(S, seed=0, grav_sig=8.0)
    state_fn = (lambda: est.state()) if est else None
    plan = build_plan(n_steps, zc)
    S.wbc.set_state(*S.robot_state())
    com0 = w.com().copy()
    sole0 = {k: S.d.site_xpos[w.sole[k]].copy() for k in "LR"}
    off_foot = np.array([sole0["L"][0], 0.0, 0.0])
    T = plan["t"][-1] if max_time is None else max_time
    n = int(T / S.m.opt.timestep)
    qa, da = S.qadr, S.dadr
    speed_lim = joint_speed_limits()
    peak_v = np.zeros(30); peak_tau = np.zeros(30)
    touchdowns = []; prev_swing = None
    com_err = []; tkeep = []
    t0 = time.time(); fell = False
    for i in range(n):
        tk = i * S.m.opt.timestep
        sw = swing_side(plan, tk)
        contacts = tuple(k for k in "LR" if k != sw)
        # (landing bookkeeping: foot error at the instant a swing phase ends)
        if prev_swing is not None and sw != prev_swing:
            st = [s for s in plan["steps"] if abs(s.end_time - tk) < 0.02]
            if st:
                side = "L" if st[0].side == "left" else "R"
                tgt = np.array([st[0].x, st[0].y, 0.0]) + off_foot
                touchdowns.append(float(np.linalg.norm(S.d.site_xpos[w.sole[side]][:2] - tgt[:2])))
        prev_swing = sw
        idx = lambda arr: np.array([np.interp(tk, plan["t"], arr[:, j]) for j in range(arr.shape[1])])
        c_ref = idx(plan["com"]); cv = idx(plan["com_v"]); ca = idx(plan["com_a"])
        com_des = np.array([com0[0] + c_ref[0], com0[1] + c_ref[1], com0[2]])
        def tasks(wc):
            ts = [wc.task_com(com_des, vcom_des=np.array([cv[0], cv[1], 0.0]), acc_ff=np.array([ca[0], ca[1], 0.0])),
                  wc.task_torso_orient()]
            if sw is not None:
                fp = idx(plan["foot"][sw]) + off_foot
                fp[2] = max(fp[2], 0.0)
                ts.append(wc.task_foot(sw, fp, vel_des=idx(plan["foot_v"][sw]), acc_ff=idx(plan["foot_a"][sw]), level=2))
            ts.append(wc.task_posture(level=3))
            return ts
        if est:
            est.step({k: (k in contacts) for k in "LR"})
        S.step(tasks, contacts=contacts, state_fn=state_fn)
        if i % 10 == 0:
            S.record(tk)
            w.set_state(*S.robot_state())
            com_err.append(np.linalg.norm(w.com()[:2] - com_des[:2]))
            tkeep.append(tk)
        peak_v = np.maximum(peak_v, np.abs(S.d.qvel[da]))
        peak_tau = np.maximum(peak_tau, np.abs(S.tau))
        if S.fell():
            fell = True
            if verbose: print(f"FELL at t={tk:.2f}s", flush=True)
            break
        if verbose and i % 1000 == 0:
            print(f"t={tk:.1f}s x={S.d.qpos[0]:.2f} com_err={com_err[-1]*1000 if com_err else 0:.0f}mm swing={sw}", flush=True)
    names = ACTUATED
    over_speed = {names[j]: float(peak_v[j] / speed_lim[j]) for j in range(30) if peak_v[j] > speed_lim[j]}
    res = dict(estimated_state_in_loop=estimated, actuators=actuators, com_height_m=zc, leg_bend_rad=bend, steps_planned=n_steps, fell=fell, sim_time_s=float(tk), wall_s=time.time() - t0,
               distance_m=float(S.d.qpos[0] - 0.0), plan_distance_m=float(plan["com"][-1, 0]),
               com_track_rms_mm=1000 * float(np.sqrt(np.mean(np.square(com_err)))) if com_err else None,
               com_track_max_mm=1000 * float(np.max(com_err)) if com_err else None,
               touchdown_err_mm_mean=1000 * float(np.mean(touchdowns)) if touchdowns else None,
               touchdown_err_mm_max=1000 * float(np.max(touchdowns)) if touchdowns else None,
               touchdowns=len(touchdowns),
               min_support_margin_mm=1000 * min(S.log["support_margin"]) if S.log["support_margin"] else None,
               torque_peak_vs_limit={names[j]: float(peak_tau[j] / w.tau_max[j]) for j in np.argsort(-peak_tau / w.tau_max)[:5]},
               joints_over_speed_limit=over_speed, speed_limit_rad_s=float(speed_lim[0]),
               peak_leg_speed_rad_s=float(peak_v[:12].max()))
    return res, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actuators", default="repo", choices=["repo", "strong"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--zc", type=float, default=0.85)
    ap.add_argument("--estimated", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-time", type=float, default=None)
    a = ap.parse_args()
    r, S = run(a.actuators, a.steps, zc=a.zc, max_time=a.max_time, estimated=a.estimated)
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
