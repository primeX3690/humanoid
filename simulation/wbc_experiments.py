"""Whole-body-control experiments (MuJoCo, torque control). Run:  python -m simulation.wbc_experiments"""
import sys, time, json
import numpy as np, mujoco
from simulation.wbc_sim import WBCSim, min_jerk
from kinematics.arm_kinematics import Arm, ArmParams
from model.humanoid_model import joint_qadr, joint_dadr, ACTUATED, nominal_q_act


def com_des_of(S):
    S.wbc.set_state(*S.robot_state()); c0 = S.wbc.com(); sc = S.wbc.support_center(("L", "R"))
    return np.array([sc[0], sc[1], c0[2]])


def summarize(S, label, extra=""):
    L = S.log
    com = np.array(L["com"])
    return dict(label=label, fell=bool(S.fell()), min_support_margin_mm=1000 * min(L["support_margin"]),
                max_tau_ratio=max(L["tau_max_ratio"]), max_fric_ratio=float(np.nanmax(L["fric_ratio"])) if L["fric_ratio"] else None,
                solves_ok=int(sum(1 for s in L["status"] if s == "ok")), solves=len(L["status"]),
                com_final_xy=com[-1][:2].tolist(), extra=extra)


def exp_reach(target, T=3.0, hold=1.0, wbc_levels=True, gaze=None, label="reach"):
    S = WBCSim()
    cd = com_des_of(S); p0 = S.tcp_pos("R"); Rd = np.eye(3)
    n = int((T + hold) / 0.001)
    herr = []
    for i in range(n):
        t = i * 0.001; s, v = min_jerk(t, T); pd = p0 + (target - p0) * s; vd = (target - p0) * v
        def tasks(w):
            ts = [w.task_com(cd), w.task_torso_orient(), w.task_hand("R", pd, Rd, vel_des=vd, level=2)]
            if gaze is not None:
                ts.append(w.task_gaze(gaze, level=3))
            ts.append(w.task_posture(level=4))
            return ts
        S.step(tasks)
        if i % 20 == 0:
            S.record(t); herr.append(np.linalg.norm(S.tcp_pos("R") - pd))
        if S.fell():
            break
    r = summarize(S, label)
    r["hand_err_final_mm"] = 1000 * float(np.linalg.norm(S.tcp_pos("R") - target))
    r["com_xy_dev_max_mm"] = 1000 * float(np.max(np.linalg.norm(np.array(S.log["com"])[:, :2] - cd[:2], axis=1)))
    r["tracking_err_rms_mm"] = 1000 * float(np.sqrt(np.mean(np.square(herr))))
    return S, r


def exp_baseline_pd(target, T=3.0, hold=1.0):
    """Naive controller: arm-only IK (fixed torso) + joint-space PD on EVERYTHING (no WBC, no balance task)."""
    S = WBCSim()
    m, d = S.m, S.d
    qa, da = S.qadr, S.dadr
    arm = Arm(ArmParams(side="R"))
    # target in torso frame (torso at nominal pose)
    tb = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")].copy()
    Rtor = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")].reshape(3, 3).copy()
    Td = np.eye(4); Td[:3, 3] = Rtor.T @ (target - tb); Td[:3, :3] = Rtor.T @ np.eye(3)
    names = ACTUATED
    ridx = [names.index(f"R_{n}") for n in ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_yaw", "wr_pitch", "wr_roll"]]
    q_nom = nominal_q_act()
    q_goal, conv, err = arm.ik(Td, q_nom[ridx])
    q_start = q_nom.copy(); q_end = q_nom.copy(); q_end[ridx] = q_goal
    kp = np.array([300.0] * 30); kd = np.array([30.0] * 30)
    n = int((T + hold) / 0.001)
    wbc = S.wbc
    for i in range(n):
        t = i * 0.001; s, v = min_jerk(t, T)
        qd_des = q_start + (q_end - q_start) * s; vd = (q_end - q_start) * v
        q = d.qpos[qa]; qd = d.qvel[da]
        # gravity compensation via inverse dynamics bias on the true model (generous to the baseline)
        tau = kp * (qd_des - q) + kd * (vd - qd) + d.qfrc_bias[da]
        d.ctrl[:30] = np.clip(tau, -wbc.tau_max, wbc.tau_max); d.ctrl[30:32] = 0.004
        mujoco.mj_step(m, d)
        if i % 20 == 0:
            S.tau = d.ctrl[:30].copy(); S.record(t)
        if S.fell():
            break
    r = summarize(S, "PD baseline (no WBC)", extra=f"IK converged={conv}")
    r["hand_err_final_mm"] = 1000 * float(np.linalg.norm(S.tcp_pos("R") - target))
    r["com_xy_dev_max_mm"] = None
    return S, r


def exp_priority_infeasible():
    """Hand target outside reach -> hierarchy must sacrifice the hand, never balance."""
    tgt = np.array([1.10, -0.10, 0.90])
    S, r = exp_reach(tgt, T=3.0, hold=1.5, label="infeasible reach (1.1 m)")
    return S, r


def exp_push(F=45.0, dur=0.15, use_wbc=True):
    S = WBCSim()
    cd = com_des_of(S)
    torso_id = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    T = 3.0; n = int(T / 0.001)
    q_nom = nominal_q_act(); kp = 300.0; kd = 30.0
    for i in range(n):
        t = i * 0.001
        S.ext_force = (torso_id, np.array([F, 0.0, 0.0])) if 1.0 <= t < 1.0 + dur else (torso_id, np.zeros(3))
        if use_wbc:
            S.step(lambda w: [w.task_com(cd), w.task_torso_orient(), w.task_posture(level=4)])
        else:
            q = S.d.qpos[S.qadr]; qd = S.d.qvel[S.dadr]
            tau = kp * (q_nom - q) + kd * (-qd) + S.d.qfrc_bias[S.dadr]
            S.d.ctrl[:30] = np.clip(tau, -S.wbc.tau_max, S.wbc.tau_max); S.d.ctrl[30:32] = 0.004
            S.d.xfrc_applied[torso_id, :3] = S.ext_force[1]
            mujoco.mj_step(S.m, S.d); S.tau = S.d.ctrl[:30].copy()
        if i % 10 == 0:
            S.record(t)
        if S.fell():
            break
    com = np.array(S.log["com"])
    r = summarize(S, ("WBC" if use_wbc else "PD baseline") + f" push {F:.0f}N x {dur*1000:.0f}ms")
    r["com_peak_dev_mm"] = 1000 * float(np.max(np.linalg.norm(com[:, :2] - cd[:2], axis=1)))
    r["com_final_dev_mm"] = 1000 * float(np.linalg.norm(com[-1][:2] - cd[:2]))
    return S, r


if __name__ == "__main__":
    out = []
    tgt = np.array([0.38, -0.12, 0.88])
    for fn, kw in [(exp_reach, dict(target=tgt, gaze=np.array([0.42, -0.10, 0.82]), label="WBC reach+gaze"))]:
        t0 = time.time(); S, r = fn(**kw); r["wall_s"] = time.time() - t0; out.append(r); print(r); sys.stdout.flush()
    S, r = exp_baseline_pd(tgt); out.append(r); print(r); sys.stdout.flush()
    S, r = exp_priority_infeasible(); out.append(r); print(r); sys.stdout.flush()
    for F in (45.0, 90.0):
        for w in (True, False):
            S, r = exp_push(F=F, use_wbc=w); out.append(r); print(r); sys.stdout.flush()
    json.dump(out, open("results/wbc_experiments.json", "w"), indent=1)
