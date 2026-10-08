"""Whole-body controller regression tests (MuJoCo torque control). Each runs a few simulated seconds."""
import numpy as np, mujoco, pytest
from simulation.wbc_sim import WBCSim, min_jerk
from model.humanoid_model import make_model, stand_pose, joint_qadr, ACTUATED


def _cd(S):
    S.wbc.set_state(*S.robot_state()); c = S.wbc.com(); sc = S.wbc.support_center(("L", "R"))
    return np.array([sc[0], sc[1], c[2]])


def test_model_mass_and_dof():
    m, d = make_model(table=False)
    assert m.nv == 6 + 30 + 2 and abs(m.body_subtreemass[1] - 25.0) < 0.1


def test_dynamics_consistency_of_qp_solution():
    """QP's qdd/tau/f must satisfy M qdd + h = S^T tau + Jc^T f (rows of non-gripper dofs)."""
    S = WBCSim(table=False); w = S.wbc
    q, qd = S.robot_state(); cd = _cd(S)
    out = w.solve(q, qd, [w.task_com(cd), w.task_torso_orient(), w.task_posture()])
    assert out["status"] == "ok"
    c = w._constraints(("L", "R"))
    x = np.concatenate([out["qdd"], out["tau"], out["f"]])
    nd = w.nv - len(w.grip_dofs)
    assert np.linalg.norm(c["Aeq"][:nd] @ x - c["beq"][:nd]) < 1e-3
    assert np.all(out["tau"] <= w.tau_max + 1e-6) and np.all(out["tau"] >= -w.tau_max - 1e-6)
    f = out["f"].reshape(-1, 3)
    assert np.all(f[:, 2] >= -1e-6)
    assert np.all(np.abs(f[:, 0]) <= w.mu * f[:, 2] + 1e-6)


def test_quiet_standing_holds_com():
    S = WBCSim(table=False); cd = _cd(S)
    for i in range(2000):
        S.step(lambda w: [w.task_com(cd), w.task_torso_orient(), w.task_posture(level=4)])
    S.wbc.set_state(*S.robot_state())
    assert not S.fell()
    assert np.linalg.norm(S.wbc.com()[:2] - cd[:2]) < 0.005


def test_reach_while_balancing_and_strict_priority():
    """Feasible reach -> small hand error. Infeasible reach -> hand error large, CoM stays put (balance has priority)."""
    S = WBCSim(table=False); cd = _cd(S); p0 = S.tcp_pos("R")
    for tgt, ok in ((p0 + np.array([0.12, 0.0, 0.10]), True), (np.array([1.2, -0.1, 0.9]), False)):
        S = WBCSim(table=False); cd = _cd(S); p0 = S.tcp_pos("R")
        for i in range(3500):
            t = i * 0.001; s, v = min_jerk(t, 2.5); pd = p0 + (tgt - p0) * s; vd = (tgt - p0) * v
            S.step(lambda w: [w.task_com(cd), w.task_torso_orient(), w.task_hand("R", pd, np.eye(3), vel_des=vd, level=2), w.task_posture(level=4)])
        S.wbc.set_state(*S.robot_state())
        err = np.linalg.norm(S.tcp_pos("R") - tgt)
        com_dev = np.linalg.norm(S.wbc.com()[:2] - cd[:2])
        assert not S.fell()
        assert com_dev < 0.02                       # balance never sacrificed
        if ok:
            assert err < 0.01
        else:
            assert err > 0.3                         # unreachable target is NOT magically reached by tipping over


def test_survives_moderate_push():
    S = WBCSim(table=False); cd = _cd(S)
    tid = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    for i in range(2500):
        t = i * 0.001
        S.ext_force = (tid, np.array([55.0, 0, 0]) if 0.5 <= t < 0.65 else np.zeros(3))
        S.step(lambda w: [w.task_com(cd), w.task_torso_orient(), w.task_posture(level=4)])
    S.wbc.set_state(*S.robot_state())
    assert not S.fell() and np.linalg.norm(S.wbc.com()[:2] - cd[:2]) < 0.01
