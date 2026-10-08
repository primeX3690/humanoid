import numpy as np, mujoco, casadi as ca
from planning.centroidal_jump_to import JumpOptimizer, JumpConfig, leg_torques


def test_planar_leg_torque_matches_mujoco_standing():
    from model.humanoid_model import make_model, stand_pose, ACTUATED
    from control.whole_body_qp import WholeBodyController
    m, d = make_model(table=False); stand_pose(m, d)
    w = WholeBodyController(); q, qd = d.qpos.copy(), np.zeros(m.nv)
    w.set_state(q, qd); c = w.com(); cd = np.array([w.support_center(("L", "R"))[0], 0, c[2]])
    out = w.solve(q, qd, [w.task_com(cd), w.task_torso_orient(), w.task_posture()])
    knee_mj = out["tau"][ACTUATED.index("L_knee")]
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    hip, ank = d.xpos[bid("L_thigh")], d.xpos[bid("L_foot")]
    f = out["f"].reshape(-1, 3).sum(0) / 2
    _, tk = leg_torques(hip[0] - ank[0], hip[2] - ank[2], f[0], f[2], 0.45)
    assert abs(float(tk) - knee_mj) < 0.15 * abs(knee_mj)       # within 15 % (measured: ~6 %)


def test_jump_is_dynamically_consistent_and_within_limits():
    o = JumpOptimizer(); r = o.solve(0.4, T_flight_min=0.3)
    assert r["ok"]
    v = o.verify(r)                                  # independent NumPy re-integration
    assert abs(v["final_err_x"]) < 1e-3 and abs(v["final_err_z"]) < 1e-3 and v["final_speed"] < 1e-3
    assert v["max_fric_ratio"] <= o.c.mu + 1e-3
    assert v["peak_knee_torque_Nm_both_legs"] <= o.c.tau_knee + 1.0
    assert v["flight_time"] >= 0.3 - 1e-6 and v["apex_gain_m"] > 0.05


def test_demanding_jump_hits_the_actuator_limit():
    """With the torque limit tightened, a high jump must become infeasible (the limit is really enforced)."""
    cfg = JumpConfig(tau_knee=100.0, tau_hip=100.0)
    o = JumpOptimizer(cfg); r = o.solve(0.0, T_flight_min=0.6)
    assert not r["ok"]
