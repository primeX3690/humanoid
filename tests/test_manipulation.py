import numpy as np
import pytest
from kinematics.arm_kinematics import Arm, ArmParams
from manipulation.grasp_planner import GripperSpec, box_grasps, cylinder_grasps, rank_by_reachability, frame_from, force_closure_2contact
from manipulation.admittance import Admittance, ForceRegulator
from manipulation.loco_manipulation import payload_com_shift, max_static_payload, static_joint_torque, bimanual_box_ik

AR, AL = Arm(ArmParams(side="R")), Arm(ArmParams(side="L"))
G8 = GripperSpec()


def test_cube_gets_grasps_wide_object_does_not():
    cube = box_grasps([0.3, -0.2, 0.12], np.eye(3), np.array([0.025, 0.025, 0.025]), G8, table_z=0.095)
    assert len(cube) >= 4 and all(abs(g.width - 0.05) < 1e-9 for g in cube)
    assert box_grasps([0.3, -0.2, 0.12], np.eye(3), np.array([0.06, 0.06, 0.06]), G8) == []          # 12 cm > 8 cm opening
    slab = box_grasps([0.3, -0.2, 0.12], np.eye(3), np.array([0.07, 0.015, 0.02]), G8)               # only the 3 cm side fits
    assert slab and {g.parts["axis"] for g in slab} == {1, 2}                                       # the 3 cm and 4 cm sides fit, the 14 cm side does not


def test_tcp_frame_is_orthonormal_and_z_opposes_approach():
    T = frame_from(np.array([0, 0, -1.0]), np.array([1, 0, 0.0]), np.array([0.3, 0, 0.1]))
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3)) and np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(T[:3, 2], [0, 0, 1.0]) and np.allclose(T[:3, 1], [1, 0, 0])


def test_table_blocks_low_grasps_and_topdown_ranks_first_for_cube():
    low = box_grasps([0.3, -0.2, 0.10], np.eye(3), np.array([0.025] * 3), G8, table_z=0.095)           # jaw tips would touch the table
    assert low == []
    gs = box_grasps([0.3, -0.2, 0.12], np.eye(3), np.array([0.025] * 3), G8, table_z=0.095)
    best = max(gs, key=lambda g: g.score); assert best.approach[2] == pytest.approx(-1.0)


def test_cylinder_grasps_and_force_closure():
    g = cylinder_grasps([0.3, -0.2, 0.12], 0.03, 0.12, G8)
    assert len(g) == 16 and cylinder_grasps([0.3, -0.2, 0.12], 0.05, 0.12, G8) == []
    p1, p2 = np.array([-0.025, 0, 0.]), np.array([0.025, 0, 0.])
    assert force_closure_2contact(np.array([1, 0, 0.]), np.array([-1, 0, 0.]), p1, p2, 0.5)
    assert not force_closure_2contact(np.array([0, 1, 0.]), np.array([0, -1, 0.]), p1, p2, 0.5)


def test_reachability_ranking_returns_ik_solutions_that_reach_the_grasp():
    gs = box_grasps([0.28, -0.22, 0.12], np.eye(3), np.array([0.025] * 3), G8, table_z=0.095)
    q0 = np.array([-0.4, -0.15, 0.0, -0.8, 0.0, 0.0, 0.0])
    ranked = rank_by_reachability(gs, AR, q0)
    assert len(ranked) >= 1
    for r in ranked[:3]:
        assert np.linalg.norm(AR.fk(r.parts["q_grasp"])[:3, 3] - r.T[:3, 3]) < 2e-3 and r.parts["manip"] > 0.01
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))


def test_admittance_steady_state_and_force_regulator_converges_on_a_spring_wall():
    a = Admittance(M=2, D=60, K=200, dt=0.005)
    for _ in range(600): x, v = a.step([10.0, 0, 0])
    assert x[0] == pytest.approx(10.0 / 200, rel=0.02) and abs(v[0]) < 1e-3
    kw, z_wall, dt = 2000.0, 0.0, 0.005                     # stiff spring wall at z=0, hand approaches from above
    reg = ForceRegulator([0, 0, 1.0], dt=dt); z = 0.01; z_ref = 0.0; f_hist = []
    for k in range(800):
        F = np.array([0, 0, kw * max(z_wall - z, 0.0)])
        disp = reg.step(F, 10.0); z_ref = 0.01 + disp[2]
        z += 0.5 * (z_ref - z)                              # inner position loop: fast first-order tracking
        f_hist.append(F[2])
    assert abs(f_hist[-1] - 10.0) < 0.2 and max(f_hist) < 14.0   # settles on 10 N, overshoot < 40 %


def test_payload_com_shift_and_max_payload_vs_arm_extension():
    new, d = payload_com_shift(25.0, [0, 0, 0.9], 5.0, [0.4, 0, 1.0])
    assert d[0] == pytest.approx(5 * 0.4 / 30) and d[2] == pytest.approx(5 * 0.1 / 30)
    tucked = np.array([0.0, -0.12, 0.0, -2.0, 0.0, 0.0, 0.0]); out = np.array([-1.57, -0.12, 0.0, -0.05, 0.0, 0.0, 0.0])
    lim = np.array([25.3, 25.3, 25.3, 25.3, 8.4, 8.4, 8.4])
    mt, mo = max_static_payload(AR, tucked, lim), max_static_payload(AR, out, lim)
    assert mo < mt and 0.2 < mo < mt < 20.0
    tau = static_joint_torque(AR, out, np.array([0, 0, -mo * 9.81])); assert np.max(np.abs(tau) / lim) == pytest.approx(1.0, abs=0.02)


def test_bimanual_box_carry_ik_keeps_both_hands_on_the_box():
    Tb = np.eye(4); Tb[:3, 3] = [0.30, 0.0, 0.05]
    qL0 = np.array([-0.9, 0.15, 0.0, -1.2, 0, 0, 0]); qR0 = np.array([-0.9, -0.15, 0.0, -1.2, 0, 0, 0])
    qL, qR, ok, err, poses = bimanual_box_ik({"L": AL, "R": AR}, Tb, 0.12, qL0, qR0)
    assert ok and err < 1e-3
    pL, pR = AL.fk(qL)[:3, 3], AR.fk(qR)[:3, 3]
    assert np.linalg.norm(pL - pR) == pytest.approx(0.24, abs=2e-3)
