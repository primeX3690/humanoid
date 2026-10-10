import numpy as np
import pytest
from kinematics.arm_kinematics import Arm, ArmParams
from manipulation.arm_collision import (ArmCollisionChecker, Sphere, Box, Capsule, seg_seg_distance, arm_capsules, point_box_sdf)
from manipulation.motion_planner import plan, JointSpace, path_length, trajectory

AL, AR = Arm(ArmParams(side="L")), Arm(ArmParams(side="R"))
REST_R = np.array([0.0, -0.12, 0.0, -0.35, 0.0, 0.0, 0.0])


def test_seg_seg_and_box_distance_basics():
    d = seg_seg_distance(np.array([0, 0, 0.]), np.array([1, 0, 0.]), np.array([0.5, 1, 0.]), np.array([0.5, 2, 0.]))
    assert d == pytest.approx(1.0)
    assert seg_seg_distance(np.array([0, 0, 0.]), np.array([1, 0, 0.]), np.array([0, 1, 0.]), np.array([1, 1, 0.])) == pytest.approx(1.0)
    assert point_box_sdf([2, 0, 0], np.zeros(3), np.ones(3)) == pytest.approx(1.0)
    assert point_box_sdf([0.5, 0, 0], np.zeros(3), np.ones(3)) < 0


def test_rest_pose_is_collision_free_and_arm_through_torso_is_not():
    ck = ArmCollisionChecker({"L": AL, "R": AR})
    assert not ck.in_collision({"R": REST_R})
    q_in = np.array([0.0, 1.0, 0.0, -2.0, 0, 0, 0])            # roll the right arm into the torso
    qs = [np.array([0.0, r, 0.0, -2.0, 0, 0, 0]) for r in np.linspace(0.0, 0.3, 4)]
    assert any(ck.in_collision({"R": q}) for q in qs) or any(ck.in_collision({"R": q}) for q in [np.array([-0.3, 0.3, 0.0, -2.4, 0, 0, 0])])


def test_arm_vs_obstacle_clearance_matches_geometry():
    q = np.array([-1.2, -0.1, 0.0, -0.6, 0.0, 0.0, 0.0]); caps = arm_capsules(AR, q)
    obs = Sphere(caps[2].b + np.array([0.0, 0.0, 0.0]), 0.05)          # sphere centred on the TCP
    ck = ArmCollisionChecker({"R": AR}, [obs]); assert ck.in_collision({"R": q})
    far = Sphere(caps[2].b + np.array([0.0, 0.0, 0.5]), 0.05)
    from manipulation.arm_collision import capsule_obstacle_distance
    assert min(capsule_obstacle_distance(c, far) for c in caps) > 0.2
    assert capsule_obstacle_distance(caps[2], far) == pytest.approx(0.5 - 0.05 - 0.07, abs=1e-6)


def _reach_pair():
    q0 = np.array([-0.2, -0.15, 0.0, -0.5, 0.0, 0.0, 0.0])
    q1, ok, err = AR.ik(_pose(AR, q0, [0.18, -0.20, 0.0]), q0, pos_only=True)
    return q0, q1


def _pose(arm, q, delta):
    T = arm.fk(q).copy(); T[:3, 3] += np.asarray(delta); return T


def test_planner_finds_collision_free_path_around_obstacle_blocking_straight_line():
    q0 = np.array([-0.2, -0.15, 0.0, -0.5, 0.0, 0.0, 0.0]); q1 = np.array([-1.3, -0.15, 0.0, -0.5, 0.0, 0.0, 0.0])
    mid = 0.5 * (arm_capsules(AR, q0)[2].b + arm_capsules(AR, q1)[2].b)
    ck = ArmCollisionChecker({"R": AR}, [Sphere(mid, 0.07)])
    sp = JointSpace(ck, "R")
    assert not sp.edge_free(q0, q1)                                   # straight joint-space line collides
    path, ok = plan(ck, "R", q0, q1, seed=3)
    assert ok and np.allclose(path[0], q0) and np.allclose(path[-1], q1)
    assert all(sp.free(path[i] + (path[i + 1] - path[i]) * s) for i in range(len(path) - 1) for s in np.linspace(0, 1, 15))
    assert np.all(path >= AR.q_lo - 1e-9) and np.all(path <= AR.q_hi + 1e-9)


def test_start_in_collision_returns_failure_and_free_space_is_direct():
    q = np.array([-1.2, -0.1, 0.0, -0.6, 0.0, 0.0, 0.0])
    ck = ArmCollisionChecker({"R": AR}, [Sphere(arm_capsules(AR, q)[2].b, 0.05)])
    assert plan(ck, "R", q, REST_R)[1] is False
    free = ArmCollisionChecker({"R": AR}); p, ok = plan(free, "R", REST_R, q)
    assert ok and len(p) == 2


def test_trajectory_respects_speed_limits_and_endpoints():
    path = np.array([REST_R, REST_R + 0.4, REST_R + 0.1])
    t, Q, V = trajectory(path, qd_max=1.5)
    assert np.abs(V).max() <= 1.5 + 1e-6 and np.allclose(Q[-1], path[-1]) and np.allclose(V[-1], 0, atol=1e-6)
    assert np.all(np.diff(t) > 0)
