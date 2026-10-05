import numpy as np
import pytest
from planning.footstep_planner import GaitParams, plan_footsteps, plan_turning_footsteps, heading_profile
from dynamics.gain_scheduled_walk_simulator import simulate_walk_gain_scheduled
from dynamics.lipm import LIPMParams
from simulation.joint_trajectory_generator import generate_joint_trajectory

DEFAULT_ZC = LIPMParams().com_height_m


def test_zero_turn_rate_reproduces_plan_footsteps_exactly():
    gait = GaitParams(n_steps=6)
    a, b = plan_footsteps(gait), plan_turning_footsteps(gait, 0.0)
    for x, y in zip(a, b):
        assert x.x == pytest.approx(y.x) and x.y == pytest.approx(y.y)
        assert y.heading == 0.0


def test_turning_gait_traces_the_commanded_heading():
    gait = GaitParams(n_steps=6)
    fs = plan_turning_footsteps(gait, np.deg2rad(5))
    assert fs[-1].heading == pytest.approx(np.deg2rad(30))
    assert [s.heading for s in fs] == sorted(s.heading for s in fs)


def test_turning_within_joint_limits_converges_ik():
    """Real, physical limit: leg_ik.py's own hip_yaw joint limit is
    +-0.6 rad (~34 deg) - a turn that stays within it must converge."""
    gait = GaitParams(n_steps=6)
    fs = plan_turning_footsteps(gait, np.deg2rad(5))  # 30 deg total, within limit
    walk = simulate_walk_gain_scheduled(gait, fs, DEFAULT_ZC)
    jt = generate_joint_trajectory(walk, gait)
    assert np.all(jt.left_ik_converged) and np.all(jt.right_ik_converged)
    assert np.max(np.abs(jt.left_joint_angles[:, 0])) <= 0.6 + 1e-9


def test_turning_beyond_hip_yaw_limit_honestly_fails_not_silently():
    gait = GaitParams(n_steps=6)
    fs = plan_turning_footsteps(gait, np.deg2rad(8))  # 48 deg total, exceeds +-34 deg limit
    walk = simulate_walk_gain_scheduled(gait, fs, DEFAULT_ZC)
    jt = generate_joint_trajectory(walk, gait)
    assert not (np.all(jt.left_ik_converged) and np.all(jt.right_ik_converged))


def test_heading_profile_matches_footstep_headings_and_is_piecewise_constant():
    gait = GaitParams(n_steps=6)
    fs = plan_turning_footsteps(gait, np.deg2rad(5))
    t, heading = heading_profile(gait, fs)
    assert heading[0] == fs[0].heading
    assert heading[-1] == fs[-1].heading
    assert np.all(np.diff(heading) >= -1e-12)  # monotonically non-decreasing (turning one way)
