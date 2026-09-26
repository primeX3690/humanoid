"""tests/test_swing_trajectory.py — verifies the minimum-jerk swing foot
trajectory's real boundary conditions (exact endpoints, zero velocity at
liftoff/touchdown, positive ground clearance mid-swing)."""
import numpy as np
import pytest

from trajectory.swing_foot import swing_foot_position, swing_foot_trajectory, SwingFootParams


def test_trajectory_starts_and_ends_exactly_at_targets():
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.3, 0.05, 0.0])
    params = SwingFootParams()
    p0 = swing_foot_position(0.0, start, end, params)
    p1 = swing_foot_position(1.0, start, end, params)
    assert np.allclose(p0, start, atol=1e-9)
    assert np.allclose(p1, end, atol=1e-9)


def test_zero_velocity_at_liftoff_and_touchdown():
    """Minimum-jerk's defining property: velocity (and acceleration) must
    be exactly zero at both endpoints - checked via finite differences,
    not just asserted from the formula."""
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.3, 0.0, 0.0])
    params = SwingFootParams()
    eps = 1e-5
    v_start = (swing_foot_position(eps, start, end, params) -
               swing_foot_position(0.0, start, end, params)) / eps
    v_end = (swing_foot_position(1.0, start, end, params) -
             swing_foot_position(1.0 - eps, start, end, params)) / eps
    assert np.linalg.norm(v_start) < 1e-2
    assert np.linalg.norm(v_end) < 1e-2


def test_positive_ground_clearance_mid_swing():
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.3, 0.0, 0.0])
    params = SwingFootParams(step_height_m=0.05)
    p_mid = swing_foot_position(0.5, start, end, params)
    assert p_mid[2] > 0.04  # near the peak of the sine profile at s=0.5


def test_full_trajectory_is_monotonic_in_horizontal_progress():
    """Horizontal (x) position should never move backward during a
    straight-line forward swing - a real gait doesn't have the foot
    overshoot and come back."""
    traj = swing_foot_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]),
                                  SwingFootParams(), n_samples=100)
    assert np.all(np.diff(traj[:, 0]) >= -1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
