"""tests/test_lipm.py — verifies the LIPM discretization against the
textbook closed-form continuous-time solution, not just "it runs"."""
import numpy as np
import pytest

from dynamics.lipm import LIPM, LIPMParams, GRAVITY


def test_natural_frequency_matches_closed_form():
    params = LIPMParams(com_height_m=0.8)
    lipm = LIPM(params)
    expected = np.sqrt(GRAVITY / 0.8)
    assert abs(lipm.natural_frequency() - expected) < 1e-12


def test_zmp_equals_com_when_no_acceleration():
    """x_zmp = x_com - (zc/g) x_com_ddot; at zero acceleration the ZMP must
    equal the CoM position exactly - the simplest possible check of the
    ZMP formula's sign and scaling."""
    params = LIPMParams(com_height_m=0.8)
    lipm = LIPM(params)
    state = np.array([0.35, 0.0, 0.0])
    assert abs(lipm.zmp(state) - 0.35) < 1e-12


def test_zmp_moves_opposite_to_forward_acceleration():
    """Standard, textbook LIPM sign convention: accelerating the CoM FORWARD
    pushes the ZMP BACKWARD relative to the CoM (this is why leaning forward
    to start walking works) - if this sign were flipped, the preview
    controller upstream would be unstable in the opposite direction."""
    params = LIPMParams(com_height_m=0.8)
    lipm = LIPM(params)
    state = np.array([0.0, 0.0, 1.0])  # accelerating forward (+x)
    assert lipm.zmp(state) < 0.0


def test_discretization_matches_continuous_freefall_solution():
    """With jerk=0 (constant acceleration a0), the discretized LIPM's
    position must match the closed-form x(t) = x0 + v0*t + 0.5*a0*t^2
    to numerical precision - this is a direct check that A/B were built
    correctly, not just plausible-looking."""
    params = LIPMParams(com_height_m=0.8, dt=0.01)
    lipm = LIPM(params)
    x0, v0, a0 = 0.1, 0.05, 0.3
    traj = lipm.free_fall_check(np.array([x0, v0, a0]), n_steps=50)
    t = np.arange(50) * params.dt
    expected_pos = x0 + v0 * t + 0.5 * a0 * t ** 2
    assert np.allclose(traj[:, 0], expected_pos, atol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
