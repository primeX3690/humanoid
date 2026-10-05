"""tests/test_gearbox_backlash.py — verifies actuators/gearbox_backlash.py:
the standard backlash nonlinearity behaves correctly on simple synthetic
inputs (monotonic motion follows exactly, a direction reversal freezes
output until the gap is re-crossed), and quantifies its real effect on
an actual gait's knee trajectory."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory
from actuators.gearbox_backlash import apply_backlash, backlash_position_error, DEFAULT_BACKLASH_RAD


def test_monotonic_motion_tracks_with_constant_half_gap_offset():
    """Once the gap is taken up in one direction and motion stays
    monotonic, the load should track the motor exactly, offset by
    exactly half the backlash width (whichever face is engaged)."""
    b = np.deg2rad(1.0)
    motor = np.linspace(0, 1.0, 200)  # steadily increasing, plenty to cross the gap
    load = apply_backlash(motor, backlash_rad=b)
    # after the first few samples (gap taken up), offset should be constant = -b/2
    offset = load[20:] - motor[20:]
    assert np.allclose(offset, -b / 2.0, atol=1e-9)


def test_direction_reversal_freezes_output_until_gap_recrossed():
    """The defining behavior of backlash: reverse the motor direction,
    and the load must NOT move at all until the motor has crossed back
    through the full gap width."""
    b = np.deg2rad(2.0)
    # motor moves up then reverses
    motor = np.concatenate([np.linspace(0, 1.0, 100), np.linspace(1.0, 0.9, 50)])
    load = apply_backlash(motor, backlash_rad=b)
    reversal_start = 100
    # load must stay frozen at its value right at the reversal point until
    # the motor has moved back by at least the full gap width (b)
    frozen_value = load[reversal_start]
    still_frozen = np.abs(motor[reversal_start] - motor[reversal_start:]) < b
    assert np.all(load[reversal_start:][still_frozen] == frozen_value)


def test_max_position_error_equals_half_the_backlash_width():
    """Theoretical property of this model: the load can never lag the
    motor by more than half the total gap width, in either direction."""
    b = np.deg2rad(0.5)
    t = np.linspace(0, 4 * np.pi, 500)
    motor = np.sin(t)  # oscillating, exercises many direction reversals
    err = backlash_position_error(motor, backlash_rad=b)
    assert np.max(np.abs(err)) <= b / 2.0 + 1e-9
    assert np.max(np.abs(err)) > 0.4 * (b / 2.0)  # and it actually gets close to that bound


def test_zero_backlash_reproduces_input_exactly():
    b = 0.0
    motor = np.sin(np.linspace(0, 4 * np.pi, 200)) + np.linspace(0, 1, 200)
    load = apply_backlash(motor, backlash_rad=b)
    assert np.allclose(load, motor)


def test_real_gait_knee_backlash_error_is_small_and_bounded():
    """Applied to this project's own gait: the resulting position error
    should be small (a fraction of a degree, given the stated 0.3-degree
    assumed backlash) and bounded by the theoretical half-gap limit -
    real, measured numbers, not just 'it runs'."""
    gait = GaitParams(n_steps=6)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    knee = jt.left_joint_angles[:, 3]
    err = backlash_position_error(knee)
    max_err_deg = np.rad2deg(np.max(np.abs(err)))
    assert max_err_deg == pytest.approx(np.rad2deg(DEFAULT_BACKLASH_RAD) / 2.0, abs=1e-6)
    assert max_err_deg < 1.0  # sub-degree, given the assumed backlash width


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
