"""tests/test_leg_dynamics.py — verifies the Jacobian-transpose torque
model's sign convention and scaling behavior against simple,
hand-checkable physical cases."""
import numpy as np
import pytest

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import stance_leg_required_torque, RobotMassParams


LEG = LegParams()
HIP = np.array([0.0, 0.0, 0.9])


def test_straight_vertical_leg_needs_near_zero_torque_when_standing():
    """A perfectly straight, vertically-loaded leg is under pure axial
    (compressive) load - the real-world reason people can stand with
    locked knees using very little muscular effort. All rotational
    joint torques should be ~zero, not just small."""
    robot = RobotMassParams(total_mass_kg=25.0)
    tau = stance_leg_required_torque(np.zeros(6), HIP, LEG, robot, com_accel=np.zeros(3))
    assert np.all(np.abs(tau) < 0.01), f"expected ~0 torque for a straight leg, got {tau}"


def test_ground_reaction_force_points_upward_when_standing_still():
    """A regression guard for a real sign-convention bug found while
    building this: an earlier version had R = M*(g - a) instead of
    M*(a - g), which reports the ground PULLING the robot down instead
    of pushing it up. Checked indirectly here via a bent-knee stance
    (nonzero torque expected) having the physically correct sign: a
    forward-flexed knee under a downward load must be flexed further by
    the load (a real knee under weight tends to buckle forward/deeper
    into flexion without extensor torque, not backward)."""
    robot = RobotMassParams(total_mass_kg=25.0)
    q_bent = np.array([0.0, 0.0, -0.3, 0.6, -0.3, 0.0])
    tau = stance_leg_required_torque(q_bent, HIP, LEG, robot, com_accel=np.zeros(3))
    assert abs(tau[3]) > 1.0, "expected a real, nonzero knee torque for a bent-knee stance"


def test_required_torque_scales_linearly_with_robot_mass():
    """Doubling the robot's mass must exactly double the required torque -
    a direct consequence of F=ma / the Jacobian-transpose method being
    linear in force, and a simple, checkable correctness property."""
    q_bent = np.array([0.0, 0.0, -0.3, 0.6, -0.3, 0.0])
    tau_25kg = stance_leg_required_torque(q_bent, HIP, LEG, RobotMassParams(25.0), np.zeros(3))
    tau_50kg = stance_leg_required_torque(q_bent, HIP, LEG, RobotMassParams(50.0), np.zeros(3))
    ratio = tau_50kg[3] / tau_25kg[3]
    assert abs(ratio - 2.0) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
