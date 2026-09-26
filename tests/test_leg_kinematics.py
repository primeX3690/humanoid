"""tests/test_leg_kinematics.py — verifies the 6-DOF leg FK/IK via
round-trip consistency and known-configuration checks, since there's no
external reference formula to check against directly (see
kinematics/leg_ik.py's docstring for why numerical IK was chosen)."""
import numpy as np
import pytest

from kinematics.leg_fk import forward_kinematics, LegParams
from kinematics.leg_ik import inverse_kinematics, JOINT_LIMITS_RAD


LEG = LegParams()
HIP = np.array([0.0, 0.0, 0.8])


def test_straight_leg_reaches_full_extension():
    """All-zero joint angles must place the foot directly below the hip at
    exactly thigh+shank distance - the simplest possible FK check."""
    pos, R = forward_kinematics(np.zeros(6), HIP, LEG)
    expected = HIP + np.array([0.0, 0.0, -(LEG.thigh_length_m + LEG.shank_length_m)])
    assert np.allclose(pos, expected, atol=1e-9)
    assert np.allclose(R, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("target_offset", [
    (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.05, 0.0),
    (0.15, -0.08, 0.03), (0.05, 0.03, -0.02), (0.2, 0.0, 0.1),
])
def test_ik_round_trip_matches_target(target_offset):
    """For a range of reachable targets, IK's output must reproduce the
    target position via the SAME forward kinematics to sub-millimeter
    precision - the actual correctness bar for numerical IK."""
    target_pos = HIP + np.array([0.0, 0.0, -0.7]) + np.array(target_offset)
    target_R = np.eye(3)
    result = inverse_kinematics(target_pos, target_R, HIP, LEG)
    assert result.converged, f"IK failed to converge, residual={result.residual_error}"
    fk_pos, _ = forward_kinematics(result.joint_angles, HIP, LEG)
    assert np.linalg.norm(fk_pos - target_pos) < 1e-4


def test_knee_never_bends_backward():
    """Real knees only flex one direction. Across a spread of reachable
    targets, the solved knee_pitch must always respect the joint limit
    (>= 0), not just happen to by luck."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        offset = rng.uniform(-0.15, 0.15, size=3)
        offset[2] = rng.uniform(-0.1, 0.05)
        target_pos = HIP + np.array([0.0, 0.0, -0.75]) + offset
        result = inverse_kinematics(target_pos, np.eye(3), HIP, LEG)
        assert result.joint_angles[3] >= JOINT_LIMITS_RAD[3, 0] - 1e-9


def test_unreachable_target_fails_to_converge_not_silently_wrong():
    """A target far beyond the leg's maximum reach must be reported as
    NOT converged with a real, nonzero residual - not silently returned
    as if it were satisfied."""
    max_reach = LEG.thigh_length_m + LEG.shank_length_m
    unreachable = HIP + np.array([0.0, 0.0, -(max_reach + 0.3)])
    result = inverse_kinematics(unreachable, np.eye(3), HIP, LEG)
    assert not result.converged
    assert result.residual_error > 0.05


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
