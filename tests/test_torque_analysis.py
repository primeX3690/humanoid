"""tests/test_torque_analysis.py — locks in the real, specific finding
from docs/BUGS_FOUND.md: a small hobby servo (MX-106) is nowhere near
powerful enough for this gait's hip/knee torque demands, while a real
research-grade actuator (H54P-200) is enough for the hip but still
slightly short at the knee - a specific, actionable hardware-selection
finding, not a vague "motors might not be strong enough"."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory
from simulation.torque_analysis import analyze_torque_feasibility
from actuators.motor_specs import DYNAMIXEL_MX106, DYNAMIXEL_H54P_200


@pytest.fixture(scope="module")
def walk_and_joints():
    gait = GaitParams(n_steps=8)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    return walk, jt, gait


def test_small_hobby_servo_is_not_feasible_for_this_gait(walk_and_joints):
    """Regression guard for the real finding: a Dynamixel MX-106 (a
    genuinely popular choice in hobbyist biped builds) is not remotely
    strong enough for a 25kg humanoid's hip or knee, even measured
    against its STALL (momentary max) rating, not just continuous."""
    walk, jt, gait = walk_and_joints
    result = analyze_torque_feasibility(walk, jt, gait, motor=DYNAMIXEL_MX106)
    assert not result.hip_feasibility.torque_ok_stall
    assert not result.knee_feasibility.torque_ok_stall


def test_research_grade_actuator_covers_hip_but_not_quite_knee(walk_and_joints):
    """Regression guard for the more actionable finding: a real,
    commercially-available high-torque actuator (Dynamixel PRO Plus
    H54P-200) DOES have enough continuous torque margin for the hip
    (>1x), but falls just short at the knee (<1x) - meaning the knee,
    not the hip, is this gait's binding actuator-selection constraint."""
    walk, jt, gait = walk_and_joints
    result = analyze_torque_feasibility(walk, jt, gait, motor=DYNAMIXEL_H54P_200)
    assert result.hip_feasibility.torque_ok_continuous
    assert result.hip_feasibility.peak_torque_margin_vs_continuous > 1.0
    assert not result.knee_feasibility.torque_ok_continuous, (
        "expected the knee to still exceed even this powerful actuator's continuous rating - "
        "if this now passes, the torque model or gait tuning changed and this finding needs "
        "re-checking, not silently dropping"
    )


def test_both_reference_motors_have_enough_speed_margin(walk_and_joints):
    """Unlike torque, this gait's joint speeds are modest relative to
    real actuators' no-load speed ratings - checked explicitly so a
    future speed-side regression (e.g. a much faster gait) is caught."""
    walk, jt, gait = walk_and_joints
    for motor in (DYNAMIXEL_MX106, DYNAMIXEL_H54P_200):
        result = analyze_torque_feasibility(walk, jt, gait, motor=motor)
        assert result.hip_feasibility.speed_ok
        assert result.knee_feasibility.speed_ok


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
