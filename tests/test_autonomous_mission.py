"""tests/test_autonomous_mission.py - verifies planning/autonomous_mission.py:
goal -> gait planning accuracy, autonomous (non-oracle) disturbance detection,
automatic recovery with the joint N-step scheme, and the supervisor's safe-stop.
Each 'first version failed here' note below is a real bug caught while building it."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams, plan_footsteps
from planning.autonomous_mission import run_mission, plan_gait_for_goal, MissionConfig


@pytest.mark.parametrize("goal", [1.5, 2.4, 3.0, 4.2])
def test_undisturbed_mission_reaches_goal_accurately(goal):
    """First version added an extra closing step and overshot a 3 m goal by 0.36 m
    with no disturbance at all."""
    r = run_mission(goal)
    assert r.status == "REACHED"
    assert r.goal_error_m < 0.2
    assert r.disturbances_detected == 0
    assert r.total_violation_samples == 0


def test_step_count_scales_with_goal_distance():
    assert plan_gait_for_goal(1.5).n_steps < plan_gait_for_goal(3.0).n_steps < plan_gait_for_goal(6.0).n_steps
    last = plan_footsteps(plan_gait_for_goal(3.0))[-1]
    assert abs(last.x - 3.0) <= GaitParams().step_length_m / 2 + 1e-9


def test_disturbance_is_detected_without_being_told():
    """The controller is never given the push time - detection must come from its own
    capture-point divergence."""
    r = run_mission(3.0, [(3.9, 0.0, 1.0)])
    assert r.disturbances_detected == 1
    assert any("disturbance detected" in msg for _, msg in r.events)
    detect_time = next(t for t, msg in r.events if "disturbance detected" in msg)
    assert detect_time == pytest.approx(3.9, abs=0.05)


def test_one_push_triggers_exactly_one_detection_and_consecutive_recovery_steps():
    """First version detected ONE push four times (supervisor kept comparing against the
    nominal plan it had deliberately left) and recovered footsteps 3, 5, 7 instead of 3, 4, 5
    (start_time == t[k] skipped the next footstep)."""
    r = run_mission(3.0, [(3.9, 0.0, 1.0)])
    assert r.disturbances_detected == 1
    rec = [msg for _, msg in r.events if msg.startswith("recovery footstep")]
    indices = [int(m.split()[2]) for m in rec]
    assert indices == list(range(indices[0], indices[0] + len(indices)))
    assert len(indices) == MissionConfig().recovery_steps


@pytest.mark.parametrize("mag", [0.5, 1.0, 1.5])
def test_mission_still_reaches_goal_after_a_push(mag):
    r = run_mission(3.0, [(3.9, 0.0, mag)])
    assert r.status == "REACHED"
    assert r.goal_error_m < 0.25


def test_two_separate_pushes_are_each_handled():
    r = run_mission(3.0, [(3.9, 0.0, 0.6), (7.5, 0.3, -0.5)])
    assert r.status == "REACHED"
    assert r.disturbances_detected == 2
    assert r.goal_error_m < 0.25


def test_remaining_footsteps_are_shifted_to_stay_consistent_after_recovery():
    r = run_mission(3.0, [(3.9, 0.0, 1.0)])
    nominal = plan_footsteps(plan_gait_for_goal(3.0))
    last_rec = max(int(m.split()[2]) for _, m in r.events if m.startswith("recovery footstep"))
    off_x = r.footsteps[last_rec].x - nominal[last_rec].x
    off_y = r.footsteps[last_rec].y - nominal[last_rec].y
    for j in range(last_rec + 1, len(nominal)):
        assert r.footsteps[j].x == pytest.approx(nominal[j].x + off_x)
        assert r.footsteps[j].y == pytest.approx(nominal[j].y + off_y)


def test_unrecoverable_disturbance_ends_in_safe_stop_not_a_silent_failure():
    r = run_mission(3.0, [(3.9, 0.0, 2.5)])
    assert r.status == "SAFE_STOP"
    assert any("SAFE STOP" in msg for _, msg in r.events)
    assert r.goal_error_m > 1.0


def test_safe_stop_threshold_is_configurable():
    tolerant = MissionConfig(abort_violation_s=5.0)
    r = run_mission(3.0, [(3.9, 0.0, 2.5)], config=tolerant)
    assert r.status == "REACHED" or r.total_violation_samples > 95


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
