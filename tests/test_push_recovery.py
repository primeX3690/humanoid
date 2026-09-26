"""tests/test_push_recovery.py — verifies the capture point formula
against simple analytic cases, and locks in the real (modest, not
magical) push-recovery benefit found in docs/BUGS_FOUND.md."""
import numpy as np
import pytest

from control.capture_point import capture_point, capture_point_2d, clip_to_reachable_step
from planning.footstep_planner import GaitParams
from simulation.push_recovery_simulator import simulate_push_recovery


def test_capture_point_matches_formula():
    assert abs(capture_point(0.0, 2.0, omega=5.0) - 0.4) < 1e-12
    assert abs(capture_point(1.0, 0.0, omega=5.0) - 1.0) < 1e-12  # no velocity -> capture point = position


def test_capture_point_2d_is_independent_per_axis():
    cp = capture_point_2d(np.array([0.0, 0.0]), np.array([1.0, 2.0]), omega=2.0)
    assert np.allclose(cp, [0.5, 1.0])


def test_clip_leaves_reachable_targets_untouched():
    target = np.array([0.1, 0.05])
    stance = np.array([0.0, 0.0])
    clipped = clip_to_reachable_step(target, stance, max_step_length_m=0.5)
    assert np.allclose(clipped, target)


def test_clip_actually_clips_unreachable_targets():
    target = np.array([2.0, 0.0])
    stance = np.array([0.0, 0.0])
    clipped = clip_to_reachable_step(target, stance, max_step_length_m=0.5)
    assert abs(np.linalg.norm(clipped - stance) - 0.5) < 1e-9


def test_recovery_reduces_cascading_violations_for_a_moderate_push():
    """Regression guard for the real finding: capture-point recovery
    cannot undo the INSTANTANEOUS ZMP transient at the moment of a push
    (the support polygon hasn't moved yet - see docs/BUGS_FOUND.md), but
    DOES measurably reduce violations in steps AFTER the recovered
    footstep, compared to ignoring the push entirely - a real, modest,
    not-magical benefit."""
    gait = GaitParams(n_steps=8)
    push_time = 3.9
    push_vel = np.array([0.0, 1.0])
    r_norec = simulate_push_recovery(gait, push_time, push_vel, use_recovery=False)
    r_rec = simulate_push_recovery(gait, push_time, push_vel, use_recovery=True)

    later_norec = np.sum(r_norec.walk.t[~r_norec.walk.zmp_in_support] > 4.8)
    later_rec = np.sum(r_rec.walk.t[~r_rec.walk.zmp_in_support] > 4.8)
    assert later_rec < later_norec, (
        f"expected recovery to reduce post-recovery-step violations "
        f"({later_rec} vs {later_norec} without recovery)"
    )


def test_undisturbed_walk_is_unaffected_by_the_push_recovery_machinery():
    """Sanity check: with zero push velocity, the push-recovery simulator
    must reproduce the same stable, zero-violation walk as the plain
    walk_simulator - the recovery code path must not itself introduce
    instability when there's nothing to recover from."""
    gait = GaitParams(n_steps=8)
    r = simulate_push_recovery(gait, push_time_s=3.9, push_velocity_xy=np.zeros(2), use_recovery=True)
    assert np.sum(~r.walk.zmp_in_support) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
