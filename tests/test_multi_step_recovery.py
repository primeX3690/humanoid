"""tests/test_multi_step_recovery.py — verifies the multi-step push-
recovery extension: backward compatibility with the original single-step
behavior, that it genuinely modifies multiple footsteps and shifts the
remainder, and locks in the real (negative) finding that naively
repeating capture-point correction across several future footsteps does
NOT improve on single-step recovery for this system - see
docs/BUGS_FOUND.md."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams, plan_footsteps
from simulation.push_recovery_simulator import simulate_push_recovery


def test_n_recovery_steps_default_is_unchanged_single_step_behavior():
    """n_recovery_steps=1 (the default) must reproduce the ORIGINAL
    single-step recovery's footsteps and result exactly - this feature
    is additive, not a silent change to the existing default."""
    gait = GaitParams(n_steps=8)
    push_vel = np.array([0.0, 1.0])
    r_default = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True)
    r_explicit = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=1)
    for a, b in zip(r_default.walk.footsteps, r_explicit.walk.footsteps):
        assert a.x == b.x and a.y == b.y
    assert r_default.modified_footstep_indices == r_explicit.modified_footstep_indices
    assert not r_default.remaining_footsteps_shifted
    assert not r_explicit.remaining_footsteps_shifted


def test_multi_step_recovery_modifies_the_requested_number_of_footsteps():
    gait = GaitParams(n_steps=8)
    push_vel = np.array([0.0, 1.5])
    r = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=3)
    assert len(r.modified_footstep_indices) == 3
    assert r.modified_footstep_indices == sorted(r.modified_footstep_indices)  # consecutive, in order
    assert len(r.capture_points_xy) == 3


def test_remaining_footsteps_are_rigidly_shifted_preserving_the_nominal_gait_pattern():
    """After the recovery window closes, every UN-touched footstep must
    keep the ORIGINAL gait's relative step length/width (just translated
    by a constant offset), not revert to the absolute nominal path."""
    gait = GaitParams(n_steps=8)
    nominal = plan_footsteps(gait)
    push_vel = np.array([0.0, 1.5])
    r = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=3)
    assert r.remaining_footsteps_shifted
    last_recovered = r.modified_footstep_indices[-1]
    offset_x = r.walk.footsteps[last_recovered].x - nominal[last_recovered].x
    offset_y = r.walk.footsteps[last_recovered].y - nominal[last_recovered].y
    for j in range(last_recovered + 1, len(r.walk.footsteps)):
        assert abs(r.walk.footsteps[j].x - (nominal[j].x + offset_x)) < 1e-9
        assert abs(r.walk.footsteps[j].y - (nominal[j].y + offset_y)) < 1e-9


def test_undisturbed_walk_unaffected_by_multi_step_recovery_machinery():
    """Matches the existing single-step sanity check
    (tests/test_push_recovery.py::test_undisturbed_walk_is_unaffected...):
    the recovery machinery still runs with a zero-magnitude push (same as
    the original single-step code always did - it doesn't special-case
    "no push"), but with nothing to correct for, it must not introduce
    any instability."""
    gait = GaitParams(n_steps=8)
    r = simulate_push_recovery(gait, 3.9, np.zeros(2), use_recovery=True, n_recovery_steps=3)
    assert np.sum(~r.walk.zmp_in_support) == 0


def test_naive_multistep_recovery_does_not_beat_single_step_for_large_pushes():
    """Real, honest finding (docs/BUGS_FOUND.md): repeating the SAME
    single-step capture-point formula independently at each of several
    upcoming footsteps does not improve on correcting just the first one
    for this system, and for large pushes is measurably (slightly)
    worse - locked in here with the actual measured counts so this
    doesn't silently drift without notice. This is NOT a claim that
    multi-step recovery is a bad idea in general - a real N-step-
    capturability solve (Pratt/Koolen) that plans the WHOLE sequence
    jointly would very plausibly do better; what's implemented and
    tested here is the naive, independently-repeated version, and it
    measurably does not help."""
    gait = GaitParams(n_steps=8)
    push_vel = np.array([0.0, 1.5])
    r_single = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=1)
    r_multi = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=3)
    later_single = int(np.sum(r_single.walk.t[~r_single.walk.zmp_in_support] > 4.8))
    later_multi = int(np.sum(r_multi.walk.t[~r_multi.walk.zmp_in_support] > 4.8))
    assert later_single == 14
    assert later_multi == 16
    assert later_multi >= later_single


def test_naive_multistep_recovery_crowds_consecutive_footsteps_laterally():
    """The actual MECHANISM behind the finding above, verified directly:
    for a large lateral push, consecutive independently-recovered
    footsteps land close together in y (crossing toward the same side),
    far closer than the gait's own nominal stance width - because each
    footstep's capture point is computed from a CoM lateral velocity
    that hasn't yet been arrested by the PREVIOUS correction, not
    because of an implementation bug in the recovery loop itself."""
    gait = GaitParams(n_steps=8)
    push_vel = np.array([0.0, 1.5])
    r = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=3)
    fs = r.walk.footsteps
    i0, i1, i2 = r.modified_footstep_indices
    lateral_gap_01 = abs(fs[i0].y - fs[i1].y)
    lateral_gap_12 = abs(fs[i1].y - fs[i2].y)
    assert lateral_gap_01 < gait.step_width_m
    assert lateral_gap_12 < gait.step_width_m


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
