"""tests/test_n_step_capture_planner.py — verifies the joint, blended
N-step recovery scheme actually solves the crowding problem the naive
multi-step version had, and genuinely outperforms both single-step and
naive multi-step recovery for large pushes - measured, not assumed."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams
from simulation.push_recovery_simulator import simulate_push_recovery
from control.n_step_capture_planner import alpha_schedule


def test_alpha_schedule_decays_geometrically_from_one():
    alphas = alpha_schedule(4, decay_ratio=0.5)
    assert alphas[0] == pytest.approx(1.0)
    assert np.allclose(alphas, [1.0, 0.5, 0.25, 0.125])


def test_backward_compatible_when_n_recovery_steps_is_one():
    """use_blended_targets must have no effect when there's only one
    footstep to recover - nothing to decay across."""
    gait = GaitParams(n_steps=8)
    push_vel = np.array([0.0, 1.0])
    r_plain = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=1)
    r_blended = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True,
                                        n_recovery_steps=1, use_blended_targets=True)
    for a, b in zip(r_plain.walk.footsteps, r_blended.walk.footsteps):
        assert a.x == pytest.approx(b.x) and a.y == pytest.approx(b.y)


def test_joint_blended_recovery_spaces_consecutive_footsteps_further_apart_than_naive():
    """The direct fix for the naive version's documented crowding
    problem (docs/BUGS_FOUND.md): consecutive recovered footsteps'
    lateral spacing must be measurably larger under the blended scheme
    than under the naive independently-repeated one, for the same push."""
    gait = GaitParams(n_steps=8)
    push_vel = np.array([0.0, 1.5])
    r_naive = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=3)
    r_joint = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True,
                                      n_recovery_steps=3, use_blended_targets=True)
    fs_naive, idx_naive = r_naive.walk.footsteps, r_naive.modified_footstep_indices
    fs_joint, idx_joint = r_joint.walk.footsteps, r_joint.modified_footstep_indices
    gap_naive = abs(fs_naive[idx_naive[0]].y - fs_naive[idx_naive[1]].y)
    gap_joint = abs(fs_joint[idx_joint[0]].y - fs_joint[idx_joint[1]].y)
    assert gap_joint > gap_naive


def test_joint_blended_recovery_beats_both_single_step_and_naive_multistep_for_large_pushes():
    """The real, measured payoff (docs/BUGS_FOUND.md): for large pushes
    where naive multi-step measurably UNDERPERFORMED single-step, the
    joint blended scheme measurably BEATS both. Locked in with the
    actual counts so this doesn't silently drift."""
    gait = GaitParams(n_steps=8)
    for push_mag, expected_single, expected_naive, expected_joint in [
        (1.5, 14, 16, 12),
        (2.0, 19, 21, 16),
    ]:
        push_vel = np.array([0.0, push_mag])
        r_single = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=1)
        r_naive = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True, n_recovery_steps=3)
        r_joint = simulate_push_recovery(gait, 3.9, push_vel, use_recovery=True,
                                          n_recovery_steps=3, use_blended_targets=True)
        later_single = int(np.sum(r_single.walk.t[~r_single.walk.zmp_in_support] > 4.8))
        later_naive = int(np.sum(r_naive.walk.t[~r_naive.walk.zmp_in_support] > 4.8))
        later_joint = int(np.sum(r_joint.walk.t[~r_joint.walk.zmp_in_support] > 4.8))
        assert later_single == expected_single
        assert later_naive == expected_naive
        assert later_joint == expected_joint
        assert later_joint < later_single
        assert later_joint < later_naive


def test_undisturbed_walk_unaffected_by_blended_recovery_machinery():
    gait = GaitParams(n_steps=8)
    r = simulate_push_recovery(gait, 3.9, np.zeros(2), use_recovery=True,
                                n_recovery_steps=3, use_blended_targets=True)
    assert np.sum(~r.walk.zmp_in_support) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
