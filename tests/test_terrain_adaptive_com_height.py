"""tests/test_terrain_adaptive_com_height.py — verifies the terrain-
adaptive hip-height fix: reproduces the original constant-height model
exactly on flat ground, fixes the documented 10cm-step-down IK failure,
leaves the 10cm-step-up case unaffected, and locks in the real,
measured new (smaller, but still finite) failure boundary rather than
claiming the leg-reach problem is solved for arbitrary drop depth."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams, apply_terrain
from terrain.terrain_profile import StepTerrain, FlatTerrain
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory
from dynamics.lipm import LIPMParams
from dynamics.terrain_adaptive_com_height import hip_height_profile, compute_zc_profile

DEFAULT_ZC = LIPMParams().com_height_m


def _walk_with_terrain(step_height_m: float, n_steps: int = 6):
    gait = GaitParams(n_steps=n_steps)
    walk = simulate_walk(gait)
    walk.footsteps[:] = apply_terrain(walk.footsteps, StepTerrain(step_x=0.9, step_height_m=step_height_m))
    return gait, walk


def test_flat_ground_reproduces_the_original_constant_height_exactly():
    """On flat ground (the only case the original model was ever
    verified against), the adaptive hip-height profile must equal
    default_zc EVERYWHERE, and produce IDENTICAL joint angles to the
    original constant-height path - this must be a strict superset of
    the old behavior, not just 'similar'."""
    gait = GaitParams(n_steps=6)
    walk = simulate_walk(gait)
    t, hip_z = hip_height_profile(gait, walk.footsteps, DEFAULT_ZC)
    assert np.allclose(hip_z, DEFAULT_ZC)

    jt_old = generate_joint_trajectory(walk, gait)
    jt_new = generate_joint_trajectory(walk, gait, hip_height_profile_m=hip_z)
    assert np.allclose(jt_old.left_joint_angles, jt_new.left_joint_angles)
    assert np.allclose(jt_old.right_joint_angles, jt_new.right_joint_angles)


def test_zc_profile_diagnostic_matches_hip_height_profile_at_each_landing():
    """The closed-form per-footstep diagnostic (compute_zc_profile)
    reports the target hip height MINUS THAT PHASE'S OWN STANCE foot's
    height (the quantity its derivation is actually about - see the
    module docstring's vertical_reach_needed derivation). It must agree
    with what the continuous min-formula actually produces once measured
    the same way: hip_at_landing - stance_foot_z == local_zc_diagnostic.
    (Comparing hip_at_landing directly to the LANDING foot's height
    instead of the STANCE foot's is exactly the mistake an earlier
    version of this test made - the min-formula's hip height at a
    landing instant is set by default_zc above whichever foot is
    LOWER, which at that exact instant is the just-landed foot, not the
    still-planted stance foot the diagnostic is measured against.)"""
    from dynamics.terrain_adaptive_com_height import _support_foot_z
    gait, walk = _walk_with_terrain(-0.1)
    zc_diag = compute_zc_profile(gait, walk.footsteps, DEFAULT_ZC)
    stance_z = _support_foot_z(gait, walk.footsteps)
    t, hip_z = hip_height_profile(gait, walk.footsteps, DEFAULT_ZC)
    for i, step in enumerate(walk.footsteps):
        idx = int(np.searchsorted(t, step.end_time))
        idx = min(idx, len(t) - 1)
        assert abs((hip_z[idx] - stance_z[i]) - zc_diag[i]) < 1e-6


def test_stepping_down_10cm_now_converges_with_adaptive_height():
    """The headline fix: tests/test_terrain.py's
    test_stepping_down_exceeds_current_leg_reach_margin documents this
    FAILING with the original constant-height model. With the adaptive
    hip-height profile plugged in instead, IK must converge for the
    entire walk."""
    gait, walk = _walk_with_terrain(-0.1)
    t, hip_z = hip_height_profile(gait, walk.footsteps, DEFAULT_ZC)
    jt = generate_joint_trajectory(walk, gait, hip_height_profile_m=hip_z)
    assert np.all(jt.left_ik_converged)
    assert np.all(jt.right_ik_converged)


def test_stepping_up_10cm_still_converges_with_adaptive_height():
    """Regression guard for docs/BUGS_FOUND.md bug #7's first-attempt
    failure: an earlier ramp-based version of this fix broke the
    previously-working stepping-UP case. The shipped min-formula must
    not reintroduce that regression."""
    gait, walk = _walk_with_terrain(0.1)
    t, hip_z = hip_height_profile(gait, walk.footsteps, DEFAULT_ZC)
    jt = generate_joint_trajectory(walk, gait, hip_height_profile_m=hip_z)
    assert np.all(jt.left_ik_converged)
    assert np.all(jt.right_ik_converged)


def test_original_nonadaptive_stepping_down_failure_is_still_locked_in():
    """docs/SCOPE.md and tests/test_terrain.py's existing regression
    guard must remain true for the ORIGINAL, non-adaptive code path -
    this new module is additive (opt-in via hip_height_profile_m), not a
    silent change to generate_joint_trajectory's default behavior."""
    gait, walk = _walk_with_terrain(-0.1)
    jt = generate_joint_trajectory(walk, gait)  # no hip_height_profile_m passed
    assert not (np.all(jt.left_ik_converged) and np.all(jt.right_ik_converged))


@pytest.mark.parametrize("step_height_m,expect_converge", [
    (-0.12, True),
    (-0.14, False),
])
def test_new_failure_boundary_is_real_and_measured_not_claimed_solved(step_height_m, expect_converge):
    """Honest characterization, not an oversold fix: adaptive height
    measurably pushes the old 10cm limit out to at least 12cm, but a
    14cm drop still fails - locked in here so this specific, real
    boundary doesn't silently drift without the test noticing (see
    docs/BUGS_FOUND.md bug #7)."""
    gait, walk = _walk_with_terrain(step_height_m)
    t, hip_z = hip_height_profile(gait, walk.footsteps, DEFAULT_ZC)
    jt = generate_joint_trajectory(walk, gait, hip_height_profile_m=hip_z)
    converged = bool(np.all(jt.left_ik_converged) and np.all(jt.right_ik_converged))
    assert converged == expect_converge


def test_flat_terrain_profile_hip_height_is_unaffected_by_calling_apply_terrain():
    """apply_terrain with FlatTerrain (all-zero heights) must give the
    exact same hip-height profile as never calling it at all - purely
    additive, matching the rest of the terrain module's own convention."""
    gait = GaitParams(n_steps=6)
    walk = simulate_walk(gait)
    flat_steps = apply_terrain(walk.footsteps, FlatTerrain())
    t1, h1 = hip_height_profile(gait, walk.footsteps, DEFAULT_ZC)
    t2, h2 = hip_height_profile(gait, flat_steps, DEFAULT_ZC)
    assert np.allclose(h1, h2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
