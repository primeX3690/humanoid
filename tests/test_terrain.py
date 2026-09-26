"""tests/test_terrain.py — verifies terrain-aware footstep placement is
purely additive (flat-ground behavior unchanged by default) and locks in
the real finding that stepping UP works with the current leg-length
margin but stepping DOWN exceeds it (see docs/BUGS_FOUND.md)."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams, plan_footsteps, apply_terrain, foot_target_trajectories
from terrain.terrain_profile import FlatTerrain, StepTerrain, RampTerrain
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory


def test_flat_terrain_is_unchanged_from_original_default_behavior():
    gait = GaitParams(n_steps=6)
    footsteps = plan_footsteps(gait)
    flat_applied = apply_terrain(footsteps, FlatTerrain())
    for orig, flat in zip(footsteps, flat_applied):
        assert orig.z == 0.0
        assert flat.z == 0.0
        assert orig.x == flat.x and orig.y == flat.y


def test_step_terrain_places_footsteps_at_correct_height():
    gait = GaitParams(n_steps=6)
    footsteps = plan_footsteps(gait)
    terrain = StepTerrain(step_x=0.9, step_height_m=0.1)
    stepped = apply_terrain(footsteps, terrain)
    for step in stepped:
        expected_z = 0.1 if step.x >= 0.9 else 0.0
        assert abs(step.z - expected_z) < 1e-9


def test_ramp_terrain_interpolates_linearly():
    ramp = RampTerrain(ramp_start_x=1.0, ramp_end_x=2.0, rise_m=0.2)
    assert ramp.height_at(0.5, 0.0) == 0.0
    assert abs(ramp.height_at(1.5, 0.0) - 0.1) < 1e-9
    assert ramp.height_at(2.5, 0.0) == 0.2


def test_swing_trajectory_correctly_climbs_a_step():
    gait = GaitParams(n_steps=6)
    footsteps = plan_footsteps(gait)
    stepped = apply_terrain(footsteps, StepTerrain(step_x=0.9, step_height_m=0.1))
    t, left_xyz, right_xyz = foot_target_trajectories(gait, stepped)
    # the foot landing AFTER the step must end its swing at z=0.1, not 0
    post_step_footsteps = [s for s in stepped if s.x >= 0.9]
    assert len(post_step_footsteps) > 0
    first_post_step = post_step_footsteps[0]
    idx = int(np.searchsorted(t, first_post_step.end_time)) - 1
    traj = left_xyz if first_post_step.side == "left" else right_xyz
    assert abs(traj[idx, 2] - 0.1) < 0.01


def test_stepping_up_within_leg_reach_margin_still_converges():
    """Locks in the real, positive finding: a modest (10cm) step UP is
    within the leg-length margin established in docs/BUGS_FOUND.md (0.9m
    reach vs 0.8m CoM height) - IK must converge for every timestep."""
    gait = GaitParams(n_steps=6)
    walk = simulate_walk(gait)
    walk.footsteps[:] = apply_terrain(walk.footsteps, StepTerrain(step_x=0.9, step_height_m=0.1))
    jt = generate_joint_trajectory(walk, gait)
    assert np.all(jt.left_ik_converged)
    assert np.all(jt.right_ik_converged)


def test_stepping_down_exceeds_current_leg_reach_margin():
    """Regression guard for the real, honest limitation found: stepping
    DOWN needs MORE leg reach (foot farther from a hip that doesn't drop
    with it, since the LIPM's CoM-height model doesn't yet adapt to
    terrain - see terrain/terrain_profile.py's module docstring), and a
    10cm step down exceeds the current leg's reach margin. If this ever
    starts passing, either the leg length or the terrain-CoM-height
    coupling changed and this finding needs re-checking, not silently
    dropping."""
    gait = GaitParams(n_steps=6)
    walk = simulate_walk(gait)
    walk.footsteps[:] = apply_terrain(walk.footsteps, StepTerrain(step_x=0.9, step_height_m=-0.1))
    jt = generate_joint_trajectory(walk, gait)
    assert not (np.all(jt.left_ik_converged) and np.all(jt.right_ik_converged)), (
        "expected stepping down to exceed the current leg's reach margin - "
        "if this now converges, the underlying model changed and this "
        "finding needs re-verifying, not silently dropping"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
