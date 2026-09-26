"""tests/test_joint_trajectory_generator.py — integration test for the
full pipeline: CoM trajectory + footstep plan -> leg IK -> joint angles.
This is the test that caught the landing-position bookkeeping bug (see
docs/BUGS_FOUND.md) - a full end-to-end tracking-error check across an
entire multi-step walk, not just a single IK call in isolation."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams, foot_target_trajectories
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory


def test_full_walk_ik_converges_at_every_timestep():
    gait = GaitParams(n_steps=8)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    assert np.all(jt.left_ik_converged), (
        f"{np.sum(~jt.left_ik_converged)} left-leg IK failures across the walk")
    assert np.all(jt.right_ik_converged), (
        f"{np.sum(~jt.right_ik_converged)} right-leg IK failures across the walk")


def test_actual_foot_position_tracks_planned_target_throughout_walk():
    """The real end-to-end correctness check: does the ACTUAL foot
    position (from forward kinematics on the solved joint angles) match
    the PLANNED foot target at every single timestep of an 8-step walk?
    This is the test that caught the landing-position bug, where most
    steps' foot target silently froze at an earlier step's position
    (max error was 732mm before the fix - a completely broken walk that
    every earlier, narrower test missed)."""
    gait = GaitParams(n_steps=8)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    t, left_target, right_target = foot_target_trajectories(gait, walk.footsteps)

    left_err_mm = np.linalg.norm(jt.left_foot_xyz - left_target, axis=1) * 1000
    right_err_mm = np.linalg.norm(jt.right_foot_xyz - right_target, axis=1) * 1000

    assert left_err_mm.max() < 1.0, f"max left foot tracking error {left_err_mm.max():.2f}mm"
    assert right_err_mm.max() < 1.0, f"max right foot tracking error {right_err_mm.max():.2f}mm"


def test_each_footstep_lands_at_its_planned_xy():
    """Direct, specific check per footstep (not just an aggregate max
    error): at the end of each step's swing, the achieved foot position
    must match that footstep's planned (x, y)."""
    gait = GaitParams(n_steps=8)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    t = jt.t
    for step in walk.footsteps:
        idx = int(np.searchsorted(t, step.end_time)) - 1
        achieved = jt.left_foot_xyz[idx] if step.side == "left" else jt.right_foot_xyz[idx]
        assert abs(achieved[0] - step.x) < 0.005, (
            f"{step.side} step at planned x={step.x:.3f} landed at x={achieved[0]:.3f}")
        assert abs(achieved[1] - step.y) < 0.005


def test_swing_foot_clears_ground_stance_foot_does_not():
    """Sanity-check the whole pipeline produces an actual walking motion:
    the swinging foot should lift off the ground (z > 0) at some point
    mid-step, while the stance foot stays essentially at z=0 throughout."""
    gait = GaitParams(n_steps=4)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    t = jt.t

    first_step = walk.footsteps[0]  # side="right"
    mid_t = (first_step.start_time + first_step.end_time) / 2
    mid_idx = int(np.searchsorted(t, mid_t))
    assert jt.right_foot_xyz[mid_idx, 2] > 0.02, "swinging (right) foot should have ground clearance mid-step"
    assert abs(jt.left_foot_xyz[mid_idx, 2]) < 0.005, "stance (left) foot should stay on the ground"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
