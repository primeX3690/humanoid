"""tests/test_gain_scheduled_lipm.py — verifies
dynamics/gain_scheduled_walk_simulator.py: exact reproduction of the
original fixed-height simulate_walk on flat terrain (the critical
regression check, since re-solving Riccati per segment must not change
anything when every segment's zc is identical), correct segment
boundaries, and the real, honestly-reported finding that gain-scheduling
does NOT improve ZMP tracking accuracy for this project's footstep
planner - because the horizontal ZMP reference itself never depended on
terrain height to begin with."""
import numpy as np
import pytest

from planning.footstep_planner import (
    GaitParams, plan_footsteps, apply_terrain, zmp_reference_trajectory, support_polygon_at,
)
from terrain.terrain_profile import StepTerrain
from simulation.walk_simulator import simulate_walk
from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from dynamics.gain_scheduled_walk_simulator import simulate_walk_gain_scheduled, _phase_boundaries

DEFAULT_ZC = LIPMParams().com_height_m


def _fixed_zc_run(gait, footsteps, zc):
    """Minimal, independent re-implementation of the ORIGINAL fixed-zc
    walk loop (matching simulate_walk's own logic) but accepting
    arbitrary (possibly terrain-modified) footsteps directly, so it can
    be compared against the gain-scheduled version on the SAME terrain -
    simulate_walk itself always calls plan_footsteps with no terrain."""
    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    n = len(t)
    params = LIPMParams(dt=gait.dt, com_height_m=zc)
    lipm_x, lipm_y = LIPM(params), LIPM(params)
    cfg = PreviewControllerConfig()
    ctrl_x, ctrl_y = ZMPPreviewController(lipm_x, cfg), ZMPPreviewController(lipm_y, cfg)
    N = ctrl_x.N
    zx_pad = np.concatenate([zx_ref, np.full(N + 1, zx_ref[-1])])
    zy_pad = np.concatenate([zy_ref, np.full(N + 1, zy_ref[-1])])
    x_state, y_state = np.zeros(3), np.zeros(3)
    zmp_x, zmp_y = np.zeros(n), np.zeros(n)
    for k in range(n):
        zmp_x[k], zmp_y[k] = lipm_x.zmp(x_state), lipm_y.zmp(y_state)
        jx = ctrl_x.compute_jerk(x_state, zx_pad[k:k + N + 1])
        jy = ctrl_y.compute_jerk(y_state, zy_pad[k:k + N + 1])
        x_state, y_state = lipm_x.step(x_state, jx), lipm_y.step(y_state, jy)
    return t, zx_ref, zy_ref, zmp_x, zmp_y


def test_flat_terrain_reproduces_original_fixed_zc_simulate_walk_exactly():
    """The critical regression check: on flat terrain, every segment's
    zc is identical to default_zc, so re-solving Riccati per segment and
    carrying the integral-error state across boundaries must reproduce
    simulate_walk's own output EXACTLY, not just approximately - any
    divergence here means the segment-chaining (state AND integral
    error) isn't actually seamless."""
    gait = GaitParams(n_steps=6)
    footsteps = plan_footsteps(gait)
    baseline = simulate_walk(gait)
    scheduled = simulate_walk_gain_scheduled(gait, footsteps, DEFAULT_ZC)
    assert np.allclose(baseline.com_x, scheduled.com_x, atol=1e-9)
    assert np.allclose(baseline.com_y, scheduled.com_y, atol=1e-9)
    assert np.allclose(baseline.zmp_x, scheduled.zmp_x, atol=1e-9)
    assert np.allclose(baseline.zmp_y, scheduled.zmp_y, atol=1e-9)


def test_phase_boundaries_align_with_footstep_landing_times():
    gait = GaitParams(n_steps=6)
    footsteps = plan_footsteps(gait)
    n = len(zmp_reference_trajectory(gait, footsteps)[0])
    boundaries = _phase_boundaries(gait, footsteps, n, gait.dt)
    assert boundaries[0] == 0
    assert boundaries[-1] == n
    landing_indices = {int(round(s.end_time / gait.dt)) for s in footsteps}
    assert landing_indices.issubset(set(boundaries))


def test_gain_scheduling_does_not_improve_horizontal_zmp_tracking_for_this_planner():
    """Real, honest, somewhat counter-intuitive finding (see
    docs/BUGS_FOUND.md): gain-scheduling the controller to the
    terrain-adaptive zc does NOT reduce horizontal ZMP tracking error
    for a step-terrain scenario - if anything it's marginally worse than
    just leaving the controller at one fixed nominal zc throughout. The
    root cause, verified directly: this project's footstep planner's
    x/y ZMP reference is completely independent of terrain HEIGHT (only
    footstep x/y positions matter, and apply_terrain only changes z) -
    so there was never a real horizontal mismatch for gain-scheduling to
    fix in the first place; switching controller gains mid-walk merely
    introduces its own small transient. This locks in the actual
    measured comparison so the finding doesn't silently drift."""
    gait = GaitParams(n_steps=6)
    footsteps = apply_terrain(plan_footsteps(gait), StepTerrain(step_x=0.9, step_height_m=-0.1))

    t, zx_ref, zy_ref, zmp_x_fixed, _ = _fixed_zc_run(gait, footsteps, DEFAULT_ZC)
    walk_gs = simulate_walk_gain_scheduled(gait, footsteps, DEFAULT_ZC)

    err_fixed = zmp_x_fixed - zx_ref
    err_scheduled = walk_gs.zmp_x - walk_gs.zmp_ref_x
    # both stay small (well within a reasonable ZMP-tracking margin) and
    # comparable in magnitude - gain scheduling doesn't meaningfully win here
    assert np.max(np.abs(err_fixed)) < 0.06
    assert np.max(np.abs(err_scheduled)) < 0.06
    assert np.max(np.abs(err_scheduled)) >= np.max(np.abs(err_fixed)) - 0.005  # not a clear win


def test_horizontal_zmp_reference_is_independent_of_terrain_height():
    """The root-cause fact behind the finding above, verified directly:
    apply_terrain only changes footstep.z, so the x/y ZMP reference this
    planner generates is bit-for-bit identical with or without a terrain
    height applied."""
    gait = GaitParams(n_steps=6)
    flat = plan_footsteps(gait)
    stepped = apply_terrain(plan_footsteps(gait), StepTerrain(step_x=0.9, step_height_m=-0.1))
    _, zx1, zy1 = zmp_reference_trajectory(gait, flat)
    _, zx2, zy2 = zmp_reference_trajectory(gait, stepped)
    assert np.allclose(zx1, zx2)
    assert np.allclose(zy1, zy2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
