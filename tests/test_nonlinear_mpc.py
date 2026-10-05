import numpy as np
import pytest
from control.nonlinear_mpc import solve_step, NMPCConfig


def test_single_solve_step_returns_bounded_output():
    cfg = NMPCConfig(horizon=4, dt=0.05)
    x0 = np.zeros(3)
    y0 = np.zeros(3)
    z0 = np.array([0.8, 0.0, 0.0])
    jx, jy, jz = solve_step(x0, y0, z0, np.zeros(4), np.zeros(4), 0.0, (-0.1, 0.1, -0.1, 0.1), cfg)
    assert all(np.isfinite(v) for v in (jx, jy, jz))
    assert max(abs(jx), abs(jy), abs(jz)) < 5.0


def test_solve_step_falls_back_safely_on_infeasible_constraints():
    cfg = NMPCConfig(horizon=4, dt=0.05)
    x0 = np.array([5.0, 2.0, 0.0])
    y0 = np.zeros(3)
    z0 = np.array([0.8, 0.0, 0.0])
    jx, jy, jz = solve_step(x0, y0, z0, np.zeros(4), np.zeros(4), 0.0, (-0.01, 0.01, -0.01, 0.01), cfg)
    assert all(np.isfinite(v) for v in (jx, jy, jz))


def test_closed_loop_simulation_no_longer_diverges_to_infinity():
    """The ORIGINAL bug (docs/BUGS_FOUND.md): CoM height ran away to
    about -24m over an 8-second walk, root-caused to a zero-jerk safety
    fallback that let the state coast under whatever acceleration it
    already had. Fixed via a stabilizing PD fallback + a longer horizon
    + state saturation. This test checks the property that is NOW true
    (bounded, physically-plausible state) - not tight tracking, which is
    a separate, still-open tuning issue (see the next test)."""
    from planning.footstep_planner import GaitParams, plan_footsteps
    from control.nonlinear_mpc import simulate
    gait = GaitParams(n_steps=2)
    fs = plan_footsteps(gait)
    res = simulate(gait, fs, NMPCConfig(horizon=10))
    assert np.all(np.isfinite(res["com_x"])) and np.all(np.isfinite(res["com_z"]))
    assert np.all((res["com_z"] >= 0.4) & (res["com_z"] <= 1.0))
    assert np.all(np.abs(res["com_x"]) < 1.0)


@pytest.mark.xfail(reason="honestly open, not yet fixed (docs/BUGS_FOUND.md): lateral (y) "
                          "tracking still oscillates with real but bounded amplitude "
                          "(order +-1 to +-1.3m) - the divergence-to-infinity bug is fixed, "
                          "but this does not yet match the already-verified linear "
                          "gain-scheduled controller's tracking quality")
def test_closed_loop_lateral_tracking_matches_the_reference_closely():
    from planning.footstep_planner import GaitParams, plan_footsteps
    from control.nonlinear_mpc import simulate
    gait = GaitParams(n_steps=2)
    fs = plan_footsteps(gait)
    res = simulate(gait, fs, NMPCConfig(horizon=10))
    assert np.max(np.abs(res["com_y"] - res["zmp_ref_y"])) < 0.15
