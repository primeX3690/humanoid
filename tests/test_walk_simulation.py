"""tests/test_walk_simulation.py — checks the ONE criterion that actually
matters for a biped: does the ZMP stay inside the support polygon for the
ENTIRE walk? This is not a soft/approximate check - leaving the support
polygon means the modeled robot would physically tip over."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams
from simulation.walk_simulator import simulate_walk


def test_zmp_stays_in_support_polygon_for_entire_walk():
    result = simulate_walk(GaitParams(n_steps=8))
    n_violations = np.sum(~result.zmp_in_support)
    assert n_violations == 0, (
        f"ZMP left the support polygon at {n_violations}/{len(result.t)} timesteps "
        f"(max violation {result.max_zmp_margin_violation_m*1000:.1f}mm) - "
        f"this walk would physically fall over"
    )
    assert result.max_zmp_margin_violation_m == 0.0


def test_stability_margin_is_real_not_a_knife_edge():
    """Regression guard for the tuning found in docs/BUGS_FOUND.md: checks
    a real, positive minimum distance from the ZMP to the support polygon
    boundary across a longer 16-step walk, not just zero violations (which
    could technically pass at exactly zero margin, an unstable-in-practice
    knife edge)."""
    from planning.footstep_planner import support_polygon_at
    gait = GaitParams(n_steps=16)
    result = simulate_walk(gait)
    margins = []
    for k in range(len(result.t)):
        xmin, xmax, ymin, ymax = support_polygon_at(result.t[k], gait, result.footsteps)
        mx = min(result.zmp_x[k] - xmin, xmax - result.zmp_x[k])
        my = min(result.zmp_y[k] - ymin, ymax - result.zmp_y[k])
        margins.append(min(mx, my))
    min_margin_mm = min(margins) * 1000
    assert min_margin_mm > 5.0, (
        f"minimum ZMP-to-support-polygon-edge margin is only {min_margin_mm:.2f}mm "
        f"across a 16-step walk - too close to a knife edge for the nominal "
        f"(disturbance-free) case this model represents"
    )


def test_overly_aggressive_tracking_gain_reduces_stability_not_improves_it():
    """Regression guard for the specific, counterintuitive finding in
    docs/BUGS_FOUND.md: cranking Q up by 4 orders of magnitude from the
    tuned default does NOT improve stability - it makes the non-minimum-
    phase transient large enough to leave the support polygon. If this
    test ever fails, the earlier tuning finding no longer holds and needs
    re-investigating, not silently ignoring."""
    from control.zmp_preview_controller import PreviewControllerConfig
    gait = GaitParams(n_steps=8)
    result_tuned = simulate_walk(gait, preview_cfg=PreviewControllerConfig(Q=30, R=1.0))
    result_aggressive = simulate_walk(gait, preview_cfg=PreviewControllerConfig(Q=1e6, R=1.0))
    assert np.sum(~result_tuned.zmp_in_support) == 0
    assert np.sum(~result_aggressive.zmp_in_support) > 0, (
        "expected the overly-aggressive Q=1e6 gain to reproduce the known "
        "support-polygon violation from the non-minimum-phase transient - "
        "if it no longer does, something else about the model changed"
    )


def test_com_ends_near_final_footstep():
    """After walking, the CoM should have actually traveled forward with
    the feet, landing near the last footstep's x position - not stayed at
    the origin (which would mean the controller isn't actually driving
    forward progress, just holding balance in place)."""
    gait = GaitParams(n_steps=8, step_length_m=0.3)
    result = simulate_walk(gait)
    final_com_x = result.com_x[-1]
    final_foot_x = result.footsteps[-1].x
    assert abs(final_com_x - final_foot_x) < 0.15, (
        f"CoM ended at x={final_com_x:.3f}, expected near the final foot "
        f"x={final_foot_x:.3f} (within a reasonable ZMP-to-CoM offset)"
    )
    assert final_com_x > 1.0, "CoM barely moved forward across 8 steps - not a real walk"


def test_no_nan_or_divergence_over_a_longer_walk():
    """A longer walk (16 steps) must not blow up numerically - a basic but
    real robustness check most control-theory bugs would fail immediately."""
    result = simulate_walk(GaitParams(n_steps=16))
    assert np.all(np.isfinite(result.com_x))
    assert np.all(np.isfinite(result.com_y))
    assert np.all(np.isfinite(result.zmp_x))
    assert np.all(np.isfinite(result.zmp_y))
    assert np.max(np.abs(result.com_x)) < 100.0, "CoM diverged - controller is unstable at this config"


def test_lateral_zmp_shifts_toward_support_foot_each_step():
    """A real walking gait must shift weight side-to-side onto whichever
    foot is on the ground; the lateral ZMP reference must alternate sign
    (or at least move meaningfully) between right-support and
    left-support steps - not stay pinned at y=0 the whole time."""
    result = simulate_walk(GaitParams(n_steps=4))
    right_support_zmp_y = result.zmp_ref_y[result.zmp_ref_y < -0.05]
    left_support_zmp_y = result.zmp_ref_y[result.zmp_ref_y > 0.05]
    assert len(right_support_zmp_y) > 0, "no right-foot-support phase found in ZMP reference"
    assert len(left_support_zmp_y) > 0, "no left-foot-support phase found in ZMP reference"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
