"""
dynamics/gain_scheduled_walk_simulator.py

Closes the honestly-stated remaining half of docs/SCOPE.md item 4:
dynamics/terrain_adaptive_com_height.py fixed the hip-HEIGHT REFERENCE
consumed by leg IK, but explicitly said it did NOT couple the varying
zc into the LIPM/ZMP-preview-control horizontal (x, y) tracking loop
itself - that loop kept using ONE fixed nominal zc for its own gain
design. This module does that coupling, via GAIN SCHEDULING (not a
full nonlinear MPC re-derivation, which remains a separate, harder,
still-open research direction - see below for exactly why this is a
smaller, genuinely different thing).

WHY GAIN SCHEDULING IS VALID HERE (not hand-waved): dynamics/lipm.py's
CoM plant dynamics (the A, B matrices governing how [position, velocity,
acceleration] evolve under a jerk input) do NOT depend on zc AT ALL -
only the OUTPUT equation (ZMP = x - (zc/g)*xddot) does. So a
change in zc across a footstep-phase boundary does not require
re-deriving or blending any PLANT dynamics - it only means the
CONTROLLER's own internal model of "how CoM state maps to ZMP" (used
inside the preview-control Riccati design) should use the zc value that
is ACTUALLY correct for the upcoming phase, rather than one fixed
nominal value used everywhere. Since zc here only changes a handful of
times per walk (once per footstep, from
dynamics/terrain_adaptive_com_height.py's own zc profile), building a
FRESH LIPM + ZMPPreviewController for each new zc value (each a cheap,
one-off 4x4 Riccati solve) and CHAINING the physical [pos, vel, acc]
state across the switch is exact for the plant and a standard,
legitimate simplification for the controller gains - not the same as
solving a single, fully general nonlinear MPC over a continuously
varying height (which would additionally need to co-optimize CoM height
itself as a decision variable coupled to horizontal dynamics - a
substantially larger problem this module does not attempt).
"""
from __future__ import annotations

import numpy as np

from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from planning.footstep_planner import GaitParams, Footstep, zmp_reference_trajectory, support_polygon_at
from dynamics.terrain_adaptive_com_height import compute_zc_profile
from simulation.walk_simulator import WalkResult


def _phase_boundaries(gait: GaitParams, footsteps: list[Footstep], n: int, dt: float) -> list[int]:
    """Sample indices at which the CoM height model should switch to the
    NEXT footstep's zc value - at each footstep's END time (start of the
    following double support), giving the full double-support window as
    lead time before that height is actually needed for the next swing -
    matching the timing convention already used in
    dynamics/terrain_adaptive_com_height.py."""
    boundaries = [0]
    for step in footsteps:
        idx = int(round(step.end_time / dt))
        if 0 < idx < n:
            boundaries.append(idx)
    boundaries.append(n)
    return sorted(set(boundaries))


def simulate_walk_gain_scheduled(gait: GaitParams, footsteps: list[Footstep],
                                  default_zc: float, lipm_params: LIPMParams | None = None,
                                  preview_cfg: PreviewControllerConfig | None = None) -> WalkResult:
    """Same interface/output shape as simulation.walk_simulator.simulate_walk,
    but zc is allowed to vary per footstep phase (from
    dynamics.terrain_adaptive_com_height.compute_zc_profile) instead of
    being fixed for the whole walk. On FLAT terrain (all zc equal to
    default_zc), this must reproduce simulate_walk's own output exactly -
    verified in tests/test_gain_scheduled_lipm.py."""
    base_params = lipm_params or LIPMParams(dt=gait.dt)
    preview_cfg = preview_cfg or PreviewControllerConfig()

    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    n = len(t)
    dt = gait.dt
    zc_profile = compute_zc_profile(gait, footsteps, default_zc)

    x_state = np.zeros(3)
    y_state = np.zeros(3)
    com_x, com_y = np.zeros(n), np.zeros(n)
    zmp_x, zmp_y = np.zeros(n), np.zeros(n)
    e_x, e_y = 0.0, 0.0  # integral tracking error - a physical quantity belonging to the
                          # ongoing control loop, not to any one segment's particular gains;
                          # carried across segment boundaries just like x_state/y_state so a
                          # gain switch alone can't introduce a spurious discontinuity (see
                          # tests/test_gain_scheduled_lipm.py's flat-terrain exact-match check)

    boundaries = _phase_boundaries(gait, footsteps, n, dt)
    # which footstep's zc applies to each segment [boundaries[s], boundaries[s+1])
    for s in range(len(boundaries) - 1):
        k_start, k_end = boundaries[s], boundaries[s + 1]
        # the zc in force during this segment is whichever footstep's swing
        # STARTS at or after k_start (i.e. the upcoming/current phase) -
        # falling back to the last footstep's zc for the tail after the walk ends
        phase_idx = 0
        for i, step in enumerate(footsteps):
            if step.start_time >= t[k_start] - 1e-9:
                phase_idx = i
                break
        else:
            phase_idx = len(footsteps) - 1
        zc = zc_profile[phase_idx]

        params = LIPMParams(dt=base_params.dt, com_height_m=zc, gravity=base_params.gravity)
        lipm_x = LIPM(params)
        lipm_y = LIPM(params)
        ctrl_x = ZMPPreviewController(lipm_x, preview_cfg)
        ctrl_y = ZMPPreviewController(lipm_y, preview_cfg)
        ctrl_x._e = e_x
        ctrl_y._e = e_y
        N = ctrl_x.N
        zx_pad = np.concatenate([zx_ref, np.full(N + 1, zx_ref[-1])])
        zy_pad = np.concatenate([zy_ref, np.full(N + 1, zy_ref[-1])])

        for k in range(k_start, k_end):
            com_x[k], com_y[k] = x_state[0], y_state[0]
            zmp_x[k] = lipm_x.zmp(x_state)
            zmp_y[k] = lipm_y.zmp(y_state)
            jerk_x = ctrl_x.compute_jerk(x_state, zx_pad[k:k + N + 1])
            jerk_y = ctrl_y.compute_jerk(y_state, zy_pad[k:k + N + 1])
            x_state = lipm_x.step(x_state, jerk_x)
            y_state = lipm_y.step(y_state, jerk_y)
        e_x, e_y = ctrl_x._e, ctrl_y._e

    zmp_in_support = np.zeros(n, dtype=bool)
    max_violation = 0.0
    for k in range(n):
        xmin, xmax, ymin, ymax = support_polygon_at(t[k], gait, footsteps)
        inside = (xmin <= zmp_x[k] <= xmax) and (ymin <= zmp_y[k] <= ymax)
        zmp_in_support[k] = inside
        if not inside:
            vx = max(xmin - zmp_x[k], zmp_x[k] - xmax, 0.0)
            vy = max(ymin - zmp_y[k], zmp_y[k] - ymax, 0.0)
            max_violation = max(max_violation, vx, vy)

    return WalkResult(
        t=t, com_x=com_x, com_y=com_y, zmp_x=zmp_x, zmp_y=zmp_y,
        zmp_ref_x=zx_ref, zmp_ref_y=zy_ref,
        zmp_in_support=zmp_in_support, max_zmp_margin_violation_m=max_violation,
        footsteps=footsteps,
    )
