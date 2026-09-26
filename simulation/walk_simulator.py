"""
simulation/walk_simulator.py

Ties the LIPM (dynamics/lipm.py), the ZMP preview controller
(control/zmp_preview_controller.py), and the footstep/ZMP reference
planner (planning/footstep_planner.py) together into a full walking
simulation, and checks the ONE criterion that actually matters for a
real biped: is the ZMP inside the support polygon at every instant?
(If it ever leaves the support polygon, the "robot" would physically
tip over - this is not a soft metric, it's the real stability
criterion this entire model exists to satisfy.)

Runs two independent LIPM+preview-controller instances, one per
horizontal axis (sagittal x, lateral y) - the standard decoupling
assumption for this class of model (see docs/SCOPE.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from planning.footstep_planner import (
    GaitParams, plan_footsteps, zmp_reference_trajectory, support_polygon_at,
)


@dataclass
class WalkResult:
    t: np.ndarray
    com_x: np.ndarray
    com_y: np.ndarray
    zmp_x: np.ndarray
    zmp_y: np.ndarray
    zmp_ref_x: np.ndarray
    zmp_ref_y: np.ndarray
    zmp_in_support: np.ndarray  # bool array - the real stability check, per timestep
    max_zmp_margin_violation_m: float  # 0.0 if always stable; >0 = how far ZMP left the polygon
    footsteps: list


def simulate_walk(gait: GaitParams | None = None,
                   lipm_params: LIPMParams | None = None,
                   preview_cfg: PreviewControllerConfig | None = None) -> WalkResult:
    gait = gait or GaitParams()
    lipm_params = lipm_params or LIPMParams(dt=gait.dt)
    preview_cfg = preview_cfg or PreviewControllerConfig()

    footsteps = plan_footsteps(gait)
    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    n = len(t)

    lipm_x = LIPM(lipm_params)
    lipm_y = LIPM(lipm_params)
    ctrl_x = ZMPPreviewController(lipm_x, preview_cfg)
    ctrl_y = ZMPPreviewController(lipm_y, preview_cfg)

    N = ctrl_x.N
    # pad the reference so the preview window never runs off the end of the
    # planned trajectory - hold the final value, matching a robot that has
    # finished walking and is standing still on its last footstep.
    zx_pad = np.concatenate([zx_ref, np.full(N + 1, zx_ref[-1])])
    zy_pad = np.concatenate([zy_ref, np.full(N + 1, zy_ref[-1])])

    x_state = np.zeros(3)
    y_state = np.zeros(3)
    com_x, com_y = np.zeros(n), np.zeros(n)
    zmp_x, zmp_y = np.zeros(n), np.zeros(n)

    for k in range(n):
        com_x[k], com_y[k] = x_state[0], y_state[0]
        zmp_x[k] = lipm_x.zmp(x_state)
        zmp_y[k] = lipm_y.zmp(y_state)

        jerk_x = ctrl_x.compute_jerk(x_state, zx_pad[k:k + N + 1])
        jerk_y = ctrl_y.compute_jerk(y_state, zy_pad[k:k + N + 1])
        x_state = lipm_x.step(x_state, jerk_x)
        y_state = lipm_y.step(y_state, jerk_y)

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
