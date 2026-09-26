"""
simulation/push_recovery_simulator.py

Injects a real external push (a velocity impulse to the CoM, representing
e.g. a shove) mid-walk, and compares two outcomes using the SAME
preview-control walk (simulation/walk_simulator.py is untouched by this
module - this is a separate, additive simulation path):

  1. NO RECOVERY: the nominal, pre-planned footsteps are kept exactly as
     planned, ignoring the disturbance - showing what the base LIPM+
     preview-control system (verified stable for the NOMINAL, undisturbed
     case in test_walk_simulation.py) actually does when hit by a real
     unplanned disturbance it was never designed to handle.

  2. WITH RECOVERY: the instant the push is detected, the very next
     upcoming footstep's landing position is replaced with the capture
     point (control/capture_point.py) computed from the post-push CoM
     state, clipped to a kinematically reachable single-step distance.

This directly answers the open SCOPE.md question about disturbance
rejection - not by asserting the system is robust, but by measuring,
for a specific real push magnitude, whether it stays stable, and
showing that a real, standard recovery mechanism (capture point) can
recover ZMP-in-support-polygon stability that the unrecovered case
loses.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from control.capture_point import capture_point_2d, clip_to_reachable_step
from planning.footstep_planner import (
    GaitParams, Footstep, plan_footsteps, zmp_reference_trajectory,
    foot_target_trajectories, support_polygon_at, _support_foot_positions,
)
from simulation.walk_simulator import WalkResult

MAX_STEP_LENGTH_M = 0.5  # stated kinematic limit on a single recovery step


@dataclass
class PushRecoveryResult:
    walk: WalkResult
    push_time_s: float
    push_velocity_xy: np.ndarray
    recovery_used: bool
    modified_footstep_index: int | None
    capture_point_xy: np.ndarray | None


def _simulate(gait: GaitParams, footsteps_initial: list[Footstep],
              push_time_s: float, push_velocity_xy: np.ndarray,
              use_recovery: bool, lipm_params: LIPMParams, preview_cfg: PreviewControllerConfig
              ) -> PushRecoveryResult:
    footsteps = copy.deepcopy(footsteps_initial)
    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    n = len(t)

    lipm_x = LIPM(lipm_params)
    lipm_y = LIPM(lipm_params)
    ctrl_x = ZMPPreviewController(lipm_x, preview_cfg)
    ctrl_y = ZMPPreviewController(lipm_y, preview_cfg)
    N = ctrl_x.N

    zx_pad = np.concatenate([zx_ref, np.full(N + 1, zx_ref[-1])])
    zy_pad = np.concatenate([zy_ref, np.full(N + 1, zy_ref[-1])])

    x_state = np.zeros(3)
    y_state = np.zeros(3)
    com_x, com_y = np.zeros(n), np.zeros(n)
    zmp_x, zmp_y = np.zeros(n), np.zeros(n)

    push_idx = int(round(push_time_s / gait.dt))
    pushed = False
    modified_idx = None
    cp_xy = None

    for k in range(n):
        com_x[k], com_y[k] = x_state[0], y_state[0]
        zmp_x[k] = lipm_x.zmp(x_state)
        zmp_y[k] = lipm_y.zmp(y_state)

        if k == push_idx and not pushed:
            x_state[1] += push_velocity_xy[0]
            y_state[1] += push_velocity_xy[1]
            pushed = True

            if use_recovery:
                omega = lipm_x.natural_frequency()
                cp_xy = capture_point_2d(np.array([x_state[0], y_state[0]]),
                                          np.array([x_state[1], y_state[1]]), omega)
                # find the next footstep whose swing hasn't started yet
                for i, step in enumerate(footsteps):
                    if step.start_time > t[k]:
                        stance_now = _support_foot_positions(gait, footsteps)[i]
                        clipped = clip_to_reachable_step(cp_xy, np.array(stance_now), MAX_STEP_LENGTH_M)
                        footsteps[i] = Footstep(x=float(clipped[0]), y=float(clipped[1]),
                                                 side=step.side, start_time=step.start_time,
                                                 end_time=step.end_time)
                        modified_idx = i
                        break
                # re-derive the reference from here on using the modified footsteps
                t2, zx_ref2, zy_ref2 = zmp_reference_trajectory(gait, footsteps)
                zx_pad2 = np.concatenate([zx_ref2, np.full(N + 1, zx_ref2[-1])])
                zy_pad2 = np.concatenate([zy_ref2, np.full(N + 1, zy_ref2[-1])])
                zx_pad[k:] = zx_pad2[k:len(zx_pad)]
                zy_pad[k:] = zy_pad2[k:len(zy_pad)]
                zx_ref[k:] = zx_ref2[k:len(zx_ref)]
                zy_ref[k:] = zy_ref2[k:len(zy_ref)]

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

    walk = WalkResult(t=t, com_x=com_x, com_y=com_y, zmp_x=zmp_x, zmp_y=zmp_y,
                       zmp_ref_x=zx_ref, zmp_ref_y=zy_ref, zmp_in_support=zmp_in_support,
                       max_zmp_margin_violation_m=max_violation, footsteps=footsteps)
    return PushRecoveryResult(walk=walk, push_time_s=push_time_s, push_velocity_xy=push_velocity_xy,
                               recovery_used=use_recovery, modified_footstep_index=modified_idx,
                               capture_point_xy=cp_xy)


def simulate_push_recovery(gait: GaitParams, push_time_s: float, push_velocity_xy: np.ndarray,
                            use_recovery: bool, lipm_params: LIPMParams | None = None,
                            preview_cfg: PreviewControllerConfig | None = None) -> PushRecoveryResult:
    lipm_params = lipm_params or LIPMParams(dt=gait.dt)
    preview_cfg = preview_cfg or PreviewControllerConfig()
    footsteps = plan_footsteps(gait)
    return _simulate(gait, footsteps, push_time_s, push_velocity_xy, use_recovery,
                      lipm_params, preview_cfg)
