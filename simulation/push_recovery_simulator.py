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

MULTI-STEP RECOVERY (`n_recovery_steps` > 1, closing docs/SCOPE.md's
"multi-step (not just single-step) push-recovery replanning" gap):
after the first recovered footstep, the loop below KEEPS GOING -
each time the simulation reaches the start of another upcoming
footstep's swing (up to `n_recovery_steps` of them total), it
recomputes a FRESH capture point from the ACTUALLY-SIMULATED CoM state
at that moment (not a hand-rolled prediction of where the CoM "should"
be - the real closed-loop preview-controller dynamics, already running
against the previously-modified reference, have already carried the
state forward correctly by the time we get there) and replaces that
footstep too. This directly addresses the single-step version's
documented weakness (see docs/BUGS_FOUND.md): with only one footstep
corrected, every subsequent PRE-PLANNED footstep stays at its original
ABSOLUTE position, so for a large enough push the walk becomes
"consistent with a robot that was never pushed" one step too early,
reintroducing tracking error the single correction didn't buy back.
Once the recovery window closes, every STILL-NOMINAL footstep after it
is RIGIDLY SHIFTED (same x/y offset applied to all of them) by however
far the last recovered footstep ended up from where it was originally
planned - preserving the ORIGINAL gait's relative step pattern (length,
width, alternation) while making the rest of the walk consistent with
where the robot actually now is, instead of asking it to snap back to
the old absolute path.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from control.capture_point import capture_point_2d, clip_to_reachable_step
from control.n_step_capture_planner import blended_recovery_target, alpha_schedule
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
    modified_footstep_index: int | None       # kept for backward compatibility: the FIRST recovered footstep
    capture_point_xy: np.ndarray | None        # kept for backward compatibility: the FIRST capture point
    modified_footstep_indices: list[int]       # every footstep the recovery loop replaced, in order
    capture_points_xy: list[np.ndarray]        # the capture point computed for each of those, same order
    n_recovery_steps_requested: int
    remaining_footsteps_shifted: bool          # whether the post-recovery-window rigid shift was applied


def _simulate(gait: GaitParams, footsteps_initial: list[Footstep],
              push_time_s: float, push_velocity_xy: np.ndarray,
              use_recovery: bool, lipm_params: LIPMParams, preview_cfg: PreviewControllerConfig,
              n_recovery_steps: int = 1, use_blended_targets: bool = False, decay_ratio: float = 0.5,
              ) -> PushRecoveryResult:
    footsteps = copy.deepcopy(footsteps_initial)
    original_footsteps = copy.deepcopy(footsteps_initial)  # nominal positions, for the closing rigid shift
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
    modified_indices: list[int] = []
    cp_list: list[np.ndarray] = []
    shifted = False
    alphas = alpha_schedule(n_recovery_steps, decay_ratio) if use_blended_targets else None

    def re_derive_reference(k: int) -> None:
        nonlocal zx_ref, zy_ref, zx_pad, zy_pad
        t2, zx_ref2, zy_ref2 = zmp_reference_trajectory(gait, footsteps)
        zx_pad2 = np.concatenate([zx_ref2, np.full(N + 1, zx_ref2[-1])])
        zy_pad2 = np.concatenate([zy_ref2, np.full(N + 1, zy_ref2[-1])])
        zx_pad[k:] = zx_pad2[k:len(zx_pad)]
        zy_pad[k:] = zy_pad2[k:len(zy_pad)]
        zx_ref[k:] = zx_ref2[k:len(zx_ref)]
        zy_ref[k:] = zy_ref2[k:len(zy_ref)]

    def recover_footstep(i: int, k: int) -> None:
        omega = lipm_x.natural_frequency()
        com_xy = np.array([x_state[0], y_state[0]])
        com_vel_xy = np.array([x_state[1], y_state[1]])
        cp_xy = capture_point_2d(com_xy, com_vel_xy, omega)
        stance_now = np.array(_support_foot_positions(gait, footsteps)[i])
        step = footsteps[i]

        if use_blended_targets:
            # JOINT N-step target: blend the immediate capture point with
            # this footstep's own NOMINAL (properly-alternating, correctly-
            # spaced) position, decaying back toward nominal over the
            # recovery window - see control/n_step_capture_planner.py for
            # why this structurally avoids the naive version's same-side
            # crowding failure mode.
            alpha = float(alphas[len(modified_indices)])
            nominal_xy = np.array([original_footsteps[i].x, original_footsteps[i].y])
            target = blended_recovery_target(com_xy, com_vel_xy, omega, nominal_xy,
                                              stance_now, alpha, MAX_STEP_LENGTH_M)
        else:
            target = clip_to_reachable_step(cp_xy, stance_now, MAX_STEP_LENGTH_M)

        footsteps[i] = Footstep(x=float(target[0]), y=float(target[1]), side=step.side,
                                 start_time=step.start_time, end_time=step.end_time, z=step.z)
        modified_indices.append(i)
        cp_list.append(cp_xy)
        re_derive_reference(k)

    def shift_remaining_footsteps(last_recovered_idx: int, k: int) -> None:
        """Preserve the ORIGINAL gait's relative pattern (step length,
        width, alternation) for everything after the recovery window,
        while making it consistent with where the recovered footstep
        actually ended up, instead of snapping back to the absolute
        pre-push path - see module docstring."""
        nonlocal shifted
        offset_x = footsteps[last_recovered_idx].x - original_footsteps[last_recovered_idx].x
        offset_y = footsteps[last_recovered_idx].y - original_footsteps[last_recovered_idx].y
        if abs(offset_x) < 1e-12 and abs(offset_y) < 1e-12:
            return  # nothing to shift (e.g. capture point coincided with the nominal target)
        for j in range(last_recovered_idx + 1, len(footsteps)):
            nominal = original_footsteps[j]
            footsteps[j] = Footstep(x=nominal.x + offset_x, y=nominal.y + offset_y, side=nominal.side,
                                     start_time=nominal.start_time, end_time=nominal.end_time, z=nominal.z)
        shifted = True
        re_derive_reference(k)

    for k in range(n):
        com_x[k], com_y[k] = x_state[0], y_state[0]
        zmp_x[k] = lipm_x.zmp(x_state)
        zmp_y[k] = lipm_y.zmp(y_state)

        if k == push_idx and not pushed:
            x_state[1] += push_velocity_xy[0]
            y_state[1] += push_velocity_xy[1]
            pushed = True

            if use_recovery:
                # find the next footstep whose swing hasn't started yet
                # single-step mode (n_recovery_steps=1, the default) stops after
                # ONE recovered footstep and never shifts the rest - byte-for-byte
                # the original behavior. The elif branch below (multi-step) is
                # what keeps going and eventually shifts - new, additive, never
                # reached when n_recovery_steps=1.
                for i, step in enumerate(footsteps):
                    if step.start_time > t[k]:
                        recover_footstep(i, k)
                        break

        elif pushed and use_recovery and modified_indices and len(modified_indices) < n_recovery_steps:
            last_idx = modified_indices[-1]
            next_idx = last_idx + 1
            if next_idx < len(footsteps):
                next_step = footsteps[next_idx]
                # trigger exactly once, on the sample where we cross into this
                # footstep's swing start - using the ACTUALLY-SIMULATED state
                # at that moment (see module docstring), not a hand-predicted one
                if t[k] >= next_step.start_time and (k == 0 or t[k - 1] < next_step.start_time):
                    recover_footstep(next_idx, k)
                    if len(modified_indices) == n_recovery_steps and next_idx + 1 < len(footsteps):
                        shift_remaining_footsteps(next_idx, k)

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
    return PushRecoveryResult(
        walk=walk, push_time_s=push_time_s, push_velocity_xy=push_velocity_xy,
        recovery_used=use_recovery,
        modified_footstep_index=(modified_indices[0] if modified_indices else None),
        capture_point_xy=(cp_list[0] if cp_list else None),
        modified_footstep_indices=modified_indices,
        capture_points_xy=cp_list,
        n_recovery_steps_requested=n_recovery_steps,
        remaining_footsteps_shifted=shifted,
    )


def simulate_push_recovery(gait: GaitParams, push_time_s: float, push_velocity_xy: np.ndarray,
                            use_recovery: bool, lipm_params: LIPMParams | None = None,
                            preview_cfg: PreviewControllerConfig | None = None,
                            n_recovery_steps: int = 1, use_blended_targets: bool = False,
                            decay_ratio: float = 0.5) -> PushRecoveryResult:
    """n_recovery_steps=1 (the default, matching the original single-step
    behavior bit-for-bit) replans only the very next footstep. >1 keeps
    replanning subsequent footsteps too, then rigidly shifts whatever's
    left - see module docstring for why.

    use_blended_targets=True switches from the naive "independently
    repeat the capture-point formula" multi-step strategy (measured to
    crowd consecutive footsteps toward the same side - see
    docs/BUGS_FOUND.md) to the joint, blended N-step scheme in
    control/n_step_capture_planner.py, which explicitly decays each
    footstep's target back toward its own nominal (alternating,
    correctly-spaced) position over the recovery window. Has no effect
    when n_recovery_steps=1 (nothing to decay across)."""
    lipm_params = lipm_params or LIPMParams(dt=gait.dt)
    preview_cfg = preview_cfg or PreviewControllerConfig()
    footsteps = plan_footsteps(gait)
    return _simulate(gait, footsteps, push_time_s, push_velocity_xy, use_recovery,
                      lipm_params, preview_cfg, n_recovery_steps=n_recovery_steps,
                      use_blended_targets=use_blended_targets, decay_ratio=decay_ratio)
