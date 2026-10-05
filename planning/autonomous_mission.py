"""
planning/autonomous_mission.py

Mission-level autonomy for the walking stack: given ONLY a goal (walk
`goal_distance_m` forward), the system plans the gait, executes it in
closed loop, detects disturbances ON ITS OWN (no oracle telling it when a
push happened - unlike simulation/push_recovery_simulator.py, which is
told the push time), triggers the joint N-step recovery
(control/n_step_capture_planner.py) automatically, and a supervisor
decides whether to continue, finish, or SAFE-STOP - with no human in the
loop after the goal is set.

Supervisor state machine:

    WALKING --(divergence detected)--> RECOVERING --(settled)--> WALKING
    WALKING --(all planned steps done)--> REACHED
    any     --(ZMP outside the support polygon for > `abort_violation_s`)--> SAFE_STOP (aborted)

Disturbance detection is by CAPTURE-POINT DIVERGENCE: the divergent
component of motion xi = x + xdot/omega is compared with the same
quantity along the nominal plan; when the distance exceeds
`detect_threshold_m` the disturbance is declared. This is the standard
signal used on real legged robots (it is what a state estimator would
feed the recovery logic), and it is computed from the robot's own
(simulated) state, not from knowledge of the push.

HONEST SCOPE (what "autonomous" means here, and what it does not):
  * covered   : goal -> gait planning, closed-loop execution, disturbance
                detection, automatic recovery, abort/safe-stop supervision.
  * NOT covered: perception (no cameras/LiDAR), mapping/SLAM, obstacle
                avoidance, terrain classification, and turning (hip yaw is
                still commanded to zero in kinematics/leg_ik.py, so goals
                are straight-line distances only). State is taken from the
                simulator, i.e. a perfect state estimator is ASSUMED; a real
                robot needs an IMU + kinematic state estimator here. These
                are the components a real unmanned deployment would add.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace

import numpy as np

from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from control.capture_point import capture_point_2d
from control.n_step_capture_planner import blended_recovery_target, alpha_schedule
from planning.footstep_planner import (
    GaitParams, Footstep, plan_footsteps, zmp_reference_trajectory,
    support_polygon_at, _support_foot_positions,
)

MAX_STEP_LENGTH_M = 0.5


@dataclass
class MissionConfig:
    detect_threshold_m: float = 0.10       # capture-point deviation that counts as a disturbance
    recovery_steps: int = 3                # N for the blended N-step recovery
    decay_ratio: float = 0.5
    abort_violation_s: float = 0.6         # continuous ZMP-outside-support time that triggers safe stop
    goal_tolerance_m: float = 0.25


@dataclass
class MissionResult:
    status: str                            # "REACHED" | "SAFE_STOP"
    goal_distance_m: float
    final_com_x: float
    n_steps_planned: int
    disturbances_detected: int
    recovery_steps_executed: int
    total_violation_samples: int
    events: list = field(default_factory=list)   # (time_s, message)
    t: np.ndarray | None = None
    com_x: np.ndarray | None = None
    com_y: np.ndarray | None = None
    footsteps: list | None = None

    @property
    def goal_error_m(self) -> float:
        return abs(self.goal_distance_m - self.final_com_x)


def plan_gait_for_goal(goal_distance_m: float, template: GaitParams | None = None) -> GaitParams:
    """Choose the number of steps so the last footstep lands nearest the goal
    (first step is half length, as in plan_footsteps). An earlier version added
    one extra closing step and overshot a 3 m goal by 0.36 m with no disturbance
    at all (caught by tests/test_autonomous_mission.py's goal-accuracy test)."""
    template = template or GaitParams()
    L = template.step_length_m
    n = 1 + int(math.floor((goal_distance_m - L / 2.0) / L + 0.5))   # last footstep lands nearest the goal
    return replace(template, n_steps=max(2, n))


def run_mission(goal_distance_m: float, disturbances: list[tuple[float, float, float]] | None = None,
                config: MissionConfig | None = None, gait_template: GaitParams | None = None,
                lipm_params: LIPMParams | None = None,
                preview_cfg: PreviewControllerConfig | None = None) -> MissionResult:
    """`disturbances` = [(time_s, vx, vy), ...] velocity impulses applied by the
    ENVIRONMENT (the test harness); the mission controller is never told about them."""
    cfg = config or MissionConfig()
    gait = plan_gait_for_goal(goal_distance_m, gait_template)
    lipm_params = lipm_params or LIPMParams(dt=gait.dt)
    preview_cfg = preview_cfg or PreviewControllerConfig()
    disturbances = sorted(disturbances or [])

    nominal = plan_footsteps(gait)
    footsteps = copy.deepcopy(nominal)
    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    n = len(t)
    dt = gait.dt

    lipm_x, lipm_y = LIPM(lipm_params), LIPM(lipm_params)
    ctrl_x, ctrl_y = ZMPPreviewController(lipm_x, preview_cfg), ZMPPreviewController(lipm_y, preview_cfg)
    N = ctrl_x.N
    omega = lipm_x.natural_frequency()
    zx_pad = np.concatenate([zx_ref, np.full(N + 1, zx_ref[-1])])
    zy_pad = np.concatenate([zy_ref, np.full(N + 1, zy_ref[-1])])

    # nominal (undisturbed) plant run gives the reference capture-point trajectory the
    # supervisor compares against - computed once, offline, from the plan itself
    nom_xi = _nominal_capture_point(gait, nominal, lipm_params, preview_cfg)

    x_state, y_state = np.zeros(3), np.zeros(3)
    com_x, com_y = np.zeros(n), np.zeros(n)
    state = "WALKING"
    events: list = []
    detected = 0
    recovered_steps: list[int] = []
    all_recovered: set[int] = set()
    ref_offset = np.zeros(2)     # capture-point baseline offset accepted after a completed recovery
    alphas = alpha_schedule(cfg.recovery_steps, cfg.decay_ratio)
    violation_run = 0
    total_violations = 0
    status = "REACHED"
    k_end = n
    dist_iter = iter(disturbances)
    next_dist = next(dist_iter, None)

    def rederive(k: int) -> None:
        nonlocal zx_pad, zy_pad
        _, zx2, zy2 = zmp_reference_trajectory(gait, footsteps)
        zx_pad2 = np.concatenate([zx2, np.full(N + 1, zx2[-1])])
        zy_pad2 = np.concatenate([zy2, np.full(N + 1, zy2[-1])])
        zx_pad[k:] = zx_pad2[k:len(zx_pad)]
        zy_pad[k:] = zy_pad2[k:len(zy_pad)]

    def recover_next(k: int, idx: int | None = None) -> None:
        """Replan footstep `idx` (or, if None, the first footstep whose swing has not started).
        The continuing-window call passes idx explicitly: at the sample where footstep i+1's
        swing begins its start_time equals t[k], so the "first not-yet-started" search would
        skip it and land on i+2 (the first version recovered steps 3, 5, 7 instead of 3, 4, 5)."""
        for i, step in enumerate(footsteps):
            if (idx is None and step.start_time > t[k] and i not in recovered_steps) or i == idx:
                alpha = float(alphas[min(len(recovered_steps), len(alphas) - 1)])
                stance = np.array(_support_foot_positions(gait, footsteps)[i])
                nominal_xy = np.array([nominal[i].x, nominal[i].y])
                target = blended_recovery_target(
                    np.array([x_state[0], y_state[0]]), np.array([x_state[1], y_state[1]]),
                    omega, nominal_xy, stance, alpha, MAX_STEP_LENGTH_M)
                footsteps[i] = Footstep(x=float(target[0]), y=float(target[1]), side=step.side,
                                         start_time=step.start_time, end_time=step.end_time, z=step.z)
                recovered_steps.append(i)
                all_recovered.add(i)
                rederive(k)
                events.append((float(t[k]), f"recovery footstep {i} -> ({target[0]:.2f}, {target[1]:.2f})"))
                return

    def rebaseline_and_shift(k: int, xi_now: np.ndarray) -> None:
        """Recovery finished: accept the current gait as the new baseline. The still-
        nominal footsteps are shifted rigidly by the last recovered footstep's offset
        (same rule as simulation/push_recovery_simulator.py) and the capture-point
        reference is offset to match, otherwise the supervisor would keep comparing
        against a plan the robot has deliberately left and re-trigger forever (the first
        version of this file detected ONE push four times for exactly that reason)."""
        nonlocal ref_offset
        last = max(recovered_steps)
        off = np.array([footsteps[last].x - nominal[last].x, footsteps[last].y - nominal[last].y])
        for j in range(last + 1, len(footsteps)):
            nm = nominal[j]
            footsteps[j] = Footstep(x=nm.x + off[0], y=nm.y + off[1], side=nm.side,
                                     start_time=nm.start_time, end_time=nm.end_time, z=nm.z)
        rederive(k)
        ref_offset = xi_now - nom_xi[k]

    for k in range(n):
        com_x[k], com_y[k] = x_state[0], y_state[0]

        # --- environment applies disturbances (mission controller is not informed) ---
        while next_dist is not None and t[k] >= next_dist[0]:
            x_state[1] += next_dist[1]
            y_state[1] += next_dist[2]
            next_dist = next(dist_iter, None)

        # --- supervisor: capture-point divergence from the nominal plan ---
        xi = capture_point_2d(np.array([x_state[0], y_state[0]]),
                               np.array([x_state[1], y_state[1]]), omega)
        dev = float(np.linalg.norm(xi - nom_xi[k] - ref_offset))

        if state == "WALKING" and dev > cfg.detect_threshold_m:
            state = "RECOVERING"
            detected += 1
            recovered_steps.clear()
            events.append((float(t[k]), f"disturbance detected (capture-point deviation {dev:.2f} m)"))
            recover_next(k)
        elif state == "RECOVERING":
            # keep replanning upcoming footsteps until the window is used up or state has settled
            if len(recovered_steps) < cfg.recovery_steps:
                last = recovered_steps[-1] if recovered_steps else -1
                if last + 1 < len(footsteps) and t[k] >= footsteps[last + 1].start_time and \
                        (k == 0 or t[k - 1] < footsteps[last + 1].start_time):
                    recover_next(k, last + 1)
            window_done = len(recovered_steps) >= cfg.recovery_steps
            last_landed = bool(recovered_steps) and t[k] >= footsteps[recovered_steps[-1]].end_time
            if window_done and last_landed:
                rebaseline_and_shift(k, xi)
                state = "WALKING"
                events.append((float(t[k]), "recovery window complete - plan shifted, resuming walking"))

        # --- ZMP safety monitor ---
        zmp = np.array([lipm_x.zmp(x_state), lipm_y.zmp(y_state)])
        xmin, xmax, ymin, ymax = support_polygon_at(t[k], gait, footsteps)
        inside = xmin <= zmp[0] <= xmax and ymin <= zmp[1] <= ymax
        if not inside:
            violation_run += 1
            total_violations += 1
        else:
            violation_run = 0
        if violation_run * dt > cfg.abort_violation_s:
            status = "SAFE_STOP"
            events.append((float(t[k]), "ZMP outside support polygon too long - SAFE STOP"))
            k_end = k + 1
            break

        jx = ctrl_x.compute_jerk(x_state, zx_pad[k:k + N + 1])
        jy = ctrl_y.compute_jerk(y_state, zy_pad[k:k + N + 1])
        x_state, y_state = lipm_x.step(x_state, jx), lipm_y.step(y_state, jy)

    if status == "REACHED":
        events.append((float(t[-1]), "goal sequence completed"))

    return MissionResult(
        status=status, goal_distance_m=goal_distance_m, final_com_x=float(com_x[k_end - 1]),
        n_steps_planned=gait.n_steps, disturbances_detected=detected,
        recovery_steps_executed=len(all_recovered),
        total_violation_samples=total_violations, events=events,
        t=t[:k_end], com_x=com_x[:k_end], com_y=com_y[:k_end], footsteps=footsteps,
    )


def _nominal_capture_point(gait: GaitParams, footsteps: list[Footstep], lipm_params: LIPMParams,
                            preview_cfg: PreviewControllerConfig) -> np.ndarray:
    """Capture point (x, y) at every sample of the UNDISTURBED plan."""
    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    lx, ly = LIPM(lipm_params), LIPM(lipm_params)
    cx, cy = ZMPPreviewController(lx, preview_cfg), ZMPPreviewController(ly, preview_cfg)
    N = cx.N
    zx_pad = np.concatenate([zx_ref, np.full(N + 1, zx_ref[-1])])
    zy_pad = np.concatenate([zy_ref, np.full(N + 1, zy_ref[-1])])
    om = lx.natural_frequency()
    xs, ys = np.zeros(3), np.zeros(3)
    out = np.zeros((len(t), 2))
    for k in range(len(t)):
        out[k] = capture_point_2d(np.array([xs[0], ys[0]]), np.array([xs[1], ys[1]]), om)
        xs = lx.step(xs, cx.compute_jerk(xs, zx_pad[k:k + N + 1]))
        ys = ly.step(ys, cy.compute_jerk(ys, zy_pad[k:k + N + 1]))
    return out
