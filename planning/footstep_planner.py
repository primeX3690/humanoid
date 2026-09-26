"""
planning/footstep_planner.py

Generates a footstep sequence (alternating left/right foot placements)
and the corresponding piecewise-constant ZMP reference trajectory that
control/zmp_preview_controller.py tracks.

IMPORTANT convention, gotten wrong on the first pass and fixed after a
failing test caught it (see docs/BUGS_FOUND.md): during a single-support
phase, the ZMP reference must sit under the STATIONARY (stance) foot -
i.e. the foot that already landed on a PREVIOUS step - not under the
SWINGING foot's upcoming target. Getting this backwards makes the
reference "lead" the actual stance foot by a full step, which the
preview controller then dutifully (and correctly, for a wrong target)
tracks - straight out of the real support polygon. Real biped
gait-pattern generators track the same convention used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaitParams:
    step_length_m: float = 0.3      # forward distance per step
    step_width_m: float = 0.12      # lateral distance between feet (half-width each side of centerline) -
                                     # kept narrow deliberately: see docs/BUGS_FOUND.md for why a wider
                                     # stance combined with a short double-support phase is dynamically
                                     # infeasible for this LIPM to track without leaving the support polygon
    step_duration_s: float = 0.8    # time per single-support phase
    double_support_s: float = 0.4   # time per double-support phase (both feet down, ZMP ramps between them) -
                                     # tuned together with step_width_m/Q, not picked independently (see below)
    n_steps: int = 8                # number of steps to plan
    dt: float = 0.01


@dataclass
class Footstep:
    x: float
    y: float
    side: str  # "left" or "right" - which foot LANDS here
    start_time: float   # start of the single-support phase during which this foot swings TO this position
    end_time: float     # this foot lands (touches down) at end_time
    z: float = 0.0      # terrain height at this footstep (0.0 = flat ground, the default/original
                         # behavior everywhere this field isn't explicitly set - see terrain/terrain_profile.py)


def plan_footsteps(gait: GaitParams) -> list[Footstep]:
    """Alternating straight-line walk. footsteps[i] is the i-th foot to
    land; during [start_time, end_time) that foot is SWINGING toward this
    position while the OPPOSITE foot (wherever it last landed, or its
    initial stance position for the very first couple of steps) is the
    stance/support foot bearing weight - see zmp_reference_trajectory()."""
    steps = []
    t = gait.double_support_s  # initial double support before the first swing begins
    x = 0.0
    for i in range(gait.n_steps):
        side = "right" if i % 2 == 0 else "left"
        x += gait.step_length_m if i > 0 else gait.step_length_m / 2.0  # shorter first step, standard practice
        y = -gait.step_width_m / 2.0 if side == "right" else gait.step_width_m / 2.0
        steps.append(Footstep(x=x, y=y, side=side,
                               start_time=t, end_time=t + gait.step_duration_s))
        t += gait.step_duration_s + gait.double_support_s
    return steps


def _support_foot_positions(gait: GaitParams, footsteps: list[Footstep]) -> list[tuple[float, float]]:
    """support[i] = position of the STATIONARY foot during footsteps[i]'s
    single-support (swing) phase - i.e. wherever the OPPOSITE foot last
    landed (or its initial stance spot, for the first step on each side)."""
    initial = {
        "left": (0.0, gait.step_width_m / 2.0),
        "right": (0.0, -gait.step_width_m / 2.0),
    }
    last_pos = dict(initial)
    support = []
    for step in footsteps:
        opposite = "left" if step.side == "right" else "right"
        support.append(last_pos[opposite])
        last_pos[step.side] = (step.x, step.y)
    return support


def zmp_reference_trajectory(gait: GaitParams, footsteps: list[Footstep]
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (time, zmp_x_ref, zmp_y_ref) sampled at gait.dt.

    Reference rule (standard simple preview-control gait generator input):
      - single support during footsteps[i]'s swing: ZMP = the STANCE foot
        position (the foot that is NOT moving - see _support_foot_positions)
      - EVERY double-support window (including the initial stand-to-walk
        transition and each inter-step transition) LINEARLY RAMPS the ZMP
        reference from the old stance foot to the new one across the full
        double-support duration, rather than jumping instantly. This
        matters: an earlier version of this function set the reference to
        jump to the new foot the INSTANT double support began, which
        demands an instantaneous lateral weight-shift no real controller
        can track without leaving the support polygon - caught by
        tests/test_walk_simulation.py's ZMP-in-support-polygon check (see
        docs/BUGS_FOUND.md). Real footstep-plan-to-ZMP-reference generators
        ramp for exactly this reason.
      - after the last footstep lands: ZMP holds at that last foot
    """
    support = _support_foot_positions(gait, footsteps)
    total_time = footsteps[-1].end_time + gait.double_support_s
    n = int(round(total_time / gait.dt)) + 1
    t = np.arange(n) * gait.dt
    zx = np.zeros(n)
    zy = np.zeros(n)

    def ramp(mask, t_start, t_end, from_pos, to_pos):
        if not np.any(mask):
            return
        frac = np.clip((t[mask] - t_start) / max(t_end - t_start, 1e-9), 0.0, 1.0)
        zx[mask] = from_pos[0] + frac * (to_pos[0] - from_pos[0])
        zy[mask] = from_pos[1] + frac * (to_pos[1] - from_pos[1])

    # initial double support: ramp from centered stance to the first support foot
    lead_mask = t < footsteps[0].start_time
    ramp(lead_mask, 0.0, footsteps[0].start_time, (0.0, 0.0), support[0])

    for i, step in enumerate(footsteps):
        ss_mask = (t >= step.start_time) & (t < step.end_time)
        zx[ss_mask] = support[i][0]
        zy[ss_mask] = support[i][1]

        # double support after this step lands: ramp from the old stance
        # foot to this step's foot (which becomes the new stance for i+1)
        ds_start = step.end_time
        ds_end = footsteps[i + 1].start_time if i + 1 < len(footsteps) else step.end_time + gait.double_support_s
        ds_mask = (t >= ds_start) & (t < ds_end)
        ramp(ds_mask, ds_start, ds_end, support[i], (step.x, step.y))

    tail_mask = t >= footsteps[-1].end_time + gait.double_support_s
    zx[tail_mask] = footsteps[-1].x
    zy[tail_mask] = footsteps[-1].y

    return t, zx, zy


def support_polygon_at(t_query: float, gait: GaitParams, footsteps: list[Footstep]
                        ) -> tuple[float, float, float, float]:
    """Returns (x_min, x_max, y_min, y_max) of the support polygon (the
    convex region the ZMP must stay inside for the robot not to tip) at a
    given time, using the SAME stance-foot convention as
    zmp_reference_trajectory (see module docstring for why this matters)."""
    FOOT_LEN, FOOT_WIDTH = 0.24, 0.10  # a plausible adult-scale foot footprint, meters
    support = _support_foot_positions(gait, footsteps)

    on_ground = []
    if t_query < footsteps[0].start_time:
        on_ground = [(0.0, gait.step_width_m / 2.0), (0.0, -gait.step_width_m / 2.0)]
    else:
        for i, step in enumerate(footsteps):
            if step.start_time <= t_query < step.end_time:
                on_ground = [support[i]]  # single support: only the stance foot
                break
            ds_start = step.end_time
            ds_end = footsteps[i + 1].start_time if i + 1 < len(footsteps) else step.end_time + gait.double_support_s
            if ds_start <= t_query < ds_end:
                on_ground = [support[i], (step.x, step.y)]  # double support: old stance + just-landed foot
                break
        if not on_ground:
            on_ground = [(footsteps[-1].x, footsteps[-1].y)]

    xs = [p[0] for p in on_ground]
    ys = [p[1] for p in on_ground]
    return (min(xs) - FOOT_LEN / 2, max(xs) + FOOT_LEN / 2,
            min(ys) - FOOT_WIDTH / 2, max(ys) + FOOT_WIDTH / 2)


def foot_target_trajectories(gait: GaitParams, footsteps: list[Footstep]
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t, left_xyz, right_xyz), each foot's full 3D target
    position (including height) at every sampled time - the per-foot
    counterpart to zmp_reference_trajectory's single combined ZMP
    reference. During a foot's own single-support (swing) phase, its
    target follows trajectory.swing_foot's minimum-jerk + ground-clearance
    profile from where it last stood to where it's landing; the other
    (stance) foot's target holds at its current fixed ground position the
    entire time. Feeds kinematics/leg_ik.py per leg per timestep in
    simulation/joint_trajectory_generator.py.
    """
    from trajectory.swing_foot import swing_foot_position, SwingFootParams
    swing_params = SwingFootParams()

    total_time = footsteps[-1].end_time + gait.double_support_s
    n = int(round(total_time / gait.dt)) + 1
    t = np.arange(n) * gait.dt

    last_pos = {
        "left": np.array([0.0, gait.step_width_m / 2.0, 0.0]),
        "right": np.array([0.0, -gait.step_width_m / 2.0, 0.0]),
    }
    left_xyz = np.zeros((n, 3))
    right_xyz = np.zeros((n, 3))

    step_idx_at_time = np.full(n, -1)
    for i, step in enumerate(footsteps):
        mask = (t >= step.start_time) & (t < step.end_time)
        step_idx_at_time[mask] = i

    # Landing updates are triggered by a PHASE TRANSITION (this step's swing
    # index no longer being active), not by comparing a sample's time
    # against step.end_time with a tolerance - an earlier version used a
    # `t >= step.end_time - dt/2` threshold that silently failed for most
    # steps because footstep end_times accumulate floating-point error
    # (e.g. 1.6 + 0.8 = 2.4000000000000004), so `t < step.end_time` masks
    # sometimes include, sometimes exclude, the sample nominally AT
    # end_time depending on rounding - a real bug that left most steps'
    # landing position stuck at an earlier step's position for the rest of
    # the walk (see docs/BUGS_FOUND.md). Transition detection is immune to
    # this because it only compares step_idx_at_time to itself, never to a
    # raw floating-point time value.
    prev_i = -1
    for k in range(n):
        tk = t[k]
        i = step_idx_at_time[k]

        if prev_i >= 0 and i != prev_i:
            landed_step = footsteps[prev_i]
            last_pos[landed_step.side] = np.array([landed_step.x, landed_step.y, landed_step.z])
        prev_i = i

        if i >= 0:
            step = footsteps[i]
            side = step.side
            s = (tk - step.start_time) / (step.end_time - step.start_time)
            start_pos = last_pos[side]
            end_pos = np.array([step.x, step.y, step.z])
            swing_pos = swing_foot_position(s, start_pos, end_pos, swing_params)
            if side == "left":
                left_xyz[k] = swing_pos
                right_xyz[k] = last_pos["right"]
            else:
                right_xyz[k] = swing_pos
                left_xyz[k] = last_pos["left"]
        else:
            left_xyz[k] = last_pos["left"]
            right_xyz[k] = last_pos["right"]

    return t, left_xyz, right_xyz


def apply_terrain(footsteps: list[Footstep], terrain) -> list[Footstep]:
    """Returns a NEW footstep list with each footstep's z set from the
    terrain profile (terrain/terrain_profile.py) at its (x, y) - purely
    additive: plan_footsteps() itself is untouched and still produces
    flat-ground (z=0) footsteps by default, so every existing caller and
    test is unaffected unless it explicitly opts into this function.
    """
    new_steps = []
    for step in footsteps:
        z = terrain.height_at(step.x, step.y)
        new_steps.append(Footstep(x=step.x, y=step.y, side=step.side,
                                   start_time=step.start_time, end_time=step.end_time, z=z))
    return new_steps
