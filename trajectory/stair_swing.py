"""Stair-safe swing foot: LIFT first, then move forward at apex height, then lower - so the toe clears the nosing.

The v2 swing profile blends height and forward motion together (z = min-jerk(start->end) + small sin^2 hump). Going UP a 12-15 cm riser that
drags the toe through the stair edge. Here the foot rises to  max(z_start, z_end) + clearance  during the first `lift_frac` of the swing
(forward motion only starts after `move_start`), travels horizontally at apex, and descends in the last `lower_frac`. Position AND velocity
are continuous and the foot is at rest at lift-off and touch-down.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def mj(u):
    u = np.clip(u, 0.0, 1.0); return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


@dataclass
class StairSwingParams:
    clearance_m: float = 0.04
    lift_frac: float = 0.35
    lower_frac: float = 0.30
    move_start: float = 0.22          # forward motion begins here (the toe is already above the riser)
    move_end: float = 0.78


def stair_swing_position(s, start, end, p: StairSwingParams = StairSwingParams()):
    start = np.asarray(start, float); end = np.asarray(end, float); s = float(np.clip(s, 0, 1))
    apex = max(start[2], end[2]) + p.clearance_m
    if s < p.lift_frac: z = start[2] + (apex - start[2]) * mj(s / p.lift_frac)
    elif s > 1 - p.lower_frac: z = apex + (end[2] - apex) * mj((s - (1 - p.lower_frac)) / p.lower_frac)
    else: z = apex
    u = mj((s - p.move_start) / (p.move_end - p.move_start))
    return np.array([*(start[:2] + u * (end[:2] - start[:2])), z])


def toe_heel_clearance(traj, terrain, foot_len=0.20, margin=0.003):
    """min over the swing of (sole height - highest terrain under the foot's x-extent). >= -margin means no collision with the stairs."""
    worst = np.inf
    for p in traj:
        under = max(terrain.height_at(p[0] + dx, p[1]) for dx in np.linspace(-foot_len / 2, foot_len / 2, 9))
        worst = min(worst, p[2] - under)
    return float(worst)
