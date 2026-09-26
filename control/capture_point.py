"""
control/capture_point.py

Instantaneous Capture Point (ICP) push recovery - Pratt, Carff, Drakunov,
Goswami, "Capture Point: A Step toward Humanoid Push Recovery" (2006).
This is the standard real theoretical basis for push recovery on
LIPM-based bipeds (including real robots like Atlas): if the NEXT
footstep is placed exactly at the capture point, the robot's CoM will
asymptotically come to rest without falling, using the SAME LIPM natural
frequency (omega = sqrt(g/zc)) already computed in dynamics/lipm.py -
push recovery on this model is "free" in the sense that it reuses the
exact same physics, not a separate ad-hoc heuristic.

    x_capture = x_com + x_com_dot / omega

Physical intuition: the capture point is where you'd have to place your
foot RIGHT NOW so that, given your current CoM position and velocity, an
(idealized, undamped) LIPM would swing to a stop exactly over that foot -
the same idea as "step in the direction you're falling, far enough,
before you fall over," made mathematically precise.
"""
from __future__ import annotations

import numpy as np


def capture_point(com_pos: float, com_vel: float, omega: float) -> float:
    """1D capture point (apply once per axis - sagittal and lateral are
    independent under the same decoupled-axis assumption the rest of this
    LIPM model already makes)."""
    return com_pos + com_vel / omega


def capture_point_2d(com_xy: np.ndarray, com_vel_xy: np.ndarray, omega: float) -> np.ndarray:
    return com_xy + com_vel_xy / omega


def clip_to_reachable_step(target_xy: np.ndarray, stance_foot_xy: np.ndarray,
                            max_step_length_m: float) -> np.ndarray:
    """A capture point computed from a large disturbance can demand a step
    farther than the leg can physically reach in one stride - clip it to a
    circle of radius max_step_length_m around the CURRENT stance foot
    (not the nominal next-footstep target), since that's the real
    kinematic constraint: the swing leg starts from wherever the stance
    foot currently is. An unclipped-but-unreachable "solution" would be
    dishonest - this is a real, physical, stated limit on how much any
    single-step recovery can achieve, not a detail to skip."""
    offset = target_xy - stance_foot_xy
    dist = np.linalg.norm(offset)
    if dist <= max_step_length_m or dist < 1e-9:
        return target_xy
    return stance_foot_xy + offset / dist * max_step_length_m
