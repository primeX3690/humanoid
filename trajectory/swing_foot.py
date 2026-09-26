"""
trajectory/swing_foot.py

Generates the swing foot's 3D trajectory during a single-support phase:
horizontal (x,y) motion follows a MINIMUM-JERK profile (Flash & Hogan,
1985 - the standard smooth point-to-point trajectory shape used
throughout robotics and cited in human-arm/leg-motion studies, chosen
because it has exactly zero velocity AND zero acceleration at both
endpoints, which is what actually matters here: the foot must be
momentarily at rest the instant it lifts off and the instant it touches
down, or the touchdown would be a real impact rather than a controlled
step), while vertical (z) motion adds a sinusoidal ground-clearance
"hump" on top so the foot doesn't drag through the floor mid-swing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _min_jerk_blend(s: float) -> float:
    """10s^3 - 15s^4 + 6s^5: the standard minimum-jerk blend function.
    blend(0)=0, blend(1)=1, blend'(0)=blend'(1)=0, blend''(0)=blend''(1)=0."""
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


@dataclass
class SwingFootParams:
    step_height_m: float = 0.05  # peak ground clearance mid-swing


def swing_foot_position(s: float, start_pos: np.ndarray, end_pos: np.ndarray,
                         params: SwingFootParams) -> np.ndarray:
    """s: normalized swing-phase time in [0, 1] (0 = liftoff, 1 = touchdown).
    start_pos/end_pos: (3,) liftoff and landing positions (z typically 0 for both).

    The height (z) clearance profile uses sin^2(pi*s), NOT sin(pi*s): a
    plain sin(pi*s) profile (an earlier version of this function) has the
    same peak clearance but a NONZERO vertical velocity at s=0 and s=1
    (derivative = step_height*pi*cos(pi*s), which is +/-step_height*pi at
    the endpoints, not zero) - meaning the foot would still be moving
    vertically at touchdown, a real impact rather than a soft landing.
    sin^2(pi*s) has zero derivative at both endpoints (d/ds = step_height
    * pi * sin(2*pi*s), which is exactly zero at s=0, 0.5, and 1), giving
    a genuinely soft touchdown - caught by
    tests/test_swing_trajectory.py::test_zero_velocity_at_liftoff_and_touchdown,
    not assumed from the formula (see docs/BUGS_FOUND.md)."""
    s = float(np.clip(s, 0.0, 1.0))
    blend = _min_jerk_blend(s)
    xy = start_pos[:2] + blend * (end_pos[:2] - start_pos[:2])
    z_base = start_pos[2] + blend * (end_pos[2] - start_pos[2])
    z_clearance = params.step_height_m * np.sin(np.pi * s) ** 2
    return np.array([xy[0], xy[1], z_base + z_clearance])


def swing_foot_trajectory(start_pos: np.ndarray, end_pos: np.ndarray,
                           params: SwingFootParams, n_samples: int) -> np.ndarray:
    """Convenience: full trajectory as an (n_samples, 3) array, s sampled
    uniformly over [0, 1] inclusive of both endpoints."""
    s_vals = np.linspace(0.0, 1.0, n_samples)
    return np.array([swing_foot_position(s, start_pos, end_pos, params) for s in s_vals])
