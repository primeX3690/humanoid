"""
actuators/gearbox_backlash.py

Closes the other half of the phrase docs/SCOPE.md always paired
together but never modeled: "motor inductance, gearbox
BACKLASH/friction". `actuators/actuator_dynamics.py` modeled the
continuous electrical/friction dynamics; backlash is fundamentally
different in kind - a position-dependent, hysteretic DEAD-ZONE effect
in the mechanical gear train, not a continuous current/torque one - so
it needs its own model, in the position domain, not the torque domain.

MODEL (standard "backlash nonlinearity" - see e.g. Nordin & Gutman,
"Controlling mechanical systems with backlash - a survey", Automatica
2002): the motor-side (input) position theta_m and the load/output-side
(output) position theta_l are connected through a gap of total angular
width `backlash_rad`. The output only moves when the input has taken up
the full gap and is pushing against one face of it:

    if theta_m - theta_l > b/2:  theta_l := theta_m - b/2   (upper face engaged)
    if theta_m - theta_l < -b/2: theta_l := theta_m + b/2   (lower face engaged)
    otherwise: theta_l unchanged (input moving freely inside the gap)

This is exactly the discrete-time update implemented below (a
first-order hold each timestep, matching how every other simulation
loop in this project advances state).

SOURCED VS ASSUMED backlash value - stated honestly, not blurred: a
web search for the MX-106's own backlash spec returned a widely
mirrored community database entry of "20 degrees", which directly
CONTRADICTS Robotis's own advertised "360 degree position control
WITHOUT DEAD ZONE" feature and its 0.088-degree encoder resolution - 20
degrees of backlash would make that resolution meaningless. This is
almost certainly a data-entry error in that source (possibly a
mismatched or unrelated field), and using it anyway just because it was
the first number found would be worse than admitting no reliable
published backlash figure exists for this actuator. Instead this module
uses an EXPLICITLY STATED, conservative literature-typical value for a
small precision robotic gearbox (0.3 degrees / 0.00524 rad) as an
assumption, not a confirmed datasheet number - flagged with the same
honesty this project has used every other time a number couldn't be
sourced (e.g. the electrical time constant in actuator_dynamics.py).
"""
from __future__ import annotations

import numpy as np

# Explicitly an ASSUMPTION, not a sourced MX-106 spec - see module docstring
_DEFAULT_BACKLASH_DEG = 0.3
DEFAULT_BACKLASH_RAD = np.deg2rad(_DEFAULT_BACKLASH_DEG)


def apply_backlash(motor_position_rad: np.ndarray, backlash_rad: float = DEFAULT_BACKLASH_RAD,
                    initial_load_position_rad: float | None = None) -> np.ndarray:
    """Given a MOTOR-side (input) position trajectory, returns the
    resulting LOAD-side (output, i.e. actual joint) position trajectory
    after passing through a backlash gap of total width `backlash_rad`.
    `initial_load_position_rad` defaults to motor_position_rad[0] (gap
    centered at the start - a stated, reasonable initial condition)."""
    n = len(motor_position_rad)
    load_position = np.empty(n)
    half_gap = backlash_rad / 2.0
    load_position[0] = (motor_position_rad[0] if initial_load_position_rad is None
                         else initial_load_position_rad)
    for k in range(1, n):
        gap = motor_position_rad[k] - load_position[k - 1]
        if gap > half_gap:
            load_position[k] = motor_position_rad[k] - half_gap
        elif gap < -half_gap:
            load_position[k] = motor_position_rad[k] + half_gap
        else:
            load_position[k] = load_position[k - 1]
    return load_position


def backlash_position_error(motor_position_rad: np.ndarray, backlash_rad: float = DEFAULT_BACKLASH_RAD
                             ) -> np.ndarray:
    """load_position - motor_position at every sample - the actual joint
    tracking error backlash alone introduces (assuming otherwise-perfect
    motor-side position tracking, i.e. isolating backlash's contribution
    from any other source of error)."""
    load = apply_backlash(motor_position_rad, backlash_rad)
    return load - motor_position_rad
