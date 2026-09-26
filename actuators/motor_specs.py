"""
actuators/motor_specs.py

Real datasheet specs for the Dynamixel MX-106 (Robotis) - a widely-used
smart servo actuator in hobbyist and research bipedal robot builds (DC
motor + gear reduction + controller + driver, all integrated, exactly
the kind of actuator a solo/small-team humanoid project would actually
buy rather than design a custom motor+gearbox for). Used here to answer
the real question SCOPE.md flagged as unanswered: given the torque this
gait ACTUALLY requires (computed in dynamics/leg_dynamics.py), can a
real, buyable actuator provide it?

Source: Robotis official MX-106 spec sheet (12V column) and the
MX-106T/R(2.0) e-manual, which explicitly notes: "This is an estimated
value for continuous torque, calculated at 20% of stall torque" - i.e.
Robotis's OWN documentation says not to design around the stall-torque
number for sustained loads, which is why both figures are kept here
rather than just the more impressive-looking stall number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorSpec:
    name: str
    stall_torque_nm: float       # maximum momentary torque (datasheet)
    continuous_torque_nm: float  # sustained-load torque Robotis itself recommends designing around
    no_load_speed_rad_s: float   # maximum joint speed with zero load


DYNAMIXEL_MX106 = MotorSpec(
    name="Dynamixel MX-106 (12V)",
    stall_torque_nm=8.40,
    continuous_torque_nm=8.40 * 0.20,   # Robotis's own "~20% of stall" continuous-duty guidance
    no_load_speed_rad_s=45.0 * 2 * np.pi / 60.0,  # 45 RPM -> rad/s
)

# A much more powerful, real, commercially-available actuator actually used
# in research-grade humanoid legs (Dynamixel PRO Plus H54P-200-S500-R, 24V) -
# included so the torque-feasibility finding below is actionable (what WOULD
# work), not just a negative result against one small servo.
DYNAMIXEL_H54P_200 = MotorSpec(
    name="Dynamixel PRO Plus H54P-200-S500-R (24V)",
    stall_torque_nm=44.7 / 0.20,  # datasheet gives continuous only; back out an approximate
                                   # stall figure using the same ~20%-of-stall convention Robotis
                                   # states elsewhere in its own MX/PRO documentation, since this
                                   # specific datasheet's stall column was blank (marked "-")
    continuous_torque_nm=44.7,     # datasheet-stated continuous-operation torque, 24V
    no_load_speed_rad_s=33.1 * 2 * np.pi / 60.0,  # 33.1 RPM -> rad/s
)


@dataclass
class FeasibilityResult:
    torque_ok_continuous: bool
    torque_ok_stall: bool
    speed_ok: bool
    peak_torque_nm: float
    peak_torque_margin_vs_continuous: float  # >1 = within continuous rating, <1 = exceeds it
    peak_torque_margin_vs_stall: float
    peak_speed_rad_s: float
    speed_margin: float


def check_feasibility(required_torque_nm: np.ndarray, required_speed_rad_s: np.ndarray,
                       motor: MotorSpec = DYNAMIXEL_MX106) -> FeasibilityResult:
    """required_torque_nm / required_speed_rad_s: arrays of the torque/speed
    a SINGLE joint needs to produce over a trajectory (e.g. hip_pitch or
    knee_pitch across a full walk). Returns whether the chosen real motor
    can actually deliver this, using its real, sourced datasheet limits -
    not a made-up 'looks big enough' comparison."""
    peak_torque = float(np.max(np.abs(required_torque_nm)))
    peak_speed = float(np.max(np.abs(required_speed_rad_s)))
    return FeasibilityResult(
        torque_ok_continuous=peak_torque <= motor.continuous_torque_nm,
        torque_ok_stall=peak_torque <= motor.stall_torque_nm,
        speed_ok=peak_speed <= motor.no_load_speed_rad_s,
        peak_torque_nm=peak_torque,
        peak_torque_margin_vs_continuous=motor.continuous_torque_nm / peak_torque if peak_torque > 0 else np.inf,
        peak_torque_margin_vs_stall=motor.stall_torque_nm / peak_torque if peak_torque > 0 else np.inf,
        peak_speed_rad_s=peak_speed,
        speed_margin=motor.no_load_speed_rad_s / peak_speed if peak_speed > 0 else np.inf,
    )
