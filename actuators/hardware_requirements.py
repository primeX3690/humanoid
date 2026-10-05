"""
actuators/hardware_requirements.py

Addresses the "physical hardware itself" item on docs/SCOPE.md's
remaining-gaps list - HONESTLY: this cannot build hardware, or replace
mechanical/electrical/structural engineering (stress analysis, PCB
design, wiring, thermal management, none of which this module attempts
or claims to). What it DOES do, legitimately: pull together every
number this project's own simulation and analysis has ALREADY computed
(actuator selection from actuators/motor_specs.py, peak torque from
dynamics/leg_dynamics.py, robot mass assumption from
dynamics/leg_dynamics.RobotMassParams, backlash/electrical assumptions
from actuators/gearbox_backlash.py and actuators/actuator_dynamics.py)
into one coherent hardware REQUIREMENTS summary - the concrete,
actionable output of everything analyzed so far, and an equally
concrete, honest list of what a real build would ALSO need that this
project has not addressed at all (sensing, structure, wiring, thermal).

Every number below is either a DIRECT real-datasheet value (sourced
elsewhere in this project) or SIMPLE, stated arithmetic on those values
(e.g. actuator count x unit mass) - nothing here is a new, independent
engineering estimate invented for this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from actuators.motor_specs import DYNAMIXEL_H54P_200, MotorSpec
from actuators.gearbox_backlash import DEFAULT_BACKLASH_RAD
from actuators.actuator_dynamics import mx106_electrical_params
from dynamics.leg_dynamics import RobotMassParams

# Real, sourced H54P-200-S500-R datasheet values not already captured in
# actuators/motor_specs.py's MotorSpec (which only tracks torque/speed) -
# found via web search of Robotis/reseller spec sheets (generationrobots.com,
# funduinoshop.com, besomi.com all agree on these figures)
H54P_200_WEIGHT_KG = 0.855
H54P_200_CONTINUOUS_CURRENT_A = 9.3
H54P_200_NO_LOAD_CURRENT_A = 1.65
H54P_200_VOLTAGE_V = 24.0

N_LEG_JOINTS_PER_LEG = 6   # hip yaw/roll/pitch, knee pitch, ankle pitch/roll - dynamics/rigid_body_leg_6dof.py
N_LEGS = 2
N_ACTUATORS = N_LEG_JOINTS_PER_LEG * N_LEGS  # LEGS ONLY - see docstring's stated scope limit


@dataclass
class HardwareRequirements:
    n_actuators: int
    actuator_model: str
    actuator_unit_mass_kg: float
    total_actuator_mass_kg: float
    assumed_total_robot_mass_kg: float
    actuator_mass_fraction_of_total: float
    peak_required_torque_nm: float
    actuator_continuous_torque_nm: float
    torque_margin: float
    worst_case_simultaneous_power_w: float
    per_joint_backlash_deg: float
    electrical_time_constant_ms: float


def derive_requirements(peak_required_torque_nm: float = 32.2,
                         robot_mass: RobotMassParams | None = None,
                         motor: MotorSpec = DYNAMIXEL_H54P_200) -> HardwareRequirements:
    """`peak_required_torque_nm` defaults to the stance-leg peak already
    measured and documented in docs/BUGS_FOUND.md (32.2 N*m) - passed as
    a parameter, not hardcoded silently, so a future re-measurement
    (e.g. a heavier robot_mass) can be fed straight through."""
    robot_mass = robot_mass or RobotMassParams()
    total_actuator_mass = N_ACTUATORS * H54P_200_WEIGHT_KG
    worst_case_power = N_ACTUATORS * H54P_200_VOLTAGE_V * H54P_200_CONTINUOUS_CURRENT_A
    backlash_deg = 0.3  # DEFAULT_BACKLASH_RAD in degrees - see gearbox_backlash.py for the honest sourcing note
    tau_e_ms = 1000.0 * mx106_electrical_params().inductance_henry / mx106_electrical_params().resistance_ohm

    return HardwareRequirements(
        n_actuators=N_ACTUATORS,
        actuator_model=motor.name,
        actuator_unit_mass_kg=H54P_200_WEIGHT_KG,
        total_actuator_mass_kg=total_actuator_mass,
        assumed_total_robot_mass_kg=robot_mass.total_mass_kg,
        actuator_mass_fraction_of_total=total_actuator_mass / robot_mass.total_mass_kg,
        peak_required_torque_nm=peak_required_torque_nm,
        actuator_continuous_torque_nm=motor.continuous_torque_nm,
        torque_margin=motor.continuous_torque_nm / peak_required_torque_nm,
        worst_case_simultaneous_power_w=worst_case_power,
        per_joint_backlash_deg=backlash_deg,
        electrical_time_constant_ms=tau_e_ms,
    )


# Real gaps this project's software has NOT addressed at all - stated plainly,
# not implied to be "handled" by anything above. Each is a substantial,
# separate engineering effort in its own right.
UNMODELED_HARDWARE_REQUIREMENTS = [
    "Structural design: link geometry, material selection, stress/fatigue "
    "analysis for the loads dynamics/rigid_body_leg_6dof.py and "
    "dynamics/foot_inertia.py compute - this project models the LOADS, not "
    "the structure that must carry them.",
    "Sensing: an IMU for real orientation/balance feedback (this project's "
    "ZMP and CoM state are computed from an idealized model, not measured "
    "sensor data), joint position feedback beyond what the actuators' own "
    "internal encoders provide, and foot force/torque sensors for a REAL "
    "ZMP measurement (as opposed to the LIPM-model-computed one this "
    "project uses throughout).",
    "Power system: battery sizing (capacity, chemistry, discharge rate) "
    "against the worst-case power draw derived below, voltage regulation, "
    "wiring gauge and connector selection for the real currents involved.",
    "Thermal management: sustained near-continuous-torque operation "
    "(docs/BUGS_FOUND.md's stance-leg finding) generates real heat in the "
    "actuator windings - no thermal analysis or cooling provision exists "
    "in this project.",
    "Foot/sole design: compliance, sensing, and ground-contact mechanics - "
    "dynamics/foot_inertia.py models the foot's INERTIA, not its contact "
    "surface or structure.",
    "Arms, torso, head, and any other body segment beyond the two legs - "
    "entirely outside this project's scope; N_ACTUATORS above counts LEG "
    "joints only.",
]
