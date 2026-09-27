"""
actuators/actuator_dynamics.py

Closes the last item on docs/SCOPE.md's original open-questions list:
"actuator dynamics (motor inductance, gearbox backlash/friction)".
`actuators/motor_specs.py` already answers whether a real motor can
produce enough STEADY torque; this module answers the DYNAMIC question
it explicitly left open - can the motor's electrical response (finite
inductance -> finite current rise time) and its own internal friction
losses keep up with how FAST the required torque changes over a gait,
not just its peak magnitude.

SOURCED ELECTRICAL PARAMETERS for the Dynamixel MX-106 (12V) - the same
actuator actuators/motor_specs.py already uses - derived from real,
published datasheet numbers (Robotis / CrustCrawler MX-106 spec sheet):
operating voltage 12.0V, stall current 5.2A, stall torque 8.40 N*m,
no-load current 0.17A, no-load speed 45 RPM, gear ratio 225:1.

Torque constant: Kt = stall_torque / (stall_current - no_load_current)
= 8.40 / (5.2 - 0.17) = 1.670 N*m/A - this is the standard formula
motor manufacturers themselves use (see e.g. Bodine Electric's "Motor
Constants for Gearmotors" technical note), not an invented one;
subtracting no-load current (rather than dividing raw stall torque by
raw stall current) accounts for the fact that some of the stall current
is already spent overcoming internal friction before producing any
output torque at all.

Back-EMF constant: set Kb = Kt (numerically, in consistent SI units) -
the standard ideal-permanent-magnet-motor convention (the same
convention Bodine's own note uses: it derives Ke directly from Kt via a
units conversion, rather than an independent back-EMF measurement).

REAL, HONEST FINDING from combining these with the OTHER published
number (no-load speed): calibrating Kt/Kb from the stall-torque data
point this way, and using it to PREDICT no-load speed via
V = I_noload*R + Kb*omega, gives 66.4 RPM - a 47% OVER-prediction
of the datasheet's real 45 RPM. This is a genuine, quantified limitation
of the simple single-Kt=Kb, single-friction-term lumped DC-motor model
applied to a real, low-cost integrated smart-servo: it cannot
simultaneously fit both the stall-torque and no-load-speed operating
points with one linear model (real efficiency and friction are
speed-dependent - iron losses, brush friction, gearbox mesh losses -
in ways a single Coulomb-friction term doesn't capture). This module
deliberately calibrates to the STALL/HIGH-TORQUE regime (since that's
the regime `dynamics/leg_dynamics.py`'s feasibility checks care about
most - can the actuator survive holding the robot's weight), and states
this tradeoff explicitly rather than hiding it: the no-load SPEED
predictions this module would make are known to run high.

Coulomb friction (output-referred): at true no-load, steady speed,
output torque is zero, so the ENTIRE no-load motor torque
(Kt * I_noload = 1.670 * 0.17 = 0.284 N*m) is being spent overcoming
friction. This is a genuine, sourced friction-torque estimate - not a
guessed "10% derating" - about 3.4% of the actuator's rated stall
torque.

Motor inductance: NOT published for the MX-106 (a fully-integrated
smart-servo module; Robotis does not expose raw motor-winding
parameters, only the servo's overall input/output behavior). Using an
explicitly-stated, cited approximation instead of inventing a number:
a directly-measured electrical time constant for a comparable small
PMDC motor, tau_e = 11.5 ms (University of Utah ECE 3510 "DC Motor
Characteristics" lab measurement) - see `_TYPICAL_TAU_E_S` below. This
is a real limitation stated plainly, not hidden: the ACTUAL MX-106
winding inductance could differ from this stand-in value.

HONEST, STATED SCOPE LIMIT (not modeled here): gearbox BACKLASH (the
other half of docs/SCOPE.md's phrase "motor inductance/friction") is
NOT modeled - backlash is a position-dependent, hysteretic dead-zone
effect fundamentally different from the continuous friction/electrical
dynamics modeled here, and would need its own separate treatment
(typically a dead-zone element in the position control loop, not the
current/torque loop this module addresses).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sourced MX-106 (12V) datasheet numbers - same actuator as actuators/motor_specs.py
_V_NOMINAL = 12.0
_I_STALL_A = 5.2
_TAU_STALL_NM = 8.40
_I_NOLOAD_A = 0.17
_N_NOLOAD_RPM = 45.0
_GEAR_RATIO = 225.0

# Stated approximation (see module docstring) - MX-106 inductance is not published
_TYPICAL_TAU_E_S = 0.0115  # 11.5 ms, measured for a comparable small PMDC motor
                            # (Univ. of Utah ECE 3510 DC Motor Characteristics lab)


@dataclass
class MotorElectricalParams:
    name: str
    resistance_ohm: float
    inductance_henry: float
    kt_nm_per_a: float          # torque constant, output-referred
    kb_v_per_rads: float        # back-EMF constant, output-referred (= kt in SI units, stated convention)
    coulomb_friction_nm: float  # output-referred, sourced from no-load current (see docstring)
    v_max: float
    predicted_no_load_speed_rad_s: float  # see docstring - a known OVER-prediction, kept visible, not hidden


def mx106_electrical_params() -> MotorElectricalParams:
    R = _V_NOMINAL / _I_STALL_A
    Kt = _TAU_STALL_NM / (_I_STALL_A - _I_NOLOAD_A)
    Kb = Kt
    tau_coulomb = Kt * _I_NOLOAD_A
    L = _TYPICAL_TAU_E_S * R
    predicted_no_load = (_V_NOMINAL - _I_NOLOAD_A * R) / Kb
    return MotorElectricalParams(
        name="Dynamixel MX-106 (12V) - electrical model",
        resistance_ohm=R, inductance_henry=L, kt_nm_per_a=Kt, kb_v_per_rads=Kb,
        coulomb_friction_nm=tau_coulomb, v_max=_V_NOMINAL,
        predicted_no_load_speed_rad_s=predicted_no_load,
    )


def current_step_response_analytic(params: MotorElectricalParams, voltage_v: float,
                                    omega_rad_s: float, t: np.ndarray, i0: float = 0.0) -> np.ndarray:
    """Exact closed-form solution for CONSTANT voltage and CONSTANT
    speed: a linear first-order ODE, L dI/dt + R*I = V - Kb*omega, whose
    solution is I(t) = I_ss + (I0 - I_ss) * exp(-t/tau_e). Used to
    verify the numerical integrator below against a known-correct
    reference, not just checked against itself."""
    tau_e = params.inductance_henry / params.resistance_ohm
    i_ss = (voltage_v - params.kb_v_per_rads * omega_rad_s) / params.resistance_ohm
    return i_ss + (i0 - i_ss) * np.exp(-t / tau_e)


def simulate_current_response(params: MotorElectricalParams, voltage_v: np.ndarray,
                               omega_rad_s: np.ndarray, dt: float, i0: float = 0.0) -> np.ndarray:
    """RK4 integration of L dI/dt = V - I*R - Kb*omega for arbitrary
    (piecewise, per-timestep) voltage and speed arrays - the general
    case current_step_response_analytic's closed form doesn't cover."""
    n = len(voltage_v)
    current = np.zeros(n)
    current[0] = i0
    L, R, Kb = params.inductance_henry, params.resistance_ohm, params.kb_v_per_rads

    def didt(i, v, w):
        return (v - i * R - Kb * w) / L

    for k in range(n - 1):
        v, w = voltage_v[k], omega_rad_s[k]
        i = current[k]
        k1 = didt(i, v, w)
        k2 = didt(i + 0.5 * dt * k1, v, w)
        k3 = didt(i + 0.5 * dt * k2, v, w)
        k4 = didt(i + dt * k3, v, w)
        current[k + 1] = i + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return current


def output_torque(params: MotorElectricalParams, current_a: np.ndarray,
                   omega_rad_s: np.ndarray, friction_deadband_rad_s: float = 1e-3) -> np.ndarray:
    """tau_output = Kt*I - coulomb_friction*sign(omega), with a small
    deadband around zero speed (a common, stated simplification - real
    static/stiction friction near zero speed is more complex than a
    clean sign() discontinuity, and isn't modeled further here)."""
    friction = params.coulomb_friction_nm * np.where(
        np.abs(omega_rad_s) > friction_deadband_rad_s, np.sign(omega_rad_s), 0.0)
    return params.kt_nm_per_a * current_a - friction


def track_required_torque(required_torque_nm: np.ndarray, omega_rad_s: np.ndarray,
                           params: MotorElectricalParams, dt: float
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a REQUIRED output-torque trajectory (e.g. from
    dynamics/foot_inertia.py or dynamics/leg_dynamics.py over a real
    gait) and the joint's own speed trajectory, simulate what the
    actuator ACTUALLY delivers under an idealized feedforward voltage
    driver: at each step, command whatever voltage would produce the
    required current in STEADY STATE (V = I_required*R + Kb*omega),
    clamped to +-v_max, then integrate the REAL current response
    (with its finite L/R electrical lag) forward - a stated, simple,
    explicit modeling choice for the voltage command law, not a claim
    about the Dynamixel's actual (proprietary, unpublished) internal
    current-control firmware.

    Returns (achieved_torque_nm, current_a, voltage_used_v)."""
    n = len(required_torque_nm)
    i_required = (required_torque_nm + params.coulomb_friction_nm *
                  np.sign(omega_rad_s)) / params.kt_nm_per_a
    v_command = np.clip(i_required * params.resistance_ohm + params.kb_v_per_rads * omega_rad_s,
                         -params.v_max, params.v_max)
    current = simulate_current_response(params, v_command, omega_rad_s, dt)
    achieved_torque = output_torque(params, current, omega_rad_s)
    return achieved_torque, current, v_command
