"""Mass / power / thermal budget for the whole robot. Every ASSUMED number is a named argument: replace with measurements."""
from __future__ import annotations
import numpy as np
from actuators.joint_actuator import spec_for_joint, ActuatorSpec

ACTUATED_30 = ([f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")] + ["waist_yaw", "waist_pitch"]
               + [f"{s}_{j}" for s in "LR" for j in ("sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_yaw", "wr_pitch", "wr_roll")] + ["neck_yaw", "neck_pitch"])


def actuator_mass(names=ACTUATED_30, spec_fn=spec_for_joint):
    per = {n: spec_fn(n).mass_kg for n in names}
    return sum(per.values()), per


def mass_budget(model_mass_kg=25.0, names=ACTUATED_30, spec_fn=spec_for_joint, battery_kg=1.5, electronics_kg=1.5, gripper_kg=0.4, structure_frac=0.20):
    """Compare ACTUATOR mass (datasheet) with the simulated robot's TOTAL mass; what is left (after ~30 min battery, electronics,
    grippers) must hold the structure, which needs >= structure_frac of the total (ASSUMED 20 %)."""
    act, per = actuator_mass(names, spec_fn)
    left = model_mass_kg - act - electronics_kg - gripper_kg - battery_kg
    return dict(model_mass_kg=model_mass_kg, actuators_kg=act, n_actuators=len(names), electronics_kg=electronics_kg, grippers_kg=gripper_kg,
                battery_kg=battery_kg, left_for_structure_kg=left, feasible=left >= structure_frac * model_mass_kg,
                note="structure (links, brackets, covers) is typically 20-35 % of a humanoid's mass; feasible=False means the simulated mass target cannot be met with these actuators")


def avg_power_w(names=ACTUATED_30, spec_fn=spec_for_joint, leg_torque_frac=0.5, leg_speed_frac=0.25, other_torque_frac=0.15,
                idle_per_actuator_w=2.0, compute_w=45.0, sensors_w=10.0):
    """Walking-average electrical power: copper loss ~ (tau/tau_cont)^2 * P_loss_cont + positive mechanical power + holding/idle + compute."""
    P = 0.0
    for n in names:
        s: ActuatorSpec = spec_fn(n); leg = any(k in n for k in ("hip", "knee", "ankle"))
        f = leg_torque_frac if leg else other_torque_frac
        tau = f * s.tau_cont_nm; w = (leg_speed_frac if leg else 0.1) * s.w_noload_rad_s
        P += s.loss_cont_w * f * f + max(tau * w, 0.0) + idle_per_actuator_w
    return P + compute_w + sensors_w


def battery_for(avg_w, runtime_min=30.0, usable=0.8, eff=0.9, wh_per_kg=180.0):
    wh = avg_w * runtime_min / 60.0 / (usable * eff)
    return dict(energy_wh=wh, mass_kg=wh / wh_per_kg, runtime_min=runtime_min)


def time_to_derate_s(spec: ActuatorSpec, torque_nm):
    """Closed form for constant torque from ambient: time until the winding reaches T_derate (inf if the steady state stays below)."""
    f = torque_nm / spec.tau_cont_nm
    dT_ss = (f * f) * (spec.t_derate_c - spec.t_amb_c)
    d = spec.t_derate_c - spec.t_amb_c
    if dT_ss <= d: return float("inf")
    return float(-spec.thermal_tau_s * np.log(1.0 - d / dT_ss))
