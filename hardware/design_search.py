"""
Actuator-assignment search: closes the "30 datasheet actuators = 20 kg of a 25 kg robot" blocker.

Picks, per joint GROUP (hip x6, knee x2, ankle x4, waist x2, shoulder x6, elbow x2, wrist x6, neck x2 = 30 joints), the lightest
catalogue actuator that satisfies the demand measured in simulation:
  legs/waist : peak torque from results/fullbody/motor_requirements.json at the chosen CoM height (x safety margin), speed vs no-load,
               continuous torque >= duty * peak (ASSUMED duty 0.5)
  arms       : static holding torque (manipulation.loco_manipulation) for a payload at several reach poses; continuous torque must hold it
It then reports total actuator mass, what is left for structure/battery, and the arm payload the design really achieves.
CATALOGUE: entries with verified=False are from memory of public datasheets or hypothetical - CHECK THEM before relying on a result.
"""
from __future__ import annotations
import itertools, json
from dataclasses import dataclass
import numpy as np
from actuators.joint_actuator import ActuatorSpec, PH54_200, PH54_100, MX106, QDD_REFERENCE, RPM

GROUPS = {"hip": 6, "knee": 2, "ankle": 4, "waist": 2, "shoulder": 6, "elbow": 2, "wrist": 6, "neck": 2}

XM540 = ActuatorSpec("Dynamixel XM540-W270 (UNVERIFIED numbers)", 3.2, 10.6, 30 * RPM, 0.165)
XM430 = ActuatorSpec("Dynamixel XM430-W350 (UNVERIFIED numbers)", 1.3, 4.1, 46 * RPM, 0.082)
CATALOG = {"PH54-200": (PH54_200, True), "PH54-100": (PH54_100, True), "MX-106": (MX106, True),
           "XM540": (XM540, False), "XM430": (XM430, False), "QDD-hyp": (QDD_REFERENCE, False)}


@dataclass
class Demand:
    peak_nm: float
    speed_rad_s: float = 0.0
    hold_nm: float = 0.0          # continuous holding torque (arms)


def leg_demands(req_rows, com_height):
    r = min(req_rows, key=lambda x: abs(x["com_height_m"] - com_height))
    sp = r["leg_speed_rad_s"]
    return {"hip": Demand(max(r["hip_pitch_Nm"], r["hip_roll_Nm"], r["hip_yaw_Nm"]), sp), "knee": Demand(r["knee_Nm"], sp),
            "ankle": Demand(max(r["ankle_pitch_Nm"], r["ankle_roll_Nm"]), sp), "waist": Demand(r["waist_Nm"], 0.5)}, r


def arm_demands(payload_kg, poses=None):
    """Static torque per arm joint group for a payload held at the TCP in several poses (uses the repo's own arm kinematics)."""
    from kinematics.arm_kinematics import Arm, ArmParams
    from manipulation.loco_manipulation import static_joint_torque, G
    arm = Arm(ArmParams(side="R"))
    poses = poses or {"tucked": [0, -0.12, 0, -2.0, 0, 0, 0], "mid": [-0.8, -0.12, 0, -1.2, 0, 0, 0], "forward": [-1.57, -0.12, 0, -0.05, 0, 0, 0]}
    worst = np.zeros(7)
    for q in poses.values():
        worst = np.maximum(worst, np.abs(static_joint_torque(arm, np.array(q, float), np.array([0, 0, -payload_kg * G]))))
    return {"shoulder": Demand(0, 0, float(worst[:3].max())), "elbow": Demand(0, 0, float(worst[3])), "wrist": Demand(0, 0, float(worst[4:].max())),
            "neck": Demand(0, 0, 1.0)}, worst


def feasible(spec: ActuatorSpec, d: Demand, margin=1.15, duty=0.5):
    if d.peak_nm > 0:
        if spec.tau_peak_nm < margin * d.peak_nm or spec.tau_cont_nm < duty * d.peak_nm: return False
        if d.speed_rad_s > 0.0 and spec.w_noload_rad_s < 1.1 * d.speed_rad_s: return False
    if d.hold_nm > 0 and spec.tau_cont_nm < margin * d.hold_nm: return False
    return True


def search(req_rows, com_height=0.89, payload_kg=1.0, use_unverified=False, margin=1.15):
    cat = {k: v[0] for k, v in CATALOG.items() if v[1] or use_unverified}
    dem, row = leg_demands(req_rows, com_height); ad, worst = arm_demands(payload_kg); dem.update(ad)
    choice, why = {}, {}
    for g in GROUPS:
        ok = [(s.mass_kg, k) for k, s in cat.items() if feasible(s, dem[g], margin)]
        if not ok: return dict(feasible=False, reason=f"no catalogue actuator satisfies group '{g}' (peak {dem[g].peak_nm:.1f} Nm, hold {dem[g].hold_nm:.1f} Nm)", demands={k: vars(v) for k, v in dem.items()})
        choice[g] = min(ok)[1]
    mass = sum(GROUPS[g] * cat[choice[g]].mass_kg for g in GROUPS)
    tau = {g: cat[choice[g]].tau_peak_nm for g in GROUPS}
    return dict(feasible=True, com_height_m=com_height, payload_kg=payload_kg, choice=choice, actuator_mass_kg=mass,
                tau_limits_nm=tau, demands={k: vars(v) for k, v in dem.items()}, arm_hold_nm=worst.tolist(), unverified_used=[choice[g] for g in GROUPS if not CATALOG[choice[g]][1]])


def to_model_kwargs(design: dict) -> dict:
    """ModelParams torque fields for a design (feed into model.humanoid_model.ModelParams(**kw) / repo_actuator_params(**kw))."""
    t = design["tau_limits_nm"]
    return dict(tau_hip=t["hip"], tau_knee=t["knee"], tau_ankle=t["ankle"], tau_waist=t["waist"], tau_shoulder=t["shoulder"],
                tau_elbow=t["elbow"], tau_wrist=t["wrist"], tau_neck=t["neck"])


def self_consistent_mass(req_rows, com_height=0.89, payload_kg=1.0, use_unverified=False, margin=1.04, structure_frac=0.20,
                         electronics_kg=1.5, gripper_kg=0.4, battery_kg=1.5, ref_mass=25.0, m_lo=18.0, m_hi=60.0, step=0.25):
    """Mass and torque demand chase each other: a heavier robot needs bigger knees, which are heavier. Leg/waist torque demand is
    scaled by M/ref_mass (gravity-dominated, first order - re-measure in simulation with the new mass!). Returns the lightest M for
    which   M * (1 - structure_frac) >= actuators + electronics + grippers + battery   AND every group is feasible."""
    best = None; M = m_lo; trace = []
    while M <= m_hi:
        rows = [dict(r, **{k: r[k] * M / ref_mass for k in ("knee_Nm", "hip_pitch_Nm", "hip_roll_Nm", "hip_yaw_Nm", "ankle_pitch_Nm", "ankle_roll_Nm", "waist_Nm")}) for r in req_rows]
        d = search(rows, com_height, payload_kg, use_unverified, margin)
        if d["feasible"]:
            need = (d["actuator_mass_kg"] + electronics_kg + gripper_kg + battery_kg) / (1 - structure_frac)
            trace.append((M, d["actuator_mass_kg"], need))
            if M >= need and best is None: best = dict(d, robot_mass_kg=M, mass_needed_kg=need, structure_kg=M * structure_frac)
        M += step
    return best, trace
