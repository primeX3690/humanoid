"""
rl/biped_model.py

Generates the MuJoCo (MJCF) model of the biped used for reinforcement
learning, directly from the SAME parameters the rest of this repository
already uses - so the learned policy and the model-based stack
(LIPM/ZMP walking, rigid-body dynamics, actuator analysis) describe one
and the same robot rather than two unrelated ones:

  * link lengths / foot size        <- kinematics.leg_fk.LegParams
  * segment masses (de Leva 1996)   <- dynamics.leg_dynamics fractions
  * total robot mass                <- dynamics.leg_dynamics.RobotMassParams
  * joint chain and axis order      <- kinematics/leg_fk.py
                                       (hip yaw Z, hip roll X, hip pitch Y,
                                        knee pitch Y, ankle pitch Y, ankle roll X)
  * actuator torque limit           <- actuators.motor_specs (Dynamixel
                                       PRO Plus H54P-200, 44.7 N*m continuous)

Joint order (12 actuated DOF), per leg: hip_yaw, hip_roll, hip_pitch, knee,
ankle_pitch, ankle_roll - left leg first, then right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import (
    RobotMassParams, THIGH_MASS_FRACTION, SHANK_MASS_FRACTION, FOOT_MASS_FRACTION,
)
from actuators.motor_specs import DYNAMIXEL_H54P_200

JOINT_NAMES_PER_LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
N_JOINTS = 12

# nominal standing pose per leg (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll).
# Sign convention follows kinematics/leg_fk.py: positive pitch swings the distal segment
# backwards, so a natural bent-knee stance is hip<0, knee>0, ankle = -(hip+knee) (foot flat).
# The shank is inclined slightly more than the thigh so the ankle sits ~4.4 cm behind the
# hip: the centre of mass projects just ahead of the ankle (inside the foot, which spans
# -3..+17 cm about the ankle) while keeping the static ankle torque (m*g*offset ~ 11 N*m)
# small enough for a PD hold. Two earlier poses failed a plain zero-action PD standing test:
# a symmetric thigh/shank pose (hip -0.25, knee 0.50) put the CoM on the heel edge and fell
# backwards in ~4 s; a strongly inclined shank (hip -0.15) put the CoM 9 cm ahead of the
# ankle, needing ~21 N*m of ankle torque, and pitched forward in ~1 s.
NOMINAL_LEG_POSE = np.array([0.0, 0.0, -0.20, 0.50, -0.30, 0.0])

# Joint-level PD gains (N*m/rad, N*m*s/rad), per leg. The ankle-pitch stiffness must exceed
# the gravitational "inverted pendulum" stiffness m*g*h (~25 kg * 9.81 * 0.9 m ~ 220 N*m/rad)
# for a plain PD hold of the standing pose to be statically stable at all - a first version
# with kp_ankle = 150 fell over regardless of the nominal pose chosen.
KP_LEG = np.array([100.0, 250.0, 300.0, 300.0, 400.0, 200.0])
KD_LEG = np.array([5.0, 10.0, 12.0, 12.0, 15.0, 8.0])
ARMATURE_LEG = np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])   # reflected rotor inertia (kg*m^2)

JOINT_LIMITS_RAD = np.array([
    [-0.5, 0.5],    # hip yaw
    [-0.5, 0.5],    # hip roll
    [-1.4, 0.7],    # hip pitch
    [0.0, 2.4],     # knee
    [-0.9, 0.9],    # ankle pitch
    [-0.5, 0.5],    # ankle roll
])


@dataclass
class BipedSpec:
    leg: LegParams
    mass: RobotMassParams
    torque_limit_nm: float
    hip_lateral_offset_m: float = 0.06   # gait.step_width_m / 2 in planning/footstep_planner.py

    @property
    def segment_masses(self) -> dict:
        m = self.mass.total_mass_kg
        thigh, shank, foot = THIGH_MASS_FRACTION * m, SHANK_MASS_FRACTION * m, FOOT_MASS_FRACTION * m
        torso = m - 2.0 * (thigh + shank + foot)
        return {"thigh": thigh, "shank": shank, "foot": foot, "torso": torso}


def default_spec() -> BipedSpec:
    return BipedSpec(leg=LegParams(), mass=RobotMassParams(),
                     torque_limit_nm=DYNAMIXEL_H54P_200.continuous_torque_nm)


def _leg_xml(side: str, sign: float, spec: BipedSpec) -> str:
    m = spec.segment_masses
    leg = spec.leg
    lim = JOINT_LIMITS_RAD
    ARM = ARMATURE_LEG
    foot_len, foot_w, foot_h = leg.foot_length_m, leg.foot_width_m, leg.foot_height_m
    com_fwd = 0.35 * foot_len                      # same foot COM placement as dynamics/foot_inertia.py
    box_cx = com_fwd                                # box centered on its COM
    return f"""
      <body name="{side}_thigh" pos="0 {sign * spec.hip_lateral_offset_m:.4f} 0">
        <joint name="{side}_hip_yaw"   type="hinge" axis="0 0 1" range="{lim[0,0]} {lim[0,1]}" damping="0.5" armature="{ARM[0]}"/>
        <joint name="{side}_hip_roll"  type="hinge" axis="1 0 0" range="{lim[1,0]} {lim[1,1]}" damping="0.5" armature="{ARM[1]}"/>
        <joint name="{side}_hip_pitch" type="hinge" axis="0 1 0" range="{lim[2,0]} {lim[2,1]}" damping="0.5" armature="{ARM[2]}"/>
        <geom name="{side}_thigh_geom" type="capsule" fromto="0 0 0 0 0 {-leg.thigh_length_m}" size="0.05"
              mass="{m['thigh']:.5f}" contype="0" conaffinity="0" rgba="0.6 0.6 0.9 1"/>
        <body name="{side}_shank" pos="0 0 {-leg.thigh_length_m}">
          <joint name="{side}_knee" type="hinge" axis="0 1 0" range="{lim[3,0]} {lim[3,1]}" damping="0.5" armature="{ARM[3]}"/>
          <geom name="{side}_shank_geom" type="capsule" fromto="0 0 0 0 0 {-leg.shank_length_m}" size="0.04"
                mass="{m['shank']:.5f}" contype="0" conaffinity="0" rgba="0.5 0.5 0.8 1"/>
          <body name="{side}_foot" pos="0 0 {-leg.shank_length_m}">
            <joint name="{side}_ankle_pitch" type="hinge" axis="0 1 0" range="{lim[4,0]} {lim[4,1]}" damping="0.5" armature="{ARM[4]}"/>
            <joint name="{side}_ankle_roll"  type="hinge" axis="1 0 0" range="{lim[5,0]} {lim[5,1]}" damping="0.5" armature="{ARM[5]}"/>
            <geom name="{side}_foot_geom" type="box" pos="{box_cx:.4f} 0 {-foot_h / 2:.4f}"
                  size="{foot_len / 2:.4f} {foot_w / 2:.4f} {foot_h / 2:.4f}" mass="{m['foot']:.5f}"
                  friction="1.0 0.02 0.001" contype="1" conaffinity="1" rgba="0.3 0.3 0.3 1"/>
            <site name="{side}_foot_site" pos="{box_cx:.4f} 0 {-foot_h:.4f}" size="0.01"/>
          </body>
        </body>
      </body>"""


def build_mjcf(spec: BipedSpec | None = None, timestep: float = 0.002) -> str:
    spec = spec or default_spec()
    m = spec.segment_masses
    tl = spec.torque_limit_nm
    actuators = "\n".join(
        f'    <position name="{side}_{j}_act" joint="{side}_{j}" kp="{KP_LEG[i]}" kv="{KD_LEG[i]}" '
        f'forcelimited="true" forcerange="-{tl} {tl}"/>'
        for side in ("left", "right") for i, j in enumerate(JOINT_NAMES_PER_LEG)
    )
    return f"""<mujoco model="prime_biped">
  <compiler angle="radian" inertiafromgeom="true" balanceinertia="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" friction="1.0 0.02 0.001" contype="1" conaffinity="1"
          rgba="0.8 0.8 0.8 1"/>
    <body name="pelvis" pos="0 0 0.95">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" pos="0 0 0.20" size="0.08 0.11 0.20" mass="{m['torso']:.5f}"
            contype="1" conaffinity="0" rgba="0.9 0.5 0.2 1"/>
      <site name="imu" pos="0 0 0.05" size="0.01"/>{_leg_xml('left', +1.0, spec)}{_leg_xml('right', -1.0, spec)}
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def nominal_standing_pelvis_height(spec: BipedSpec | None = None) -> float:
    """Pelvis height (m) with both feet flat at the nominal pose - used to
    place the robot on the floor at reset."""
    spec = spec or default_spec()
    hip_pitch, knee = NOMINAL_LEG_POSE[2], NOMINAL_LEG_POSE[3]
    thigh_dir_z = np.cos(hip_pitch)
    shank_dir_z = np.cos(hip_pitch + knee)
    return float(spec.leg.thigh_length_m * thigh_dir_z + spec.leg.shank_length_m * shank_dir_z
                 + spec.leg.foot_height_m)
